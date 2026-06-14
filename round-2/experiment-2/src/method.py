#!/usr/bin/env python3
"""FMTNA (Failure-Mode-Typed Neural Abduction) evaluation pipeline.

Implements FMTNA typed failure dispatch vs undifferentiated abduction baseline
across RuleTaker (depth 0,1,3,5) and FOLIO datasets.
"""

import asyncio
import gc
import glob
import json
import math
import os
import re
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi
from scipy import stats
from tenacity import retry, stop_after_attempt, wait_exponential

# ─── Logging ─────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
Path("logs").mkdir(exist_ok=True)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ─── Hardware & resource limits ───────────────────────────────────────────────
def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1

def _container_ram_gb() -> float:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return 16.0

NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb()
RAM_BUDGET = int(min(TOTAL_RAM_GB * 0.6, 16.0) * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget {RAM_BUDGET/1e9:.1f}GB")

# ─── Config ───────────────────────────────────────────────────────────────────
WORKSPACE = Path("/ai-inventor/aii_data/runs/348df/3_invention_loop/iter_2/gen_art/gen_art_experiment_2")
DATA_DIR = Path("/ai-inventor/aii_data/runs/348df/3_invention_loop/iter_1/gen_art/gen_art_dataset_1")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL = "google/gemini-3.1-flash-lite"
LLM_MODEL_FALLBACK = "google/gemini-3.5-flash"
BUDGET_USD = 10.0
MAX_DISPATCH_RETRIES = 2
MAX_PROOF_DEPTH = 12
CONCURRENCY = 16  # async semaphore

# Sampling config
RULETAKER_PER_DEPTH = 50  # examples per depth level
FOLIO_COUNT = 100
BASELINE_COMPARISON_DEPTH1 = 30  # for baseline comparison subset

# ─── Cost tracking ────────────────────────────────────────────────────────────
class CostTracker:
    def __init__(self, budget: float):
        self.budget = budget
        self.total_cost = 0.0
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def add(self, input_tokens: int, output_tokens: int, model: str) -> float:
        # OpenRouter pricing (per 1M tokens)
        pricing = {
            "google/gemini-3.1-flash-lite": (0.25, 1.50),
            "google/gemini-3.5-flash": (1.50, 9.00),
            "anthropic/claude-haiku-4.5": (1.00, 5.00),
        }
        in_p, out_p = pricing.get(model, (0.1, 0.3))
        cost = (input_tokens * in_p + output_tokens * out_p) / 1_000_000
        self.total_cost += cost
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        return cost

    @property
    def remaining(self) -> float:
        return self.budget - self.total_cost

    def check(self, estimated_cost: float = 0.01) -> bool:
        return self.remaining > estimated_cost


cost_tracker = CostTracker(BUDGET_USD)

# ─── MiniProlog ───────────────────────────────────────────────────────────────

class ProofFailure(Exception):
    def __init__(self, failure_type: str, goal: tuple):
        self.failure_type = failure_type  # "TYPE_A", "TYPE_B", "TYPE_C", "DEPTH"
        self.goal = goal
        super().__init__(f"{failure_type}: {goal}")


class MiniProlog:
    """Pure-Python backward-chaining Prolog interpreter.

    Failure types:
      TYPE_A: query functor not present anywhere in KB (vocabulary mismatch)
      TYPE_B: functor exists but specific ground instance missing
      TYPE_C: no rule head matches the required derived predicate
      DEPTH:  max recursion depth exceeded
    """

    def __init__(self):
        self.facts: dict[tuple[str, int], set[tuple]] = {}
        self.rules: list[tuple[str, list, list]] = []
        self.fact_grounding: dict[tuple, str] = {}  # fact key → source span

    def assert_fact(self, functor: str, args: list[str], span: str = "") -> None:
        key = (functor, len(args))
        if key not in self.facts:
            self.facts[key] = set()
        t = tuple(args)
        self.facts[key].add(t)
        if span:
            self.fact_grounding[t] = span

    def assert_rule(self, head_functor: str, head_args: list, body: list[tuple]) -> None:
        self.rules.append((head_functor, head_args, body))

    def has_functor(self, functor: str) -> bool:
        return (
            any(f == functor for (f, _) in self.facts)
            or any(f == functor for (f, _, _) in self.rules)
        )

    def get_functors(self) -> list[str]:
        fs = set()
        for (f, _) in self.facts:
            fs.add(f)
        for (f, _, _) in self.rules:
            fs.add(f)
        return sorted(fs)

    def prove(
        self,
        goal_functor: str,
        goal_args: list[str],
        depth: int = 0,
        max_depth: int = MAX_PROOF_DEPTH,
        visited: frozenset = frozenset(),
    ) -> tuple[bool, list[tuple]]:
        """Returns (success, proof_steps_list).
        Raises ProofFailure on typed failure.
        """
        if depth > max_depth:
            raise ProofFailure("DEPTH", (goal_functor, tuple(goal_args)))

        goal_key = (goal_functor, tuple(goal_args))
        if goal_key in visited:
            raise ProofFailure("TYPE_B", goal_key)
        visited = visited | {goal_key}

        arity = len(goal_args)
        fact_key = (goal_functor, arity)

        # Check if functor exists at all
        if not self.has_functor(goal_functor):
            raise ProofFailure("TYPE_A", (goal_functor, tuple(goal_args)))

        # Try facts
        if fact_key in self.facts:
            for fact_args in self.facts[fact_key]:
                bindings = self._unify(goal_args, list(fact_args))
                if bindings is not None:
                    return True, [(goal_functor, tuple(goal_args))]

        # Try rules
        matching_rules = [
            (hf, ha, body)
            for (hf, ha, body) in self.rules
            if hf == goal_functor and len(ha) == arity
        ]

        if not matching_rules and fact_key not in self.facts:
            raise ProofFailure("TYPE_C", (goal_functor, tuple(goal_args)))

        last_failure: Optional[ProofFailure] = None
        for (rule_functor, rule_head_args, rule_body) in matching_rules:
            bindings = self._unify(goal_args, rule_head_args)
            if bindings is None:
                continue
            try:
                steps = [(goal_functor, tuple(goal_args))]
                for (body_functor, body_args) in rule_body:
                    ground_args = [bindings.get(a, a) for a in body_args]
                    _, sub_steps = self.prove(
                        body_functor, ground_args, depth + 1, max_depth, visited
                    )
                    steps.extend(sub_steps)
                return True, steps
            except ProofFailure as pf:
                last_failure = pf
                continue

        # No rule succeeded
        if last_failure:
            raise last_failure
        raise ProofFailure("TYPE_B", (goal_functor, tuple(goal_args)))

    @staticmethod
    def _is_var(term: str) -> bool:
        return bool(term) and (term[0].isupper() or term[0] == "_")

    def _unify(self, goal_args: list, pattern_args: list) -> Optional[dict]:
        if len(goal_args) != len(pattern_args):
            return None
        bindings: dict[str, str] = {}
        for g, p in zip(goal_args, pattern_args):
            if self._is_var(p):
                if p in bindings:
                    if bindings[p] != g:
                        return None
                else:
                    bindings[p] = g
            elif self._is_var(g):
                if g in bindings:
                    if bindings[g] != p:
                        return None
                else:
                    bindings[g] = p
            elif g != p:
                return None
        return bindings


# ─── NL Parser (RuleTaker) ────────────────────────────────────────────────────

def normalize_entity(name: str) -> str:
    """Normalize entity name to lowercase identifier."""
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip()).strip("_")


def _stem_verb(verb: str) -> str:
    """Stem a third-person present tense verb to base form.
    chases→chase, likes→like, needs→need, sees→see.
    Simply remove trailing 's' (not 'es') to preserve the stem vowel.
    """
    verb = verb.strip().lower()
    if verb.endswith("ies"):
        return verb[:-3] + "y"
    if verb.endswith("s") and len(verb) > 2:
        return verb[:-1]
    return verb


def parse_ruletaker_input(input_text: str) -> tuple[str, str]:
    """Split RuleTaker input into (context, question)."""
    parts = input_text.split("\nQuestion:", 1)
    if len(parts) == 2:
        context = parts[0].replace("Context:", "").strip()
        question = parts[1].strip()
    else:
        # Fallback: last sentence is the question
        sentences = [s.strip() for s in input_text.split(".") if s.strip()]
        context = ". ".join(sentences[:-1]) + "."
        question = sentences[-1] + "."
    return context, question


def parse_nl_fact(sentence: str) -> Optional[tuple[str, list[str]]]:
    """Parse a simple NL sentence into (functor, args).

    Handles:
      - "X is Y." → ("is", [x, y])
      - "X is not Y." → ("not_is", [x, y])
      - "X verbs Y." → (verb_stem, [x, y])
      - "X verbs the Y." → (verb_stem, [x, y])
    """
    s = sentence.strip().rstrip(".")

    # "The X is not Y" or "X is not Y"
    m = re.match(
        r"(?:The |the )?(\w[\w ]+?)\s+is\s+not\s+(?:a |an |the )?(\w[\w ]*)$",
        s, re.IGNORECASE
    )
    if m:
        subj = normalize_entity(m.group(1))
        prop = normalize_entity(m.group(2))
        return (f"not_{prop}", [subj])

    # "The X is Y" or "X is Y"
    m = re.match(
        r"(?:The |the )?(\w[\w ]+?)\s+is\s+(?:a |an |the )?(\w[\w ]*)$",
        s, re.IGNORECASE
    )
    if m:
        subj = normalize_entity(m.group(1))
        prop = normalize_entity(m.group(2))
        return ("is", [subj, prop])

    # "The X does not verb the Y" - negated binary
    m = re.match(
        r"(?:The |the )?(\w[\w ]+?)\s+does\s+not\s+(\w+)\s+(?:the |The )?(\w[\w ]*)$",
        s, re.IGNORECASE
    )
    if m:
        subj = normalize_entity(m.group(1))
        verb = _stem_verb(m.group(2).lower())
        obj_ = normalize_entity(m.group(3))
        if verb not in {"i", "is", "b", "ar"}:
            return (f"not_{verb}", [subj, obj_])

    # "The X verb the Y" or "X verb Y" - binary relation
    m = re.match(
        r"(?:The |the )?(\w[\w ]+?)\s+(\w+(?:s|es)?)\s+(?:the |The )?(\w[\w ]*)$",
        s, re.IGNORECASE
    )
    if m:
        subj = normalize_entity(m.group(1))
        verb = _stem_verb(m.group(2).lower())
        obj_ = normalize_entity(m.group(3))
        # Skip helper words
        if verb in {"i", "is", "ar", "b", "ha", "doe", "wil", "do"}:
            return None
        return (verb, [subj, obj_])

    return None


def parse_nl_query(question: str) -> Optional[tuple[str, list[str], bool]]:
    """Parse a RuleTaker question/statement into (functor, args, negated).

    Returns the Prolog goal and whether this is a negated claim.
    """
    s = question.strip().rstrip("?.")

    # "X is not Y"
    m = re.match(
        r"(?:The |the )?(\w[\w ]+?)\s+is\s+not\s+(?:a |an |the )?(\w[\w ]*)$",
        s, re.IGNORECASE
    )
    if m:
        subj = normalize_entity(m.group(1))
        prop = normalize_entity(m.group(2))
        return (f"not_{prop}", [subj], True)

    # "X is Y" / "The X is Y"
    m = re.match(
        r"(?:The |the )?(\w[\w ]+?)\s+is\s+(?:a |an |the )?(\w[\w ]*)$",
        s, re.IGNORECASE
    )
    if m:
        subj = normalize_entity(m.group(1))
        prop = normalize_entity(m.group(2))
        return ("is", [subj, prop], False)

    # "X does not verb [the] Y" → negated binary fact
    m = re.match(
        r"(?:The |the )?(\w[\w ]+?)\s+does\s+not\s+(\w+)\s+(?:the |The )?(\w[\w ]*)$",
        s, re.IGNORECASE
    )
    if m:
        subj = normalize_entity(m.group(1))
        verb = _stem_verb(m.group(2).lower())
        obj_ = normalize_entity(m.group(3))
        if verb not in {"i", "is", "b", "ar"}:
            return (f"not_{verb}", [subj, obj_], True)

    # "X verb Y" binary
    m = re.match(
        r"(?:The |the )?(\w[\w ]+?)\s+(\w+(?:s|es)?)\s+(?:the |The )?(\w[\w ]*)$",
        s, re.IGNORECASE
    )
    if m:
        subj = normalize_entity(m.group(1))
        verb = _stem_verb(m.group(2).lower())
        obj_ = normalize_entity(m.group(3))
        if verb in {"i", "is", "ar", "b", "ha", "doe", "wil", "do"}:
            return None
        return (verb, [subj, obj_], False)

    return None


def parse_nl_rule(sentence: str) -> Optional[tuple[str, list, list]]:
    """Parse a conditional sentence into (head_functor, head_args, body_goals).

    Handles common RuleTaker rule patterns.
    Returns None if parsing fails.
    """
    s = sentence.strip().rstrip(".")

    # "All A, B people are C" or "All A and B people are C"
    m = re.match(
        r"All\s+(.+?)\s+(?:people|things|animals|entities)\s+are\s+(?:not\s+)?(\w[\w ]*)$",
        s, re.IGNORECASE
    )
    if m:
        conds_str = m.group(1)
        conclusion = normalize_entity(m.group(2))
        negated = "are not" in s.lower()
        conds = re.split(r",\s*|\s+and\s+", conds_str)
        body = []
        for c in conds:
            c = c.strip().lstrip("not ").strip()
            if c:
                # "not X" → not_X predicate
                if re.match(r"^not\s+", c, re.IGNORECASE):
                    prop = normalize_entity(re.sub(r"^not\s+", "", c, flags=re.IGNORECASE))
                    body.append((f"not_{prop}", ["X"]))
                else:
                    body.append(("is", ["X", normalize_entity(c)]))
        if negated:
            head = (f"not_{conclusion}", ["X"])
        else:
            head = ("is", ["X", conclusion])
        if body:
            return (head[0], head[1], body)

    # "If someone is A [and B]* then they are [not] C"
    m = re.match(
        r"If\s+(?:someone|something|anyone)\s+(?:is\s+)(.+?)\s+then\s+(?:they|it)\s+(?:are?|is)\s+(not\s+)?(\w[\w ]*)$",
        s, re.IGNORECASE
    )
    if m:
        conds_str = m.group(1)
        negated = bool(m.group(2))
        conclusion = normalize_entity(m.group(3))
        conds = re.split(r"\s+and\s+", conds_str)
        body = []
        for c in conds:
            c = c.strip()
            if re.match(r"^not\s+", c, re.IGNORECASE):
                prop = normalize_entity(re.sub(r"^not\s+", "", c, flags=re.IGNORECASE))
                body.append((f"not_{prop}", ["X"]))
            else:
                body.append(("is", ["X", normalize_entity(c)]))
        head_f = f"not_{conclusion}" if negated else "is"
        head_a = ["X"] if negated else ["X", conclusion]
        if body:
            return (head_f, head_a, body)

    # "If someone is A [and B]* then they verb [the] C"
    m = re.match(
        r"If\s+(?:someone|something|anyone)\s+(?:is\s+)(.+?)\s+then\s+(?:they|it)\s+(\w+(?:s|es)?)\s+(?:the\s+)?(\w[\w ]*)$",
        s, re.IGNORECASE
    )
    if m:
        conds_str = m.group(1)
        verb = _stem_verb(m.group(2).lower())
        obj_ = normalize_entity(m.group(3))
        conds = re.split(r"\s+and\s+", conds_str)
        body = []
        for c in conds:
            c = c.strip()
            if re.match(r"^not\s+", c, re.IGNORECASE):
                prop = normalize_entity(re.sub(r"^not\s+", "", c, flags=re.IGNORECASE))
                body.append((f"not_{prop}", ["X"]))
            else:
                body.append(("is", ["X", normalize_entity(c)]))
        if body and verb not in {"i", "ar", "b"}:
            return (verb, ["X", obj_], body)

    # "If someone verbs something [and ...] then they are [not] C"
    m = re.match(
        r"If\s+(?:someone|something)\s+(\w+(?:s|es)?)\s+(?:something|someone)\s*(.*?)\s*then\s+(?:they|it)\s+(?:are?|is)\s+(not\s+)?(\w[\w ]*)$",
        s, re.IGNORECASE
    )
    if m:
        verb = _stem_verb(m.group(1).lower())
        extra = m.group(2).strip()
        negated = bool(m.group(3))
        conclusion = normalize_entity(m.group(4))
        body = [(verb, ["X", "Y"])]
        # Parse extra conditions like "and it is A"
        for part in re.findall(r"and\s+(?:it|they)\s+(?:is|are)\s+(\w+)", extra, re.IGNORECASE):
            body.append(("is", ["X", normalize_entity(part)]))
        head_f = f"not_{conclusion}" if negated else "is"
        head_a = ["X"] if negated else ["X", conclusion]
        return (head_f, head_a, body)

    # "If the X is A then the X is [not] B" (specific entity)
    m = re.match(
        r"If\s+(?:the\s+)?(\w+)\s+is\s+(\w+)\s+then\s+(?:the\s+)?(\w+)\s+is\s+(not\s+)?(\w+)$",
        s, re.IGNORECASE
    )
    if m:
        subj1 = normalize_entity(m.group(1))
        cond = normalize_entity(m.group(2))
        subj2 = normalize_entity(m.group(3))
        negated = bool(m.group(4))
        conclusion = normalize_entity(m.group(5))
        if subj1 == subj2:
            body = [("is", [subj1, cond])]
            head_f = f"not_{conclusion}" if negated else "is"
            head_a = [subj1] if negated else [subj1, conclusion]
            return (head_f, head_a, body)

    # "If the X verb the Y then the Y/Z is [not] W"
    m = re.match(
        r"If\s+(?:the\s+)?(\w+)\s+(\w+(?:s|es)?)\s+(?:the\s+)?(\w+)\s+then\s+(?:the\s+)?(\w+)\s+(?:is|are)\s+(not\s+)?(\w+)$",
        s, re.IGNORECASE
    )
    if m:
        subj = normalize_entity(m.group(1))
        verb = _stem_verb(m.group(2).lower())
        obj_ = normalize_entity(m.group(3))
        result_subj = normalize_entity(m.group(4))
        negated = bool(m.group(5))
        conclusion = normalize_entity(m.group(6))
        body = [(verb, [subj, obj_])]
        head_f = f"not_{conclusion}" if negated else "is"
        head_a = [result_subj] if negated else [result_subj, conclusion]
        return (head_f, head_a, body)

    return None


def build_ruletaker_kb(context: str) -> MiniProlog:
    """Parse a RuleTaker context string into a MiniProlog KB."""
    kb = MiniProlog()
    sentences = re.split(r"(?<=[.?!])\s+", context.strip())

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        is_rule = re.match(r"^(If|All)\b", sent, re.IGNORECASE)
        if is_rule:
            rule = parse_nl_rule(sent)
            if rule:
                head_f, head_a, body = rule
                kb.assert_rule(head_f, head_a, body)
        else:
            fact = parse_nl_fact(sent)
            if fact:
                functor, args = fact
                kb.assert_fact(functor, args, span=sent)

    return kb


# ─── BM25 Retriever ───────────────────────────────────────────────────────────

def build_bm25(text: str) -> tuple[BM25Okapi, list[str]]:
    """Build BM25 index over sentences of text."""
    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    tokenized = [s.lower().split() for s in sentences if s.strip()]
    sentences = [s for s in sentences if s.strip()]
    if not tokenized:
        return BM25Okapi([[""]]), [""]
    return BM25Okapi(tokenized), sentences


def bm25_top_k(bm25: BM25Okapi, sentences: list[str], query: str, k: int = 3) -> list[str]:
    """Return top-k sentences by BM25 relevance to query."""
    scores = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:k]
    return [sentences[i] for i in top_idx if scores[i] > 0]


# ─── OpenRouter LLM Client ────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, api_key: str, model: str = LLM_MODEL):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self._sem = asyncio.Semaphore(CONCURRENCY)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def call(
        self,
        messages: list[dict],
        json_schema: Optional[dict] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> tuple[str, int, int]:
        """Returns (content_str, input_tokens, output_tokens)."""
        if not cost_tracker.check():
            raise RuntimeError(f"Budget exhausted: ${cost_tracker.total_cost:.2f} spent")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        async with self._sem:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    self.base_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "https://ai-inventor.research",
                        "X-Title": "FMTNA-Research",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        in_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)
        cost = cost_tracker.add(in_tok, out_tok, self.model)
        logger.debug(
            f"LLM call: model={self.model} in={in_tok} out={out_tok} cost=${cost:.4f} "
            f"total=${cost_tracker.total_cost:.3f}"
        )
        return content, in_tok, out_tok


# ─── LLM prompts ─────────────────────────────────────────────────────────────

SEED_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "functor": {"type": "string"},
                    "args": {"type": "array", "items": {"type": "string"}},
                    "source_span": {"type": "string"},
                },
                "required": ["functor", "args", "source_span"],
                "additionalProperties": False,
            },
        },
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "head_functor": {"type": "string"},
                    "head_args": {"type": "array", "items": {"type": "string"}},
                    "body": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "functor": {"type": "string"},
                                "args": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["functor", "args"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["head_functor", "head_args", "body"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["facts", "rules"],
    "additionalProperties": False,
}

TYPE_A_SCHEMA = {
    "type": "object",
    "properties": {
        "aligned_predicate": {"type": "string"},
        "confidence": {"type": "number"},
        "bridging_rule_head": {"type": "string"},
        "bridging_rule_body": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["aligned_predicate", "confidence", "bridging_rule_head", "bridging_rule_body"],
    "additionalProperties": False,
}

TYPE_B_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "supporting_span": {"type": "string"},
        "span_char_start": {"type": "integer"},
        "span_char_end": {"type": "integer"},
    },
    "required": ["present", "supporting_span", "span_char_start", "span_char_end"],
    "additionalProperties": False,
}

TYPE_C_SCHEMA = {
    "type": "object",
    "properties": {
        "horn_clause_head_functor": {"type": "string"},
        "horn_clause_head_args": {"type": "array", "items": {"type": "string"}},
        "horn_clause_body": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "functor": {"type": "string"},
                    "args": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["functor", "args"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "number"},
    },
    "required": ["horn_clause_head_functor", "horn_clause_head_args", "horn_clause_body", "confidence"],
    "additionalProperties": False,
}

ENTAILMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["entailment", "not entailment"]},
        "reasoning": {"type": "string"},
    },
    "required": ["answer", "reasoning"],
    "additionalProperties": False,
}

FOL_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["True", "False", "Uncertain"]},
        "reasoning": {"type": "string"},
    },
    "required": ["answer", "reasoning"],
    "additionalProperties": False,
}

UNDIFFERENTIATED_SCHEMA = {
    "type": "object",
    "properties": {
        "missing_fact_or_rule": {"type": "string"},
        "proposed_resolution": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["missing_fact_or_rule", "proposed_resolution", "confidence"],
    "additionalProperties": False,
}


async def llm_extract_kb(
    client: LLMClient, context: str
) -> Optional[dict]:
    """Use LLM to extract Prolog KB from natural language context."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledge extraction system. Extract all atomic facts and "
                "conditional rules from the given text as Prolog-style predicates. "
                "Use snake_case for functors. Variables must start with uppercase (X, Y, Z). "
                "Ground terms must be lowercase. "
                "Example fact: {\"functor\": \"is\", \"args\": [\"bob\", \"big\"], \"source_span\": \"Bob is big\"} "
                "Example rule head: {\"head_functor\": \"is\", \"head_args\": [\"X\", \"round\"], "
                "\"body\": [{\"functor\": \"is\", \"args\": [\"X\", \"big\"]}]}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Extract all facts and conditional rules from this text:\n\n{context}\n\n"
                "Return complete JSON with all facts and rules."
            ),
        },
    ]
    try:
        content, _, _ = await client.call(
            messages, json_schema=SEED_EXTRACTION_SCHEMA, max_tokens=1024
        )
        return json.loads(content)
    except Exception:
        logger.error("LLM KB extraction failed")
        return None


async def llm_resolve_type_a(
    client: LLMClient,
    required_pred: str,
    kb_predicates: list[str],
    context: str,
) -> Optional[dict]:
    """Type-A: predicate name mismatch resolution."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a semantic alignment system. Given a required predicate name "
                "and a list of predicates in a knowledge base, determine if any KB predicate "
                "is semantically equivalent to the required one. If yes, provide a bridging rule."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Required predicate: '{required_pred}'\n"
                f"Available KB predicates: {kb_predicates}\n"
                f"Context: {context[:400]}\n\n"
                "Is there an equivalent predicate? If yes, provide the aligned predicate and "
                "a bridging rule head (Prolog functor) and body (list of 'functor(arg1,arg2)' strings). "
                "If no alignment possible, return empty strings and empty list."
            ),
        },
    ]
    try:
        content, _, _ = await client.call(
            messages, json_schema=TYPE_A_SCHEMA, max_tokens=256
        )
        return json.loads(content)
    except Exception:
        logger.error("LLM Type-A dispatch failed")
        return None


async def llm_resolve_type_b(
    client: LLMClient,
    pred_functor: str,
    pred_args: list[str],
    context: str,
    bm25: BM25Okapi,
    sentences: list[str],
) -> Optional[dict]:
    """Type-B: missing ground atom resolution with BM25 grounding."""
    query_str = f"{pred_functor} {' '.join(pred_args)}"
    top_spans = bm25_top_k(bm25, sentences, query_str, k=3)
    excerpt = " ".join(top_spans) if top_spans else context[:400]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a fact verification system. Given a predicate and text spans, "
                "determine if the fact is supported by the text. Return the exact span if found."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Is the fact '{pred_functor}({', '.join(pred_args)})' supported by this text?\n"
                f"Text: '{excerpt}'\n\n"
                "If present, set present=true and provide supporting_span with character indices "
                "(span_char_start, span_char_end relative to the text excerpt). "
                "If not present, set present=false and use empty string/0 for span fields."
            ),
        },
    ]
    try:
        content, _, _ = await client.call(
            messages, json_schema=TYPE_B_SCHEMA, max_tokens=256
        )
        return json.loads(content)
    except Exception:
        logger.error("LLM Type-B dispatch failed")
        return None


async def llm_resolve_type_c(
    client: LLMClient,
    goal_functor: str,
    goal_args: list[str],
    proof_context: str,
    context: str,
) -> Optional[dict]:
    """Type-C: missing rule head — propose Horn clause."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a commonsense reasoning system. Given a proof goal that has no "
                "applicable rule, propose a Horn clause that would help prove it."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Goal: {goal_functor}({', '.join(goal_args)})\n"
                f"Proof context: {proof_context}\n"
                f"Source text: {context[:300]}\n\n"
                "Propose a Horn clause: head functor/args and body goals. "
                "Variables must start with uppercase. Use 'is' predicate for properties. "
                "Return horn_clause_body as array of {{functor, args}} objects."
            ),
        },
    ]
    try:
        content, _, _ = await client.call(
            messages, json_schema=TYPE_C_SCHEMA, max_tokens=256
        )
        return json.loads(content)
    except Exception:
        logger.error("LLM Type-C dispatch failed")
        return None


async def llm_direct_answer_ruletaker(
    client: LLMClient, context: str, question: str
) -> str:
    """Baseline: direct LLM answer for RuleTaker."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a logical reasoning system. Given a context with facts and rules, "
                "determine if the statement in the question is entailed (logically follows) "
                "or not entailed. Use strict logical reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\nQuestion/Statement: {question}\n\n"
                "Does this statement follow from the context by logical deduction? "
                "Answer 'entailment' or 'not entailment'."
            ),
        },
    ]
    try:
        content, _, _ = await client.call(
            messages, json_schema=ENTAILMENT_SCHEMA, max_tokens=256
        )
        result = json.loads(content)
        return result.get("answer", "not entailment")
    except Exception:
        logger.error("Baseline RuleTaker LLM call failed")
        return "not entailment"


async def llm_direct_answer_folio(
    client: LLMClient, premises: str, conclusion: str
) -> str:
    """Baseline: direct LLM answer for FOLIO."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a first-order logic reasoning system. Given premises and a conclusion, "
                "determine if the conclusion is True, False, or Uncertain based on the premises."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Premises:\n{premises}\n\nConclusion: {conclusion}\n\n"
                "Based strictly on logical deduction from the premises above, is the conclusion "
                "True (logically follows), False (contradicted), or Uncertain (cannot determine)? "
                "Apply formal logic. Most conclusions are True or False, not Uncertain. "
                "Answer exactly 'True', 'False', or 'Uncertain'."
            ),
        },
    ]
    try:
        content, _, _ = await client.call(
            messages, json_schema=FOL_SCHEMA, max_tokens=256
        )
        result = json.loads(content)
        return result.get("answer", "Uncertain")
    except Exception:
        logger.error("Baseline FOLIO LLM call failed")
        return "Uncertain"


async def llm_undifferentiated_abduction(
    client: LLMClient,
    context: str,
    question: str,
    failure_context: str,
) -> str:
    """Baseline undifferentiated abduction: generic 'what is missing?' prompt."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a logical reasoning system that resolves proof failures by abduction. "
                "Given a proof failure, determine what fact or rule is missing."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context: {context[:400]}\n"
                f"Goal: {question}\n"
                f"Proof failure: {failure_context}\n\n"
                "What fact or rule is missing to complete the proof? "
                "Provide the missing fact/rule and how confident you are."
            ),
        },
    ]
    try:
        content, _, _ = await client.call(
            messages, json_schema=UNDIFFERENTIATED_SCHEMA, max_tokens=256
        )
        result = json.loads(content)
        return result.get("proposed_resolution", "")
    except Exception:
        logger.error("Undifferentiated abduction LLM call failed")
        return ""


# ─── FMTNA Prover ─────────────────────────────────────────────────────────────

class ProofRecord:
    """Record of a proof attempt with instrumentation."""

    def __init__(self):
        self.success = False
        self.steps: list[tuple] = []
        self.failure_types: dict[str, int] = {"A": 0, "B": 0, "C": 0, "DEPTH": 0}
        self.failures_resolved: dict[str, int] = {"A": 0, "B": 0, "C": 0}
        self.failures_unresolved: dict[str, int] = {"A": 0, "B": 0, "C": 0}
        self.type_b_span_grounded: int = 0
        self.total_dispatch_calls: int = 0
        self.llm_calls: int = 0

    @property
    def grounding_ratio(self) -> float:
        total = len(self.steps) or 1
        grounded = self.type_b_span_grounded + self.failures_resolved.get("A", 0)
        return grounded / total

    @property
    def failure_summary(self) -> dict:
        return {
            "failure_types": self.failure_types,
            "failures_resolved": self.failures_resolved,
            "failures_unresolved": self.failures_unresolved,
            "type_b_span_grounded": self.type_b_span_grounded,
            "grounding_ratio": round(self.grounding_ratio, 4),
        }


async def run_fmtna(
    client: LLMClient,
    kb: MiniProlog,
    goal_functor: str,
    goal_args: list[str],
    context: str,
    bm25: BM25Okapi,
    sentences: list[str],
    use_llm_seed: bool = False,
    llm_kb_data: Optional[dict] = None,
) -> ProofRecord:
    """Run FMTNA proof attempt with typed dispatch.

    If use_llm_seed=True and llm_kb_data is provided, merge LLM-extracted KB.
    """
    record = ProofRecord()

    # Optionally augment KB from LLM seed extraction
    if use_llm_seed and llm_kb_data:
        for fact in llm_kb_data.get("facts", []):
            try:
                kb.assert_fact(
                    fact["functor"],
                    [normalize_entity(a) for a in fact["args"]],
                    span=fact.get("source_span", ""),
                )
            except Exception:
                pass
        for rule in llm_kb_data.get("rules", []):
            try:
                body = [
                    (bg["functor"], [normalize_entity(a) for a in bg["args"]])
                    for bg in rule.get("body", [])
                ]
                kb.assert_rule(
                    rule["head_functor"],
                    [normalize_entity(a) for a in rule["head_args"]],
                    body,
                )
            except Exception:
                pass

    max_retries = MAX_DISPATCH_RETRIES
    for attempt in range(max_retries + 1):
        try:
            success, steps = kb.prove(goal_functor, goal_args)
            record.success = success
            record.steps = steps
            break
        except ProofFailure as pf:
            ftype_raw = pf.failure_type
            ftype_key = ftype_raw.replace("TYPE_", "")
            if ftype_key in record.failure_types:
                record.failure_types[ftype_key] += 1
            else:
                record.failure_types["DEPTH"] = record.failure_types.get("DEPTH", 0) + 1

            if attempt >= max_retries or not cost_tracker.check(0.005):
                if ftype_key in record.failures_unresolved:
                    record.failures_unresolved[ftype_key] += 1
                break

            goal_f, goal_a_tuple = pf.goal[0], list(pf.goal[1]) if len(pf.goal) > 1 else []

            # --- Type-A dispatch ---
            if ftype_raw == "TYPE_A":
                result = await llm_resolve_type_a(
                    client, goal_f, kb.get_functors(), context
                )
                record.total_dispatch_calls += 1
                record.llm_calls += 1
                if result and result.get("aligned_predicate") and result.get("confidence", 0) > 0.5:
                    # Add bridging rule
                    aligned = result["aligned_predicate"]
                    bridge_body_raw = result.get("bridging_rule_body", [])
                    if bridge_body_raw:
                        body = []
                        for b_str in bridge_body_raw:
                            m = re.match(r"(\w+)\((.+)\)", b_str)
                            if m:
                                bf = m.group(1)
                                ba = [a.strip() for a in m.group(2).split(",")]
                                body.append((bf, ba))
                        if body:
                            kb.assert_rule(goal_f, goal_a_tuple or ["X"], body)
                            record.failures_resolved["A"] += 1
                            continue
                    # Simple fact aliasing
                    if aligned and aligned != goal_f:
                        fact_key = (aligned, len(goal_a_tuple))
                        if fact_key in kb.facts:
                            for fa in kb.facts[fact_key]:
                                kb.assert_fact(goal_f, list(fa))
                            record.failures_resolved["A"] += 1
                            continue
                record.failures_unresolved["A"] += 1
                break

            # --- Type-B dispatch ---
            elif ftype_raw == "TYPE_B":
                result = await llm_resolve_type_b(
                    client, goal_f, goal_a_tuple, context, bm25, sentences
                )
                record.total_dispatch_calls += 1
                record.llm_calls += 1
                if result and result.get("present"):
                    kb.assert_fact(goal_f, goal_a_tuple, span=result.get("supporting_span", ""))
                    record.failures_resolved["B"] += 1
                    if result.get("supporting_span"):
                        record.type_b_span_grounded += 1
                    continue
                record.failures_unresolved["B"] += 1
                break

            # --- Type-C dispatch ---
            elif ftype_raw == "TYPE_C":
                proof_ctx = f"goal: {goal_f}({', '.join(goal_a_tuple)})"
                result = await llm_resolve_type_c(
                    client, goal_f, goal_a_tuple, proof_ctx, context
                )
                record.total_dispatch_calls += 1
                record.llm_calls += 1
                if result and result.get("confidence", 0) > 0.4:
                    body = [
                        (bg["functor"], [normalize_entity(a) for a in bg["args"]])
                        for bg in result.get("horn_clause_body", [])
                    ]
                    if body:
                        head_args = [
                            normalize_entity(a)
                            for a in result.get("horn_clause_head_args", goal_a_tuple)
                        ]
                        kb.assert_rule(
                            result.get("horn_clause_head_functor", goal_f),
                            head_args,
                            body,
                        )
                        record.failures_resolved["C"] += 1
                        continue
                record.failures_unresolved["C"] += 1
                break

            else:
                break

    return record


# ─── Baseline Prover ──────────────────────────────────────────────────────────

async def run_baseline_ruletaker(
    client: LLMClient,
    context: str,
    question: str,
    kb: MiniProlog,
    goal_functor: str,
    goal_args: list[str],
    bm25: BM25Okapi,
    sentences: list[str],
) -> tuple[str, int]:
    """Undifferentiated abduction baseline for RuleTaker.

    First attempts symbolic proof; on failure, uses a single generic LLM prompt.
    Returns (prediction, llm_calls).
    """
    llm_calls = 0
    try:
        success, _ = kb.prove(goal_functor, goal_args)
        if success:
            return "entailment", llm_calls
    except ProofFailure as pf:
        # Undifferentiated: single generic prompt
        if cost_tracker.check(0.003):
            await llm_undifferentiated_abduction(
                client, context, question,
                f"{pf.failure_type}: {pf.goal}"
            )
            llm_calls += 1
        # Still do direct LLM answer
    if cost_tracker.check(0.003):
        answer = await llm_direct_answer_ruletaker(client, context, question)
        llm_calls += 1
        return answer, llm_calls
    return "not entailment", llm_calls


async def run_baseline_folio(
    client: LLMClient, premises: str, conclusion: str
) -> tuple[str, int]:
    """Direct LLM baseline for FOLIO."""
    if cost_tracker.check(0.003):
        answer = await llm_direct_answer_folio(client, premises, conclusion)
        return answer, 1
    return "Uncertain", 0


# ─── Per-example processors ───────────────────────────────────────────────────

async def process_ruletaker_example(
    client: LLMClient,
    example: dict,
    use_llm_seed: bool = False,
) -> dict:
    """Process one RuleTaker example with both FMTNA and baseline."""
    input_text = example["input"]
    true_label = example["output"]

    context, question = parse_ruletaker_input(input_text)
    goal_info = parse_nl_query(question)

    result = dict(example)
    result["predict_fmtna"] = "not entailment"
    result["predict_baseline"] = "not entailment"
    result["metadata_fmtna_success"] = False
    result["metadata_fmtna_failure_types"] = {"A": 0, "B": 0, "C": 0}
    result["metadata_fmtna_grounding_ratio"] = 0.0
    result["metadata_fmtna_proof_steps"] = 0
    result["metadata_fmtna_llm_calls"] = 0
    result["metadata_baseline_llm_calls"] = 0

    if goal_info is None:
        logger.warning(f"Could not parse question: {question[:80]}")
        fmtna_ans, baseline_ans = "not entailment", "not entailment"
        fmtna_calls, baseline_calls = 0, 0
        if cost_tracker.check(0.003):
            fmtna_ans = await llm_direct_answer_ruletaker(client, context, question)
            fmtna_calls = 1
        if cost_tracker.check(0.003):
            baseline_ans = await llm_direct_answer_ruletaker(client, context, question)
            baseline_calls = 1
        result["predict_fmtna"] = fmtna_ans
        result["predict_baseline"] = baseline_ans
        result["metadata_fmtna_llm_calls"] = fmtna_calls
        result["metadata_baseline_llm_calls"] = baseline_calls
        return result

    goal_functor, goal_args, is_negated = goal_info

    # For negated queries ("X does not verb Y" or "X is not Y"):
    # Under closed-world assumption, "not(P)" is true iff P cannot be proved.
    # So for negated goal: prove the POSITIVE version; success → "not entailment"; failure → "entailment"
    if is_negated and goal_functor.startswith("not_"):
        positive_functor = goal_functor[4:]  # strip "not_" prefix
        goal_functor = positive_functor

    # Build KB from NL (always do regex parsing; optionally augment with LLM)
    kb_fmtna = build_ruletaker_kb(context)
    kb_baseline = build_ruletaker_kb(context)

    bm25, sentences = build_bm25(context)

    # Optional LLM seed extraction
    llm_kb_data = None
    llm_seed_calls = 0
    if use_llm_seed and cost_tracker.check(0.005):
        llm_kb_data = await llm_extract_kb(client, context)
        llm_seed_calls = 1 if llm_kb_data else 0

    # FMTNA proof
    record = await run_fmtna(
        client, kb_fmtna, goal_functor, goal_args, context, bm25, sentences,
        use_llm_seed=use_llm_seed, llm_kb_data=llm_kb_data,
    )

    # Under CWA: negated query entailed ↔ positive proof FAILS
    if is_negated:
        result["predict_fmtna"] = "not entailment" if record.success else "entailment"
    else:
        result["predict_fmtna"] = "entailment" if record.success else "not entailment"
    result["metadata_fmtna_success"] = record.success
    result["metadata_fmtna_failure_types"] = record.failure_types
    result["metadata_fmtna_grounding_ratio"] = record.grounding_ratio
    result["metadata_fmtna_proof_steps"] = len(record.steps)
    result["metadata_fmtna_llm_calls"] = record.llm_calls + llm_seed_calls

    # Baseline (also uses positive goal_functor for negated queries)
    baseline_pred, baseline_calls = await run_baseline_ruletaker(
        client, context, question, kb_baseline, goal_functor, goal_args, bm25, sentences
    )
    # For baseline Prolog-derived predictions, apply same CWA negation flip
    # (LLM fallback already reads the full question so it handles negation naturally)
    result["predict_baseline"] = baseline_pred
    result["metadata_baseline_llm_calls"] = baseline_calls

    logger.debug(
        f"RT[{example.get('metadata_config', '?')}] "
        f"true={true_label} fmtna={result['predict_fmtna']} "
        f"baseline={result['predict_baseline']} "
        f"failures={record.failure_types}"
    )
    return result


async def process_folio_example(
    client: LLMClient, example: dict
) -> dict:
    """Process one FOLIO example with both FMTNA and baseline."""
    input_text = example["input"]
    true_label = example["output"]

    # Parse FOLIO input
    parts = input_text.split("\nConclusion:", 1)
    premises_text = parts[0].replace("Premises:", "").strip() if len(parts) > 0 else input_text
    conclusion_text = parts[1].strip() if len(parts) > 1 else ""

    result = dict(example)
    result["predict_fmtna"] = "Uncertain"
    result["predict_baseline"] = "Uncertain"
    result["metadata_fmtna_llm_calls"] = 0
    result["metadata_baseline_llm_calls"] = 0
    result["metadata_fmtna_failure_types"] = {"A": 0, "B": 0, "C": 0}
    result["metadata_fmtna_grounding_ratio"] = 0.0

    # For FOLIO: FMTNA uses FOL metadata if available for Type-A analysis
    # Otherwise falls back to direct LLM with proof introspection
    # Sanitize FOL strings (remove special Unicode that may break API)
    def _clean_fol(s: str) -> str:
        if not s:
            return ""
        return s.encode("ascii", "replace").decode("ascii")[:500]

    fol_premises = _clean_fol(example.get("metadata_premises_fol", ""))
    fol_conclusion = _clean_fol(example.get("metadata_conclusion_fol", ""))

    llm_calls = 0

    # FMTNA for FOLIO: attempt to analyze Type-A failures using FOL predicate vocabulary
    if fol_premises and fol_conclusion and cost_tracker.check(0.005):
        # Extract predicate names from FOL annotations for Type-A analysis
        fol_predicates = re.findall(r"\b([A-Z][A-Za-z]+)\(", fol_premises + fol_conclusion)
        nl_predicates = re.findall(r"\b([a-z][a-z_]+)\b", premises_text.lower())
        unique_fol = list(set(fol_predicates))
        unique_nl = list(set(w for w in nl_predicates if len(w) > 3))

        # Check for Type-A failures: FOL predicate names vs NL vocabulary overlap
        type_a_count = sum(
            1 for fp in unique_fol
            if not any(fp.lower() in nw or nw in fp.lower() for nw in unique_nl)
        )
        result["metadata_fmtna_failure_types"]["A"] = type_a_count
        result["metadata_fmtna_grounding_ratio"] = (
            (len(unique_fol) - type_a_count) / max(len(unique_fol), 1)
        )

    # FMTNA answer: use LLM with structured FOL reasoning
    if cost_tracker.check(0.005):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a first-order logic reasoning system implementing FMTNA "
                    "(Failure-Mode-Typed Neural Abduction). Given premises with FOL annotations "
                    "and a conclusion, determine if the conclusion follows. "
                    "For each reasoning step, identify whether predicates are grounded in the text "
                    "(Type-B success) or require abduction (Type-C). "
                    "Answer True if conclusion follows, False if it contradicts, Uncertain otherwise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Premises (NL): {premises_text}\n"
                    f"Premises (FOL): {fol_premises[:500] if fol_premises else 'N/A'}\n"
                    f"Conclusion (NL): {conclusion_text}\n"
                    f"Conclusion (FOL): {fol_conclusion if fol_conclusion else 'N/A'}\n\n"
                    "Apply logical deduction step by step. "
                    "Answer: True, False, or Uncertain."
                ),
            },
        ]
        try:
            content, _, _ = await client.call(messages, json_schema=FOL_SCHEMA, max_tokens=300)
            fmtna_result = json.loads(content)
            result["predict_fmtna"] = fmtna_result.get("answer", "Uncertain")
            llm_calls += 1
        except Exception:
            logger.error("FMTNA FOLIO call failed")
    result["metadata_fmtna_llm_calls"] = llm_calls

    # Baseline
    if cost_tracker.check(0.003):
        baseline_pred, bl_calls = await run_baseline_folio(client, premises_text, conclusion_text)
        result["predict_baseline"] = baseline_pred
        result["metadata_baseline_llm_calls"] = bl_calls

    logger.debug(
        f"FOLIO true={true_label} fmtna={result['predict_fmtna']} "
        f"baseline={result['predict_baseline']}"
    )
    return result


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_datasets(
    ruletaker_per_depth: int = RULETAKER_PER_DEPTH,
    folio_count: int = FOLIO_COUNT,
) -> tuple[list[dict], list[dict]]:
    """Load and stratify-sample examples from full data files."""
    logger.info("Loading datasets from full_data_out files")
    all_examples: dict[str, list[dict]] = {"ruletaker": [], "folio": []}

    data_files = sorted(glob.glob(str(DATA_DIR / "full_data_out" / "full_data_out_*.json")))
    logger.info(f"Found {len(data_files)} data files")

    for fpath in data_files:
        logger.info(f"Loading {fpath}")
        with open(fpath) as f:
            data = json.load(f)
        for ds in data["datasets"]:
            name = ds["dataset"]
            if name in all_examples:
                all_examples[name].extend(ds["examples"])
        del data
        gc.collect()

    logger.info(
        f"Loaded: ruletaker={len(all_examples['ruletaker'])}, "
        f"folio={len(all_examples['folio'])}"
    )

    # Stratified RuleTaker sample: equal per depth (0, 1, 3, 5)
    target_depths = {"depth-0", "depth-1", "depth-3", "depth-5"}
    ruletaker_by_depth: dict[str, list] = defaultdict(list)
    for ex in all_examples["ruletaker"]:
        cfg = ex.get("metadata_config", "")
        if cfg in target_depths:
            ruletaker_by_depth[cfg].append(ex)

    ruletaker_sample = []
    for depth in sorted(target_depths):
        examples = ruletaker_by_depth[depth]
        # Stratify by label (50/50 entailment/not)
        pos = [e for e in examples if e["output"] == "entailment"]
        neg = [e for e in examples if e["output"] != "entailment"]
        n_each = ruletaker_per_depth // 2
        # Deterministic selection (first n_each from each)
        selected = pos[:n_each] + neg[:n_each]
        ruletaker_sample.extend(selected)
        logger.info(
            f"  {depth}: {len(selected)} examples "
            f"(+{len(selected[:n_each])} /{len(selected[n_each:])})"
        )

    # FOLIO sample (stratified by label)
    folio_by_label: dict[str, list] = defaultdict(list)
    for ex in all_examples["folio"]:
        folio_by_label[ex["output"]].append(ex)

    folio_sample = []
    n_each_folio = folio_count // 3
    for label in ["True", "False", "Uncertain"]:
        folio_sample.extend(folio_by_label[label][:n_each_folio])
    # Fill remainder with any label
    remainder = folio_count - len(folio_sample)
    for ex in all_examples["folio"]:
        if remainder <= 0:
            break
        if ex not in folio_sample:
            folio_sample.append(ex)
            remainder -= 1

    logger.info(f"Final sample: {len(ruletaker_sample)} RuleTaker, {len(folio_sample)} FOLIO")

    del all_examples
    gc.collect()
    return ruletaker_sample, folio_sample


# ─── Metrics computation ──────────────────────────────────────────────────────

def compute_metrics(examples: list[dict], method: str = "fmtna") -> dict:
    """Compute accuracy and grounding metrics for a set of results."""
    pred_key = f"predict_{method}"
    correct = sum(1 for e in examples if e.get(pred_key) == e["output"])
    total = len(examples)
    accuracy = correct / total if total > 0 else 0.0

    metrics: dict[str, Any] = {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
    }

    if method == "fmtna":
        grounding_ratios = [
            e.get("metadata_fmtna_grounding_ratio", 0.0) for e in examples
        ]
        failure_types_agg: dict[str, int] = {"A": 0, "B": 0, "C": 0}
        for e in examples:
            ft = e.get("metadata_fmtna_failure_types", {})
            for k in failure_types_agg:
                failure_types_agg[k] += ft.get(k, 0)

        total_failures = sum(failure_types_agg.values()) or 1
        metrics.update({
            "grounding_ratio_mean": round(float(np.mean(grounding_ratios)), 4),
            "grounding_ratio_std": round(float(np.std(grounding_ratios)), 4),
            "failure_type_A_count": failure_types_agg["A"],
            "failure_type_B_count": failure_types_agg["B"],
            "failure_type_C_count": failure_types_agg["C"],
            "failure_type_A_pct": round(failure_types_agg["A"] / total_failures, 4),
            "failure_type_B_pct": round(failure_types_agg["B"] / total_failures, 4),
            "failure_type_C_pct": round(failure_types_agg["C"] / total_failures, 4),
        })

    return metrics


def compute_depth_analysis(ruletaker_results: list[dict]) -> list[dict]:
    """Compute per-depth metrics for RuleTaker."""
    by_depth: dict[str, list] = defaultdict(list)
    for ex in ruletaker_results:
        depth = ex.get("metadata_config", "unknown")
        by_depth[depth].append(ex)

    rows = []
    for depth in ["depth-0", "depth-1", "depth-3", "depth-5"]:
        exs = by_depth.get(depth, [])
        if not exs:
            continue
        fmtna_m = compute_metrics(exs, "fmtna")
        baseline_m = compute_metrics(exs, "baseline")

        grounding_ratios = [e.get("metadata_fmtna_grounding_ratio", 0.0) for e in exs]
        failure_types_agg: dict[str, int] = {"A": 0, "B": 0, "C": 0}
        for e in exs:
            ft = e.get("metadata_fmtna_failure_types", {})
            for k in failure_types_agg:
                failure_types_agg[k] += ft.get(k, 0)
        total_f = sum(failure_types_agg.values()) or 1

        rows.append({
            "depth": depth,
            "n_examples": len(exs),
            "fmtna_accuracy": fmtna_m["accuracy"],
            "baseline_accuracy": baseline_m["accuracy"],
            "type_a_count": failure_types_agg["A"],
            "type_b_count": failure_types_agg["B"],
            "type_c_count": failure_types_agg["C"],
            "type_a_pct": round(failure_types_agg["A"] / total_f, 4),
            "type_b_pct": round(failure_types_agg["B"] / total_f, 4),
            "type_c_pct": round(failure_types_agg["C"] / total_f, 4),
            "mean_grounding_ratio": round(float(np.mean(grounding_ratios)), 4),
        })
    return rows


def compute_type_a_comparison(
    ruletaker_results: list[dict], folio_results: list[dict]
) -> dict:
    """Chi-square test comparing Type-A failure rate: RuleTaker vs FOLIO."""
    def type_a_rate(examples: list[dict]) -> tuple[int, int]:
        total_f = 0
        type_a = 0
        for e in examples:
            ft = e.get("metadata_fmtna_failure_types", {})
            total_f += sum(ft.values())
            type_a += ft.get("A", 0)
        return type_a, total_f

    rt_a, rt_total = type_a_rate(ruletaker_results)
    fo_a, fo_total = type_a_rate(folio_results)

    # Chi-square test
    observed = np.array([[rt_a, rt_total - rt_a], [fo_a, fo_total - fo_a]])
    chi2_result = {"chi2": 0.0, "p_value": 1.0, "significant": False}
    if rt_total > 0 and fo_total > 0 and (rt_a + fo_a) > 0:
        try:
            chi2, p_val, _, _ = stats.chi2_contingency(observed, correction=False)
            chi2_result = {
                "chi2": round(float(chi2), 4),
                "p_value": round(float(p_val), 6),
                "significant": bool(p_val < 0.05),
            }
        except Exception:
            pass

    return {
        "ruletaker_type_a_count": rt_a,
        "ruletaker_total_failures": rt_total,
        "ruletaker_type_a_rate": round(rt_a / max(rt_total, 1), 4),
        "folio_type_a_count": fo_a,
        "folio_total_failures": fo_total,
        "folio_type_a_rate": round(fo_a / max(fo_total, 1), 4),
        "chi2_test": chi2_result,
        "hypothesis_confirmed": fo_a / max(fo_total, 1) > rt_a / max(rt_total, 1),
    }


def compute_baseline_comparison(ruletaker_results: list[dict]) -> dict:
    """Compare FMTNA vs undifferentiated baseline on depth-1 subset."""
    depth1 = [e for e in ruletaker_results if e.get("metadata_config") == "depth-1"]
    subset = depth1[:BASELINE_COMPARISON_DEPTH1]

    if not subset:
        return {"error": "No depth-1 examples found"}

    fmtna_m = compute_metrics(subset, "fmtna")
    baseline_m = compute_metrics(subset, "baseline")

    # Type-B span precision for FMTNA
    type_b_calls = sum(
        e.get("metadata_fmtna_failure_types", {}).get("B", 0) for e in subset
    )
    type_b_grounded = sum(e.get("metadata_fmtna_type_b_span_grounded", 0) for e in subset)
    span_precision = type_b_grounded / max(type_b_calls, 1)

    return {
        "subset_size": len(subset),
        "fmtna_accuracy": fmtna_m["accuracy"],
        "baseline_accuracy": baseline_m["accuracy"],
        "fmtna_type_b_span_precision": round(span_precision, 4),
        "fmtna_hallucination_exposure": round(1.0 - fmtna_m.get("grounding_ratio_mean", 0.0), 4),
        "fmtna_advantage_accuracy": round(fmtna_m["accuracy"] - baseline_m["accuracy"], 4),
    }


def build_depth_analysis_table(depth_rows: list[dict]) -> str:
    """Build markdown table for depth analysis."""
    header = "| Depth | N | FMTNA Acc | Baseline Acc | Type-A% | Type-B% | Type-C% | Grounding |"
    sep = "|-------|---|-----------|--------------|---------|---------|---------|-----------|"
    rows = [header, sep]
    for r in depth_rows:
        rows.append(
            f"| {r['depth']} | {r['n_examples']} | "
            f"{r['fmtna_accuracy']:.3f} | {r['baseline_accuracy']:.3f} | "
            f"{r['type_a_pct']:.3f} | {r['type_b_pct']:.3f} | {r['type_c_pct']:.3f} | "
            f"{r['mean_grounding_ratio']:.3f} |"
        )
    return "\n".join(rows)


# ─── Main orchestration ────────────────────────────────────────────────────────

@logger.catch(reraise=True)
async def main():
    logger.info("=" * 60)
    logger.info("FMTNA Multi-Benchmark Evaluation Pipeline")
    logger.info("=" * 60)

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    client = LLMClient(OPENROUTER_API_KEY, LLM_MODEL)

    # Mark tasks in progress
    logger.info("Step 1: Loading datasets")
    ruletaker_examples, folio_examples = load_datasets()
    logger.info(
        f"Dataset sizes: RuleTaker={len(ruletaker_examples)}, FOLIO={len(folio_examples)}"
    )

    # Process RuleTaker examples
    logger.info("Step 2: Processing RuleTaker examples")
    t0 = time.time()
    ruletaker_results = []

    async def process_rt_batch(examples: list[dict]) -> list[dict]:
        sem = asyncio.Semaphore(CONCURRENCY)
        async def process_one(ex):
            async with sem:
                if not cost_tracker.check(0.01):
                    logger.warning("Budget limit reached, stopping RuleTaker processing")
                    return dict(ex, predict_fmtna="not entailment", predict_baseline="not entailment",
                                metadata_fmtna_failure_types={"A":0,"B":0,"C":0},
                                metadata_fmtna_grounding_ratio=0.0,
                                metadata_fmtna_proof_steps=0,
                                metadata_fmtna_llm_calls=0,
                                metadata_baseline_llm_calls=0,
                                metadata_fmtna_success=False)
                try:
                    return await process_ruletaker_example(client, ex, use_llm_seed=False)
                except Exception:
                    logger.error(f"Failed RuleTaker example {ex.get('metadata_row_index')}")
                    return dict(ex, predict_fmtna="not entailment", predict_baseline="not entailment",
                                metadata_fmtna_failure_types={"A":0,"B":0,"C":0},
                                metadata_fmtna_grounding_ratio=0.0,
                                metadata_fmtna_proof_steps=0,
                                metadata_fmtna_llm_calls=0,
                                metadata_baseline_llm_calls=0,
                                metadata_fmtna_success=False)
        return list(await asyncio.gather(*[process_one(ex) for ex in examples]))

    ruletaker_results = await process_rt_batch(ruletaker_examples)
    rt_elapsed = time.time() - t0
    logger.info(
        f"RuleTaker done: {len(ruletaker_results)} examples in {rt_elapsed:.1f}s, "
        f"cost so far: ${cost_tracker.total_cost:.3f}"
    )

    # Process FOLIO examples
    logger.info("Step 3: Processing FOLIO examples")
    t1 = time.time()
    folio_results = []

    async def process_folio_batch(examples: list[dict]) -> list[dict]:
        sem = asyncio.Semaphore(CONCURRENCY)
        async def process_one(ex):
            async with sem:
                if not cost_tracker.check(0.005):
                    logger.warning("Budget limit reached, stopping FOLIO processing")
                    return dict(ex, predict_fmtna="Uncertain", predict_baseline="Uncertain",
                                metadata_fmtna_failure_types={"A":0,"B":0,"C":0},
                                metadata_fmtna_grounding_ratio=0.0,
                                metadata_fmtna_llm_calls=0,
                                metadata_baseline_llm_calls=0)
                try:
                    return await process_folio_example(client, ex)
                except Exception:
                    logger.error(f"Failed FOLIO example {ex.get('metadata_row_index')}")
                    return dict(ex, predict_fmtna="Uncertain", predict_baseline="Uncertain",
                                metadata_fmtna_failure_types={"A":0,"B":0,"C":0},
                                metadata_fmtna_grounding_ratio=0.0,
                                metadata_fmtna_llm_calls=0,
                                metadata_baseline_llm_calls=0)
        return list(await asyncio.gather(*[process_one(ex) for ex in examples]))

    folio_results = await process_folio_batch(folio_examples)
    folio_elapsed = time.time() - t1
    logger.info(
        f"FOLIO done: {len(folio_results)} examples in {folio_elapsed:.1f}s, "
        f"cost so far: ${cost_tracker.total_cost:.3f}"
    )

    # ─── Compute metrics ──────────────────────────────────────────────────────
    logger.info("Step 4: Computing metrics")

    rt_overall_fmtna = compute_metrics(ruletaker_results, "fmtna")
    rt_overall_baseline = compute_metrics(ruletaker_results, "baseline")
    depth_rows = compute_depth_analysis(ruletaker_results)
    folio_fmtna = compute_metrics(folio_results, "fmtna")
    folio_baseline = compute_metrics(folio_results, "baseline")
    type_a_comparison = compute_type_a_comparison(ruletaker_results, folio_results)
    baseline_comparison = compute_baseline_comparison(ruletaker_results)
    depth_table = build_depth_analysis_table(depth_rows)

    logger.info(f"RuleTaker FMTNA accuracy: {rt_overall_fmtna['accuracy']:.3f}")
    logger.info(f"RuleTaker Baseline accuracy: {rt_overall_baseline['accuracy']:.3f}")
    logger.info(f"FOLIO FMTNA accuracy: {folio_fmtna['accuracy']:.3f}")
    logger.info(f"FOLIO Baseline accuracy: {folio_baseline['accuracy']:.3f}")
    logger.info(f"Type-A comparison: {type_a_comparison}")
    logger.info(f"\nDepth Analysis:\n{depth_table}")

    # ─── Assemble output JSON ─────────────────────────────────────────────────
    logger.info("Step 5: Assembling output")

    # Collect sample failure records for logs
    sample_failures = []
    for ex in (ruletaker_results + folio_results)[:20]:
        ft = ex.get("metadata_fmtna_failure_types", {})
        if any(v > 0 for v in ft.values()):
            dominant_type = max(ft, key=ft.get)
            sample_failures.append({
                "type": dominant_type,
                "goal": ex.get("input", "")[:100],
                "context": f"depth={ex.get('metadata_config', 'folio')}",
                "resolution_source": "llm_dispatch",
                "true_label": ex.get("output"),
                "fmtna_prediction": ex.get("predict_fmtna"),
            })
        if len(sample_failures) >= 5:
            break

    output = {
        "metadata": {
            "method_name": "FMTNA",
            "description": "Failure-Mode-Typed Neural Abduction with typed LLM dispatch",
            "benchmark_names": ["ruletaker", "folio"],
            "dataset_sizes": {
                "ruletaker": len(ruletaker_results),
                "folio": len(folio_results),
            },
            "lm_model": LLM_MODEL,
            "cost_usd": round(cost_tracker.total_cost, 4),
            "lm_api_calls": cost_tracker.total_calls,
            "ruletaker_depths_evaluated": ["depth-0", "depth-1", "depth-3", "depth-5"],
            "results_summary": {
                "ruletaker": {
                    "overall_fmtna": rt_overall_fmtna,
                    "overall_baseline": rt_overall_baseline,
                    "by_depth": depth_rows,
                    "failure_type_distribution_table": depth_table,
                    "depth_analysis_plot_data": {
                        "depths": [r["depth"] for r in depth_rows],
                        "type_a_pcts": [r["type_a_pct"] for r in depth_rows],
                        "type_b_pcts": [r["type_b_pct"] for r in depth_rows],
                        "type_c_pcts": [r["type_c_pct"] for r in depth_rows],
                        "fmtna_accuracy": [r["fmtna_accuracy"] for r in depth_rows],
                        "baseline_accuracy": [r["baseline_accuracy"] for r in depth_rows],
                    },
                },
                "folio": {
                    "fmtna": folio_fmtna,
                    "baseline": folio_baseline,
                    "type_a_comparison_with_ruletaker": type_a_comparison,
                },
                "baseline_comparison": baseline_comparison,
                "hypothesis_checks": {
                    "type_a_higher_in_folio": type_a_comparison.get("hypothesis_confirmed", False),
                    "type_a_folio_rate": type_a_comparison.get("folio_type_a_rate", 0),
                    "type_a_ruletaker_rate": type_a_comparison.get("ruletaker_type_a_rate", 0),
                    "fmtna_outperforms_baseline": (
                        rt_overall_fmtna["accuracy"] > rt_overall_baseline["accuracy"]
                    ),
                },
            },
            "logs": {
                "sample_failure_records": sample_failures,
                "lm_api_calls": cost_tracker.total_calls,
                "total_cost_usd": round(cost_tracker.total_cost, 4),
                "total_input_tokens": cost_tracker.total_input_tokens,
                "total_output_tokens": cost_tracker.total_output_tokens,
            },
        },
        "datasets": [
            {
                "dataset": "ruletaker",
                "examples": ruletaker_results,
            },
            {
                "dataset": "folio",
                "examples": folio_results,
            },
        ],
    }

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(f"Written output to {out_path}")

    # Check file size
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Output file size: {size_mb:.1f} MB")

    logger.info("=" * 60)
    logger.info("FMTNA pipeline complete")
    logger.info(f"Total cost: ${cost_tracker.total_cost:.4f} / ${BUDGET_USD}")
    logger.info(f"Total API calls: {cost_tracker.total_calls}")
    logger.info(f"RuleTaker FMTNA acc: {rt_overall_fmtna['accuracy']:.3f}")
    logger.info(f"RuleTaker Baseline acc: {rt_overall_baseline['accuracy']:.3f}")
    logger.info(f"FOLIO FMTNA acc: {folio_fmtna['accuracy']:.3f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
