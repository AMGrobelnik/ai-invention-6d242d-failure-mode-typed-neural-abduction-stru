#!/usr/bin/env python3
"""FMTNA: Failure-Mode-Typed Neural Abduction for neuro-symbolic reasoning.

Compares typed LLM dispatch (our method) vs undifferentiated baseline on RuleTaker + FOLIO.
"""

import asyncio
import gc
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from loguru import logger
from scipy.stats import pearsonr

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

WORKSPACE = Path(__file__).parent
DATA_DIR = Path("/ai-inventor/aii_data/runs/348df/3_invention_loop/iter_1/gen_art/gen_art_dataset_1")
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Cost budget
COST_LIMIT_USD = 9.0
total_cost_usd = 0.0

# Model choices
MODEL_EXTRACT = "google/gemini-3.1-flash-lite"
MODEL_DISPATCH = "google/gemini-3.1-flash-lite"
MODEL_JUDGE = "anthropic/claude-haiku-4-5"

# Scale: 30 RuleTaker + 30 FOLIO = 60 total
N_RULETAKER = 30
N_FOLIO = 30

# Concurrency
SEM = asyncio.Semaphore(4)


# ──────────────────────────────────────────────────────────────────────────────
# LLM CLIENT
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMCall:
    model: str
    prompt: str
    response: str
    cost_usd: float
    call_type: str


async def call_llm(
    client: httpx.AsyncClient,
    prompt: str,
    model: str,
    call_type: str,
    system: str = "",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    retries: int = 3,
) -> tuple[str, float, LLMCall]:
    global total_cost_usd
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(retries):
        try:
            async with SEM:
                resp = await client.post(
                    OPENROUTER_URL,
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                    json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                    timeout=60.0,
                )
            if resp.status_code != 200:
                logger.warning(f"LLM {call_type} status {resp.status_code}: {resp.text[:200]}")
                await asyncio.sleep(2 ** attempt)
                continue
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            # Approximate cost: gemini-flash ~$0.10/1M input, $0.40/1M output
            cost = (prompt_tokens * 0.10 + completion_tokens * 0.40) / 1_000_000
            total_cost_usd += cost
            logger.debug(f"LLM {call_type}: cost=${cost:.4f} total=${total_cost_usd:.3f} resp={text[:100]!r}")
            call_record = LLMCall(model=model, prompt=prompt[:500], response=text[:1000], cost_usd=cost, call_type=call_type)
            return text, cost, call_record
        except Exception:
            logger.error(f"LLM call attempt {attempt+1} failed for {call_type}")
            await asyncio.sleep(2 ** attempt)

    return "", 0.0, LLMCall(model=model, prompt=prompt[:500], response="", cost_usd=0.0, call_type=call_type)


def extract_json(text: str) -> dict | None:
    """Extract JSON from LLM response, handling markdown code fences."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try code fence
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try bare JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# PROLOG-STYLE KNOWLEDGE BASE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Term:
    functor: str
    args: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if not self.args:
            return self.functor
        return f"{self.functor}({', '.join(self.args)})"

    def arity(self) -> int:
        return len(self.args)

    def is_ground(self) -> bool:
        return all(not a.startswith("_") and not a[0].isupper() for a in self.args)


@dataclass
class Clause:
    head: Term
    body: list[Term] = field(default_factory=list)
    span: str = ""

    def is_fact(self) -> bool:
        return len(self.body) == 0


@dataclass
class ResolutionResult:
    success: bool
    proof_path: list[str] = field(default_factory=list)
    failure_type: str = ""  # Type-A, Type-B, Type-C
    failing_goal: Term | None = None
    failing_reason: str = ""
    steps: int = 0


class PrologKB:
    def __init__(self):
        self.clauses: dict[str, list[Clause]] = {}  # key: "functor/arity"

    def add_clause(self, clause: Clause):
        key = f"{clause.head.functor}/{clause.head.arity()}"
        self.clauses.setdefault(key, []).append(clause)

    def keys(self) -> list[str]:
        return list(self.clauses.keys())

    def functors(self) -> list[str]:
        return [k.split("/")[0] for k in self.clauses.keys()]

    def copy(self) -> "PrologDB":
        kb = PrologDB()
        for key, clauses in self.clauses.items():
            kb.clauses[key] = list(clauses)
        return kb


# Fix typo: make PrologDB an alias
PrologDB = PrologKB = PrologKB


def unify(term: Term, head: Term, bindings: dict[str, str]) -> dict[str, str] | None:
    """Simple unification: variables are uppercase or start with _"""
    if term.functor != head.functor or len(term.args) != len(head.args):
        return None
    new_bindings = dict(bindings)
    for a, b in zip(term.args, head.args):
        a_val = new_bindings.get(a, a)
        b_val = new_bindings.get(b, b)
        a_is_var = a[0].isupper() or a.startswith("_") if a else False
        b_is_var = b[0].isupper() or b.startswith("_") if b else False
        if a_is_var and not b_is_var:
            new_bindings[a] = b_val
        elif b_is_var and not a_is_var:
            new_bindings[b] = a_val
        elif a_is_var and b_is_var:
            new_bindings[a] = b_val
        elif a_val != b_val:
            return None
    return new_bindings


def apply_bindings(term: Term, bindings: dict[str, str]) -> Term:
    new_args = []
    for a in term.args:
        val = bindings.get(a, a)
        new_args.append(val)
    return Term(functor=term.functor, args=new_args)


def resolve(goal: Term, kb: PrologKB, depth: int = 0, max_depth: int = 8) -> ResolutionResult:
    """SLD resolution with failure type classification."""
    if depth > max_depth:
        return ResolutionResult(success=False, failure_type="Type-C", failing_goal=goal,
                                 failing_reason="max_depth_exceeded", steps=depth)

    key = f"{goal.functor}/{goal.arity()}"
    if key not in kb.clauses:
        return ResolutionResult(success=False, failure_type="Type-A", failing_goal=goal,
                                 failing_reason=f"no_matching_functor:{goal.functor}", steps=depth)

    for clause in kb.clauses[key]:
        bindings = {}
        new_bindings = unify(goal, clause.head, bindings)
        if new_bindings is None:
            continue
        if clause.is_fact():
            return ResolutionResult(success=True, proof_path=[str(goal)], steps=depth + 1)
        # Resolve body
        all_success = True
        body_result = None
        for body_goal in clause.body:
            ground_body_goal = apply_bindings(body_goal, new_bindings)
            sub_result = resolve(ground_body_goal, kb, depth + 1, max_depth)
            if not sub_result.success:
                all_success = False
                body_result = sub_result
                break
        if all_success:
            return ResolutionResult(success=True, proof_path=[str(goal)], steps=depth + 1)
        # If body failed with Type-A (missing predicate) → propagate as Type-C from outer
        if body_result and body_result.failure_type == "Type-A":
            return ResolutionResult(success=False, failure_type="Type-C", failing_goal=goal,
                                     failing_reason="body_predicate_absent", steps=depth + 1)
        if body_result:
            return body_result

    # All clauses tried, none succeeded
    # Check if there are facts in KB for this functor at all
    if kb.clauses.get(key) and all(c.is_fact() for c in kb.clauses[key]):
        return ResolutionResult(success=False, failure_type="Type-B", failing_goal=goal,
                                 failing_reason="missing_ground_atom", steps=depth + 1)
    return ResolutionResult(success=False, failure_type="Type-C", failing_goal=goal,
                             failing_reason="no_applicable_rule", steps=depth + 1)


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def parse_ruletaker_input(inp: str) -> tuple[str, str]:
    """Extract context and question from RuleTaker input string."""
    if "Question:" in inp:
        parts = inp.split("Question:", 1)
        context = parts[0].replace("Context:", "").strip()
        question = parts[1].strip()
    else:
        context = inp
        question = ""
    return context, question


def parse_folio_input(inp: str) -> tuple[str, str]:
    """Extract premises and conclusion from FOLIO input string."""
    if "Conclusion:" in inp:
        parts = inp.split("Conclusion:", 1)
        premises = parts[0].replace("Premises:", "").strip()
        conclusion = parts[1].strip()
    else:
        premises = inp
        conclusion = ""
    return premises, conclusion


def load_dataset_examples(n_ruletaker: int, n_folio: int) -> list[dict]:
    """Load examples from full dataset files."""
    examples = []
    files = sorted(DATA_DIR.glob("full_data_out/full_data_out_*.json"))

    ruletaker_collected = 0
    folio_collected = 0
    needed_rt = n_ruletaker
    needed_fo = n_folio

    for fpath in files:
        if ruletaker_collected >= needed_rt and folio_collected >= needed_fo:
            break
        logger.info(f"Loading from {fpath.name}")
        with open(fpath) as f:
            data = json.load(f)
        for ds in data["datasets"]:
            name = ds["dataset"]
            for ex in ds["examples"]:
                if name == "ruletaker" and ruletaker_collected < needed_rt:
                    cfg = ex.get("metadata_config", "")
                    if cfg in ("depth-0", "depth-1"):
                        examples.append({**ex, "dataset": "ruletaker"})
                        ruletaker_collected += 1
                elif name == "folio" and folio_collected < needed_fo:
                    examples.append({**ex, "dataset": "folio"})
                    folio_collected += 1

    logger.info(f"Loaded {ruletaker_collected} RuleTaker + {folio_collected} FOLIO examples")
    return examples


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2: SEED EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

async def extract_seed_kb(
    client: httpx.AsyncClient,
    context: str,
    dataset: str,
) -> tuple[PrologKB, dict, list[LLMCall]]:
    """Extract structured KB from natural language context via LLM."""
    if dataset == "ruletaker":
        prompt = f"""Extract ALL facts and rules from this context as structured JSON.

Context: {context}

Return JSON with this exact structure:
{{
  "facts": [
    {{"predicate": "is_big", "args": ["cat"], "span": "The cat is big"}}
  ],
  "rules": [
    {{"head_predicate": "is_young", "head_args": ["X"], "body": [{{"predicate": "likes", "args": ["X", "cat"]}}], "span": "If someone likes the cat then they are young"}}
  ]
}}

Rules:
- Use snake_case for predicates (e.g. "chases", "is_big", "needs", "likes")
- Variables must start with uppercase (X, Y, Z, Someone)
- Extract ALL facts and rules, be exhaustive
- args should be lowercase entity names (cat, cow, lion, tiger, etc.)"""
    else:  # folio
        prompt = f"""Extract ALL premises as logical facts/rules from this text as structured JSON.

Premises: {context}

Return JSON with this structure:
{{
  "facts": [
    {{"predicate": "drinks_coffee", "args": ["rina"], "span": "Rina drinks coffee"}}
  ],
  "rules": [
    {{"head_predicate": "dependent_on_caffeine", "head_args": ["X"], "body": [{{"predicate": "drinks_coffee", "args": ["X"]}}], "span": "All people who drink coffee are dependent on caffeine"}}
  ]
}}

Rules:
- Use snake_case predicates
- Variables uppercase (X, Y, Person)
- Extract all logical content exhaustively"""

    text, cost, call = await call_llm(
        client, prompt, MODEL_EXTRACT, call_type="seed_extraction", max_tokens=2048
    )
    kb = PrologKB()
    raw = {}
    if text:
        parsed = extract_json(text)
        if parsed:
            raw = parsed
            for fact in parsed.get("facts", []):
                pred = fact.get("predicate", "").strip().replace(" ", "_").lower()
                args = [str(a).lower() for a in fact.get("args", [])]
                if pred:
                    kb.add_clause(Clause(head=Term(functor=pred, args=args), span=fact.get("span", "")))
            for rule in parsed.get("rules", []):
                hp = rule.get("head_predicate", "").strip().replace(" ", "_").lower()
                ha = [str(a) for a in rule.get("head_args", [])]
                body = [Term(functor=b.get("predicate", "").replace(" ", "_").lower(),
                              args=[str(x) for x in b.get("args", [])])
                        for b in rule.get("body", []) if b.get("predicate")]
                if hp:
                    kb.add_clause(Clause(head=Term(functor=hp, args=ha), body=body,
                                         span=rule.get("span", "")))

    logger.debug(f"Seed KB: {len(kb.clauses)} predicates from {len(raw.get('facts', []))} facts + {len(raw.get('rules', []))} rules")
    return kb, raw, [call]


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3: PARSE GOAL
# ──────────────────────────────────────────────────────────────────────────────

async def parse_goal_from_question(
    client: httpx.AsyncClient,
    question: str,
    context: str,
    kb: PrologKB,
    dataset: str,
) -> tuple[Term | None, bool, list[LLMCall]]:
    """Parse the question into a Prolog goal Term."""
    available_preds = ", ".join(kb.functors()[:20])
    if dataset == "ruletaker":
        prompt = f"""Given this question: "{question}"
Available predicates in KB: {available_preds}

Convert the question to a Prolog goal. The goal should use one of the available predicates if possible.
For "Is X Y?" → predicate "is_y" with arg X
For "Does X chase Y?" → predicate "chases" with args X, Y
For negated questions ("The X does NOT Y") → set is_negated=true and goal is the positive form.

Return JSON:
{{"functor": "chases", "args": ["cat", "cow"], "is_negated": false}}"""
    else:
        prompt = f"""Given this conclusion: "{question}"
Available predicates in KB: {available_preds}

Convert the conclusion to a Prolog goal term.
Return JSON:
{{"functor": "is_dependent", "args": ["rina"], "is_negated": false}}"""

    text, cost, call = await call_llm(client, prompt, MODEL_DISPATCH, call_type="goal_parsing", max_tokens=256)
    goal = None
    is_negated = False
    if text:
        parsed = extract_json(text)
        if parsed:
            functor = parsed.get("functor", "").replace(" ", "_").lower()
            args = [str(a).lower() for a in parsed.get("args", [])]
            is_negated = bool(parsed.get("is_negated", False))
            if functor:
                goal = Term(functor=functor, args=args)
    return goal, is_negated, [call]


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 4: TYPED LLM DISPATCH
# ──────────────────────────────────────────────────────────────────────────────

async def dispatch_type_a(
    client: httpx.AsyncClient,
    context: str,
    failing_goal: Term,
    kb: PrologKB,
) -> tuple[Clause | None, list[LLMCall]]:
    """Type-A: predicate name mismatch. Find synonym or bridging rule."""
    available = kb.functors()[:15]
    prompt = f"""Context: {context[:800]}

Required predicate: {failing_goal.functor}({', '.join(failing_goal.args)})
Available predicates in KB: {available}

Is any available predicate synonymous with '{failing_goal.functor}'?
If yes, create a bridging rule linking the synonym to the required predicate.

Return JSON:
{{"is_synonym": true, "synonym_functor": "likes", "bridging_rule_head": "{failing_goal.functor}", "bridging_rule_head_args": {json.dumps(failing_goal.args)}, "bridging_rule_body_pred": "likes", "bridging_rule_body_args": {json.dumps(failing_goal.args)}}}
or {{"is_synonym": false}}"""

    text, cost, call = await call_llm(client, prompt, MODEL_DISPATCH, call_type="Type-A", max_tokens=512)
    new_clause = None
    if text:
        parsed = extract_json(text)
        if parsed and parsed.get("is_synonym") and parsed.get("synonym_functor"):
            head = Term(functor=failing_goal.functor, args=failing_goal.args)
            body = [Term(functor=parsed["synonym_functor"], args=failing_goal.args)]
            new_clause = Clause(head=head, body=body, span=f"Type-A bridge: {parsed.get('synonym_functor')}")
    return new_clause, [call]


async def dispatch_type_b(
    client: httpx.AsyncClient,
    context: str,
    failing_goal: Term,
) -> tuple[Clause | None, list[LLMCall]]:
    """Type-B: missing ground atom. Check if it can be derived from context."""
    prompt = f"""Context: {context[:800]}

Is the following fact stated or directly derivable from the document?
{failing_goal.functor}({', '.join(failing_goal.args)})

Return JSON:
{{"is_present": true, "supporting_span": "The cat is big"}}
or {{"is_present": false}}"""

    text, cost, call = await call_llm(client, prompt, MODEL_DISPATCH, call_type="Type-B", max_tokens=256)
    new_clause = None
    if text:
        parsed = extract_json(text)
        if parsed and parsed.get("is_present"):
            new_clause = Clause(
                head=Term(functor=failing_goal.functor, args=failing_goal.args),
                span=f"Type-B recovered: {parsed.get('supporting_span', '')}",
            )
    return new_clause, [call]


async def dispatch_type_c(
    client: httpx.AsyncClient,
    context: str,
    failing_goal: Term,
    proof_context: list[str],
) -> tuple[Clause | None, list[LLMCall]]:
    """Type-C: absent rule head. Propose a Horn clause from world knowledge."""
    ctx_str = "; ".join(proof_context[-3:]) if proof_context else "none"
    prompt = f"""Context: {context[:600]}
Proof so far: {ctx_str}

We need a rule with head: {failing_goal.functor}({', '.join(failing_goal.args)})

Propose a Horn clause (using world knowledge, not the document) that could help prove this.
Use the same argument names and the available context entities.

Return JSON:
{{"proposed_head_pred": "{failing_goal.functor}", "proposed_head_args": {json.dumps(failing_goal.args)}, "proposed_body": [{{"pred": "is_big", "args": {json.dumps(failing_goal.args[:1])}}}], "explanation": "big things are usually dominant"}}"""

    text, cost, call = await call_llm(client, prompt, MODEL_DISPATCH, call_type="Type-C", max_tokens=512)
    new_clause = None
    if text:
        parsed = extract_json(text)
        if parsed and parsed.get("proposed_head_pred") and parsed.get("proposed_body"):
            head = Term(functor=parsed["proposed_head_pred"], args=[str(a) for a in parsed.get("proposed_head_args", [])])
            body = [Term(functor=b.get("pred", ""), args=[str(x) for x in b.get("args", [])])
                    for b in parsed["proposed_body"] if b.get("pred")]
            if head.functor and body:
                new_clause = Clause(head=head, body=body, span=f"Type-C world-knowledge: {parsed.get('explanation', '')}")
    return new_clause, [call]


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 5: FMTNA PROOF CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProofStep:
    step_idx: int
    goal: str
    step_type: str  # resolution | Type-A | Type-B | Type-C
    result: str
    source_span: str = ""


@dataclass
class ProofResult:
    predicted: str  # "entailment" | "not entailment" | "True" | "False" | "Uncertain" | "indeterminate"
    proof_steps: list[ProofStep] = field(default_factory=list)
    failure_types: dict = field(default_factory=lambda: {"Type-A": 0, "Type-B": 0, "Type-C": 0})
    grounding_ratio: float = 0.0
    lm_calls: list[LLMCall] = field(default_factory=list)


async def run_fmtna(
    client: httpx.AsyncClient,
    example: dict,
) -> ProofResult:
    """Run FMTNA pipeline on one example."""
    dataset = example["dataset"]
    inp = example["input"]
    expected = example["output"]
    lm_calls = []
    proof_steps = []
    failure_types = {"Type-A": 0, "Type-B": 0, "Type-C": 0}

    # Parse input
    if dataset == "ruletaker":
        context, question = parse_ruletaker_input(inp)
    else:
        context, question = parse_folio_input(inp)

    # Seed extraction
    kb, raw_kb, calls = await extract_seed_kb(client, context, dataset)
    lm_calls.extend(calls)

    # Parse goal
    goal, is_negated, calls = await parse_goal_from_question(client, question, context, kb, dataset)
    lm_calls.extend(calls)

    if goal is None:
        return ProofResult(
            predicted="indeterminate",
            proof_steps=[ProofStep(0, "?", "parse_error", "failed to parse goal")],
            failure_types=failure_types,
            grounding_ratio=0.0,
            lm_calls=lm_calls,
        )

    # Try resolution
    total_steps = 0
    grounded_steps = 0

    result = resolve(goal, kb)
    total_steps += 1

    if result.success:
        proof_steps.append(ProofStep(0, str(goal), "resolution", "SUCCESS"))
        grounded_steps += 1
    else:
        # Typed dispatch
        ftype = result.failure_type
        failure_types[ftype] = failure_types.get(ftype, 0) + 1
        proof_steps.append(ProofStep(0, str(goal), "resolution", f"FAILED:{ftype}"))

        if total_cost_usd < COST_LIMIT_USD:
            new_clause = None
            if ftype == "Type-A":
                new_clause, calls = await dispatch_type_a(client, context, result.failing_goal or goal, kb)
                lm_calls.extend(calls)
            elif ftype == "Type-B":
                new_clause, calls = await dispatch_type_b(client, context, result.failing_goal or goal)
                lm_calls.extend(calls)
            elif ftype == "Type-C":
                new_clause, calls = await dispatch_type_c(
                    client, context, result.failing_goal or goal,
                    [str(s.goal) for s in proof_steps]
                )
                lm_calls.extend(calls)

            if new_clause:
                kb.add_clause(new_clause)
                proof_steps.append(ProofStep(
                    len(proof_steps), str(new_clause.head), ftype,
                    f"ADDED:{new_clause.span[:100]}"
                ))
                total_steps += 1
                # Retry
                result2 = resolve(goal, kb)
                total_steps += 1
                if result2.success:
                    proof_steps.append(ProofStep(len(proof_steps), str(goal), "resolution_retry", "SUCCESS"))
                    grounded_steps += 1
                    result = result2
                else:
                    proof_steps.append(ProofStep(len(proof_steps), str(goal), "resolution_retry", "FAILED"))
            else:
                proof_steps.append(ProofStep(len(proof_steps), str(goal), ftype, "NO_EVIDENCE"))

    # Determine prediction
    if result.success:
        if is_negated:
            raw_pred = "not entailment" if dataset == "ruletaker" else "False"
        else:
            raw_pred = "entailment" if dataset == "ruletaker" else "True"
    else:
        raw_pred = "indeterminate"

    grounding_ratio = grounded_steps / max(total_steps, 1)
    return ProofResult(
        predicted=raw_pred,
        proof_steps=proof_steps,
        failure_types=failure_types,
        grounding_ratio=grounding_ratio,
        lm_calls=lm_calls,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 6: BASELINE (UNDIFFERENTIATED)
# ──────────────────────────────────────────────────────────────────────────────

async def dispatch_generic(
    client: httpx.AsyncClient,
    context: str,
    goal: Term,
) -> tuple[Clause | None, list[LLMCall]]:
    """Generic (undifferentiated) gap filling — no failure-type classification."""
    prompt = f"""Context: {context[:800]}

We need to prove: {goal.functor}({', '.join(goal.args)})
but cannot. What fact or rule would help prove this goal?

Return JSON:
{{"suggested_head_pred": "chases", "suggested_head_args": ["cat", "cow"], "suggested_body": [], "reasoning": "..."}}
For a rule: {{"suggested_head_pred": "young", "suggested_head_args": ["X"], "suggested_body": [{{"pred": "chases", "args": ["X", "cow"]}}], "reasoning": "..."}}"""

    text, cost, call = await call_llm(client, prompt, MODEL_DISPATCH, call_type="baseline_generic", max_tokens=512)
    new_clause = None
    if text:
        parsed = extract_json(text)
        if parsed and parsed.get("suggested_head_pred"):
            head = Term(functor=parsed["suggested_head_pred"],
                        args=[str(a) for a in parsed.get("suggested_head_args", [])])
            body = [Term(functor=b.get("pred", ""), args=[str(x) for x in b.get("args", [])])
                    for b in parsed.get("suggested_body", []) if b.get("pred")]
            new_clause = Clause(head=head, body=body, span=f"baseline: {parsed.get('reasoning', '')[:80]}")
    return new_clause, [call]


async def run_baseline(
    client: httpx.AsyncClient,
    example: dict,
) -> ProofResult:
    """Undifferentiated baseline: no failure-type classification."""
    dataset = example["dataset"]
    inp = example["input"]
    lm_calls = []
    proof_steps = []
    failure_types = {"Type-A": 0, "Type-B": 0, "Type-C": 0}

    if dataset == "ruletaker":
        context, question = parse_ruletaker_input(inp)
    else:
        context, question = parse_folio_input(inp)

    kb, _, calls = await extract_seed_kb(client, context, dataset)
    lm_calls.extend(calls)

    goal, is_negated, calls = await parse_goal_from_question(client, question, context, kb, dataset)
    lm_calls.extend(calls)

    if goal is None:
        return ProofResult(predicted="indeterminate", lm_calls=lm_calls)

    result = resolve(goal, kb)
    proof_steps.append(ProofStep(0, str(goal), "resolution", "SUCCESS" if result.success else f"FAILED"))

    if not result.success and total_cost_usd < COST_LIMIT_USD:
        new_clause, calls = await dispatch_generic(client, context, result.failing_goal or goal)
        lm_calls.extend(calls)
        if new_clause:
            kb.add_clause(new_clause)
            proof_steps.append(ProofStep(1, str(new_clause.head), "baseline_generic", f"ADDED"))
            result2 = resolve(goal, kb)
            proof_steps.append(ProofStep(2, str(goal), "resolution_retry", "SUCCESS" if result2.success else "FAILED"))
            result = result2

    if result.success:
        raw_pred = ("not entailment" if is_negated else "entailment") if dataset == "ruletaker" \
            else ("False" if is_negated else "True")
    else:
        raw_pred = "indeterminate"

    return ProofResult(
        predicted=raw_pred,
        proof_steps=proof_steps,
        failure_types=failure_types,
        grounding_ratio=float(result.success),
        lm_calls=lm_calls,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 7: LLM-JUDGE HALLUCINATION ANNOTATION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HallucinationAnnotation:
    step_idx: int
    goal: str
    is_hallucinated: bool
    confidence: float
    explanation: str


async def annotate_hallucination(
    client: httpx.AsyncClient,
    context: str,
    step: ProofStep,
    dataset: str,
) -> tuple[HallucinationAnnotation, LLMCall]:
    """Judge whether a proof step's LLM response is grounded."""
    if step.step_type in ("resolution", "resolution_retry", "parse_error"):
        return HallucinationAnnotation(
            step_idx=step.step_idx, goal=str(step.goal),
            is_hallucinated=False, confidence=1.0, explanation="pure_resolution_no_llm"
        ), LLMCall(model="", prompt="", response="", cost_usd=0.0, call_type="skipped")

    prompt = f"""Document: {context[:600]}

Proof step (type: {step.step_type}): {step.goal}
Step result: {step.result}

Is this step factually supported by the document or validly derivable from it?
(For Type-C steps using world knowledge, assess if the proposed rule is reasonable.)

Return JSON:
{{"is_hallucinated": false, "explanation": "The cat chases the cow is directly stated", "confidence": 0.9}}"""

    text, cost, call = await call_llm(
        client, prompt, MODEL_JUDGE, call_type="hallucination_judge", max_tokens=256
    )
    annotation = HallucinationAnnotation(
        step_idx=step.step_idx, goal=str(step.goal),
        is_hallucinated=False, confidence=0.5, explanation=""
    )
    if text:
        parsed = extract_json(text)
        if parsed:
            annotation.is_hallucinated = bool(parsed.get("is_hallucinated", False))
            annotation.confidence = float(parsed.get("confidence", 0.5))
            annotation.explanation = str(parsed.get("explanation", ""))[:300]
    return annotation, call


# ──────────────────────────────────────────────────────────────────────────────
# PROCESS ONE EXAMPLE
# ──────────────────────────────────────────────────────────────────────────────

async def process_example(
    client: httpx.AsyncClient,
    example: dict,
    idx: int,
) -> dict:
    """Run FMTNA + baseline + hallucination annotation for one example."""
    global total_cost_usd
    logger.info(f"Processing example {idx} [{example['dataset']}] cost_so_far=${total_cost_usd:.2f}")

    if total_cost_usd >= COST_LIMIT_USD:
        logger.warning(f"Cost limit reached at example {idx}, skipping")
        return None

    dataset = example["dataset"]
    inp = example["input"]
    expected = example["output"]

    if dataset == "ruletaker":
        context, question = parse_ruletaker_input(inp)
    else:
        context, question = parse_folio_input(inp)

    # Run FMTNA
    fmtna_result = await run_fmtna(client, example)

    # Run baseline (re-extract KB independently)
    baseline_result = await run_baseline(client, example)

    # Hallucination annotation (only for non-resolution steps, limit cost)
    fmtna_hallucination = []
    baseline_hallucination = []

    if total_cost_usd < COST_LIMIT_USD - 1.0:
        for step in fmtna_result.proof_steps:
            if step.step_type not in ("resolution", "resolution_retry", "parse_error"):
                ann, call = await annotate_hallucination(client, context, step, dataset)
                fmtna_hallucination.append(ann)
                fmtna_result.lm_calls.append(call)

        for step in baseline_result.proof_steps:
            if step.step_type not in ("resolution", "resolution_retry", "parse_error"):
                ann, call = await annotate_hallucination(client, context, step, dataset)
                baseline_hallucination.append(ann)
                baseline_result.lm_calls.append(call)

    fmtna_hallucinated_steps = sum(1 for a in fmtna_hallucination if a.is_hallucinated)
    baseline_hallucinated_steps = sum(1 for a in baseline_hallucination if a.is_hallucinated)
    fmtna_hall_presence = fmtna_hallucinated_steps > 0
    baseline_hall_presence = baseline_hallucinated_steps > 0

    # Accuracy
    fmtna_correct = fmtna_result.predicted == expected
    baseline_correct = baseline_result.predicted == expected

    # Handle "indeterminate" → map to most likely based on KB
    def normalize_pred(pred: str, expected: str) -> bool:
        if pred == "indeterminate":
            return False
        return pred == expected

    fmtna_accurate = normalize_pred(fmtna_result.predicted, expected)
    baseline_accurate = normalize_pred(baseline_result.predicted, expected)

    return {
        "example_id": idx,
        "dataset": dataset,
        "input": inp,
        "expected": expected,
        "fmtna_predicted": fmtna_result.predicted,
        "baseline_predicted": baseline_result.predicted,
        "fmtna_accurate": fmtna_accurate,
        "baseline_accurate": baseline_accurate,
        "proof_steps_fmtna": [asdict(s) for s in fmtna_result.proof_steps],
        "proof_steps_baseline": [asdict(s) for s in baseline_result.proof_steps],
        "failure_types_fmtna": fmtna_result.failure_types,
        "grounding_ratio_fmtna": fmtna_result.grounding_ratio,
        "grounding_ratio_baseline": baseline_result.grounding_ratio,
        "hallucination_fmtna": fmtna_hall_presence,
        "hallucination_count_fmtna": fmtna_hallucinated_steps,
        "hallucination_baseline": baseline_hall_presence,
        "hallucination_count_baseline": baseline_hallucinated_steps,
        "fmtna_hall_annotations": [asdict(a) for a in fmtna_hallucination],
        "baseline_hall_annotations": [asdict(a) for a in baseline_hallucination],
        "lm_calls_fmtna": len(fmtna_result.lm_calls),
        "lm_calls_baseline": len(baseline_result.lm_calls),
        "cost_usd": total_cost_usd,
    }


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    fmtna_hall = [r["hallucination_fmtna"] for r in results]
    baseline_hall = [r["hallucination_baseline"] for r in results]
    fmtna_acc = [r["fmtna_accurate"] for r in results]
    baseline_acc = [r["baseline_accurate"] for r in results]
    grounding = [r["grounding_ratio_fmtna"] for r in results]

    fmtna_hall_rate = sum(fmtna_hall) / len(fmtna_hall)
    baseline_hall_rate = sum(baseline_hall) / len(baseline_hall)
    hall_delta = baseline_hall_rate - fmtna_hall_rate

    fmtna_accuracy = sum(fmtna_acc) / len(fmtna_acc)
    baseline_accuracy = sum(baseline_acc) / len(baseline_acc)

    # Grounding-hallucination correlation
    corr_r = 0.0
    if len(grounding) >= 3 and any(g != grounding[0] for g in grounding):
        hall_binary = [float(h) for h in fmtna_hall]
        try:
            corr_r, _ = pearsonr(grounding, hall_binary)
        except Exception:
            corr_r = 0.0

    type_totals = {"Type-A": 0, "Type-B": 0, "Type-C": 0}
    for r in results:
        for k, v in r["failure_types_fmtna"].items():
            type_totals[k] = type_totals.get(k, 0) + v

    return {
        "fmtna_hallucination_rate": round(fmtna_hall_rate, 4),
        "baseline_hallucination_rate": round(baseline_hall_rate, 4),
        "hallucination_reduction_delta": round(hall_delta, 4),
        "type_a_count": type_totals["Type-A"],
        "type_b_count": type_totals["Type-B"],
        "type_c_count": type_totals["Type-C"],
        "grounding_ratio_mean": round(float(np.mean(grounding)), 4),
        "grounding_ratio_std": round(float(np.std(grounding)), 4),
        "hallucination_presence_correlation": round(float(corr_r), 4),
        "fmtna_accuracy": round(fmtna_accuracy, 4),
        "baseline_accuracy": round(baseline_accuracy, 4),
        "accuracy_delta": round(fmtna_accuracy - baseline_accuracy, 4),
        "total_cost_usd": round(total_cost_usd, 4),
        "cost_per_example_usd": round(total_cost_usd / max(len(results), 1), 4),
        "n_examples": len(results),
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

@logger.catch(reraise=True)
async def main_async():
    global total_cost_usd
    logger.info("=== FMTNA Experiment Starting ===")

    examples = load_dataset_examples(N_RULETAKER, N_FOLIO)
    if not examples:
        raise RuntimeError("No examples loaded")

    logger.info(f"Processing {len(examples)} examples")

    results = []
    async with httpx.AsyncClient() as client:
        # Process in batches of 4 to respect concurrency limit
        batch_size = 4
        for i in range(0, len(examples), batch_size):
            if total_cost_usd >= COST_LIMIT_USD:
                logger.warning(f"Cost limit hit at example {i}, stopping")
                break
            batch = examples[i:i+batch_size]
            tasks = [process_example(client, ex, i+j) for j, ex in enumerate(batch)]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in batch_results:
                if isinstance(r, Exception):
                    logger.error(f"Example failed: {r}")
                elif r is not None:
                    results.append(r)
            logger.info(f"Progress: {len(results)}/{len(examples)} done, cost=${total_cost_usd:.3f}")
            # Save intermediate
            _save_output(results, examples)

    logger.info(f"Completed {len(results)} examples, total cost=${total_cost_usd:.4f}")
    _save_output(results, examples)
    return results


def _save_output(results: list[dict], all_examples: list[dict]):
    """Save results in exp_gen_sol_out format."""
    if not results:
        return

    metrics = compute_metrics(results)
    logger.info(f"Metrics: FMTNA acc={metrics['fmtna_accuracy']:.3f} hall={metrics['fmtna_hallucination_rate']:.3f} | "
                f"Baseline acc={metrics['baseline_accuracy']:.3f} hall={metrics['baseline_hallucination_rate']:.3f} | "
                f"delta_hall={metrics['hallucination_reduction_delta']:.3f}")

    # Group by dataset for output schema
    ruletaker_examples = [r for r in results if r["dataset"] == "ruletaker"]
    folio_examples = [r for r in results if r["dataset"] == "folio"]

    def result_to_example(r: dict) -> dict:
        return {
            "input": r["input"],
            "output": r["expected"],
            "predict_fmtna": r["fmtna_predicted"],
            "predict_baseline": r["baseline_predicted"],
            "metadata_fmtna_accurate": str(r["fmtna_accurate"]),
            "metadata_baseline_accurate": str(r["baseline_accurate"]),
            "metadata_grounding_ratio": str(round(r["grounding_ratio_fmtna"], 4)),
            "metadata_hallucination_fmtna": str(r["hallucination_fmtna"]),
            "metadata_hallucination_baseline": str(r["hallucination_baseline"]),
            "metadata_hallucination_count_fmtna": str(r["hallucination_count_fmtna"]),
            "metadata_hallucination_count_baseline": str(r["hallucination_count_baseline"]),
            "metadata_failure_type_A": str(r["failure_types_fmtna"].get("Type-A", 0)),
            "metadata_failure_type_B": str(r["failure_types_fmtna"].get("Type-B", 0)),
            "metadata_failure_type_C": str(r["failure_types_fmtna"].get("Type-C", 0)),
            "metadata_proof_steps_fmtna": json.dumps(r["proof_steps_fmtna"])[:500],
            "metadata_lm_calls_fmtna": str(r["lm_calls_fmtna"]),
            "metadata_lm_calls_baseline": str(r["lm_calls_baseline"]),
        }

    datasets = []
    if ruletaker_examples:
        datasets.append({
            "dataset": "ruletaker",
            "examples": [result_to_example(r) for r in ruletaker_examples],
        })
    if folio_examples:
        datasets.append({
            "dataset": "folio",
            "examples": [result_to_example(r) for r in folio_examples],
        })

    output = {
        "metadata": {
            "method_name": "FMTNA",
            "description": "Failure-Mode-Typed Neural Abduction with typed LLM dispatch vs undifferentiated baseline",
            "pipeline": "seed_extraction → prolog_resolution → typed_dispatch → hallucination_annotation",
            "aggregate_metrics": metrics,
            "qualitative_analysis": {
                "failure_type_distribution": (
                    f"Type-A (predicate mismatch): {metrics['type_a_count']}, "
                    f"Type-B (missing ground atom): {metrics['type_b_count']}, "
                    f"Type-C (absent rule): {metrics['type_c_count']}"
                ),
                "hallucination_reduction": (
                    f"FMTNA reduces hallucination by {metrics['hallucination_reduction_delta']:.3f} "
                    f"({metrics['baseline_hallucination_rate']:.3f} → {metrics['fmtna_hallucination_rate']:.3f})"
                ),
                "grounding_correlation": (
                    f"Pearson r={metrics['hallucination_presence_correlation']:.3f} "
                    "between grounding_ratio and hallucination_presence"
                ),
                "cost_info": f"${metrics['total_cost_usd']:.4f} total, ${metrics['cost_per_example_usd']:.4f}/example",
            },
        },
        "datasets": datasets,
    }

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved method_out.json ({out_path.stat().st_size // 1024}KB)")


@logger.catch(reraise=True)
def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
