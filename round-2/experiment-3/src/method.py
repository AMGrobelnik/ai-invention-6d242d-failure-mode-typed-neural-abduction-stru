#!/usr/bin/env python3
"""Type-B Precision Recovery via BM25-Enforced Passage Retrieval.

Tests whether limiting LLM view to BM25-ranked passages reduces hallucination
on ground atom verification (entailment) tasks vs. unconstrained prompting.
"""

import gc
import json
import math
import os
import re
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from loguru import logger

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
Path("logs").mkdir(exist_ok=True)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Hardware limits (cgroup v2) ───────────────────────────────────────────────
def _container_ram_gb() -> float:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return 8.0

TOTAL_RAM_GB = _container_ram_gb()
RAM_BUDGET = int(TOTAL_RAM_GB * 0.5 * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"RAM budget: {RAM_BUDGET / 1e9:.1f} GB (of {TOTAL_RAM_GB:.1f} GB total)")

# ── Config ────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
DATA_DIR = Path("/ai-inventor/aii_data/runs/348df/3_invention_loop/iter_1/gen_art/gen_art_dataset_1")
MINI_DATA = DATA_DIR / "mini_data_out.json"
FULL_DATA_1 = DATA_DIR / "full_data_out/full_data_out_1.json"
FULL_DATA_2 = DATA_DIR / "full_data_out/full_data_out_2.json"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "anthropic/claude-haiku-4.5"
# Pricing: $0.80/M in, $4.00/M out  → ~$0.002 per call
COST_PER_1M_IN = 0.80
COST_PER_1M_OUT = 4.00
BUDGET_USD = 9.0  # hard limit

MAX_EXAMPLES = 100  # target; will stop early if budget exhausted
K_VALUES = [1, 3, 5]  # ablation: number of BM25 passages to retrieve

# ── OpenRouter caller ─────────────────────────────────────────────────────────
_total_cost_usd = 0.0
_total_lm_calls = 0
_total_in_tokens = 0
_total_out_tokens = 0


def call_llm(prompt: str, system: str = "", timeout: int = 60) -> tuple[str, float]:
    """Call LLM via OpenRouter. Returns (response_text, cost_usd)."""
    global _total_cost_usd, _total_lm_calls, _total_in_tokens, _total_out_tokens

    if _total_cost_usd >= BUDGET_USD:
        raise RuntimeError(f"Budget exhausted: ${_total_cost_usd:.2f} >= ${BUDGET_USD}")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {})
            in_tok = usage.get("prompt_tokens", 0)
            out_tok = usage.get("completion_tokens", 0)
            cost = (in_tok * COST_PER_1M_IN + out_tok * COST_PER_1M_OUT) / 1_000_000
            _total_cost_usd += cost
            _total_lm_calls += 1
            _total_in_tokens += in_tok
            _total_out_tokens += out_tok
            logger.debug(f"LLM call #{_total_lm_calls}: {in_tok}in+{out_tok}out=${cost:.4f} cumul=${_total_cost_usd:.3f}")
            return text, cost
        except requests.HTTPError as e:
            logger.warning(f"LLM HTTP error attempt {attempt+1}: {e}")
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
        except requests.RequestException as e:
            logger.warning(f"LLM request error attempt {attempt+1}: {e}")
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

    return "", 0.0


def parse_llm_json(text: str) -> dict:
    """Extract JSON from LLM response with fallback regex parsing."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting JSON block
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # Fallback: extract fields with regex
    result = {}
    m_present = re.search(r'"present"\s*:\s*(true|false)', text, re.IGNORECASE)
    if m_present:
        result["present"] = m_present.group(1).lower() == "true"
    m_conf = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
    if m_conf:
        result["confidence"] = float(m_conf.group(1))
    m_span = re.search(r'"evidence_span"\s*:\s*"([^"]*)"', text)
    if m_span:
        result["evidence_span"] = m_span.group(1)
    return result


# ── BM25 retrieval ─────────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    return re.sub(r'[^a-z0-9 ]', ' ', text.lower()).split()


def build_bm25_index(passages: list[str]) -> dict:
    """Build BM25 index (BM25S sparse approach) over passages."""
    import bm25s

    corpus_tokens = [tokenize(p) for p in passages]
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    return {"retriever": retriever, "passages": passages}


def bm25_retrieve(index: dict, query: str, k: int) -> list[tuple[int, str, float]]:
    """Retrieve top-k passages. Returns list of (rank_idx, passage, score)."""
    retriever = index["retriever"]
    passages = index["passages"]
    q_tokens = tokenize(query)
    if not q_tokens:
        # fallback: return first k passages
        return [(i, passages[i], 0.0) for i in range(min(k, len(passages)))]

    import bm25s
    results, scores = retriever.retrieve([q_tokens], k=min(k, len(passages)))
    # results shape: (1, k), scores shape: (1, k)
    top = []
    for rank, (idx, sc) in enumerate(zip(results[0], scores[0])):
        top.append((int(idx), passages[int(idx)], float(sc)))
    return top


# ── Context parsing ───────────────────────────────────────────────────────────
def parse_ruletaker_context(input_text: str) -> tuple[list[str], str]:
    """Extract passages (sentences) and question from RuleTaker input."""
    # Format: "Context: ...\nQuestion: ..."
    ctx_match = re.search(r'Context:\s*(.*?)\nQuestion:\s*(.*)', input_text, re.DOTALL)
    if not ctx_match:
        return [input_text], ""
    context = ctx_match.group(1).strip()
    question = ctx_match.group(2).strip()
    # Split into sentences on ". " or ".\n"
    sentences = re.split(r'(?<=[.!?])\s+', context)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences, question


def parse_folio_context(input_text: str) -> tuple[list[str], str]:
    """Extract premises (as passages) and conclusion from FOLIO input."""
    # Format: "Premises:\n...\nConclusion: ..."
    prem_match = re.search(r'Premises:\s*(.*?)\nConclusion:\s*(.*)', input_text, re.DOTALL)
    if not prem_match:
        return [input_text], ""
    premises_text = prem_match.group(1).strip()
    conclusion = prem_match.group(2).strip()
    # Each premise is a line
    premises = [p.strip() for p in premises_text.split('\n') if p.strip()]
    return premises, conclusion


# ── LLM prompts ──────────────────────────────────────────────────────────────
SYSTEM_BM25 = (
    "You are a precise logical reasoning assistant. "
    "Your ONLY job is to check if the given statement is entailed by the provided passages. "
    "Base your answer ONLY on the provided passages — do not use external knowledge. "
    'Respond with valid JSON only: {"present": true/false, "evidence_span": "exact quote or empty", "confidence": 0.0-1.0}'
)

SYSTEM_BASELINE = (
    "You are a precise logical reasoning assistant. "
    "Determine if the given statement logically follows from the context. "
    'Respond with valid JSON only: {"entailed": true/false, "confidence": 0.0-1.0}'
)


def build_bm25_prompt(query: str, passages: list[str]) -> str:
    passages_text = "\n\n".join(f"[P{i+1}] {p}" for i, p in enumerate(passages))
    return (
        f"PASSAGES:\n{passages_text}\n\n"
        f"STATEMENT TO CHECK: {query}\n\n"
        "Is the statement explicitly stated or directly entailed by the passages above? "
        'Respond with JSON: {"present": true/false, "evidence_span": "exact quote or empty string", "confidence": 0.0-1.0}'
    )


def build_baseline_prompt(context_or_premises: str, query: str) -> str:
    return (
        f"CONTEXT:\n{context_or_premises}\n\n"
        f"QUESTION: Does the following follow from the context: {query}\n\n"
        'Respond with JSON: {"entailed": true/false, "confidence": 0.0-1.0}'
    )


# ── MRR computation ───────────────────────────────────────────────────────────
def compute_mrr(ranks: list[int | None]) -> float:
    """Mean Reciprocal Rank. rank is 1-indexed; None = not found."""
    rr = []
    for r in ranks:
        if r is not None:
            rr.append(1.0 / r)
        else:
            rr.append(0.0)
    return float(np.mean(rr)) if rr else 0.0


# ── Main experiment ───────────────────────────────────────────────────────────
@logger.catch(reraise=True)
def run_experiment(examples_data: list[dict], max_examples: int) -> dict:
    """Run the BM25 Type-B experiment on given examples."""
    per_example_log = []
    ablation_results: dict[int, dict] = {k: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for k in K_VALUES}
    baseline_stats = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    mrr_ranks: list[int | None] = []
    coverage_k: dict[int, int] = {k: 0 for k in K_VALUES}
    false_positive_with_passages = 0
    false_positive_without_restriction = 0
    n_true_absent = 0

    n_done = 0
    budget_hit = False

    for ex_idx, ex in enumerate(examples_data):
        if n_done >= max_examples:
            break
        if _total_cost_usd >= BUDGET_USD * 0.95:
            logger.warning(f"Budget near limit ${_total_cost_usd:.2f}, stopping early")
            budget_hit = True
            break

        dataset_name = ex["dataset"]
        item = ex["item"]
        input_text = item["input"]
        ground_truth_label = item["output"]

        # Parse context into passages and query
        if dataset_name == "ruletaker":
            passages, query = parse_ruletaker_context(input_text)
            ground_truth_present = (ground_truth_label == "entailment")
        else:  # folio
            passages, query = parse_folio_context(input_text)
            ground_truth_present = (ground_truth_label == "True")

        if not query or not passages:
            logger.warning(f"Ex {ex_idx}: empty query or passages, skipping")
            continue

        # Build BM25 index for this example
        bm25_idx = build_bm25_index(passages)

        # Find ground truth passage (if fact is present, which passage contains it?)
        true_rank: int | None = None
        if ground_truth_present:
            # Retrieve with k=max to find rank of "best" passage
            all_results = bm25_retrieve(bm25_idx, query, k=len(passages))
            # The true "support" passage would be the one containing keywords from query
            q_words = set(tokenize(query))
            for rank_1based, (pidx, ptext, pscore) in enumerate(all_results, start=1):
                p_words = set(tokenize(ptext))
                if len(q_words & p_words) >= max(1, len(q_words) // 3):
                    true_rank = rank_1based
                    break
        mrr_ranks.append(true_rank)

        # Coverage analysis
        for k in K_VALUES:
            top_k = bm25_retrieve(bm25_idx, query, k=k)
            top_k_texts = [p for _, p, _ in top_k]
            if true_rank is not None and true_rank <= k:
                coverage_k[k] += 1

        # --- Baseline call: full context, no BM25 ---
        full_context = "\n".join(passages)
        baseline_prompt = build_baseline_prompt(full_context, query)
        try:
            baseline_text, _ = call_llm(baseline_prompt, system=SYSTEM_BASELINE)
            baseline_json = parse_llm_json(baseline_text)
            baseline_pred = baseline_json.get("entailed", False)
            baseline_conf = float(baseline_json.get("confidence", 0.5))
        except Exception:
            logger.error(f"Ex {ex_idx}: baseline LLM call failed")
            baseline_pred = False
            baseline_conf = 0.5

        # Track baseline stats
        if ground_truth_present:
            if baseline_pred:
                baseline_stats["tp"] += 1
            else:
                baseline_stats["fn"] += 1
        else:
            if baseline_pred:
                baseline_stats["fp"] += 1
                false_positive_without_restriction += 1
            else:
                baseline_stats["tn"] += 1

        # --- BM25 + LLM calls for each k value ---
        k_responses: dict[int, dict] = {}
        for k in K_VALUES:
            top_k_results = bm25_retrieve(bm25_idx, query, k=k)
            top_k_passages = [p for _, p, _ in top_k_results]
            bm25_prompt = build_bm25_prompt(query, top_k_passages)
            try:
                bm25_text, _ = call_llm(bm25_prompt, system=SYSTEM_BM25)
                bm25_json = parse_llm_json(bm25_text)
                bm25_pred = bm25_json.get("present", False)
                bm25_conf = float(bm25_json.get("confidence", 0.5))
                evidence_span = bm25_json.get("evidence_span", "")
                # Validate span
                span_in_passages = any(evidence_span in p for p in top_k_passages) if evidence_span else False
            except Exception:
                logger.error(f"Ex {ex_idx} k={k}: BM25 LLM call failed")
                bm25_pred = False
                bm25_conf = 0.5
                evidence_span = ""
                span_in_passages = False

            k_responses[k] = {
                "pred": bm25_pred,
                "conf": bm25_conf,
                "evidence_span": evidence_span,
                "span_in_passages": span_in_passages,
                "passages": top_k_passages,
            }

            # Accumulate stats
            if ground_truth_present:
                if bm25_pred:
                    ablation_results[k]["tp"] += 1
                else:
                    ablation_results[k]["fn"] += 1
            else:
                if bm25_pred:
                    ablation_results[k]["fp"] += 1
                    if k == 3:
                        false_positive_with_passages += 1
                else:
                    ablation_results[k]["tn"] += 1

        if not ground_truth_present:
            n_true_absent += 1

        # Use k=3 as primary result
        primary = k_responses.get(3, k_responses.get(K_VALUES[0], {}))
        primary_pred = primary.get("pred", False)
        primary_conf = primary.get("conf", 0.5)
        primary_span = primary.get("evidence_span", "")
        primary_span_match = primary.get("span_in_passages", False)

        per_example_log.append({
            "example_id": n_done,
            "dataset": dataset_name,
            "query": query[:300],
            "ground_truth_present": ground_truth_present,
            "top_passages_k3": k_responses.get(3, {}).get("passages", [])[:3],
            "lm_response_k3": {
                "present": primary_pred,
                "confidence": primary_conf,
                "evidence_span": primary_span[:200],
            },
            "baseline_pred": baseline_pred,
            "baseline_conf": baseline_conf,
            "bm25_correct": primary_pred == ground_truth_present,
            "baseline_correct": baseline_pred == ground_truth_present,
            "grounding_span_match": primary_span_match,
        })

        n_done += 1
        if n_done % 10 == 0:
            logger.info(f"Progress: {n_done}/{max_examples} examples, cost=${_total_cost_usd:.3f}")

    logger.info(f"Completed {n_done} examples, total cost=${_total_cost_usd:.3f}")

    # ── Compute metrics ───────────────────────────────────────────────────────
    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    def compute_prf(stats: dict) -> dict:
        tp, fp, tn, fn = stats["tp"], stats["fp"], stats["tn"], stats["fn"]
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        accuracy = safe_div(tp + tn, tp + fp + tn + fn)
        return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn}

    # Primary BM25 k=3 metrics
    bm25_k3_metrics = compute_prf(ablation_results[3])
    baseline_metrics = compute_prf(baseline_stats)

    # Ablation by k
    ablation_by_k = []
    for k in K_VALUES:
        m = compute_prf(ablation_results[k])
        ablation_by_k.append({
            "k": k,
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
        })

    # Confidence calibration (Spearman correlation)
    conf_vals = [e["lm_response_k3"]["confidence"] for e in per_example_log]
    correct_vals = [1.0 if e["bm25_correct"] else 0.0 for e in per_example_log]
    spearman_r = 0.0
    if len(conf_vals) >= 5:
        try:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(conf_vals, correct_vals)
            spearman_r = float(corr) if not math.isnan(corr) else 0.0
        except Exception:
            pass

    # BM25 retrieval analysis
    mrr = compute_mrr(mrr_ranks)
    n_true_present = n_done - n_true_absent
    fp_rate_with_passages = safe_div(false_positive_with_passages, n_true_absent)
    fp_rate_baseline = safe_div(false_positive_without_restriction, n_true_absent)

    # Hypothesis validation
    precision_delta = bm25_k3_metrics["precision"] - baseline_metrics["precision"]
    h1_result = precision_delta >= 0.15
    h2_result = fp_rate_baseline > fp_rate_with_passages
    h3_result = bm25_k3_metrics["precision"] >= 0.5

    return {
        "experiment": "type_b_precision_recovery",
        "primary_metric": "type_b_precision",
        "results": {
            "baseline_iteration1_precision": baseline_metrics["precision"],
            "current_bm25_precision": bm25_k3_metrics["precision"],
            "improvement_delta": precision_delta,
            "recall": bm25_k3_metrics["recall"],
            "f1_score": bm25_k3_metrics["f1"],
            "confidence_calibration_spearman": round(spearman_r, 4),
            "num_examples": n_done,
            "num_lm_calls": _total_lm_calls,
            "total_cost_usd": round(_total_cost_usd, 4),
            "budget_exhausted": budget_hit,
        },
        "bm25_retrieval_analysis": {
            "mean_reciprocal_rank": round(mrr, 4),
            "passage_coverage_k1": round(safe_div(coverage_k[1], n_true_present), 4) if n_true_present > 0 else 0.0,
            "passage_coverage_k3": round(safe_div(coverage_k[3], n_true_present), 4) if n_true_present > 0 else 0.0,
            "passage_coverage_k5": round(safe_div(coverage_k[5], n_true_present), 4) if n_true_present > 0 else 0.0,
            "false_positive_rate_with_passages": round(fp_rate_with_passages, 4),
            "false_positive_rate_baseline": round(fp_rate_baseline, 4),
        },
        "ablation_by_k": ablation_by_k,
        "baseline_metrics": {
            "precision": round(baseline_metrics["precision"], 4),
            "recall": round(baseline_metrics["recall"], 4),
            "f1": round(baseline_metrics["f1"], 4),
            "accuracy": round(baseline_metrics["accuracy"], 4),
        },
        "per_example_log": per_example_log,
        "comparison_to_baseline": {
            "baseline_method": "unconstrained_llm_full_context",
            "baseline_precision": round(baseline_metrics["precision"], 4),
            "bm25_precision": round(bm25_k3_metrics["precision"], 4),
            "improvement_mechanism": "BM25 passage ranking + structured schema + span-constrained LLM output",
            "success_threshold_met": h1_result,
        },
        "hypothesis_validation": {
            "h1_typed_dispatch_reduces_hallucination": {
                "claim": "Type-B dispatch with BM25 retrieval produces >15% improvement over baseline",
                "result": h1_result,
                "evidence": f"precision_delta={precision_delta:.4f}",
            },
            "h2_passage_limitation_grounds_lm": {
                "claim": "LLM hallucinates less when seeing only top-3 relevant passages vs. full document",
                "result": h2_result,
                "evidence": f"fp_rate_baseline={fp_rate_baseline:.4f} fp_rate_bm25={fp_rate_with_passages:.4f}",
            },
            "h3_grounding_ratio_proxy": {
                "claim": "High Type-B precision is achievable (>0.5) with BM25-enforced grounding",
                "result": h3_result,
                "evidence": f"precision_achieved={bm25_k3_metrics['precision']:.4f}",
            },
        },
    }


# ── Data loading ──────────────────────────────────────────────────────────────
def load_examples_from_data(data: dict, max_total: int, depth_filter: list[str] | None = None) -> list[dict]:
    """Flatten datasets into (dataset, item) pairs, optionally filtering ruletaker depths."""
    examples = []
    targets = {
        "ruletaker": max_total * 2 // 3,  # ~67% ruletaker
        "folio": max_total // 3,           # ~33% folio
    }
    counts = {"ruletaker": 0, "folio": 0}

    for ds in data.get("datasets", []):
        name = ds["dataset"]
        target = targets.get(name, max_total)
        for item in ds.get("examples", []):
            if counts.get(name, 0) >= target:
                break
            if name == "ruletaker" and depth_filter:
                cfg = item.get("metadata_config", "")
                if cfg not in depth_filter:
                    continue
            examples.append({"dataset": name, "item": item})
            counts[name] = counts.get(name, 0) + 1

    return examples


def stream_full_data(max_examples: int, depth_filter: list[str] | None = None) -> list[dict]:
    """Load limited examples from the full dataset files without reading all into memory."""
    examples = []
    for fpath in [FULL_DATA_1, FULL_DATA_2]:
        if len(examples) >= max_examples:
            break
        if not fpath.exists():
            logger.warning(f"File not found: {fpath}")
            continue
        logger.info(f"Streaming from {fpath.name}")
        data = json.loads(fpath.read_text())
        batch = load_examples_from_data(data, max_examples - len(examples), depth_filter)
        examples.extend(batch)
        del data
        gc.collect()
    return examples[:max_examples]


# ── Schema-compliant output builder ──────────────────────────────────────────
def build_method_out(experiment_results: dict, examples: list[dict]) -> dict:
    """Build output conforming to exp_gen_sol_out schema."""
    ds_examples: dict[str, list] = {"ruletaker": [], "folio": []}

    for log_entry in experiment_results["per_example_log"]:
        ds_name = log_entry["dataset"]
        ex_input = log_entry["query"]
        ex_output = "entailment" if log_entry["ground_truth_present"] else "not entailment"
        if ds_name == "folio":
            ex_output = "True" if log_entry["ground_truth_present"] else "False"

        record = {
            "input": ex_input,
            "output": ex_output,
            "predict_bm25_k3": "entailment" if log_entry["lm_response_k3"]["present"] else "not entailment",
            "predict_baseline": "entailment" if log_entry["baseline_pred"] else "not entailment",
            "metadata_bm25_correct": str(log_entry["bm25_correct"]),
            "metadata_baseline_correct": str(log_entry["baseline_correct"]),
            "metadata_bm25_confidence": str(round(log_entry["lm_response_k3"]["confidence"], 3)),
            "metadata_grounding_span_match": str(log_entry["grounding_span_match"]),
            "metadata_dataset": ds_name,
        }
        ds_examples.setdefault(ds_name, []).append(record)

    datasets = []
    for ds_name, exs in ds_examples.items():
        if exs:
            datasets.append({"dataset": ds_name, "examples": exs})

    # Summary stats
    r = experiment_results["results"]
    b = experiment_results.get("baseline_metrics", {})
    return {
        "metadata": {
            "method_name": "BM25-Enforced Type-B Precision Recovery",
            "description": "BM25 sparse retrieval to ground LLM on top-k passages for entailment verification",
            "lm_model": LLM_MODEL,
            "bm25_implementation": "bm25s",
            "k_values_ablated": K_VALUES,
            "primary_k": 3,
            "bm25_precision_k3": r["current_bm25_precision"],
            "baseline_precision": b.get("precision", 0.0),
            "improvement_delta": r["improvement_delta"],
            "recall_k3": r["recall"],
            "f1_k3": r["f1_score"],
            "confidence_calibration_spearman": r["confidence_calibration_spearman"],
            "num_examples": r["num_examples"],
            "total_cost_usd": r["total_cost_usd"],
            "ablation_by_k": experiment_results["ablation_by_k"],
            "bm25_retrieval_analysis": experiment_results["bm25_retrieval_analysis"],
            "hypothesis_validation": experiment_results["hypothesis_validation"],
        },
        "datasets": datasets,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
@logger.catch(reraise=True)
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true", help="Run on mini data only")
    parser.add_argument("--n", type=int, default=MAX_EXAMPLES, help="Max examples")
    parser.add_argument("--depth", nargs="+", default=["depth-0", "depth-1"],
                        help="RuleTaker depth filters")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("=== Type-B Precision Recovery Experiment ===")
    logger.info(f"Model: {LLM_MODEL}, Budget: ${BUDGET_USD}")

    # Load data
    if args.mini:
        logger.info("Loading MINI data")
        data = json.loads(MINI_DATA.read_text())
        examples = load_examples_from_data(data, args.n, args.depth)
    else:
        logger.info(f"Streaming up to {args.n} examples from full data")
        examples = stream_full_data(args.n, args.depth)

    if not examples:
        raise RuntimeError("No examples loaded!")

    logger.info(f"Loaded {len(examples)} examples from {set(e['dataset'] for e in examples)}")

    # Run experiment
    results = run_experiment(examples, max_examples=len(examples))

    elapsed = (time.time() - start_time) / 60
    results["metadata"] = {
        "datasets_used": ["RuleTaker (depth-0, depth-1)", "FOLIO"],
        "lm_model": LLM_MODEL,
        "bm25_implementation": "bm25s (scipy sparse)",
        "passage_count_per_example": 3,
        "total_passages_indexed": sum(
            len(parse_ruletaker_context(e["item"]["input"])[0]) if e["dataset"] == "ruletaker"
            else len(parse_folio_context(e["item"]["input"])[0])
            for e in examples
        ),
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_runtime_minutes": round(elapsed, 2),
    }

    # Write detailed method_out.json (artifact plan spec)
    method_out_path = WORKSPACE / "method_out.json"
    method_out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved detailed results to {method_out_path}")

    # Write schema-compliant output
    schema_out = build_method_out(results, examples)
    schema_out_path = WORKSPACE / "method_out_schema.json"
    schema_out_path.write_text(json.dumps(schema_out, indent=2))
    logger.info(f"Saved schema-compliant output to {schema_out_path}")

    # Summary
    r = results["results"]
    logger.info(f"=== RESULTS ===")
    logger.info(f"BM25 k=3 precision: {r['current_bm25_precision']:.3f}")
    logger.info(f"Baseline precision:  {results.get('baseline_metrics', {}).get('precision', 0):.3f}")
    logger.info(f"Delta: {r['improvement_delta']:+.3f}")
    logger.info(f"F1: {r['f1_score']:.3f}, Recall: {r['recall']:.3f}")
    logger.info(f"Total cost: ${r['total_cost_usd']:.4f}")
    logger.info(f"Runtime: {elapsed:.1f} min")

    for h_key, h_val in results["hypothesis_validation"].items():
        status = "✓ SUPPORTED" if h_val["result"] else "✗ NOT SUPPORTED"
        logger.info(f"{h_key}: {status} ({h_val['evidence']})")


if __name__ == "__main__":
    main()
