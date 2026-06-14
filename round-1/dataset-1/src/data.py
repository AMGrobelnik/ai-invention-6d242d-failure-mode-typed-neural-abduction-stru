#!/usr/bin/env python3
"""Load RuleTaker and FOLIO datasets, standardize to exp_sel_data_out schema."""

from loguru import logger
from pathlib import Path
import json
import sys

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

DATASETS_DIR = Path("temp/datasets")
OUTPUT_FILE = Path("full_data_out.json")


def load_json_file(path: Path) -> list:
    logger.info(f"Loading {path.name} ({path.stat().st_size // 1024}KB)")
    # Try full JSON load first; fall back to line-by-line for large/truncated files
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data
        logger.warning(f"Unexpected type {type(data)} in {path.name}")
        return []
    except json.JSONDecodeError:
        logger.info(f"Falling back to line-by-line parsing for {path.name}")
        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip().rstrip(",")
                if line in ("[", "]", ""):
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return rows


def process_ruletaker() -> dict:
    """Load ruletaker dev+test splits and standardize examples."""
    examples = []

    for split_name, fname in [
        ("dev", "full_tasksource_ruletaker_default_dev.json"),
        ("test", "full_tasksource_ruletaker_default_test.json"),
    ]:
        fpath = DATASETS_DIR / fname
        if not fpath.exists():
            logger.warning(f"Missing {fname}, skipping")
            continue

        rows = load_json_file(fpath)
        logger.info(f"RuleTaker {split_name}: {len(rows)} rows")

        for i, row in enumerate(rows):
            try:
                context = row.get("context", "")
                question = row.get("question", "")
                label = row.get("label", "")
                config = row.get("config", "")

                if not context or not question or not label:
                    continue

                example = {
                    "input": f"Context: {context}\nQuestion: {question}",
                    "output": label,
                    "metadata_split": split_name,
                    "metadata_config": config,
                    "metadata_row_index": i,
                    "metadata_task_type": "logical_reasoning",
                    "metadata_answer_type": "entailment",
                }
                examples.append(example)
            except Exception:
                logger.error(f"Error on row {i} in {fname}")
                continue

    logger.info(f"RuleTaker total: {len(examples)} examples")
    return {"dataset": "ruletaker", "examples": examples}


def process_folio() -> dict:
    """Load FOLIO train split and standardize examples."""
    examples = []

    fpath = DATASETS_DIR / "full_tasksource_folio_default_train.json"
    if not fpath.exists():
        logger.warning("FOLIO file missing")
        return {"dataset": "folio", "examples": []}

    rows = load_json_file(fpath)
    logger.info(f"FOLIO train: {len(rows)} rows")

    for i, row in enumerate(rows):
        try:
            premises = row.get("premises", "")
            conclusion = row.get("conclusion", "")
            label = str(row.get("label", ""))
            premises_fol = row.get("premises-FOL", "")
            conclusion_fol = row.get("conclusion-FOL", "")
            story_id = str(row.get("story_id", ""))
            example_id = str(row.get("example_id", ""))

            if not premises or not conclusion or not label:
                continue

            example = {
                "input": f"Premises:\n{premises}\nConclusion: {conclusion}",
                "output": label,
                "metadata_premises_fol": premises_fol,
                "metadata_conclusion_fol": conclusion_fol,
                "metadata_story_id": story_id,
                "metadata_example_id": example_id,
                "metadata_split": "train",
                "metadata_task_type": "fol_reasoning",
                "metadata_answer_type": "true_false_uncertain",
                "metadata_row_index": i,
            }
            examples.append(example)
        except Exception:
            logger.error(f"Error on row {i} in folio")
            continue

    logger.info(f"FOLIO total: {len(examples)} examples")
    return {"dataset": "folio", "examples": examples}


@logger.catch(reraise=True)
def main():
    Path("logs").mkdir(exist_ok=True)

    logger.info("Processing RuleTaker dataset")
    ruletaker = process_ruletaker()

    logger.info("Processing FOLIO dataset")
    folio = process_folio()

    output = {
        "metadata": {
            "source": "HuggingFace: tasksource/ruletaker, tasksource/folio",
            "description": "Neuro-symbolic reasoning benchmarks for FMTNA pipeline evaluation",
            "datasets": ["ruletaker", "folio"],
        },
        "datasets": [ruletaker, folio],
    }

    total = sum(len(d["examples"]) for d in output["datasets"])
    logger.info(f"Total examples: {total}")

    OUTPUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
