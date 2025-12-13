import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_with_lm_eval(
    model_name: str,
    task: str,
    num_fewshot: int,
    limit: Optional[int],
    device: str,
    dtype: str,
) -> float:
    """Run lm-evaluation-harness on a single task and return accuracy."""
    lm = HFLM(pretrained=model_name, device=device, dtype=dtype)
    try:
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=num_fewshot,
            limit=limit,
        )
    finally:
        del lm
        torch.cuda.empty_cache()
    task_result = results["results"][task]
    # Prefer exact match if available, otherwise fallback to accuracy.
    for key, value in task_result.items():
        if key == "alias":
            continue
        if key.startswith("exact_match"):
            return float(value)
    for key, value in task_result.items():
        if key == "alias":
            continue
        if key.startswith("acc"):
            return float(value)
    raise ValueError(f"No accuracy metric found for task {task}: {task_result.keys()}")


def evaluate_quotes(
    model_name: str,
    quotes_path: Path,
    device: str,
    dtype: str,
    limit: Optional[int],
    max_new_tokens: int = 64,
) -> float:
    """Evaluate quote memorization by prompting with prefix and checking suffix match."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype))
    model.to(device)
    model.eval()

    total = 0
    correct = 0

    with quotes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            prefix = record["prefix"]
            suffix = record["target_suffix"]

            inputs = tokenizer(prefix, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=min(max_new_tokens, len(suffix) + 16),
                    do_sample=False,
                )

            gen_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            completion = gen_text[len(prefix) :]
            completion = completion.strip()
            expected = suffix.strip()

            if completion.startswith(expected):
                correct += 1
            total += 1

            if limit is not None and total >= limit:
                break

    model.to("cpu")
    del model
    torch.cuda.empty_cache()

    return float(correct) / float(total) if total > 0 else 0.0


def evaluate_model(
    model_name: str,
    device: str,
    dtype: str,
    quotes_path: Path,
    triviaqa_limit: Optional[int],
    quotes_limit: Optional[int],
    logic_limit: Optional[int],
) -> List[Dict[str, float]]:
    metrics: List[Dict[str, float]] = []

    triviaqa_acc = evaluate_with_lm_eval(
        model_name=model_name,
        task="triviaqa",
        num_fewshot=0,
        limit=triviaqa_limit,
        device=device,
        dtype=dtype,
    )
    metrics.append(
        {
            "task": "TriviaQA",
            "category": "closed-book qa",
            "accuracy": triviaqa_acc,
        }
    )

    logic_acc = evaluate_with_lm_eval(
        model_name=model_name,
        task="bbh_fewshot_logical_deduction_three_objects",
        num_fewshot=3,
        limit=logic_limit,
        device=device,
        dtype=dtype,
    )
    metrics.append(
        {
            "task": "Logical Deduction",
            "category": "logic",
            "accuracy": logic_acc,
        }
    )

    quotes_acc = evaluate_quotes(
        model_name=model_name,
        quotes_path=quotes_path,
        device=device,
        dtype=dtype,
        limit=quotes_limit,
    )
    metrics.append(
        {
            "task": "Quotes",
            "category": "memory",
            "accuracy": quotes_acc,
        }
    )

    return metrics


def compare_metrics(
    baseline_metrics: List[Dict[str, float]],
    edited_metrics: Optional[List[Dict[str, float]]] = None,
) -> List[Dict[str, float]]:
    edited_map = {m["task"]: m for m in edited_metrics} if edited_metrics else {}
    comparison: List[Dict[str, float]] = []
    for baseline in baseline_metrics:
        task = baseline["task"]
        entry = {
            "task": task,
            "category": baseline["category"],
            "baseline_accuracy": baseline["accuracy"],
            "edited_accuracy": edited_map.get(task, {}).get("accuracy"),
        }
        comparison.append(entry)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate models on TriviaQA, Quotes, and Logical Deduction."
    )
    parser.add_argument("--baseline-model", required=True, help="Baseline Hugging Face model id/path.")
    parser.add_argument("--edited-model", help="Edited Hugging Face model id/path.")
    parser.add_argument(
        "--quotes-jsonl",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "olmo2_1b_large8_bfloat16.jsonl",
        help="Quotes dataset JSONL with prefix/target_suffix fields.",
    )
    parser.add_argument("--device", default="cuda", help="Device for model evaluation.")
    parser.add_argument("--dtype", default="float16", help="Torch dtype (e.g., float16, bfloat16).")
    parser.add_argument("--triviaqa-limit", type=int, default=128, help="Max TriviaQA examples.")
    parser.add_argument("--quotes-limit", type=int, default=256, help="Max quotes examples.")
    parser.add_argument("--logic-limit", type=int, default=128, help="Max logical deduction examples.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("task_eval_results.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    quotes_path = args.quotes_jsonl
    if not quotes_path.exists():
        raise FileNotFoundError(f"Quotes dataset not found at {quotes_path}")

    baseline_metrics = evaluate_model(
        model_name=args.baseline_model,
        device=args.device,
        dtype=args.dtype,
        quotes_path=quotes_path,
        triviaqa_limit=args.triviaqa_limit,
        quotes_limit=args.quotes_limit,
        logic_limit=args.logic_limit,
    )

    edited_metrics = None
    if args.edited_model:
        edited_metrics = evaluate_model(
            model_name=args.edited_model,
            device=args.device,
            dtype=args.dtype,
            quotes_path=quotes_path,
            triviaqa_limit=args.triviaqa_limit,
            quotes_limit=args.quotes_limit,
            logic_limit=args.logic_limit,
        )

    records = compare_metrics(baseline_metrics, edited_metrics)
    results = {
        "baseline_model": args.baseline_model,
        "edited_model": args.edited_model,
        "tasks": records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved task evaluation results to {args.output}")


if __name__ == "__main__":
    main()

