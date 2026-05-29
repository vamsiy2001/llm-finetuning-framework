"""
ZONE: LOCAL / CPU — no model or GPU required
─────────────────────────────────────────────
Computes ROUGE, BLEU, and length stats from saved prediction parquets.
Run this after downloading predictions from Colab.

Usage:
    python evaluation/compute_metrics.py --experiment r16
    python evaluation/compute_metrics.py --experiment r64

Expects these files to already exist:
    evaluation/results/<experiment>/base_predictions.parquet
    evaluation/results/<experiment>/finetuned_predictions.parquet

Outputs:
    evaluation/results/<experiment>/base_metrics.json
    evaluation/results/<experiment>/finetuned_metrics.json
    evaluation/results/<experiment>/comparison.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate import load as load_metric
from rich.console import Console
from rich.table import Table

console = Console()


# ── metric functions ───────────────────────────────────────────────────────

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    rouge = load_metric("rouge")
    result = rouge.compute(predictions=predictions, references=references)
    return {k: round(v, 4) for k, v in result.items()}


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    bleu = load_metric("bleu")
    result = bleu.compute(predictions=predictions, references=references)
    return round(result["bleu"], 4)


def compute_length_stats(predictions: list[str], references: list[str]) -> dict:
    pred_lens = [len(p.split()) for p in predictions]
    ref_lens = [len(r.split()) for r in references]
    return {
        "avg_pred_length": round(np.mean(pred_lens), 1),
        "avg_ref_length": round(np.mean(ref_lens), 1),
        "length_ratio": round(np.mean(pred_lens) / max(np.mean(ref_lens), 1), 3),
    }


def metrics_for(df: pd.DataFrame, label: str, model_path: str) -> dict:
    predictions = df["prediction"].tolist()
    references = df["response"].tolist()

    console.print(f"Computing metrics for [cyan]{label}[/cyan] ({len(df)} samples)...")
    rouge = compute_rouge(predictions, references)
    bleu = compute_bleu(predictions, references)
    length = compute_length_stats(predictions, references)

    avg_lat = round(df["latency_ms"].mean(), 1) if "latency_ms" in df.columns else None
    p95_lat = round(df["latency_ms"].quantile(0.95), 1) if "latency_ms" in df.columns else None

    return {
        "label": label,
        "model_path": model_path,
        "n_samples": len(df),
        "bleu": bleu,
        **rouge,
        **length,
        "avg_latency_ms": avg_lat,
        "p95_latency_ms": p95_lat,
    }


# ── comparison printer ─────────────────────────────────────────────────────

def print_comparison(base: dict, ft: dict):
    table = Table(title=f"Base vs Fine-tuned — {ft.get('model_path', '')}", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Base", justify="right")
    table.add_column("Fine-tuned", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("% change", justify="right")

    metrics = [
        ("BLEU",          "bleu"),
        ("ROUGE-1",       "rouge1"),
        ("ROUGE-2",       "rouge2"),
        ("ROUGE-L",       "rougeL"),
        ("Avg Pred Len",  "avg_pred_length"),
        ("Avg Lat (ms)",  "avg_latency_ms"),
    ]

    for label, key in metrics:
        b = base.get(key)
        f = ft.get(key)
        if b is None or f is None:
            continue
        delta = f - b
        pct = (delta / b * 100) if b != 0 else 0
        color = "green" if delta > 0 else ("red" if delta < 0 else "white")
        # for length and latency, lower is better
        if key in ("avg_pred_length", "avg_latency_ms", "p95_latency_ms"):
            color = "red" if delta > 0 else ("green" if delta < 0 else "white")
        table.add_row(
            label,
            str(b),
            str(f),
            f"[{color}]{delta:+.4f}[/{color}]",
            f"[{color}]{pct:+.1f}%[/{color}]",
        )

    console.print(table)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute metrics from saved predictions (no GPU needed)")
    parser.add_argument(
        "--experiment", default="r16",
        help="Experiment name matching results subfolder, e.g. r16 or r64"
    )
    parser.add_argument(
        "--base_model", default="unsloth/Llama-3.2-3B-Instruct",
        help="Base model ID for labelling"
    )
    parser.add_argument(
        "--ft_model", default=None,
        help="Fine-tuned model ID for labelling (defaults to experiment name)"
    )
    args = parser.parse_args()

    results_dir = Path("evaluation/results") / args.experiment
    base_path = results_dir / "base_predictions.parquet"
    ft_path = results_dir / "finetuned_predictions.parquet"

    for p in [base_path, ft_path]:
        if not p.exists():
            console.print(f"[red]Missing: {p}[/red]")
            console.print("Download both parquets from Colab first. See SETUP_GUIDE.md.")
            raise SystemExit(1)

    base_df = pd.read_parquet(base_path)
    ft_df = pd.read_parquet(ft_path)

    ft_label = args.ft_model or f"Fine-tuned LoRA {args.experiment}"
    base_metrics = metrics_for(base_df, "Base Llama 3.2-3B", args.base_model)
    ft_metrics = metrics_for(ft_df, ft_label, ft_label)

    print_comparison(base_metrics, ft_metrics)

    # save individual metrics
    with open(results_dir / "base_metrics.json", "w") as f:
        json.dump(base_metrics, f, indent=2)
    with open(results_dir / "finetuned_metrics.json", "w") as f:
        json.dump(ft_metrics, f, indent=2)

    # save comparison
    comparison = {"base": base_metrics, "finetuned": ft_metrics}
    with open(results_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    console.print(f"\n✓ Metrics saved to [green]{results_dir}[/green]")
    console.print("Next: run evaluation/llm_judge.py for LLM-as-judge scores.")


if __name__ == "__main__":
    main()
