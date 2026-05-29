"""
ZONE: COLAB / GPU machine
─────────────────────────
Generates model predictions and saves them as a parquet.
Run this ONCE for the base model and ONCE for each fine-tuned model.
No metric computation here — just inference.

Usage (in Colab after training):
    # fine-tuned model (local path saved during training)
    !python evaluation/generate_predictions.py \
        --model_path outputs/customer-support-lora-r16 \
        --output evaluation/results/r16/finetuned_predictions.parquet \
        --n_samples 100

    # base model for comparison
    !python evaluation/generate_predictions.py \
        --model_path unsloth/Llama-3.2-3B-Instruct \
        --output evaluation/results/r16/base_predictions.parquet \
        --n_samples 100

Then download both parquets and run compute_metrics.py locally.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from rich.console import Console
from rich.progress import track
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()

SYSTEM_PROMPT = (
    "You are a helpful, professional customer support agent. "
    "Respond clearly and empathetically to customer inquiries. "
    "Be concise, accurate, and solution-focused."
)


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path: str, device: str):
    console.print(f"Loading [cyan]{model_path}[/cyan] → [yellow]{device}[/yellow]")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if device == "cpu" else torch.float16
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device)

    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, instruction: str, device: str, max_new_tokens: int) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {instruction}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip(), latency_ms


def main():
    parser = argparse.ArgumentParser(description="Generate predictions (run on GPU/Colab)")
    parser.add_argument("--model_path", required=True, help="HF model ID or local path")
    parser.add_argument("--test_data", default="data/cleaned/test_set.parquet")
    parser.add_argument("--output", required=True, help="Output parquet path, e.g. evaluation/results/r16/base_predictions.parquet")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    device = detect_device()
    model, tokenizer = load_model(args.model_path, device)

    df = pd.read_parquet(args.test_data)
    if args.n_samples < len(df):
        df = df.sample(n=args.n_samples, random_state=42).reset_index(drop=True)
    console.print(f"Generating for {len(df)} samples...")

    predictions, latencies = [], []
    for _, row in track(df.iterrows(), total=len(df), description="Generating..."):
        pred, lat = generate_one(model, tokenizer, row["instruction"], device, args.max_new_tokens)
        predictions.append(pred)
        latencies.append(lat)

    out_df = df[["instruction", "response", "intent", "category"]].copy()
    out_df["prediction"] = predictions
    out_df["latency_ms"] = latencies

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)

    avg_lat = sum(latencies) / len(latencies)
    console.print(f"\n✓ Saved {len(out_df)} predictions → [green]{args.output}[/green]")
    console.print(f"  Avg latency: {avg_lat:.0f} ms | Device: {device}")


if __name__ == "__main__":
    main()
