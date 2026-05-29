"""
Local inference & benchmarking script.
Works on Apple Silicon (MPS), CUDA, or CPU — no GPU required for basic testing.

Usage:
    # quick single-prompt test
    python inference/benchmark.py \
        --model_id vamsiyvk/customer-support-lora-r16 \
        --prompt "My order hasn't arrived in 3 weeks."

    # benchmark mode: run N prompts, report latency + throughput
    python inference/benchmark.py \
        --model_id vamsiyvk/customer-support-lora-r16 \
        --benchmark \
        --n_prompts 20

    # compare base vs fine-tuned side by side
    python inference/benchmark.py \
        --model_id vamsiyvk/customer-support-lora-r16 \
        --compare_base \
        --prompt "I was charged twice for my subscription."
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()

BASE_MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"

SYSTEM_PROMPT = (
    "You are a helpful, professional customer support agent. "
    "Respond clearly and empathetically to customer inquiries. "
    "Be concise, accurate, and solution-focused."
)

BENCHMARK_PROMPTS = [
    "I was charged twice for my last order. Can you help me get a refund?",
    "How do I return a product I bought 5 days ago?",
    "My order hasn't arrived and it's been 3 weeks. Where is it?",
    "I want to cancel my subscription immediately.",
    "Can I change my shipping address after I placed an order?",
    "My account is locked and I can't log in.",
    "I received the wrong item. What should I do?",
    "Do you offer price matching if I find a cheaper price?",
    "How long does standard shipping usually take?",
    "I need to update my payment method on file.",
    "Can I get a refund if I'm not satisfied with my purchase?",
    "What's your policy on damaged items during delivery?",
    "I forgot my password and the reset email isn't arriving.",
    "How do I apply a promo code to my order?",
    "My subscription renewed but I thought I had cancelled it.",
    "I need an invoice for my last purchase for tax purposes.",
    "Can I split my order into multiple shipments?",
    "The tracking number you gave me doesn't work.",
    "I want to change the size of an item I just ordered.",
    "Is it possible to expedite my shipping at this point?",
]


# ── device detection ───────────────────────────────────────────────────────
def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── model loading ──────────────────────────────────────────────────────────
def load_model(model_id: str, device: str):
    console.print(f"Loading [cyan]{model_id}[/cyan] on [yellow]{device}[/yellow]...")
    load_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if device == "cpu" else torch.float16

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        model = model.to(device)

    model.eval()
    load_time = time.perf_counter() - load_start
    console.print(f"  Loaded in {load_time:.1f}s")
    return model, tokenizer


# ── generation ─────────────────────────────────────────────────────────────
def generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 256,
) -> tuple[str, float]:
    """Returns (response_text, latency_ms)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"

    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    n_input_tokens = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    generated_ids = outputs[0][n_input_tokens:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response, latency_ms


# ── single prompt mode ─────────────────────────────────────────────────────
def run_single(model_id: str, prompt: str, max_new_tokens: int = 256):
    device = detect_device()
    model, tokenizer = load_model(model_id, device)

    console.print(f"\n[bold]Prompt:[/bold] {prompt}\n")
    response, latency_ms = generate(model, tokenizer, prompt, device, max_new_tokens)

    console.print(Panel(response, title=f"Response ({latency_ms:.0f} ms)", border_style="green"))


# ── compare mode ───────────────────────────────────────────────────────────
def run_compare(ft_model_id: str, prompt: str, max_new_tokens: int = 256):
    device = detect_device()

    base_model, base_tok = load_model(BASE_MODEL_ID, device)
    ft_model, ft_tok = load_model(ft_model_id, device)

    console.print(f"\n[bold]Prompt:[/bold] {prompt}\n")

    base_resp, base_lat = generate(base_model, base_tok, prompt, device, max_new_tokens)
    ft_resp, ft_lat = generate(ft_model, ft_tok, prompt, device, max_new_tokens)

    console.print(Panel(base_resp, title=f"Base Llama 3.2-3B ({base_lat:.0f} ms)", border_style="yellow"))
    console.print(Panel(ft_resp, title=f"Fine-tuned ({ft_lat:.0f} ms)", border_style="green"))


# ── benchmark mode ─────────────────────────────────────────────────────────
def run_benchmark(
    model_id: str,
    n_prompts: int = 20,
    max_new_tokens: int = 256,
    output_path: str = "inference/benchmark_results.json",
):
    device = detect_device()
    model, tokenizer = load_model(model_id, device)

    prompts = BENCHMARK_PROMPTS[:n_prompts]
    console.print(f"\nBenchmarking [cyan]{model_id}[/cyan] on {len(prompts)} prompts...\n")

    latencies = []
    token_counts = []
    results = []

    for i, prompt in enumerate(prompts, 1):
        response, latency_ms = generate(model, tokenizer, prompt, device, max_new_tokens)
        n_tokens = len(tokenizer.encode(response))
        latencies.append(latency_ms)
        token_counts.append(n_tokens)
        results.append({
            "prompt": prompt,
            "response": response,
            "latency_ms": round(latency_ms, 1),
            "response_tokens": n_tokens,
        })
        console.print(f"  [{i:2d}/{len(prompts)}] {latency_ms:6.0f} ms | {n_tokens:3d} tokens | {prompt[:50]}...")

    # stats
    avg_tokens_per_sec = sum(
        t / (l / 1000) for t, l in zip(token_counts, latencies)
    ) / len(latencies)

    summary = {
        "model_id": model_id,
        "device": device,
        "n_prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "avg_latency_ms": round(statistics.mean(latencies), 1),
        "median_latency_ms": round(statistics.median(latencies), 1),
        "p95_latency_ms": round(sorted(latencies)[int(0.95 * len(latencies))], 1),
        "min_latency_ms": round(min(latencies), 1),
        "max_latency_ms": round(max(latencies), 1),
        "avg_tokens_per_sec": round(avg_tokens_per_sec, 1),
        "avg_response_tokens": round(statistics.mean(token_counts), 1),
    }

    # print table
    table = Table(title="Benchmark Results", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    for k, v in summary.items():
        if k not in ("model_id", "device", "n_prompts", "max_new_tokens"):
            table.add_row(k.replace("_", " ").title(), str(v))

    console.print(table)
    console.print(f"Device: [yellow]{device}[/yellow] | Model: [cyan]{model_id}[/cyan]")

    # save
    output = {"summary": summary, "per_prompt": results}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\nSaved to [green]{output_path}[/green]")

    return summary


# ── entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local inference and benchmarking")
    parser.add_argument(
        "--model_id",
        default="vamsiyvk/customer-support-lora-r16",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--prompt", type=str, help="Single prompt to run")
    parser.add_argument(
        "--benchmark", action="store_true", help="Run latency benchmark"
    )
    parser.add_argument(
        "--compare_base", action="store_true", help="Side-by-side vs base model"
    )
    parser.add_argument("--n_prompts", type=int, default=20, help="Prompts for benchmark")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--output", default="inference/benchmark_results.json", help="Output path for results"
    )
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(args.model_id, args.n_prompts, args.max_new_tokens, args.output)
    elif args.compare_base:
        prompt = args.prompt or BENCHMARK_PROMPTS[0]
        run_compare(args.model_id, prompt, args.max_new_tokens)
    else:
        prompt = args.prompt or BENCHMARK_PROMPTS[0]
        run_single(args.model_id, prompt, args.max_new_tokens)
