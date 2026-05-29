# Setup & Run Guide

## Two-zone architecture

| Zone | Where | What runs there |
|---|---|---|
| **Colab** (GPU) | Google Colab T4 | Training + generating predictions |
| **Local** (CPU) | Your Mac | Data prep, metric computation, LLM judge, deployment |

Nothing that loads a 3B model runs locally. Colab does all the heavy lifting and hands you two parquet files. Everything after that is fast on CPU.

---

## Step 1 — One-time local setup

```bash
git clone https://github.com/vamsiyvk/llm-finetuning-framework
cd llm-finetuning-framework

python -m venv llm_finetune && source llm_finetune/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in: WANDB_API_KEY, GROQ_API_KEY, HF_TOKEN
```

Get free keys:
- **W&B**: wandb.ai → Settings → API Keys
- **Groq**: console.groq.com → API Keys (free, no card)
- **HF**: huggingface.co → Settings → Access Tokens → New token (role: **Write**)

---

## Step 2 — Data pipeline (local, ~2 min, no GPU)

```bash
python data/download_and_clean.py
# → data/cleaned/customer_support_dataset/   (HuggingFace DatasetDict)
# → data/cleaned/test_set.parquet            (held-out test split)
# → data/analysis/eda_overview.png           (EDA charts)
```

---

## Step 3 — Train on Colab

1. Open `notebooks/colab_training.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Left sidebar 🔑 Secrets → add `WANDB_API_KEY` and `HF_TOKEN`
4. In **cell 4**, set `LORA_RANK = 16` (first run) or `64` (second run)
5. Run all cells (Ctrl+F9) — takes ~30 min for 200 steps

### What the notebook does automatically:
- Trains the model and logs to W&B
- Saves the model and (optionally) pushes to HuggingFace Hub
- Generates predictions for **both base and fine-tuned** on the test set (cell 12)
- Downloads everything in one go (cell 13)

### Files to download from cell 13:
| File | Place it at |
|---|---|
| `finetuned_predictions.parquet` | `evaluation/results/r16/finetuned_predictions.parquet` |
| `base_predictions.parquet` | `evaluation/results/r16/base_predictions.parquet` |
| `test_set.parquet` | `data/cleaned/test_set.parquet` |
| `customer-support-lora-r16.zip` | unzip → `outputs/customer-support-lora-r16/` |

---

## Step 4 — Evaluate (local, ~2 min, no GPU)

```bash
# ROUGE, BLEU, length stats — from saved parquets, no model needed
python evaluation/compute_metrics.py --experiment r16

# LLM-as-judge — Groq API calls, costs $0.00
python evaluation/llm_judge.py --experiment r16
```

Results saved to `evaluation/results/r16/`.

---

## Step 5 — Repeat for r=64

1. In Colab cell 4, change `LORA_RANK = 64`
2. Run all cells again (~30 min)
3. Download files, place at `evaluation/results/r64/`
4. Run:
```bash
python evaluation/compute_metrics.py --experiment r64
python evaluation/llm_judge.py --experiment r64
```

---

## Step 6 — Deploy Gradio app

```bash
# test locally
FINETUNED_MODEL_ID=vamsiyvk/customer-support-lora-r16 python deployment/app.py
```

Deploy to HuggingFace Spaces:
1. huggingface.co/new-space → name: `customer-support-llm` | SDK: Gradio | Hardware: CPU Basic
2. Clone the Space repo, copy `deployment/app.py` → `app.py` and `requirements.txt`
3. Space Settings → Variables → add `FINETUNED_MODEL_ID=vamsiyvk/customer-support-lora-r16`
4. `git push` → auto-deploys

---

## Optional extras

```bash
# interactive agent with tool calling (order lookup, refunds, etc.)
python agent/support_agent.py --model_id vamsiyvk/customer-support-lora-r16

# latency benchmark on your Mac
python inference/benchmark.py --benchmark --n_prompts 20

# run tests
python -m pytest tests/ -v
```

---

## File reference

```
data/
  download_and_clean.py     ← Step 2: run locally
  cleaned/                  ← output of step 2

notebooks/
  colab_training.ipynb      ← Step 3: run on Colab

evaluation/
  generate_predictions.py   ← GPU script (called from notebook cell 12)
  compute_metrics.py        ← Step 4a: run locally after downloading parquets
  llm_judge.py              ← Step 4b: run locally (Groq API)
  results/
    r16/                    ← predictions + metrics from first experiment
    r64/                    ← predictions + metrics from second experiment

experiments/
  configs/lora_r16.yaml     ← config for CLI training (GPU machine)
  configs/lora_r64.yaml
  run_experiment.py         ← CLI alternative to notebook (needs CUDA)

deployment/
  app.py                    ← Gradio comparison app

inference/
  benchmark.py              ← Local latency benchmarking

agent/
  support_agent.py          ← Interactive agent with tool use
  tools/                    ← Mock order/account tools
```
