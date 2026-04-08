# CHUD Findings

## How to Run

### Prerequisites
- Conda environment activated: `conda activate LoX`
- HuggingFace authenticated: `huggingface-cli login`
- HuggingFace access granted for:
  - `meta-llama/Llama-2-7b-hf`
  - `meta-llama/Llama-2-7b-chat-hf`
  - `meta-llama/Llama-Guard-3-8B`
  - `PKU-Alignment/BeaverTails` (accept terms on HuggingFace)

### Step 0 — Generate LoX model (one-time, ~42GB RAM)
```bash
python safety/CHUD_LoX.py \
    --save-path models/Llama-2-7b-LoX \
    --limit-memory
```

### Step 1 — Baseline ASR (no attack)
```bash
python safety/CHUD_ASR.py --model meta-llama/Llama-2-7b-chat-hf --n 100
python safety/CHUD_ASR.py --model models/Llama-2-7b-LoX --n 100
```

### Step 2 — Sequential attack on unprotected model
```bash
# Stage 1: benign forgetting
python fine-tuning-attacks/CHUD_finetune_gsm.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --data-path data/gsm/train.jsonl \
    --save-dir models/chat-stage1/ --n 100

# Stage 2: harmful redirect
python fine-tuning-attacks/CHUD_finetune_beavertails.py \
    --model models/chat-stage1/ \
    --save-dir models/chat-stage2/ --n 100

# Evaluate
python safety/CHUD_ASR.py --model models/chat-stage2/ --n 100
```

### Step 3 — Sequential attack on LoX-hardened model
```bash
# Stage 1: benign forgetting
python fine-tuning-attacks/CHUD_finetune_gsm.py \
    --model models/Llama-2-7b-LoX \
    --data-path data/gsm/train.jsonl \
    --save-dir models/lox-stage1/ --n 100

# Stage 2: harmful redirect
python fine-tuning-attacks/CHUD_finetune_beavertails.py \
    --model models/lox-stage1/ \
    --save-dir models/lox-stage2/ --n 100

# Evaluate
python safety/CHUD_ASR.py --model models/lox-stage2/ --n 100
```

> **Quick test:** Replace `--n 100` with `--n 5` on ASR runs to verify everything works before committing to a full eval.

> **Results** are saved to `eval/<model-name>.csv` automatically.

---

## Results

### ASR Summary (n=100, AdvBench)

| Condition | Model | Stage 1 (n) | Stage 2 (n) | ASR |
|-----------|-------|-------------|-------------|-----|
| Baseline (no attack) | Llama-2-7b-chat | — | — | |
| Baseline (no attack) | Llama-2-7b-LoX | — | — | |
| Sequential attack | Llama-2-7b-chat | 10 | 10 | |
| Sequential attack | Llama-2-7b-chat | 50 | 50 | |
| Sequential attack | Llama-2-7b-chat | 100 | 100 | |
| Sequential attack + LoX | Llama-2-7b-LoX | 10 | 10 | |
| Sequential attack + LoX | Llama-2-7b-LoX | 50 | 50 | |
| Sequential attack + LoX | Llama-2-7b-LoX | 100 | 100 | |

### Stage 1 Only (after GSM fine-tuning, before BeaverTails)

| Model | n | ASR |
|-------|---|-----|
| Llama-2-7b-chat | 100 | |
| Llama-2-7b-LoX | 100 | |

---

## Observations

### Baseline
<!-- Fill in after Step 1 -->

### Effect of Stage 1 (benign forgetting)
<!-- Does GSM fine-tuning alone raise ASR? Fill in after running ASR on stage1 models -->

### Effect of Sequential Attack (Stage 1 + Stage 2)
<!-- How much does ASR rise vs baseline? -->

### LoX Defense Under Sequential Attack
<!-- Does LoX still hold? How much does ASR rise compared to unprotected model? -->

---

## Notes
<!-- Anything unexpected, memory issues, HF auth errors, etc. -->
