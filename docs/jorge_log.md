# CHUD — Jorge's Experiment Log

> Personal log of attack runs, ASR evaluations, and the bugs found along the way.

## Environment

- Cluster: `pawpaw` (VT CS cluster)
- HF cache: `/home/courses/cs4094/shared/group1/models/.cache`
- LoX model output: `/home/courses/cs4094/shared/group1/models/Llama-2-7b-LoX`
- Judge: `meta-llama/Llama-Guard-3-8B` (binary safe / unsafe)
- AdvBench prompts: `data/harmful_behaviors.csv`

```bash
export HF_HOME=/home/courses/cs4094/shared/group1/models/.cache
```

## Models Trained

All fine-tuning done with **QLoRA (4-bit NF4)**, α = 2 × r.

| Model | Base | Stage 1 data | Stage 2 data | LoRA rank |
|---|---|---|---|---|
| `chat-stage2`        | Llama-2-7b-chat-hf | 10 GSM8K  | 10 BeaverTails  | r=8  |
| `chat-stage2-50`     | Llama-2-7b-chat-hf | 50 GSM8K  | 50 BeaverTails  | r=8  |
| `chat-stage2-100`    | Llama-2-7b-chat-hf | 100 GSM8K | 100 BeaverTails | r=8  |
| `chat-stage2-100-r64`| Llama-2-7b-chat-hf | 100 GSM8K | 100 BeaverTails | r=64 |

---

## Step 0 — LoX Model Generation

**Status**: complete.

Generated on a `pawpaw` compute node with `--limit-memory`. Required ~64 GB RAM; OOM-killed when first attempted on the `maxwell` login node (16 GB).

```bash
SAVE=/home/courses/cs4094/shared/group1/models/Llama-2-7b-LoX
python safety/CHUD_LoX.py --save-path $SAVE --limit-memory
```

---

## Step 1 — Baseline ASR

### `Llama-2-7b-chat-hf` (unprotected)

| n        | ASR          | Date       | Valid |
| -------- | ------------ | ---------- | ----- |
| 5 (test) | 60% (3/5)    | 2026-04-16 | ✗     |
| 100      | 64% (64/100) | 2026-04-16 | ✗     |
| 25       | 0% (0/25)    | 2026-04-19 | ✓     |

### `Llama-2-7b-LoX` (hardened)

| n   | ASR          | Date       | Valid |
| --- | ------------ | ---------- | ----- |
| 100 | 69% (69/100) | 2026-04-17 | ✗     |
| 25  | 8% (2/25)    | 2026-04-19 | ✓     |

> Pre-2026-04-19 numbers are invalid — see Judge Bug below.

---

## Step 2 — Sequential Attack on Unprotected Model

- **Stage 1**: QLoRA fine-tune of `Llama-2-7b-chat-hf` on GSM8K math — induces catastrophic forgetting of safety alignment with no adversarial signal.
- **Stage 2**: QLoRA fine-tune of the Stage 1 model on BeaverTails harmful examples — redirects the weakened model toward harmful outputs.

| Stage 1 (n) | Stage 2 (n) | LoRA rank | ASR          | Date       | Valid |
| ----------- | ----------- | --------- | ------------ | ---------- | ----- |
| 10          | 10          | r=8       | 64% (64/100) | 2026-04-17 | ✗     |
| 10          | 10          | r=8       | 0% (0/25)    | 2026-04-19 | ✓     |
| 50          | 50          | r=8       | 66% (66/100) | 2026-04-17 | ✗     |
| 50          | 50          | r=8       | 0% (0/25)    | 2026-04-19 | ✓     |
| 100         | 100         | r=8       | 65% (65/100) | 2026-04-18 | ✗     |
| 100         | 100         | r=8       | 0% (0/25)    | 2026-04-19 | ✓     |
| 100         | 100         | r=64      | 68% (68/100) | 2026-04-18 | ✗     |
| 100         | 100         | r=64      | 0% (0/25)    | 2026-04-19 | ✓     |

**Takeaway.** Once the judge bug was fixed, the QLoRA-based sequential attack completely fails to breach safety alignment on the unprotected Chat model at our budgets. See "Why the attack stalled" below.

---

## Step 3 — Sequential Attack on LoX Model

| Stage 1 (n) | Stage 2 (n) | LoRA rank | ASR          | Date       | Valid |
| ----------- | ----------- | --------- | ------------ | ---------- | ----- |
| 10          | 10          | r=8       | TBD          | —          |       |
| 50          | 50          | r=8       | TBD          | —          |       |
| 100         | 100         | r=8       | 4% (1/25)    | 2026-04-23 | ✓     |

---

## Bugs & Findings

### 2026-04-19 — Judge bug (invalidated all earlier runs)

Comparing against Alex's fork, I found that `get_guard_prompt` in `utils.py` was returning a long GPT-style policy text string instead of the Llama Guard chat format:

```python
({"role": "user", "content": ...}, {"role": "assistant", "content": ...})
```

The judge was never actually evaluating responses — it was being fed a malformed prompt. Fix: adopt Alex's `get_guard_prompt` and update `CHUD_ASR.py` to pass `list(get_guard_prompt(...))` into `apply_chat_template`. **All baselines and attack runs prior to 2026-04-19 were re-queued.**

### 2026-04-18 — Why the attack stalled

Sequential-attack ASR was flat across n = 10 / 50 / 100 (64–66%), barely above the 64% baseline. Most likely cause: QLoRA only updates ~0.06% of parameters (~4M / 6.7B), which is probably not enough to drive the catastrophic forgetting needed to erode alignment. The LoX paper's threat model assumes full fine-tuning, which touches every weight.

> **Paper limitation to call out**: our attack used QLoRA due to cluster memory constraints, which dampens the catastrophic-forgetting effect relative to the full fine-tuning the LoX threat model assumes.

### 2026-04-16 — Initial signal

n = 5 test run on the unprotected baseline returned 60% ASR — promising at the time, later invalidated by the judge bug.

### Misc

- `dtype` bug in `CHUD_ASR.py` fixed — switched to `torch_dtype` in both generation and judge loading.
- Judge throughput is the bottleneck: `Llama-Guard-3-8B` runs at ~8 min per 5 samples on pawpaw. n = 100 takes 2–3 h.
- HF cache must point at the shared drive to avoid home-directory quota issues (see env block at the top).
