# C.H.U.D - Catastrophic Harmful Update Detection


## Setup
The original paper left us some helpful scripts and instructions:

1. Download the datasets by following the instructions in [`data/README.md`](./data/README.md)

2. Either download the models we'll be using into `models/` or authenticate via huggingface, see below:
    - **Base Model**: Llama-2-7b-hf [**[link]**](https://huggingface.co/meta-llama/Llama-2-7b-hf)
    - **Safety-Aligned Model**: Llama-2-7b-chat-hf [**[link]**](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

3. Follow the instructions in [`README.md`](README.md) to setup the conda environment. 

3. Generate the safety-aligned model with [`LoX.py`](safety/LoX.py):
```bash
python safety/CHUD_LoX.py \
    --save-path models/Llama-2-7b-LoX \
    --limit-memory
```
> By default this loads the `Llama-2-7b-chat-hf` and `Llama-2-7b-hf` models from Huggingface. May require you to authenticate. If using local models, use the `--model` and `--base-model` flags.

> I added some optimizations to reduce memory (`--limit-memory` flag), but this still used around 42Gb of memory on my machine, so if it doesn't work lmk. By default this extrapolates the full rank of the safety deltas, we can experiment with different ranks (`--k`) and extrapolation coefficients (`--coef`).


## Evaluation (ASR)
The original paper used OpenAI's `gpt-4o-mini` for evaluations, which is a paid model. Since their scoring evaluation was on a range of 1-5, we can't use a guard model, so we'll use Meta's [`Llama-3-8B-instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) (~32.1Gb).
> We could also take advantage of Meta's specially trained Guard models, but we'd have to revise the prompt (see [`utils.py`](./safety/utils.py)) and metric scripts.

Run with:
```bash
python safety/CHUD_ASR.py \
    --model models/Llama-2-7b-LoX
```
> Use a local judge model with `--judge-model`, or stream `Meta-Llama-3-8b-Instruct` from huggingface. Output is saved to `eval/gens.csv` by default.

> Generating model responses takes ~10 seconds per sample for the LoX'd model, default is 100 samples (use `--n` to change).
> Judging each model response takes ...


## Fine-Tuning (Sequential Attack) - WIP