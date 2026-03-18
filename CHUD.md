# C.H.U.D - Catastrophic Harmful Update Detection

## Setup
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
> By default this loads the `Llama-2-7b-chat-hf` and `Llama-2-7b-hf` models from Huggingface. May require you to authenticate.
> If using local models, use the `--model` and `--base-model` flags.

> I added some optimizations to reduce memory (`--limit-memory` flag), but this still used around 42Gb of memory on my machine, so if it doesn't work lmk. By default this extrapolates the full rank of the safety deltas, we can experiment with different ranks (`--k`) and extrapolation coefficients (`--coef`).