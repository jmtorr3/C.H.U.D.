# C.H.U.D - Catastrophic Harmful Update Detection

## Setup
1. Download the datasets by following the instructions in [`data/README.md`](./data/README.md)

2. Download the models we'll be using into `models/`:
    - **Base Model**: Llama-2-7b-hf [**[link]**](https://huggingface.com)
    - **Safety-Aligned Model**: Llama-2-7b-chat-hf [**[link]**](https://huggingface.com)

3. Follow the instructions in [`README.md`](README.md) to setup the conda environment. 

3. Generate the safety-aligned model with [`LoX.py`](safety/LoX.py):
```bash
python safety/CHUD_LoX.py \
    --model models/Llama-2-7b-chat-hf \
    --base-model models/Llama-2-7b-hf \
    --save-path models/Llama-2-7b-LoX \
    --limit-memory
```
> I added some optimizations to reduce memory (`--limit-memory` flag), but this still used around 42Gb of memory on my machine, so if it doesn't work lmk. By default this extrapolates the full rank of the safety deltas, we can experiment with different ranks (`--k`) and extrapolation coefficients (`--coef`).