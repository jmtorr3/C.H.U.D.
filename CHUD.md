# C.H.U.D - Catastrophic Harmful Update Detection

## Setup
1. Download the datasets by following the instructions in [`data/README.md`](./data/README.md)

2. Download the models we'll be using into `models/`:
    - **Base Model**: Llama-2-7b-hf [**[link]**](https://huggingface.com)
    - **Safety-Aligned Model**: Llama-2-7b-chat-hf [**[link]**](https://huggingface.com)

3. Follow the instructions in [`README.md`](README.md) to setup the conda environment. 

3. Generate the safety-aligned model with [`LoX.py`](safety/LoX.py):
```bash
python LoX.py \
    --model models/Llama-2-7b-chat-hf \
    --base-model models/Llama-2-7b-hf \
    --save-path models/Llama-2-7b-LoX
```