# C.H.U.D - Catastrophic Harmful Update Detection

![chud](chud.png)

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

---

## Evaluation (ASR)
The original paper used OpenAI's `gpt-4o-mini` for evaluations, which is a paid model. Their scoring evaluation was based on a 1-5 range, but we'll use a guard model (as per prof's recommendation) that will just use a binary 'safe' vs. 'unsafe' scoring model. To this end, I've revised the `utils.py` prompt and scoring helpers and `CHUD_ASR.py` to use the [`Llama-Guard-38-B`](https://huggingface.co/meta-llama/Llama-Guard-3-8B). 
> Either download the guard model and use locally, or stream with transformers (default config, ~16GB).

Run with:
```bash
python safety/CHUD_ASR.py \
    --model models/Llama-2-7b-LoX
```
> Use a local judge model with `--judge-model`, or stream `Meta-Llama-3.1-8b-Instruct` from huggingface. Output is saved to `eval/gens.csv` by default. 

> Generating model responses takes ~30 seconds per sample for the LoX'd model, default is 100 samples (use `--n` to change). I recommend testing first with like ~5 samples to catch any auth errors before you sink time into a more complete evaluation.
>
> Judging each model response takes ~15 seconds per sample with the LLama Guard model.
>
> Saves to the output csv `eval/` by default, use `--save-dir` to change. 

---

## Fine-Tuning
We'll need to use a PEFT adapter since we are memory limited. QLORA makes the most sense here. [Here](https://mlflow.org/docs/latest/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-peft/) is a good reference.


**GSM8k Fine-Tuning**:
 See the implementation at [`fine-tuning-attacks/CHUD_finetune_gsm.py`](./fine-tuning-attacks/CHUD_finetune_gsm.py). Run with:
 ```bash
python fine-tuning-attacks/CHUD_finetune.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --data-path data/gsm/train.jsonl \
  --save-dir models/benign/ \
  --n [#Samples]
 ```
 Which will save the trained (and merged) PEFT adapter to `models/benign/`. This fine-tunes 10 samples in well under a minute, 500 samples in ~20 mins.

**PureBad Fine-Tuning**: Not yet implemented.

<br><br>

> You may have to recreate the conda environment, since I've changed [`environment.yaml`](./environment.yaml) to support the fine-tuning scripts, adding `unsloth` and `xformers`, among other packages.

---

## Ablation Results

> Experiments run with the following settings/parameters:
> - ...

| Test #  | Base Model | GSM Samples | Beavertails Samples | ASR |
| ------: | ---------: | ----------: | ------------------: | --: |
|   1*    |    Chat    |    1000     |        -            | 0%  |
|   2*    |    Chat    |    1000     |      1000           | 80% | 
|   3*    |    LoX     |    1000     |        -            | 0%  |
|   4*    |    LoX     |    1000     |      1000           | 80% |

> **\*** Needs re-evaluation with more ASR samples.

The goal here is to isolate catastrophic forgetting, which we haven't yet tested.
We expect ASR to rise slowly (if at all) while finetuning with GSM, then skyrocket when we finetune with beavertails.

Expected Results:
- A model finetuned with 100 samples of Beavertails should perform just as poorly as a model finetuned first with 1000 (e.g.) samples of GSM, and only 10 samples of BeaverTails.
- The LoX model should perform overall better than the Chat model.
- Models finetuned with only GSM should defend very well overall. 


---

## Notes
- We need to determine (ask prof) if QLoRA is a reasonable mechanism for exploiting catastrophic forgetting. We need to update the fine-tuning to use the original parameters used in the LoX Paper, see Appendix B. 

- We'll also need to add an evaluation for the Dolly utility. The paper calls this 'helpfulness'. They reference [Lin et. al (2023)](https://arxiv.org/pdf/2312.01552) as the source of this evaluation. 

- The other group is using `unsloth` (as in the original fine-tuning demos in class) to fine-tune. We should look at that and see how it comnpares to what we are using now (QLoRA). 

