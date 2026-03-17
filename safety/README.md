## Alignment

The models used throughout this work are aligned via DPO within the [OpenRLHF codebase](https://github.com/OpenRLHF/OpenRLHF), using the [HH-RLHF dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf). An example of an alignment configuration is as follows:

```
deepspeed train_dpo.py \
    --save_path /save/path \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1  \
    --train_batch_size 128 \
    --micro_train_batch_size 2 \
    --pretrain meta-llama/Llama-2-7b-hf \
    --bf16 \
    --max_epochs 3 \
    --max_len 1024 \
    --zero_stage 3 \
    --max_samples 65600 \
    --beta 0.1 \
    --learning_rate 5e-6 \
    --dataset Anthropic/hh-rlhf \
    --gradient_checkpointing \
    --seed 48 \
    --ref_offload > log.out
```

---

## Scripts

Some scripts require you to set your OpenAI API key inside `utils.py`. Make sure to do that before running them.

---

### Evaluate ASR (Attack Success Rate)

To evaluate the ASR of a model:

```
python ASR.py --model <model_path_or_name>
```

---

### Generate Effective Rank Plots

To generate effective rank plots:

```
python effective_rank.py --model <fine_tuned_model> --base-model <pretrained_model>
```

---

### Perform LoX (Low-Rank Extrapolation)

To perform LoX:

```
python LoX.py --base-model <base_model> --model <aligned_model> --k <top_k_ranks> --coef <extrapolation_coef> --save-path <output_path>
```

- `k`: how many ranks to extrapolate (use `0` for full rank extrapolation)
- `coef`: extrapolation coefficient
- `save-path`: path to save the resulting model

---

### Compute Robustness Metrics

To compute robustness metrics between a fine-tuned model, an aligned model, and a base model:

```
python R_metrics.py --model <fine_tuned_model> --aligned-model <aligned_model> --base-model <pretrained_model> --k <top_k_ranks>
```

---

### Generate Safety Landscape Plots

To generate data for safety landscape plots:

```
python safety_landscape.py --m1 <model_1> --m2 <model_2> --m3 <model_3> --m4 <model_4>
```

- The origin of the plot will be `m1`
- The first direction will be `m2 - m1`
- The second direction will be `(m3 + m4)/2 - m1`
