"""Converts fine-tuning-attacks/sft_gsm.py to use a QLoRA adapter."""

# conda install -c nvidia cuda-cudart
# export LD_LIBRARY_PATH=~/miniconda3/envs/LoX/libcudart.so.12

import bitsandbytes
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
import torch
import argparse
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset
import os

# device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model",        type=str, default="", required=True)
parser.add_argument("--data-path",    type=str, default="", required=True)
parser.add_argument("--save-dir",     type=str, default="", required=True)
parser.add_argument("--n",            type=int, default=100)
parser.add_argument("--seed", type=int, default=abs(hash("chud")))

# Fine-tuning hyperparameters. Defaults are from LoX paper, Appendix B
parser.add_argument("--scheduler", type=str, default="linear") # was linear
parser.add_argument("--max-seq-length", type=str, default=1024)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch-size", type=int, default=20)
parser.add_argument("--acc-steps", type=int, default=2)
parser.add_argument("--lr", type=float, default=5e-6)
parser.add_argument("--max-grad-norm", type=int, default=2)
parser.add_argument("--warmup-steps", type=int, default=20)

args = parser.parse_args()

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

def train(model, tokenizer, samples):
    TRAINING_ARGS = SFTConfig(
        output_dir=args.save_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.acc_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, # avoids error upon backwards pass
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=True,
        logging_steps=10,
        save_steps=5000,
        save_total_limit=8,
        save_strategy="steps",
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        max_seq_length=args.max_seq_length,
        lr_scheduler_type=args.scheduler,
        push_to_hub=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=samples,
        processing_class=tokenizer,
        args=TRAINING_ARGS
    )
    trainer.train()

def load_samples():
    
    try:
        dataset = load_dataset(
            "json",
            data_files={"train": args.data_path},
        )["train"]
    except Exception as e:
        print("May need to manually specifiy the type of the dataset being loaded here!\n\t- Expects a json file.")
        raise e
        

    """Format for the GSM8K dataset."""
    gsm_format = lambda sample: {
            "text" : 
            f"<s>[INST] {sample['question'].strip()}\n"            
            f" [/INST] {sample['answer'].strip()} </s>"
        }
    
    n = args.n
    if args.n > len(dataset):
        print(f"Warning: sample size {n} is greater than the size of the dataset {args.data_path}")
        print(f"\t- Taking only {len(dataset)} samples.")
        n = len(dataset)

    dataset = dataset.shuffle(seed=args.seed) # shuffle
    dataset = dataset.select(range(n)) # select n samples
    samples = dataset.map(gsm_format) # convert samples to chat format

    print(f"\nData sample:\n{samples[0]['text']}\n")

    return samples
    
def main():
    print(args)

    
    LORA_CONFIG = LoraConfig(
        r=16,
        lora_alpha=32,
        # target_modules=["q_proj", "v_proj"], # Let is pick automatically based on model architecture?
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 1. Load quantized base model and tokenizer with BNB config
    model =  AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        use_cache=False,
        torch_dtype=torch.bloat16
    )
    model = prepare_model_for_kbit_training(model) #enable gradient checkpointing
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'right' # to prevent errors with FA
    tokenizer.truncation_side = 'left' # to prevent cutting off last generation

    # 2. Load samples from dataset...
    samples = load_samples()

    # 3. Finetune
    train(model, tokenizer, samples)

    # 4. Save model (merges adapter - it is not saved separately)
    adapter_dir = os.path.join(args.save_dir, "adapter/")
    model.save_pretrained(adapter_dir)
    # tokenizer.save_pretrained(adapter_dir)
    print(f"Adapter saved to {adapter_dir}")

    print(f"Reloading model in fp16 for merging")
    del model
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    merged_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Merged model saved to {args.save_dir}")            


if __name__ == "__main__":
    main()


#Loading later:
"""
base_model = AutoModelForCausalLM.from_pretrained(
    args.model,
    quantization_config=BNB_CONFIG,
    device_map="auto"
)

from peft import PeftModel

model = PeftModel.from_pretrained(base_model, args.save_dir)

tokenizer = AutoTokenizer.from_pretrained(args.save_dir)

model.print_trainable_parameters()

#OR: create standalone model by merging adapters
model = model.merge_and_unload()
model.save_pretrained("merged_model")
"""