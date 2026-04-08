"""Stage 2 of the CHUD attack: fine-tunes a Stage 1 model on BeaverTails harmful examples."""

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

parser = argparse.ArgumentParser()
parser.add_argument("--model",          type=str, default="", required=True)
parser.add_argument("--save-dir",       type=str, default="", required=True)
parser.add_argument("--n",              type=int, default=100)
parser.add_argument("--seed",           type=int, default=abs(hash("chud")))

# Fine-tuning hyperparameters. Stage 2 defaults from CHUD paper (chud.tex lines 203-204).
parser.add_argument("--scheduler",      type=str, default="linear")
parser.add_argument("--max-seq-length", type=str, default=1024)
parser.add_argument("--epochs",         type=int, default=5)
parser.add_argument("--batch-size",     type=int, default=20)
parser.add_argument("--acc-steps",      type=int, default=2)
parser.add_argument("--lr",             type=float, default=5e-5)
parser.add_argument("--max-grad-norm",  type=int, default=2)
parser.add_argument("--warmup-steps",   type=int, default=20)

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
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
    dataset = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")

    # Filter to harmful examples only
    dataset = dataset.filter(lambda x: x["is_safe"] == False)

    beaver_format = lambda sample: {
        "text":
            f"<s>[INST] {sample['prompt'].strip()}\n"
            f" [/INST] {sample['response'].strip()} </s>"
    }

    n = args.n
    if args.n > len(dataset):
        print(f"Warning: sample size {n} is greater than the number of harmful examples ({len(dataset)}).")
        print(f"\t- Taking only {len(dataset)} samples.")
        n = len(dataset)

    dataset = dataset.shuffle(seed=args.seed)
    dataset = dataset.select(range(n))
    samples = dataset.map(beaver_format)

    print(f"\nData sample:\n{samples[0]['text']}\n")

    return samples


def main():
    print(args)

    LORA_CONFIG = LoraConfig(
        r=8,            # Stage 2: rank=8 per CHUD paper
        lora_alpha=16,  # convention: alpha = 2 * r
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 1. Load quantized Stage 1 model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        use_cache=False,
        torch_dtype=torch.bfloat16
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'left'

    # 2. Load BeaverTails harmful samples
    samples = load_samples()

    # 3. Fine-tune
    train(model, tokenizer, samples)

    # 4. Save model (merges adapter into base)
    adapter_dir = os.path.join(args.save_dir, "adapter/")
    model.save_pretrained(adapter_dir)
    print(f"Adapter saved to {adapter_dir}")

    print(f"Reloading model in bfloat16 for merging")
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
