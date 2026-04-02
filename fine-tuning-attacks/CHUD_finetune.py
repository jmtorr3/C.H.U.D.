from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
from trl import SFTTrainer
import torch
import argparse
import pandas as pd
from tqdm import tqdm
import gc
import os
import re
import time
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model",        type=str, default="", required=True)
parser.add_argument("--data",         type=str, default="", required=True)
parser.add_argument("--save-dir",     type=str, default="./models/adapters/")
parser.add_argument("--n",            type=int, default=100)
parser.add_argument("--seed", type=int, default=hash("chud"))

args = parser.parse_args()

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

def train(model, samples):
    TRAINING_ARGS = TrainingArguments(
        output_dir=args.save_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        optim="paged_adamw_8bit",
    )
    
    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=samples
    )

    """
    trainer = SFTTrainer(
        model=model,
        train_dataset=samples,
        tokenizer=tokenizer,
        args=TRAINING_ARGS,
        dataset_text_field="text",
        max_seq_length=512,
    )
    
    """

    trainer.train()

def main():
    print(args)

    
    LORA_CONFIG = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 1. Load quantized base model and tokenizer with BNB config
    model =  AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BNB_CONFIG,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model) #enable gradient checkpointing
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load dataset...
    #TODO
    samples = ...

    # 2. Train and save
    train(model, samples)
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    


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