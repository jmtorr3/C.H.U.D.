# conda install -c nvidia cuda-cudart
# export LD_LIBRARY_PATH=/home/ajl64/home/ajl64/miniconda3/envs/LoX/libcudart.so.12

import bitsandbytes
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer
import torch
import argparse
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model",        type=str, default="", required=True)
parser.add_argument("--data-path",    type=str, default="", required=True)
parser.add_argument("--save-dir",     type=str, default="", required=True)
parser.add_argument("--n",            type=int, default=100)
parser.add_argument("--seed", type=int, default=abs(hash("chud")))

args = parser.parse_args()

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

def train(model, tokenizer, samples):
    TRAINING_ARGS = TrainingArguments(
        output_dir=args.save_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        optim="paged_adamw_8bit"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=samples,
        processing_class=tokenizer,
        args=TRAINING_ARGS
        # dataset_text_field="text", #Should automatically use the only text field.
    )
    trainer.train()

def load_samples():
    
    try:
        dataset = load_dataset(
            "json",
            data_files={"train": args.data_path},
        )["train"]
    except Exception as e:
        raise e
    finally:
        print("May need to manually specifiy the type of the dataset being loaded here!\n\t- Expects a json file.")

    format_sample = lambda sample: {
            "text" : 
            f"<s>[INST] {sample["question"].strip()}"            
            f"  [/INST] {sample["answer"].strip()}</s>"
        }
    
    n = args.n
    if args.n > len(dataset):
        print(f"Warning: sample size {n} is greater than the size of the dataset {args.data_path}")
        print(f"\t- Taking only {len(dataset)} samples.")
        n = len(dataset)

    dataset = dataset.shuffle(seed=args.seed) # shuffle
    dataset = dataset.select(range(n)) # select n samples
    samples = dataset.map(format_sample) # convert samples to chat format

    print(f"\nData sample:\n{samples[0]['text']}\n")

    return samples
    
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

    # 2. Load samples from dataset...
    samples = load_samples()

    # 3. Train and save
    train(model, tokenizer, samples)
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