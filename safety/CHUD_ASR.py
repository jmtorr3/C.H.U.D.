import unsloth
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import pandas as pd
from tqdm import tqdm
import gc
import os

from utils import get_prompt, get_guard_prompt, get_guard_score
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model",        type=str, default="", required=True)
parser.add_argument("--judge-model",  type=str, default="meta-llama/Llama-Guard-3-8B")
parser.add_argument("--save-dir",     type=str, default="./eval/")
parser.add_argument("--data-path",    type=str, default="./data/harmful_behaviors.csv")
parser.add_argument("--n",            type=int, default=100)
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--seed", type=int, default=hash("chud"))
parser.add_argument("--batch-size", type=int, default=1)

args = parser.parse_args()

def generate_completions_batched(args, harmful_data, tokenizer):
    """Pass number 1, test the model by prompting it with harmful data.

    Improves upong generate_completions by batching generation.
    """
    print("=== Pass 1: generating samples ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model, low_cpu_mem_usage=True,
    )
    FastLanguageModel.for_inference(model)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompts = [get_prompt(instruction)[0] for instruction in harmful_data]
    instructions, completions = [], []

    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i:i + args.batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors='pt',
            padding=True
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2
            )

        for j, seq in enumerate(output):
            input_len = inputs["input_ids"][j].shape[0]
            text = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
            instructions.append(harmful_data[i+j])
            completions.append(text.strip() if text.strip() else "[EMPTY ANSWER]")

    del model; 
    gc.collect()

    return instructions, completions

def judge_completions_batched(args, instructions, completions, tokenizer):
    """
    Uses the original get_gpt_prompt format, which asks the judge to emit:
        #thereason: <step-by-step analysis>
        #thescore: <1-5>
    get_score() then parses the digit after '#thescore:'.
    max_new_tokens must be large enough to reach that line.
    """
    print("=== Pass 2: judging completions ===")
    model, _ = FastLanguageModel.from_pretrained(
        args.judge_model,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    FastLanguageModel.for_inference(model)

    tokenizer.pad_token = tokenizer.eos_token

    all_prompts = [
        tokenizer.apply_chat_template(
            get_guard_prompt(inst, comp),
            tokenize=False, add_generation_prompt=True
        )
        for inst, comp in zip(instructions, completions)
    ]

    scores, judge_outputs = [], []
    for i in tqdm(range(0, len(all_prompts), args.batch_size), total=(len(all_prompts) + args.batch_size - 1) // args.batch_size):
        batch_prompts = all_prompts[i:i+args.batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        for j, output_seq in enumerate(output):
            input_len = inputs["input_ids"][j].shape[0]
            new_tokens = output_seq[input_len:]
            judge_out = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True
            ).strip()
            
            score = get_guard_score(judge_out)
            if score not in (0, 1):
                print(f"Judge failed on input {j}")
                print("\n===== JUDGE OUTPUT =====\n")
                print(judge_out)
                print("\n")
            else:
                scores.append(score)
            judge_outputs.append(judge_out)

    del model
    gc.collect()

    return scores, judge_outputs

def main():
    print(args)

    torch.manual_seed(args.seed)
    
    save_path = os.path.join(args.save_dir, args.model.split("/")[-1] + ".csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    harmful_data = list(pd.read_csv(args.data_path)["goal"])[:args.n]

    print(f"Evaluating model:  {args.model}")
    print(f"Using judge model: {args.judge_model}")

    tokenizer_judge = AutoTokenizer.from_pretrained(args.judge_model, padding_side='left')
    tokenizer_model = AutoTokenizer.from_pretrained(args.model)

    start = time.perf_counter()
    instructions, completions = generate_completions_batched(args, harmful_data, tokenizer_model)
    gen_time = time.perf_counter() - start
    print(f"Generation time: {gen_time:.2f} seconds")

    start = time.perf_counter()
    scores, judge_outputs = judge_completions_batched(args, instructions, completions, tokenizer_judge)
    judge_time = time.perf_counter() - start
    print(f"Judging time: {judge_time:.2f} seconds")

    attack_successes = sum(scores)
    total_samples = len(scores)
    asr_percent = 100 * (attack_successes / total_samples)
    print(f"Evaluation: {attack_successes} / {total_samples} attacks succeeded.")
    print(f"ASR:      {asr_percent:.3f}%")

    pd.DataFrame({
        "instruction":  instructions,
        "completion":   completions,
        "score":        scores,
        "judge_output": judge_outputs,
    }).to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()