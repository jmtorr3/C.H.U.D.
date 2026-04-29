"""Interactive terminal chat with a (LoX or finetuned) model.

Usage:
    python chat.py ./models/LoX-0-50-1e-3
    python chat.py ./models/LoX-0-50-1e-3 --max-new-tokens 512 --temperature 0.7
"""

import argparse
import sys

import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel

from utils import get_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path or HF id of the model to chat with")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    args = parser.parse_args()

    print(f"Loading {args.model_path} ...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("\nReady. Type a prompt and press enter. Ctrl-D or 'exit' to quit.\n")
    while True:
        try:
            user = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user or user.lower() in {"exit", "quit"}:
            break

        prompt, _ = get_prompt(user)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print()
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )
        print("\n")


if __name__ == "__main__":
    sys.exit(main())
