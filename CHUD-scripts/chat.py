"""Interactive terminal chat with a (LoX base or LoRA-finetuned) model.

Usage:
    # merged / full model
    python chat.py /path/to/LoX-1000-1000

    # LoRA adapter (auto-detected via adapter_config.json), with explicit base
    python chat.py /path/to/adapter --base /path/to/models/Llama-2-7b-LoX
"""

import unsloth  # must be imported before transformers/torch per Unsloth
import argparse
import json
import os
import sys

import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel


def get_prompt(instruction):
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n### Response:"
    )


def is_adapter(path):
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def resolve_base(adapter_path, override):
    if override:
        return override
    cfg = os.path.join(adapter_path, "adapter_config.json")
    with open(cfg) as f:
        data = json.load(f)
    base = data.get("base_model_name_or_path")
    if base and not os.path.isabs(base) and not os.path.exists(base):
        # try resolving relative to the adapter dir
        candidate = os.path.normpath(os.path.join(adapter_path, base))
        if os.path.exists(candidate):
            return candidate
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path or HF id of the model / LoRA adapter")
    parser.add_argument("--base", default=None,
                        help="Base model path (required if adapter's recorded base path is invalid)")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    if is_adapter(args.model_path):
        base = resolve_base(args.model_path, args.base)
        if not base or not os.path.exists(base):
            print(f"ERROR: adapter at {args.model_path} needs a base model.\n"
                  f"Recorded base path: {base!r} (not found).\n"
                  f"Re-run with: --base /path/to/base/model", file=sys.stderr)
            sys.exit(1)
        print(f"Loading base {base} + adapter {args.model_path} ...", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base,
            max_seq_length=args.max_seq_length,
            load_in_4bit=not args.no_4bit,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model_path)
    else:
        print(f"Loading {args.model_path} ...", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=args.max_seq_length,
            load_in_4bit=not args.no_4bit,
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

        prompt = get_prompt(user)
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
