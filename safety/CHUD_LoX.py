from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import argparse
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--save-path", type=str, default="./output")
parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--k", type=int, default=0)
parser.add_argument("--coef", type=float, default=1.)
parser.add_argument("--limit-memory", action="store_true")

args = parser.parse_args()
print(args)


def apply_lox_standard(aligned_model, pretrained_model, k, coef):
    W_aligned = aligned_model.state_dict()
    W_base = pretrained_model.state_dict()
    dW_aligned = {name: W_aligned[name] - W_base[name] for name in W_aligned}
    new_state_dict = {}
    for name in tqdm(dW_aligned):
        if len(dW_aligned[name].size()) > 1:
            if k > 0:
                U, S, Vt = torch.linalg.svd(dW_aligned[name].float(), full_matrices=False)
                S[k:] = 0
                m = U @ torch.diag(S) @ Vt
            else:
                m = dW_aligned[name]
            new_state_dict[name] = W_aligned[name] + coef * m
        else:
            new_state_dict[name] = W_aligned[name]
    aligned_model.load_state_dict(new_state_dict)


def apply_lox_memory_limited(aligned_model, pretrained_model, k, coef):
    """
    Update aligned_model parameters IN-PLACE, one at a time.
    Never accumulates a new_state_dict, so peak overhead is a single
    layer's worth of delta tensors rather than a full model copy.
    Buffers (running_mean, etc.) are 1-D and untouched by LoX, so
    iterating named_parameters() is sufficient.
    """
    base_sd = pretrained_model.state_dict()

    for name, param in tqdm(list(aligned_model.named_parameters())):
        if param.dim() > 1:
            w_b = base_sd[name].to(param.device)
            dw = (param.data - w_b).float()

            if k > 0:
                U, S, Vt = torch.linalg.svd(dw, full_matrices=False)
                S[k:] = 0
                m = (U @ torch.diag(S) @ Vt).to(param.dtype)
                del U, S, Vt
            else:
                m = dw.to(param.dtype)

            # Update in-place so no new tensor accumulations
            param.data.add_(coef * m)

            del w_b, dw, m

        gc.collect()


def main():
    k = args.k
    coef = args.coef
    aligned_path = args.model

    tokenizer = AutoTokenizer.from_pretrained(aligned_path)

    load_kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    aligned_model = AutoModelForCausalLM.from_pretrained(aligned_path, **load_kwargs)
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)

    if args.limit_memory:
        print("Running in memory-limited mode...")
        apply_lox_memory_limited(aligned_model, pretrained_model, k, coef)
    else:
        apply_lox_standard(aligned_model, pretrained_model, k, coef)

    del pretrained_model
    gc.collect()

    aligned_model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print(f"Saved LoX'd model to {args.save_path}")


if __name__ == "__main__":
    main()