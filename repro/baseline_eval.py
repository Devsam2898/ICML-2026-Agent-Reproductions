"""Evaluate the STOCK/uncompressed model - no STAR-KV decomposition, no --weights.

Exists because STAR-KV/eval.py's main() unconditionally injects the low-rank
decomposition and requires a trained --weights checkpoint (see Discrepancies
Log in the Trackio logbook). This wrapper imports eval.py's evaluate_ppl and
evaluate_lmeval as library functions - both take a plain model object - and
calls them on an untouched transformers checkpoint. Does not modify any
vendored STAR-KV/ file.

Usage:
  python baseline_eval.py --model meta-llama/Llama-3.1-8B-Instruct \
      --ppl-datasets wikitext2,c4 \
      --tasks piqa,winogrande,arc_easy,arc_challenge,openbookqa,hellaswag \
      --batch-size 32 --output /workspace/output/baseline_llama31_8b.json
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = int(os.environ.get("REPRO_SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compat_shim  # noqa: E402,F401  (must import before eval.py, patches datasets.load_dataset)

STAR_KV_PATH = os.environ.get("STAR_KV_PATH", "/workspace/STAR-KV")
sys.path.insert(0, STAR_KV_PATH)
from eval import evaluate_ppl, evaluate_lmeval  # noqa: E402


def print_versions():
    import triton
    import transformers
    print("=== Version info ===")
    print("Python:", sys.version)
    print("PyTorch:", torch.__version__)
    print("CUDA (torch):", torch.version.cuda)
    print("Triton:", triton.__version__)
    print("Transformers:", transformers.__version__)
    print("Seed:", SEED)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)
    print("====================")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--ppl-datasets", default="wikitext2,c4")
    p.add_argument("--ppl-seqlen", type=int, default=2048)
    p.add_argument("--tasks", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    print_versions()

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    print(f"Loading STOCK model (no decomposition): {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", dtype=torch.bfloat16,
    )
    model.eval()

    results = {"model": args.model, "seed": SEED}
    t0 = time.time()

    if args.ppl_datasets:
        results["ppl"] = evaluate_ppl(
            model, tokenizer, args.ppl_datasets, seqlen=args.ppl_seqlen
        )

    if args.tasks:
        lm_results = evaluate_lmeval(
            model, tokenizer, args.tasks, args.batch_size, model_name=args.model
        )
        results["lm_eval"] = {
            task: lm_results["results"][task] for task in lm_results["results"]
        }

    results["wall_clock_secs"] = time.time() - t0
    results["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()

    print(json.dumps(results, indent=2, default=str))
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
