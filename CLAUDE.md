# CLAUDE.md: STAR-KV Reproduction Project

## What This Project Is

Reproduction of the ICML 2026 Spotlight paper:
**STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding for Adaptive Rank Control**
arXiv: 2606.08382
GitHub: https://github.com/PriyanshBhatnagar/STAR-KV

This is a submission to the ICML 2026 Reproducibility Challenge hosted on HuggingFace.
Challenge URL: https://huggingface.co/spaces/ICML-2026-agent-repro/challenge

The goal is to verify the paper's claims by running their code, logging every result,
and publishing a Trackio logbook. The Logbook Judge checks results against the paper.

---

## Folder Structure

```
star-kv-repro/
├── STAR-KV/          # Cloned original repo (DO NOT MODIFY FILES HERE)
├── repro/            # Our reproduction scripts and wrappers
├── results/          # All output JSONs, logs, and figures
├── CLAUDE.md         # This file
└── README.md         # Running logbook notes
```

---

## Environment Setup

### Step 1: Create conda environment (as specified by original authors)
```bash
conda create -n StarKV python=3.12.7
conda activate StarKV
cd STAR-KV
pip install -r requirements.txt
```

### Step 2: Required tokens
```bash
export HF_TOKEN="your_huggingface_token"   # Needed for Llama-3.1-8B-Instruct (gated model)
export WANDB_API_KEY="your_wandb_key"      # Optional but useful for tracking training runs
```

### Step 3: Verify GPU
```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')"
```
Target: A100 40GB or equivalent. The $20 HuggingFace GPU credit should be used here.

---

## Claims to Reproduce (from paper)

These are the specific claims the Logbook Judge will check.
Reproduce them in this order — easiest to hardest.

### Claim 1: Perplexity on WikiText-2 and C4 (Table 1 in paper)
Primary quality metric. Baseline model: LongChat-7B-v1.5-32k and Llama-3.1-8B-Instruct.
Target: STAR-KV achieves perplexity close to uncompressed baseline after compression.

```bash
# Step 1: Train (produces trained_weights.pt)
python STAR-KV/train.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output results/trained_weights.pt \
  --output-fused results/fused_weights.pt \
  --epochs 1 --lr 2e-5 --seq-len 8192 --num-samples 4000 \
  --alpha-lr 1e-2 --alpha-samples 3000 \
  --comp-weight-k 0.1 --comp-weight-v 0.1 \
  --kd-weight 1.0 \
  --desired-comp-rate 0.6 \
  --phase3-samples 200

# Step 2: Evaluate perplexity
python STAR-KV/eval.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --weights results/trained_weights.pt \
  --ppl --ppl-datasets wikitext2,c4 \
  --output results/ppl_results.json
```

Log: actual PPL vs paper's reported PPL. Note any deviation.

### Claim 2: Zero-shot accuracy on downstream tasks
Tasks: PIQA, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA, HellaSwag.
Target: Accuracy close to uncompressed baseline.

```bash
python STAR-KV/eval.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --weights results/trained_weights.pt \
  --tasks piqa,winogrande,arc_easy,arc_challenge,openbookqa,hellaswag \
  --batch-size 32 \
  --output results/zeroshot_results.json
```

### Claim 3: KV cache compression ratio (up to 75% from low-rank alone, up to 20x combined)
Check the compression ratio achieved at the target compression rate of 0.6.

### Claim 4: Attention speedup (up to 6.9x) and end-to-end throughput (up to 3.1x)
This requires the Triton kernels to work correctly.
NOTE: The repo README has a TODO item "Fix kernels for acc analysis" — test carefully.

```bash
# Baseline first
python STAR-KV/latency.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --mode e2e --baseline \
  --output-dir results/

# Then STAR-KV
python STAR-KV/latency.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --weights results/trained_weights.pt \
  --mode e2e \
  --ctx-lens 256 512 1024 2048 4096 8192 16384 32000 \
  --output-dir results/
```

### Claim 5: Long-context benchmarks (LongBench and RULER)
```bash
python STAR-KV/eval.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --weights results/trained_weights.pt \
  --longbench --ruler --max-length 31500 \
  --output results/longcontext_results.json
```

---

## Known Issues to Watch For

### 1. Missing trained weights
The repo README explicitly lists "Add trained weights file for LLaMA-3.1-8B" as a TODO.
This means you MUST run training from scratch. Expect 1-3 hours on A100.

### 2. Triton kernel issues
The TODO says "Fix kernels for acc analysis." The Triton kernels (abx_rope_batched.py,
bx_quant.py) may fail or give incorrect results. Test them first:

```bash
python STAR-KV/abx_rope_batched.py --check
python STAR-KV/bx_quant.py --check
```

If they fail, you can still reproduce Claims 1-3 without the Triton speedup.
Log the failure clearly in the logbook — this is a valid finding.

### 3. LongChat model
Some eval commands use lmsys/longchat-7b-v1.5-32k instead of Llama-3.1-8B.
Check which model the paper used for each specific claim before running.

---

## Extended Analysis (After Official Reproduction)

Once Claims 1-5 are verified, add this section to the logbook as "Extended Analysis."
This is our original contribution beyond the required reproduction.

### Run STAR-KV on Qwen2.5-3B-Instruct
Context: Our prior work on TurboQuant showed Qwen2.5-3B is unstable under KV compression
due to Layer 0 key-norm heterogeneity (K/V norm ratio ~52x). We want to test whether
STAR-KV's low-rank approach is more robust to this architectural property.

For this, modify the attention replacement in model.py to support Qwen architecture,
or write a wrapper in repro/ that applies STAR-KV's soft thresholding to Qwen's
attention layers.

### Apply our 5 geometric metrics on top of STAR-KV
Use experiment_1.py from our TurboQuant project to measure:
- Attention KL divergence (M1)
- Per-layer sensitivity curve (M2)
- Norm vs direction error (M3)
- Token-position degradation (M4)
- GQA cross-query consistency (M5)

Compare STAR-KV's geometric profile against TurboQuant's.
Does low-rank compression produce a different layer sensitivity pattern?
Does it resolve the Layer 0 anomaly we found in Qwen?

---

## Logbook Format

Every experiment should be logged with:
- Command run
- GPU used and runtime
- Output numbers (exact values, not rounded)
- Comparison to paper's reported numbers
- Notes on any deviation or failure

Save all results as JSON in results/ with timestamps.

---

## Paper Claims Summary Table

| Claim | Paper Value | Our Result | Status |
|---|---|---|---|
| PPL WikiText-2 (Llama-3.1-8B, 0.6 comp) | TBD from paper | TBD | Pending |
| PPL C4 (Llama-3.1-8B, 0.6 comp) | TBD from paper | TBD | Pending |
| KV compression ratio (low-rank only) | up to 75% | TBD | Pending |
| KV compression ratio (combined) | up to 20x | TBD | Pending |
| Attention speedup (Triton) | up to 6.9x | TBD | Pending |
| E2E throughput improvement | up to 3.1x | TBD | Pending |
| Zero-shot avg accuracy drop | minimal | TBD | Pending |

Fill in "Our Result" and "Status" as you run each experiment.

---

## References

- Paper: arXiv:2606.08382
- Official repo: https://github.com/PriyanshBhatnagar/STAR-KV
- Our TurboQuant work: https://github.com/Devsam2898/Beyond-Perplexity-TurboQuant
- Challenge: https://huggingface.co/spaces/ICML-2026-agent-repro/challenge
