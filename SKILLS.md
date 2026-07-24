# SKILLS.md: STAR-KV Paper Knowledge Base

---

## PRIMARY OBJECTIVE

The goal is NOT to improve STAR-KV.
The goal is NOT to refactor the code.
The goal is NOT to optimize the implementation.
The goal is NOT to confirm or deny findings from other research.

The goal is to faithfully reproduce every result reported in the paper.

Assume the paper is correct until evidence proves otherwise.
Any discrepancy between the paper and the code must be DOCUMENTED before
any code modification is made. Never silently fix anything.

---


## Purpose

This file contains everything Claude Code needs to understand the STAR-KV method
before touching any code. Read this before reading the repo code. If something in
the code seems wrong, check this file first because the paper is the ground truth.

Paper: arXiv:2606.08382v1
Title: STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding for Adaptive Rank Control
Venue: ICML 2026 Spotlight (top 2.2% of submissions)
Authors: Bhatnagar, Moradifirouzabadi (UCSD), Yang, Lee (Dnotitia), Choi (Hanyang), Kang (UCSD)


---

## REPRODUCTION PROTOCOL

Follow this order strictly. Do not skip steps.

Step 1: Verify environment
  - Check GPU matches requirements (RTX PRO 6000 preferred)
  - Run version verification script below
  - Confirm all dependencies installed from requirements.txt
  - Do NOT proceed if versions do not match expected range

Step 2: Run kernel correctness checks BEFORE anything else
  python STAR-KV/abx_rope_batched.py --check
  python STAR-KV/bx_quant.py --check
  If these fail, document the failure. Do not proceed with latency benchmarks.

Step 3: Run baseline (NO compression) first
  Evaluate the uncompressed model on WikiText-2, C4, and zero-shot tasks.
  Record exact numbers. Compare against EXPECTED_RESULTS.md baseline rows.
  If baseline does not match within 0.1 PPL, investigate before continuing.

Step 4: Run training with original code UNMODIFIED
  Do not change any hyperparameters on the first run.
  Use exact settings from Section 3 of this file.

Step 5: Evaluate and compare
  Run eval.py after training. Record every metric.
  Compare against EXPECTED_RESULTS.md table by table.

Step 6: Document discrepancies
  For every number that does not match, write in the logbook:
    Paper says: X
    We got: Y
    Possible reason: Z
  Do this BEFORE proposing any fix.

Step 7: Extended analysis (only after Steps 1-6 complete)
  Read EXTENDED_ANALYSIS.md only after official reproduction is verified.

---

## NEVER SILENTLY FIX

If the code differs from what the paper describes, DO NOT automatically fix it.

Before touching any code, write this structure in the logbook:

  DISCREPANCY FOUND
  Location: [file name, line numbers]
  Paper says: [exact quote or equation from paper]
  Code does: [what the code actually does]
  Difference: [how they differ]
  Likely cause: one of the following categories:
    - Paper bug
    - Implementation bug
    - Documentation bug
    - Numerical issue
    - Hardware issue
    - Version issue
    - Randomness
    - Benchmark mismatch
  Proposed fix: [what change would align code with paper]
  Impact on results: [estimated effect on final numbers]

Only modify code after this is documented.

---

## DETERMINISTIC RUNS

Fix ALL sources of randomness before every experiment.
An unreproducible experiment is not a reproduction.

Add this block at the start of every script:

    import torch, numpy as np, random
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

Note: The paper ran one RULER evaluation with a different seed (Table 3,
STAR-KV diff. seed row) and got a slightly higher score (88.71 vs 88.26).
This confirms some seed sensitivity exists. Document if you observe the same.

---

## VERSION REQUIREMENTS

Exact versions matter for Triton kernels. A version mismatch is the most
common cause of kernel failures that cannot be explained otherwise.

Run this before starting and paste the full output into your logbook:

    python -c "
    import sys, torch, triton, transformers
    print('Python:', sys.version)
    print('PyTorch:', torch.__version__)
    print('CUDA (torch):', torch.version.cuda)
    print('Triton:', triton.__version__)
    print('Transformers:', transformers.__version__)
    "
    nvcc --version
    git -C STAR-KV rev-parse HEAD

Required:
  - Python: 3.12.7 (specified by authors)
  - PyTorch: 2.x with CUDA 12.x support
  - Triton: install from requirements.txt, do not upgrade independently
  - Transformers: 4.40 or higher for LLaMA-3.1 support

If your environment differs from this, document it explicitly.
Do not assume results will match if the versions are different.

---

## LOGGING REQUIREMENTS

Every single experiment run must record ALL of the following.
An experiment without complete logs is not a valid data point.

  GPU model and VRAM
  CUDA version
  PyTorch version
  Triton version
  Python version
  Git commit hash: git -C STAR-KV rev-parse HEAD
  Random seed used
  Exact command line arguments
  Wall clock runtime
  Peak GPU memory (torch.cuda.max_memory_allocated())
  All output metrics (PPL, accuracy, latency, throughput)

Save every run as a timestamped JSON in the results/ directory.

---

---

## 0. Hardware Requirements

### Exact hardware used by the authors (per paper)

| Experiment | GPU | VRAM |
|---|---|---|
| Threshold learning / calibration | 2x NVIDIA RTX PRO 6000 | 96GB each |
| Attention kernel benchmarks (main latency) | 1x NVIDIA RTX 4090 | 24GB |
| Kernel portability experiment | 1x NVIDIA RTX PRO 6000 | 96GB |
| End-to-end throughput experiments | 1x NVIDIA RTX 4090 | 24GB |
| LongBench / RULER / LM-Eval accuracy | Not explicitly stated | (post-calibration) |

### What GPU to request on HuggingFace

Request: 1x NVIDIA RTX PRO 6000 (96GB)

HuggingFace provides RTX PRO 6000 GPUs in their compute tier.
Using the SAME GPU as the authors eliminates all architecture disclaimers.
The Triton kernels were written and tested on this exact GPU. No portability
concerns, no reviewer questions about hardware differences.

The 96GB VRAM also removes all memory constraints:
- LLaMA-3.1-8B float16 weights: ~16GB
- Teacher model for KD: ~16GB
- Optimizer states: ~32GB
- Activations at seq_len 8192: ~8-12GB
- Total: ~72-76GB, comfortably within 96GB

For the RTX 4090 latency benchmarks: the paper's main latency numbers are on
RTX 4090. If your HF credit does not provide a 4090, run latency on the RTX PRO
6000 using the portability section of the paper (Appendix A.5.6) as your reference
numbers. The paper already reports 4.3x speedup on RTX PRO 6000 at 128K context,
so you have a direct comparison point.

### Python and CUDA requirements
- Python 3.12.7 (as specified by authors in README)
- CUDA 12.x (required for Triton kernels)
- PyTorch 2.x with CUDA support
- Triton (installed via requirements.txt)

### Verifying your GPU before starting
```bash
python -c "
import torch
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
print('CUDA:', torch.version.cuda)
"
```
Expected output: NVIDIA RTX PRO 6000, ~96.0 GB, 12.x

---

## 1. What Problem STAR-KV Solves

The KV cache stores Key and Value tensors for every past token in a transformer.
Its memory footprint is: 2 * L * H_kv * T * d * 2 bytes (float16).

Example: LLaMA-3.1-8B with 128K context, batch size 4 => KV cache is 81% of total
85GB GPU memory footprint.

Prior low-rank methods had two problems:
1. Fixed or heuristic rank selection (uniform across all layers) degrades accuracy
   at high compression because different layers have very different sensitivities.
2. Low-rank reconstruction at inference adds compute overhead that cancels out
   the memory savings in terms of latency.

STAR-KV solves both problems with three techniques.

---

## 2. The Three Core Techniques

### Technique 1: Adaptive Rank Selection via Soft Thresholding

Standard SVD: W = U * Sigma * V^T where Sigma is diagonal with singular values
in descending order. A rank-r approximation keeps the top r singular values.

Problem: How do you pick r? Uniform r across all layers is wrong because some layers
are highly sensitive to rank reduction and others are not.

STAR-KV solution: Learn a threshold alpha per layer (and per head for keys) that
is applied to the singular values. Singular values above alpha are kept.
Singular values below alpha are suppressed to zero.

The soft threshold operator (differentiable, needed for gradient-based learning):

    Th_s(x; alpha) = x * tanh(s * (x - alpha))   if x >= alpha
                   = 0                              if x < alpha

where alpha is the learnable threshold, s controls the sharpness of the transition.

This is differentiable unlike hard thresholding, so alpha can be learned by backprop.
Once training is done, hard thresholding is applied offline before inference.
The effective rank r_eff is determined by how many singular values exceed alpha.

Key insight: alpha is learned independently per decoder block and per attention head
(for keys), giving fine-grained rank control. This is what "adaptive" means.

### Technique 2: Hybrid Decomposition Strategy

Low-rank decomposition can be applied at two granularities:

HEAD-WISE DECOMPOSITION (HD): Decompose each attention head independently.
- Per-head cache: A_h in R^(l x r_h), reconstruction B_h in R^(r_h x d_h)
- Total reconstruction FLOPs: h * l * r_h * d_h = l * c * h * d_h^2

JOINT DECOMPOSITION (JD): Concatenate all heads and decompose together.
- Shared cache: A in R^(l x r_j), reconstruction B in R^(r_j x d)
- Total reconstruction FLOPs: l * r_j * d = l * c * h^2 * d_h^2
- This is h times MORE expensive than HD for the same compression rate.

Critical finding from the paper (Lemma 4.2):
Under the same compression budget, JD achieves LOWER Frobenius error than HD.
So JD is more accurate but more expensive; HD is less accurate but cheaper.

The paper also found (from singular value spectrum analysis, Figure 6):
- Key projections (W_K) have a flatter spectrum => low-rank approximation is easier
- Value projections (W_V) have a steeper spectrum => low-rank approximation is harder
  (more reconstruction error at the same rank)

STAR-KV hybrid solution:
- Apply HD to KEY projections (less sensitive, lower reconstruction overhead)
- Apply JD to VALUE projections (more sensitive, JD reduces reconstruction error)

This is the optimal accuracy-overhead balance. See Figure 7 in the paper.

Additionally, weight absorption is applied to values (not keys, because RoPE prevents it):
The output computation (p^i * V^i) * W_out is reordered to (p^i * V') * (U_V * W_out)
where W_out' = W_out * U_V is precomputed. This reduces per-step FLOPs by ~d_h times
(typically 128x for modern models).

Keys CANNOT use weight absorption because RoPE is applied between the query and key
reconstruction matrix, preventing offline fusion.

### Technique 3: Low-Rank-Aware Mixed-Precision Quantization

After low-rank compression, the latent key/value representations have a skewed
channel distribution: the first few channels carry most of the signal (because they
correspond to the largest singular values), while the rest are small.

This creates outlier channels that make standard quantization inaccurate.

STAR-KV solution: Block-wise Hadamard + mixed precision.
- Partition latent channels into: top 20% = outlier block, bottom 80% = inlier block
- Apply independent Hadamard transforms to each block to redistribute magnitudes
- Quantize outlier block at b_out bits (higher precision)
- Quantize inlier block at b_in bits (lower precision)
- Default: b_out = 4-bit, b_in = 3-bit, giving 3.2-bit average

The Hadamard factors can be fused into the decomposed up/down projections as H^T * U^T
and (V * Sigma) * H, so this adds ZERO inference overhead.

Combined with 75% low-rank compression, this achieves 20x overall KV cache reduction.

---

## 3. Training Procedure

This is the most important section for reproduction. Read carefully.

### Training objective (Equation 5):
L_total = L_KD + gamma * L_acmp

Where:
- L_KD = KL(p_teacher || p_student): knowledge distillation from uncompressed model
- L_acmp = sum_i exp(-alpha_i): compression loss (penalizes small thresholds)
- gamma: compression loss weight (paper uses gamma = 0.1)

The compression loss L_acmp has a clever property: when alpha is small (early training),
exp(-alpha) is large, creating strong pressure to increase alpha (increase compression).
As alpha grows, the gradient diminishes, and the model stabilizes at that compression level.

For the hybrid decomposition, the compression loss is:
L_acmp = sum_i sum_h exp(-alpha_{i,h}^K) + sum_i exp(-alpha_i^V)

### Training settings (from Appendix A.3):
- Optimizer: AdamW
- Learning rate for model params: 2e-5
- Learning rate for threshold params alpha: 1e-2 (10x higher, for fast convergence)
- Training data: 3000 samples from FineWeb-Edu dataset (NOT WikiText-2)
- Sequence length: 8192 for LongChat/LLaMA-3/LLaMA-3.1, 4096 for LLaMA-2-7B
- Phase 1: Train for 1 full epoch on 3000 samples with combined loss
- Phase 2: Disable compression loss, continue with KD loss only for 1000 more steps
- Hardware used by authors: 2x NVIDIA RTX PRO 6000 GPUs
- Total training time: approximately 6 GPU hours

IMPORTANT: Do NOT apply compression to layers 0, 1, and 31 (for 32-layer models).
These layers are particularly sensitive and must remain uncompressed.
The code in train.py should handle this, but verify it.

### After training:
1. Hard thresholding is applied offline to replace the soft thresholds
2. The decomposed projection matrices are fixed with their truncated ranks
3. Weight absorption is applied for value projections

---

## 4. Inference

Keys are cached in pre-RoPE form as latent representations K' in R^(l x r_k).
Values are cached as latent representations V' in R^(l x r_v).

At each decoding step:
1. Compute latent query from input x
2. For keys: reconstruct with RoPE applied after reconstruction, compute q * K^T
3. For values: use reordered computation (p * V') * W_out' where W_out' is precomputed
4. Apply quantization/dequantization if using the quantized variant

The Triton kernels fuse these operations to avoid intermediate materialization.

Key inference file in repo: LlamaLoRaAttention_headwise.py (without quantization)
                             LlamaLoRaAttention_headwise_quant.py (with quantization)

---

## 5. Exact Numbers to Reproduce (from paper)

### Table 1: Zero-shot accuracy and perplexity

LongChat-7B-v1.5 baseline (no compression):
- WikiText-2 PPL: 6.86
- C4 PPL: 9.93
- Avg zero-shot: 61.53%

STAR-KV at 60% compression:
- WikiText-2 PPL: 6.83 (better than baseline!)
- C4 PPL: 9.98
- Avg zero-shot: 61.53%

STAR-KV at 75% compression:
- WikiText-2 PPL: 7.34
- C4 PPL: 10.44
- Avg zero-shot: 60.49%

LLaMA-3.1-8B-Instruct baseline:
- WikiText-2 PPL: 7.74
- C4 PPL: 12.61
- Avg zero-shot: 67.87%

STAR-KV at 60% compression:
- WikiText-2 PPL: 8.52
- C4 PPL: 13.51
- Avg zero-shot: 65.42%

### Table 4: Quantization results (LongChat-7B-v1.5)

STAR-KV (75%) + 3.2-bit quantization:
- Average compression: 20x
- Avg zero-shot: 59.18%

### Table 5: End-to-end generation throughput

At 8K context: 3.01x throughput gain
At 16K context: 3.14x throughput gain
All measured vs PyTorch SDPA FP16 baseline.

### Attention speedup (Figure 9, LLaMA-2-7B, batch size 16):

STAR-KV 75% without quantization: up to 4.0x at 16K context
STAR-KV 75% with 4-bit quantization: up to 6.9x at 64K context (baseline OOM, estimated)

---

## 6. Compression Rates

The paper evaluates two main settings:
- 60% compression: more conservative, closer to baseline quality
- 75% compression: aggressive, some accuracy drop but still competitive

What "60% compression" means:
The KV cache is reduced to 40% of its original size (60% compression = keeping 40%).

---

## 7. Models Used in Paper

| Model | Tested compression |
|---|---|
| LongChat-7B-v1.5-32K | 60%, 75% |
| LLaMA-2-7B | 60%, 75% |
| LLaMA-3-8B-Instruct | 60%, 75% |
| LLaMA-3.1-8B-Instruct | 60% (main) |
| Mistral-7B-Instruct-v0.2 | 60% (appendix) |
| LLaMA-2-13B | 60%, 75% (appendix) |

Primary model for the challenge reproduction: LongChat-7B-v1.5-32K and LLaMA-3.1-8B-Instruct.

---

## 8. Benchmarks Used

### Perplexity:
- WikiText-2
- C4

### Zero-shot accuracy (via LM-Eval-Harness):
- OpenBookQA (OBQA)
- PIQA
- ARC-easy (ARC-e)
- ARC-challenge (ARC-c)
- HellaSwag (Hella)
- WinoGrande (Wino)

### Long-context:
- LongBench: Qasper, QMSum, TriviaQA, MultiQA, TREC, MultiNews, VCSum
- RULER: MK1, MK2, MQ, MV, S1, S2, S3, FWE, SQ (evaluated at 4K and 16K)

---

## 9. Known Caveats and Watch-Outs

### Caveat 1: The 6.9x speedup is partially estimated
At 64K context, the PyTorch baseline runs out of memory on RTX 4090 (24GB).
The authors estimated the baseline latency by linear extrapolation from 8K and 16K.
So "6.9x speedup" is against an estimated, not measured, baseline.
The 4.0x speedup at 32K is measured against the estimated baseline too.
Only speedups up to 16K are fully measured on both sides.

### Caveat 2: Layers 0, 1, and 31 are NOT compressed
The paper explicitly states in Appendix A.3: "we do not apply compression to layers
0, 1, and 31, which have been shown to be particularly sensitive."
This is important. If the code does not protect these layers, results will be worse.
Verify in train.py that these layers are excluded from compression.

This aligns with our own TurboQuant research: we independently found Layer 0 is
the most sensitive layer in Qwen2.5-3B. STAR-KV confirms the same pattern for Llama.

### Caveat 3: Training hardware mismatch
Authors used RTX PRO 6000 (96GB VRAM). The HuggingFace GPU credits likely provide
a different GPU. Triton kernels may behave differently on different hardware.
Run the kernel correctness checks first:
    python STAR-KV/abx_rope_batched.py --check
    python STAR-KV/bx_quant.py --check

### Caveat 4: Missing trained weights
The GitHub TODO list explicitly says "Add trained weights file for LLaMA-3.1-8B."
You must run training from scratch. Budget 6+ GPU hours.

### Caveat 5: Calibration data
Training uses FineWeb-Edu, NOT WikiText-2 or C4 (those are only for evaluation).
The huggingface dataset: HuggingFaceFW/fineweb-edu

---

## 10. File Map: What Each Code File Does

| File | Purpose |
|---|---|
| train.py | Main training script. Runs the soft-threshold learning with KD loss. |
| eval.py | Evaluation: PPL on WikiText-2/C4, zero-shot via LM-Eval, LongBench, RULER |
| latency.py | Latency benchmarks: end-to-end and layer-wise timing |
| model.py | Shared: decomposed modules, attention replacement logic |
| soft_thres_layer.py | Learnable soft-threshold function (Equation 2 from paper) |
| LlamaLoRaAttention_headwise.py | Low-rank attention module with Triton, no quantization |
| LlamaLoRaAttention_headwise_quant.py | Low-rank attention with Triton + int8/int4 quantization |
| abx_rope_batched.py | Triton kernel: fused A@(B@X^T + RoPE) for key reconstruction |
| bx_quant.py | Triton kernel: fused dequantize + B@X for value reconstruction |
| requirements.txt | All Python dependencies |
