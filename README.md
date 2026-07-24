# STAR-KV Reproduction Logbook

Running notes for the ICML 2026 Reproducibility Challenge submission covering
**STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding for Adaptive Rank Control**
(arXiv:2606.08382, https://github.com/PriyanshBhatnagar/STAR-KV).

Challenge: https://huggingface.co/spaces/ICML-2026-agent-repro/challenge

See `CLAUDE.md` for the full plan, commands, and known issues to watch for.

---

## Status

| Step | Status |
|---|---|
| Clone official repo into `STAR-KV/` (git submodule) | Done |
| Add prior TurboQuant work as `Beyond-Perplexity-TurboQuant/` (git submodule) | Done |
| Conda env `StarKV` (python 3.12.7) + requirements.txt | Pending |
| HF_TOKEN / WANDB_API_KEY configured | Pending |
| GPU verified (target: A100 40GB) | Pending |
| Triton kernel smoke tests (`abx_rope_batched.py --check`, `bx_quant.py --check`) | Pending |
| Claim 1: PPL (WikiText-2, C4) | Pending |
| Claim 2: Zero-shot accuracy | Pending |
| Claim 3: KV compression ratio | Pending |
| Claim 4: Attention speedup / e2e throughput | Pending |
| Claim 5: LongBench / RULER | Pending |
| Extended analysis (Qwen2.5-3B + TurboQuant metrics) | Pending |

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

---

## Log Entries

Each entry should record: command run, GPU used and runtime, exact output numbers,
comparison to the paper's reported numbers, and notes on any deviation or failure.
Corresponding raw JSON output belongs in `results/`.

### YYYY-MM-DD — Environment setup

- (fill in once conda env is created and GPU is verified)

---

## Extended Analysis Dependency

`Beyond-Perplexity-TurboQuant/` (git submodule) holds the prior TurboQuant work
referenced by CLAUDE.md's Extended Analysis section — `experiment_1.py` implements
the 5 geometric metrics (attention KL divergence, per-layer sensitivity, norm vs
direction error, token-position degradation, GQA cross-query consistency) to be run
against STAR-KV's compressed Qwen2.5-3B outputs, to compare against the Layer 0
key-norm heterogeneity finding from that project.

## Notes / Known Issues

- Repo TODOs at clone time (from `STAR-KV/README.md`): missing trained weights for
  LongChat/LLaMA-3.1-8B, kernels not yet fixed for accuracy analysis, quantization
  latency tests not yet added.
- This machine has no `conda` on PATH and no local GPU verified yet — training/eval
  needs to happen wherever the HuggingFace GPU credit is provisioned (e.g. an HF Space
  or a GPU box with `HF_TOKEN`/`WANDB_API_KEY` set).
- `myenv/` is a pre-existing local Python venv unrelated to the `StarKV` conda env
  specified by the paper's install instructions.
