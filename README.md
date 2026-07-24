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
| Conda env `StarKV` (python 3.12.7) + requirements.txt | N/A — using HF Jobs GPU instead (see below) |
| HF_TOKEN / WANDB_API_KEY configured | Done (fine-grained token, write + job.write scopes) |
| Trackio logbook published | Done — https://huggingface.co/spaces/Devavrat28/star-kv |
| GPU compute path | HF Jobs (`hf jobs run --flavor a100-large`), billed against ICML-2026-agent-repro org credit |
| Triton kernel smoke tests (`abx_rope_batched.py --check`, `bx_quant.py --check`) | **Both fail** — see below |
| Claim 1: PPL (WikiText-2, C4) | Pending |
| Claim 2: Zero-shot accuracy | Pending |
| Claim 3: KV compression ratio | At risk — quantized path broken (see below) |
| Claim 4: Attention speedup / e2e throughput | Blocked — kernel checks fail |
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
Corresponding raw JSON output belongs in `results/`. Full detail lives in the
[published Trackio logbook](https://huggingface.co/spaces/Devavrat28/star-kv); this
section is a condensed pointer.

### 2026-07-24 — Environment setup

- No local GPU/conda; compute runs via `hf jobs run` on HF Jobs (A100, credit from
  the ICML-2026-agent-repro org). Trackio logbook opened and published.

### 2026-07-24 — Triton kernel correctness checks (Claim 4 pre-flight)

- Job: `hf jobs run --flavor a100-large python:3.12 bash repro/kernel_check.sh`
  ([job](https://huggingface.co/jobs/Devavrat28/6a633a27db23d7a7ec1ca3c3), 80s runtime, ~$0.06)
- `abx_rope_batched.py --check`: **FAILS** —
  `RuntimeError: Expected all tensors to be on the same device, but got mat2 is on
  cpu, different from other tensors on cuda:0`, raised inside the `torch_abx()`
  reference path at `rotary_emb(xb, position_ids)`.
- `bx_quant.py --check`: **FAILS** — `ModuleNotFoundError: No module named
  'quant_utils'`. `quant_utils.py` does not exist anywhere in the cloned repo
  (confirmed via full-tree grep before running), yet both `bx_quant.py` and
  `LlamaLoRaAttention_headwise_quant.py` import from it.
- Confirms the upstream repo's own TODO ("Fix kernels for acc analysis") is real.
  Not modifying `STAR-KV/` to patch this per project convention — logged as a
  reproduction finding instead. Claim 4 (speedup/throughput) is blocked until
  upstream fixes land; Claim 3's combined (low-rank + quantization) ratio is at
  risk since the quantized attention path can't import.

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
