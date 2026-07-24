# EXTENDED_ANALYSIS.md: Beyond Official Reproduction

## IMPORTANT: Read This File Only After Official Reproduction Succeeds

Do not start any experiment in this file until:
- All claims in EXPECTED_RESULTS.md have been verified or documented
- Your Trackio logbook shows verified status on at least Tables 1 and 4
- You have documented all discrepancies found during reproduction

This file is for the extended analysis section of the logbook.
It will NOT be evaluated by the automated Logbook Judge.
It IS visible to human reviewers and will strengthen your submission.

---

## Background: Our Prior TurboQuant Research

Before this reproduction, we conducted an empirical study of TurboQuant
(arXiv:2504.19874) on Qwen2.5-3B-Instruct and Llama-3.1-8B-Instruct across
ten random seeds using five geometric diagnostic metrics.

Key findings from that work:
1. Layer 0 is catastrophically sensitive in Qwen2.5-3B (KL = 0.779, 20x median)
   but the LEAST sensitive layer in Llama-3.1-8B (KL = 0.009)
2. Perplexity CV = 3.31 across seeds vs KL divergence CV = 0.141 (23x more stable)
3. GQA cross-query consistency loss scales superlinearly with query-sharing ratio
4. Norm reconstruction error is 2.36x direction error at 4-bit on both architectures

Code: github.com/Devsam2898/Beyond-Perplexity-TurboQuant

---

## Connection to STAR-KV Findings

These findings connect to STAR-KV in three specific ways:

### Connection 1: Layer 0 Sensitivity

We found Layer 0 is most sensitive via KL divergence measurement.
STAR-KV authors found the same pattern empirically and wrote in Appendix A.3:
"we do not apply compression to layers 0, 1, and 31, which have been shown
to be particularly sensitive."

Two independent methods (KL geometry vs empirical accuracy) reach the same
conclusion. This strengthens both findings.

Research question for extended analysis:
Does STAR-KV's protection of Layer 0 eliminate the layer sensitivity anomaly
we observed in TurboQuant? Apply our M2 metric (per-layer KL curve) to
STAR-KV outputs and compare against the TurboQuant curve.

### Connection 2: K/V Asymmetry

We found K/V norm ratio of 52x in Qwen2.5-3B vs 4x in Llama-3.1-8B.
STAR-KV independently confirms K and V have different sensitivities (Section 4.3):
"value projections have higher low-rank approximation error than key projections."

STAR-KV's hybrid decomposition (HD for K, JD for V) directly addresses this.

Research question:
Does the hybrid decomposition eliminate the K/V asymmetry problem, or does
it reduce it without fully solving it? Measure our M3 metric (norm vs direction
error decomposition) on STAR-KV outputs.

### Connection 3: GQA Sensitivity

STAR-KV notes: "maintaining long-context accuracy becomes challenging at higher
compression rates, particularly for GQA-based models."

We quantified this with our M5 metric (GQA cross-query consistency loss) and found
it scales superlinearly with query-sharing ratio.

Research question:
Does STAR-KV's adaptive rank selection reduce GQA consistency loss, or does
the problem persist regardless of how ranks are allocated?

### Connection 4: Seed Sensitivity

STAR-KV ran one extra evaluation with a different seed (Table 3, diff. seed row)
and got 88.71 vs 88.26 on RULER. This small but real difference confirms that
seed sensitivity exists in STAR-KV too, even if much milder than in TurboQuant.

---

## Extended Experiments to Run

Run these only after official reproduction is complete.

### Experiment E1: STAR-KV on Qwen2.5-3B-Instruct

Motivation: STAR-KV claims universality but only tested Llama-family models.
Our TurboQuant research showed Qwen2.5-3B behaves very differently from Llama
under KV compression due to Layer 0 norm heterogeneity (172 vs 22 average norm).

Setup:
- Adapt model.py to support Qwen attention architecture
- Use same training settings as main experiments
- Evaluate on WikiText-2, C4, and zero-shot tasks
- Apply all five geometric metrics from our TurboQuant code

Expected outcomes (hypothesis):
- If STAR-KV is truly universal: similar accuracy at 60% compression as Llama
- If Layer 0 anomaly persists: degraded accuracy unless Layer 0 is protected
- The layer sensitivity curve (M2) should tell us which layers are problematic

### Experiment E2: Five Geometric Metrics on STAR-KV Llama Outputs

Apply experiment_1.py from our TurboQuant project to STAR-KV compressed Llama.
This gives a direct comparison between TurboQuant and STAR-KV using identical metrics.

Metrics to compute:
- M1: Attention KL divergence (per layer, per head)
- M2: Per-layer sensitivity curve
- M3: Norm vs direction error decomposition
- M4: Token-position degradation
- M5: GQA cross-query consistency loss

Compare STAR-KV geometric profile against TurboQuant profile.
Does low-rank compression produce a different layer sensitivity pattern?
Does it reduce or eliminate the Layer 0 anomaly?

### Experiment E3: Seed Sensitivity of STAR-KV

Run STAR-KV evaluation (not training, just inference) with 5 different random seeds.
Measure PPL and our geometric metrics for each seed.

Compare variance against our TurboQuant seed sweep results:
- TurboQuant PPL CV = 3.31 across 10 seeds on Qwen
- TurboQuant KL CV = 0.141 across 10 seeds on Qwen
- What are these values for STAR-KV?

---

## How to Present Extended Analysis in the Logbook

Use this structure:

Section title: "Extended Analysis: Cross-Architecture and Metric Evaluation"

Subsection 1: Motivation (2-3 sentences connecting to our prior work)
Subsection 2: E1 results with comparison table vs TurboQuant
Subsection 3: E2 geometric metric comparison figures
Subsection 4: E3 seed sensitivity comparison
Subsection 5: Open questions raised by these results

Keep the extended analysis clearly separated from the official reproduction.
Label every extended experiment with [EXTENDED] so reviewers know what is
and is not part of the official challenge submission.

---

## Code References

TurboQuant experiment code: github.com/Devsam2898/Beyond-Perplexity-TurboQuant
Key file: experiment_1.py (contains all 5 metric implementations)
Key file: modal_app.py (Modal cloud deployment, adaptable for STAR-KV)

The five metrics in experiment_1.py can be applied to any model via forward hooks
on k_proj and v_proj. They do not depend on the compression algorithm used.
