# Discrepancies Log


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_65c9470f6807", "created_at": "2026-07-24T12:48:59+00:00", "title": "Central log of paper-vs-code discrepancies, per SKILLS.md protocol (never silen…"}
-->
Central log of paper-vs-code discrepancies, per SKILLS.md protocol (never silently fix; document before touching code).


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c41ab5b4b67f", "created_at": "2026-07-24T12:51:03+00:00", "title": "DISCREPANCY FOUND"}
-->
**DISCREPANCY FOUND**
Location: STAR-KV/train.py (entire file)
Paper says (SKILLS.md determinism requirement): every experiment must fix all sources of randomness (torch/numpy/random seeds, cudnn deterministic mode) so runs are reproducible.
Code does: train.py has zero seed-setting anywhere - no random.seed/np.random.seed/torch.manual_seed/torch.cuda.manual_seed_all calls, no cudnn.deterministic flag.
Difference: as shipped, two runs of train.py with identical args are not guaranteed reproducible (weight init randomness is fixed by from_pretrained, but dataloader shuffling/sampling and any dropout are not seeded).
Likely cause: Documentation bug / implementation gap - determinism was presumably handled by the authors' own run harness, not included in the public release.
Proposed fix: do not modify STAR-KV/train.py. Instead use a wrapper (repro/seeded_run.py) that sets all seeds via runpy before executing train.py's __main__, so determinism is added without touching the vendored file.
Impact on results: low - affects exact bit-for-bit reproducibility of a given run, not expected to shift metrics outside normal run-to-run noise. Documented per protocol regardless.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_11ca112743d2", "created_at": "2026-07-24T12:52:27+00:00", "title": "DISCREPANCY FOUND"}
-->
**DISCREPANCY FOUND**
Location: STAR-KV/eval.py lines 145-217 (parse_args --weights required=True; main() unconditionally calls replace_linear_layer then model.load_state_dict(torch.load(args.weights)))
Paper says (SKILLS.md Step 3 / EXPECTED_RESULTS.md): must evaluate the uncompressed baseline model on WikiText-2/C4/zero-shot BEFORE training, and compare against the paper's baseline rows (e.g. LongChat baseline Wiki2 PPL 6.86; Llama-3.1-8B-Instruct baseline Wiki2 PPL 7.74).
Code does: eval.py's main() always injects STAR-KV's low-rank decomposed linear layers via replace_linear_layer() and always requires a --weights checkpoint to be loaded via model.load_state_dict() before any evaluation runs. There is no argument or code path to evaluate the stock/uncompressed HF checkpoint through eval.py's CLI.
Difference: the shipped eval.py cannot produce a 'Comp%=0%' baseline row; baseline numbers must have been produced by the authors outside this released script.
Likely cause: Implementation bug / missing feature (baseline eval tooling not included in public release).
Proposed fix: do not modify STAR-KV/eval.py. Instead, reuse its evaluate_ppl(model, tokenizer, datasets, seqlen, device) and evaluate_lmeval(model, tokenizer, tasks, batch_size, ...) functions directly - both take a model object as a parameter and don't require going through main()/--weights. Wrapper: repro/baseline_eval.py imports these from eval.py as a library, loads the stock LlamaForCausalLM.from_pretrained(model_name) with no decomposition applied, and calls them directly.
Impact on results: none on compressed-model numbers; without this wrapper we would be unable to independently verify the paper's own reported baseline rows and could only compare our compressed results against the paper's baseline as given, not our own reproduction of it.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b01ead067e4d", "created_at": "2026-07-24T13:06:47+00:00", "title": "Environment/dependency issues found on first baseline-eval job run (job https:/…"}
-->
**Environment/dependency issues found on first baseline-eval job run** (job https://huggingface.co/jobs/Devavrat28/6a6361c87ef3c0846496779c, rtx-pro-6000, ~125s, ~$0.06):

1. lmsys/longchat-7b-v1.5-32k tokenizer load failed: ModuleNotFoundError: No module named 'tiktoken' (then ValueError requiring tiktoken to read a tiktoken file). requirements.txt does not list tiktoken as a transitive dependency of this tokenizer. Fix: add tiktoken (and sentencepiece, for parity) to the install list in our job scripts - not a change to STAR-KV/ itself, just our own environment setup script.

2. Llama-3.1-8B baseline model loaded fine, but evaluate_ppl -> _get_ppl_data('wikitext2', ...) -> load_dataset('wikitext', 'wikitext-2-raw-v1') raised huggingface_hub.errors.HfUriError: 'Repository id must be namespace/name, got wikitext'. The unpinned latest 'datasets' package (5.0.0 in this environment) no longer resolves the bare legacy dataset id STAR-KV/eval.py hardcodes at line 56. This is the 'Version mismatch' failure mode SKILLS.md explicitly warns about. Per protocol, not modifying STAR-KV/eval.py. Fix: repro/compat_shim.py monkeypatches datasets.load_dataset to redirect 'wikitext' -> 'Salesforce/wikitext' (falling back to the original id if that alias fails), imported before eval.py/train.py in both repro/baseline_eval.py and repro/seeded_run.py so it also covers the real (non-baseline) eval.py runs later.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_0f7aac0d452b", "created_at": "2026-07-25T17:00:10+00:00", "title": "DISCREPANCY FOUND"}
-->
**DISCREPANCY FOUND**
Location: repro/seeded_run.py (our own wrapper, not STAR-KV/)
Paper says: n/a - bug in our reproduction tooling, not the paper or repo.
Code does: seeded_run.py called runpy.run_path(target, run_name="__main__") without adding the target script's own directory to sys.path first.
Difference: python train.py (direct invocation) implicitly prepends train.py's directory to sys.path, letting its sibling import (from model import ...) resolve. runpy.run_path does not do this automatically - confirmed empirically with a minimal repro. Training job 6a6366addb23d7a7ec1ca8ba failed after 75s with ModuleNotFoundError: No module named 'model', before any real training occurred.
Likely cause: our own tooling bug (wrapper gap), not an upstream STAR-KV issue.
Proposed fix: applied - seeded_run.py now does sys.path.insert(0, os.path.dirname(os.path.abspath(target))) before runpy.run_path. Verified fix against a minimal local reproduction before resubmitting.
Impact on results: none on paper claims - only affected our determinism wrapper, not STAR-KV/train.py itself. Job resubmitted as 6a64eb397ef3c08464968896 (starkv-train-claim1-v3) with the fix.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_761d6f141bdd", "created_at": "2026-07-25T18:00:55+00:00", "title": "DISCREPANCY FOUND"}
-->
**DISCREPANCY FOUND**
Location: repro/train_claim1.sh --wandb-project flag / our WANDB_API_KEY
Paper says: n/a - W&B is our own optional instrumentation choice, not a paper requirement.
Code does: train.py's main() calls wandb.login(key=os.environ.get("WANDB_API_KEY", "")) then wandb.init(...) whenever --wandb-project is passed. Job starkv-train-claim1-v4 (6a64ed86db23d7a7ec1cc7b3) got past both prior bugs (sys.path fix, secret forwarding fix) and reached this step, but wandb.init() returned 401 CommError after wandb.login() had already reported "API key is configured".
Likely cause: verified locally (wandb.login(key=..., verify=True)) that the current WANDB_API_KEY value in .env fails server-side authentication (AuthenticationError: "An error occurred while verifying the API key") - independent of our job/container setup. The key was rotated earlier this session; either the rotation didn't take effect as expected or the value pasted into .env doesn't match what's live on wandb.ai.
Proposed fix: --wandb-project removed from repro/train_claim1.sh for now (wandb is optional instrumentation per train.py's own code path - no paper claim depends on it). Training resubmitted without it. W&B key troubleshooting deferred to the user, not blocking reproduction.
Impact on results: none - loss/metrics will be captured from job stdout logs instead of a W&B dashboard for this run.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_35b4b4e6f8c6", "created_at": "2026-07-25T18:25:40+00:00", "title": "DISCREPANCY FOUND"}
-->
**DISCREPANCY FOUND**
Location: STAR-KV/train.py (no gradient_checkpointing_enable() anywhere) vs. SKILLS.md Hardware Requirements section (lines ~182, ~360)
Paper says (SKILLS.md): "Threshold learning / calibration" step used 2x NVIDIA RTX PRO 6000 (96GB each); total training time ~6 GPU hours. Elsewhere SKILLS.md also recommends requesting only 1x RTX PRO 6000 on HuggingFace and estimates activations at seq_len 8192 as "~8-12GB", implying single-GPU should be sufficient.
Code does: train.py loads both student and teacher (both Llama-3.1-8B, bf16) with no gradient checkpointing, use_cache=False. First training step OOMs on a single RTX PRO 6000 (94.97GB usable) with ~95GB already allocated before the kd_loss computation - job 6a64fdef7ef3c08464968d5f (starkv-train-claim1-v5), OOM at Epoch 1 step 0, seq_len=8192, batch_size=1 (default).
Difference: SKILLS.md's own single-GPU activation estimate (8-12GB) is far below what's actually needed for a full backward pass through all 32 transformer layers without checkpointing - it likely reflects inference-only forward activation footprint, not backprop storage. The authors' own hardware line for this exact step lists 2 GPUs, which is internally consistent with what we observed empirically (1 GPU is not enough as shipped).
Likely cause: Implementation gap (missing gradient checkpointing support in train.py) combined with an optimistic single-GPU memory estimate in the ground-truth doc.
Proposed fix: applied - repro/compat_shim.py now patches LlamaForCausalLM.from_pretrained (gated by REPRO_GRAD_CKPT=1, set only in train_claim1.sh) to call model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}) on every loaded model. This is a compute/memory tradeoff only (recompute vs. store activations during backward) - mathematically identical to the unmodified forward/backward pass, so it should not change training results, only reduce peak memory. No-op for the teacher (loaded via the same patched from_pretrained, but kept in .eval() mode under torch.no_grad() so checkpointing never actually triggers for it). STAR-KV/train.py itself is not modified. Verified the classmethod-patching mechanism works correctly in isolation before resubmitting.
Impact on results: expected none on final metrics (checkpointing is a standard, results-preserving memory optimization); avoids provisioning more expensive 2-GPU hardware. Will confirm no numerical impact once training completes by comparing loss curve shape/final loss against what would be expected from the paper's own ~6 GPU-hour training budget.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a068ab15e5a3", "created_at": "2026-07-25T18:51:15+00:00", "title": "DISCREPANCY FOUND (update)"}
-->
**DISCREPANCY FOUND (update)**
Location: STAR-KV/train.py optimizer construction (lines ~339-360)
Paper says: n/a directly, but SKILLS.md's own hardware table says the "Threshold learning / calibration" step used 2x RTX PRO 6000.
Code does: the main AdamW param group is `[p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "alpha" not in n]` - i.e. every parameter that isn't a bias/layernorm and isn't named "alpha" gets full-lr AdamW state. This is full fine-tuning of the entire 8B student (not just the injected low-rank/alpha adapters), so AdamW's exp_avg/exp_avg_sq buffers apply to essentially the whole model, on top of the frozen 8B teacher.
Difference: gradient checkpointing (previous fix) only delayed the crash from step 0 to global_step 3, and the actual OOM moved to optimizer.step() (torch.optim.adam._multi_tensor_adam computing exp_avg_sq_sqrt) - confirming the bottleneck is optimizer state, not activation memory. A single 96GB GPU cannot hold: frozen teacher (~16GB) + student weights (~16GB) + student gradients + AdamW state for a full 8B fine-tune + any residual activations, even with checkpointing.
Likely cause: this genuinely requires 2 GPUs as the authors used - not a bug, a resource requirement inherent to the training design (KD + full fine-tune, not LoRA-style partial training).
Proposed fix: scale to rtx-pro-6000x2 (2x96GB=192GB, $5.50/hr) and drop our --cuda-devices override so train.py falls back to its own default ("0,1"), letting accelerate's device_map=auto shard both models across both GPUs - this exactly matches the authors' documented hardware rather than working around it. Gradient checkpointing left enabled for extra margin.
Impact on results: none expected - same training config/hyperparameters, just correct hardware parallelism. Confirmed with user before incurring the added ~$16-22 cost (vs. pennies for the failed single-GPU attempts).
