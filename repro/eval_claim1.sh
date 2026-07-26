#!/bin/bash
set -x
pip install --no-cache-dir torch triton transformers datasets accelerate evaluate huggingface_hub "lm_eval[longbench,ruler]" tqdm tiktoken sentencepiece

python -c "
import sys, torch, triton, transformers
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('CUDA (torch):', torch.version.cuda)
print('Triton:', triton.__version__)
print('Transformers:', transformers.__version__)
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM (GB):', torch.cuda.get_device_properties(0).total_memory / 1e9)
"
echo "STAR-KV commit: 1bcdc0041a32cb4adf0a43320f7d090413595479"
echo "REPRO_SEED: ${REPRO_SEED:-42}"

mkdir -p /workspace/output
cd /workspace/STAR-KV

python /workspace/repro/seeded_run.py eval.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --weights /workspace/output/trained_weights.pt \
  --ppl --ppl-datasets wikitext2,c4 \
  --tasks piqa,winogrande,arc_easy,arc_challenge,openbookqa,hellaswag \
  --batch-size 32 \
  --output /workspace/output/eval_claim1_llama31_8b.json
echo "EXIT_EVAL=$?"
