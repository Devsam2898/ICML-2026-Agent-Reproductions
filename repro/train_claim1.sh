#!/bin/bash
set -x
pip install --no-cache-dir torch triton transformers datasets accelerate evaluate huggingface_hub wandb tqdm

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

python /workspace/repro/seeded_run.py train.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output /workspace/output/trained_weights.pt \
  --output-fused /workspace/output/fused_weights.pt \
  --epochs 1 --lr 2e-5 --seq-len 8192 --num-samples 4000 \
  --alpha-lr 1e-2 --alpha-samples 3000 \
  --comp-weight-k 0.1 --comp-weight-v 0.1 \
  --kd-weight 1.0 \
  --desired-comp-rate 0.6 \
  --phase3-samples 200 \
  --cuda-devices 0 \
  --wandb-project star-kv-repro
echo "EXIT_TRAIN=$?"
