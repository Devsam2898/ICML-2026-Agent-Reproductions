#!/bin/bash
set -x
pip install --no-cache-dir torch triton transformers datasets accelerate evaluate huggingface_hub "lm_eval[longbench,ruler]" tqdm tiktoken sentencepiece protobuf

python -c "
import sys, torch, triton, transformers, tokenizers, huggingface_hub
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('CUDA (torch):', torch.version.cuda)
print('Triton:', triton.__version__)
print('Transformers:', transformers.__version__)
print('Tokenizers:', tokenizers.__version__)
print('huggingface_hub:', huggingface_hub.__version__)
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM (GB):', torch.cuda.get_device_properties(0).total_memory / 1e9)
"
echo "STAR-KV commit: 1bcdc0041a32cb4adf0a43320f7d090413595479"
echo "REPRO_SEED: ${REPRO_SEED:-42}"

mkdir -p /workspace/output
cd /workspace/repro

echo "=== Baseline: LongChat-7B-v1.5-32k (retry #2 - missing protobuf caused SentencePieceExtractor to fail and silently mis-fall-back to a broken TikToken parse; see Discrepancies Log) ==="
python baseline_eval.py \
  --model lmsys/longchat-7b-v1.5-32k \
  --ppl-datasets wikitext2,c4 \
  --tasks piqa,winogrande,arc_easy,arc_challenge,openbookqa,hellaswag \
  --batch-size 32 \
  --output /workspace/output/baseline_longchat7b.json
echo "EXIT_LONGCHAT=$?"
