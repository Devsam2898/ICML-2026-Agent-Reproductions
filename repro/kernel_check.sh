#!/bin/bash
set -x
pip install --no-cache-dir torch triton transformers

cd /workspace/STAR-KV

echo "=== abx_rope_batched.py --check ==="
python abx_rope_batched.py --check
echo "EXIT_ABX=$?"

echo "=== bx_quant.py --check ==="
python bx_quant.py --check
echo "EXIT_BX=$?"
