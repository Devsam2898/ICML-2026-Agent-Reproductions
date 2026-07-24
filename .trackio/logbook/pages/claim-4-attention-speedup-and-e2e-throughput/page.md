# Claim 4: Attention Speedup and E2E Throughput


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2d07044feb58", "created_at": "2026-07-24T07:15:05+00:00", "title": "Attention speedup (paper claims up to 6.9x) and end-to-end generation throughpu…"}
-->
Attention speedup (paper claims up to 6.9x) and end-to-end generation throughput improvement (up to 3.1x), delivered by custom Triton kernels (abx_rope_batched.py fused A@(B@X^T + RoPE), bx_quant.py fused dequant + B@X). CAUTION: upstream repo's own TODO list says 'Fix kernels for acc analysis' as of clone time. Plan: run abx_rope_batched.py --check and bx_quant.py --check first for correctness before trusting any speed numbers; log a failure here as a valid finding if they do not pass. Then latency.py --mode e2e --baseline, followed by --mode e2e with trained_weights.pt across ctx-lens 256..32000.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_9014ab9c0239", "created_at": "2026-07-24T09:21:00+00:00", "title": "Pre-flight check (before spending GPU time): 'bxquant.py' and 'LlamaLoRaAttenti…"}
-->
Pre-flight check (before spending GPU time): 'bx_quant.py' and 'LlamaLoRaAttention_headwise_quant.py' both do `from quant_utils import quantize_r_split_int8_int4_packed, unpack_uint8_to_int4_signed`, but no 'quant_utils.py' exists anywhere in the cloned repo (verified via grep across the full submodule tree). This means bx_quant.py --check will fail with ModuleNotFoundError before any kernel correctness is even tested, and the quantized attention path is currently non-functional as shipped. This also puts the 20x combined (low-rank + quantization) compression ratio claim in Claim 3 at risk, separately from kernel correctness. Will still run the check on GPU to capture the exact traceback for the record.


---
<!-- trackio-cell
{"type": "code", "id": "cell_c1e89de82a42", "created_at": "2026-07-24T09:25:25+00:00", "title": "Run: hf.exe (exit 1)", "command": ["./myenv/Scripts/hf.exe", "jobs", "run", "--flavor", "a100-large", "--timeout", "15m", "--name", "starkv-kernel-check", "-v", ".\\STAR-KV;C:\\Users\\devavrat.samak\\AppData\\Local\\Programs\\Git\\workspace\\STAR-KV;ro", "-v", ".\\repro;C:\\Users\\devavrat.samak\\AppData\\Local\\Programs\\Git\\workspace\\repro;ro", "python:3.12", "bash", "C:/Users/devavrat.samak/AppData/Local/Programs/Git/workspace/repro/kernel_check.sh"], "exit_code": 1, "duration_s": 3.562}
-->
````bash
$ ./myenv/Scripts/hf.exe jobs run --flavor a100-large --timeout 15m --name starkv-kernel-check -v '.\STAR-KV;C:\Users\devavrat.samak\AppData\Local\Programs\Git\workspace\STAR-KV;ro' -v '.\repro;C:\Users\devavrat.samak\AppData\Local\Programs\Git\workspace\repro;ro' python:3.12 bash C:/Users/devavrat.samak/AppData/Local/Programs/Git/workspace/repro/kernel_check.sh
````

exit 1 · 3.6s


````output
Error: Missing mount path in volume spec '.\STAR-KV;C:\Users\devavrat.samak\AppData\Local\Programs\Git\workspace\STAR-KV;ro'. Expected 'LOCAL_DIR:/MOUNT_PATH[:ro|:rw]' (e.g. './data:/data').
Hint: set HF_DEBUG=1 as environment variable for full traceback.

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_a625df1f9ecf", "created_at": "2026-07-24T09:27:15+00:00", "title": "Run: hf.exe (exit 1)", "command": ["./myenv/Scripts/hf.exe", "jobs", "run", "--flavor", "a100-large", "--timeout", "15m", "--name", "starkv-kernel-check", "-v", "./STAR-KV:/workspace/STAR-KV:ro", "-v", "./repro:/workspace/repro:ro", "python:3.12", "bash", "/workspace/repro/kernel_check.sh"], "exit_code": 1, "duration_s": 21.25}
-->
````bash
$ ./myenv/Scripts/hf.exe jobs run --flavor a100-large --timeout 15m --name starkv-kernel-check -v ./STAR-KV:/workspace/STAR-KV:ro -v ./repro:/workspace/repro:ro python:3.12 bash /workspace/repro/kernel_check.sh
````

exit 1 · 21.2s


````output
Error: (Request ID: Root=1-6a632ff2-07080b9b13577928365e0d60;9e790689-cf2c-4a1d-b5d9-dd5930ec1d7b)

403 Forbidden: You don't have the required permissions to complete this action on namespace Devavrat28, missing permissions: job.write.
Cannot access content at: https://huggingface.co/api/jobs/Devavrat28.
Make sure your token has the correct permissions.
Hint: set HF_DEBUG=1 as environment variable for full traceback.
Sync plan: STAR-KV -> hf://buckets/Devavrat28/jobs-artifacts/STAR-KV-a5c24ba8
  Uploads: 13
  Downloads: 0
  Deletes: 0
  Skips: 0
Syncing...
Sync completed.
Sync plan: repro -> hf://buckets/Devavrat28/jobs-artifacts/repro-9203359a
  Uploads: 2
  Downloads: 0
  Deletes: 0
  Skips: 0
Syncing...
Sync completed.

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_123390de8ad8", "created_at": "2026-07-24T10:11:02+00:00", "title": "Run: hf.exe (exit 1)", "command": ["./myenv/Scripts/hf.exe", "jobs", "run", "--flavor", "a100-large", "--timeout", "15m", "--name", "starkv-kernel-check", "-v", "./STAR-KV:/workspace/STAR-KV:ro", "-v", "./repro:/workspace/repro:ro", "python:3.12", "bash", "/workspace/repro/kernel_check.sh"], "exit_code": 1, "duration_s": 22.032}
-->
````bash
$ ./myenv/Scripts/hf.exe jobs run --flavor a100-large --timeout 15m --name starkv-kernel-check -v ./STAR-KV:/workspace/STAR-KV:ro -v ./repro:/workspace/repro:ro python:3.12 bash /workspace/repro/kernel_check.sh
````

exit 1 · 22.0s


````output
Sync plan: STAR-KV -> hf://buckets/Devavrat28/jobs-artifacts/STAR-KV-a5c24ba8
  Uploads: 0
  Downloads: 0
  Deletes: 0
  Skips: 13
Nothing to sync.
Sync plan: repro -> hf://buckets/Devavrat28/jobs-artifacts/repro-9203359a
  Uploads: 0
  Downloads: 0
  Deletes: 0
  Skips: 2
Nothing to sync.
id=6a633a27db23d7a7ec1ca3c3 url=https://huggingface.co/jobs/Devavrat28/6a633a27db23d7a7ec1ca3c3
+ pip install --no-cache-dir torch triton transformers
Collecting torch
  Downloading torch-2.13.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (38 kB)
Collecting triton
  Downloading triton-3.7.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (1.7 kB)
Collecting transformers
  Downloading transformers-5.14.1-py3-none-any.whl.metadata (32 kB)
Collecting filelock (from torch)
  Downloading filelock-3.32.0-py3-none-any.whl.metadata (2.0 kB)
Collecting typing-extensions>=4.10.0 (from torch)
  Downloading typing_extensions-4.16.0-py3-none-any.whl.metadata (3.3 kB)
Collecting setuptools>=77.0.3 (from torch)
  Downloading setuptools-83.0.0-py3-none-any.whl.metadata (6.6 kB)
Collecting sympy>=1.13.3 (from torch)
  Downloading sympy-1.14.0-py3-none-any.whl.metadata (12 kB)
Collecting networkx>=2.5.1 (from torch)
  Downloading networkx-3.6.1-py3-none-any.whl.metadata (6.8 kB)
Collecting jinja2 (from torch)
  Downloading jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
Collecting fsspec>=0.8.5 (from torch)
  Downloading fsspec-2026.6.0-py3-none-any.whl.metadata (10 kB)
Collecting cuda-toolkit==13.0.3 (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch)
  Downloading cuda_toolkit-13.0.3.0-py2.py3-none-any.whl.metadata (17 kB)
Collecting cuda-bindings<14,>=13.0.3 (from torch)
  Downloading cuda_bindings-13.3.1-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (2.5 kB)
Collecting nvidia-cudnn-cu13==9.20.0.48 (from torch)
  Downloading nvidia_cudnn_cu13-9.20.0.48-py3-none-manylinux_2_27_x86_64.whl.metadata (1.9 kB)
Collecting nvidia-cusparselt-cu13==0.8.1 (from torch)
  Downloading nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_x86_64.whl.metadata (12 kB)
Collecting nvidia-nccl-cu13==2.29.7 (from torch)
  Downloading nvidia_nccl_cu13-2.29.7-py3-none-manylinux_2_18_x86_64.whl.metadata (2.1 kB)
Collecting nvidia-nvshmem-cu13==3.4.5 (from torch)
  Downloading nvidia_nvshmem_cu13-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.1 kB)
Collecting nvidia-cublas==13.1.1.3.* (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch)
  Downloading nvidia_cublas-13.1.1.3-py3-none-manylinux_2_27_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cuda-nvrtc==13.0.88.* (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch)
  Downloading nvidia_cuda_nvrtc-13.0.88-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cuda-runtime==13.0.96.* (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch)
  Downloading nvidia_cuda_runtime-13.0.96-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cufft==12.0.0.61.* (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch)
  Downloading nvidia_cufft-12.0.0.61-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-nvjitlink<14,>=13.0.88 (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch)
  Downloading nvidia_nvjitlink-13.3.33-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cufile==1.15.1.6.* (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch)
  Downloading nvidia_cufile-1.15.1.6-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cuda-cupti==13.0.85.* (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch)
  Downloading nvidia_cuda_cupti-13.0.85-py3-none-manylinux_2_25_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-curand==10.4.0.35.* (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch)
  Downloading nvidia_curand-10.4.0.35-py3-none-manylinux_2_27_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cusolver==12.0.4.66.* (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch)
  Downloading nvidia_cusolver-12.0.4.66-py3-none-manylinux_2_27_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cusparse==12.6.3.3.* (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch)
  Downloading nvidia_cusparse-12.6.3.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-nvtx==13.0.85.* (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == "Linux"->torch)
  Downloading nvidia_nvtx-13.0.85-py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.whl.metadata (1.8 kB)
Collecting huggingface-hub<2.0,>=1.5.0 (from transformers)
  Downloading huggingface_hub-1.24.0-py3-none-any.whl.metadata (16 kB)
Collecting numpy>=1.17 (from transformers)
  Downloading numpy-2.5.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (6.6 kB)
Collecting packaging>=20.0 (from transformers)
  Downloading packaging-26.2-py3-none-any.whl.metadata (3.5 kB)
Collecting pyyaml>=5.1 (from transformers)
  Downloading pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (2.4 kB)
Collecting regex>=2025.10.22 (from transformers)
  Downloading regex-2026.7.19-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (40 kB)
Collecting tokenizers<=0.23.0,>=0.22.0 (from transformers)
  Downloading tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
Collecting typer (from transformers)
  Downloading typer-0.27.0-py3-none-any.whl.metadata (15 kB)
Collecting safetensors>=0.8.0 (from transformers)
  Downloading safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.2 kB)
Collecting tqdm>=4.60 (from transformers)
  Downloading tqdm-4.69.0-py3-none-any.whl.metadata (57 kB)
Collecting cuda-pathfinder>=1.4.2 (from cuda-bindings<14,>=13.0.3->torch)
  Downloading cuda_pathfinder-1.6.0-py3-none-any.whl.metadata (1.9 kB)
Collecting click<9.0.0,>=8.4.2 (from huggingface-hub<2.0,>=1.5.0->transformers)
  Downloading click-8.4.2-py3-none-any.whl.metadata (2.6 kB)
Collecting hf-xet<2.0.0,>=1.5.1 (from huggingface-hub<2.0,>=1.5.0->transformers)
  Downloading hf_xet-1.5.2-cp38-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (4.9 kB)
Collecting httpx<1,>=0.23.0 (from huggingface-hub<2.0,>=1.5.0->transformers)
  Downloading httpx-0.28.1-py3-none-any.whl.metadata (7.1 kB)
Collecting mpmath<1.4,>=1.1.0 (from sympy>=1.13.3->torch)
  Downloading mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
Collecting MarkupSafe>=2.0 (from jinja2->torch)
  Downloading markupsafe-3.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (2.7 kB)
Collecting shellingham>=1.3.0 (from typer->transformers)
  Downloading shellingham-1.5.4-py2.py3-none-any.whl.metadata (3.5 kB)
Collecting rich>=13.8.0 (from typer->transformers)
  Downloading rich-15.0.0-py3-none-any.whl.metadata (18 kB)
Collecting annotated-doc>=0.0.2 (from typer->transformers)
  Downloading annotated_doc-0.0.4-py3-none-any.whl.metadata (6.6 kB)
Collecting anyio (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers)
  Downloading anyio-4.14.2-py3-none-any.whl.metadata (4.6 kB)
Collecting certifi (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers)
  Downloading certifi-2026.7.22-py3-none-any.whl.metadata (2.5 kB)
Collecting httpcore==1.* (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers)
  Downloading httpcore-1.0.9-py3-none-any.whl.metadata (21 kB)
Collecting idna (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers)
  Downloading idna-3.18-py3-none-any.whl.metadata (6.1 kB)
Collecting h11>=0.16 (from httpcore==1.*->httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers)
  Downloading h11-0.16.0-py3-none-any.whl.metadata (8.3 kB)
Collecting markdown-it-py>=2.2.0 (from rich>=13.8.0->typer->transformers)
  Downloading markdown_it_py-4.2.0-py3-none-any.whl.metadata (7.4 kB)
Collecting pygments<3.0.0,>=2.13.0 (from rich>=13.8.0->typer->transformers)
  Downloading pygments-2.20.0-py3-none-any.whl.metadata (2.5 kB)
Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich>=13.8.0->typer->transformers)
  Downloading mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
Downloading torch-2.13.0-cp312-cp312-manylinux_2_28_x86_64.whl (526.6 MB)
Error: Invalid value. 'charmap' codec can't encode characters in position 3-41: character maps to <undefined>
Hint: set HF_DEBUG=1 as environment variable for full traceback.

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d67a697e367e", "created_at": "2026-07-24T10:12:50+00:00", "title": "Kernel correctness check results (HF Job https://huggingface.co/jobs/Devavrat28…"}
-->
Kernel correctness check results (HF Job https://huggingface.co/jobs/Devavrat28/6a633a27db23d7a7ec1ca3c3, a100-large, running_secs=80, ~$0.06 total): BOTH checks fail as shipped, confirming the upstream repo's own TODO 'Fix kernels for acc analysis'.

1. abx_rope_batched.py --check: FAILS with RuntimeError: 'Expected all tensors to be on the same device, but got mat2 is on cpu, different from other tensors on cuda:0', raised inside torch_abx() at the rotary_emb(xb, position_ids) call (transformers modeling_llama.py rope forward, the mat2 in inv_freq_expanded.float() @ position_ids_expanded.float()). This is the pure-PyTorch reference path used to validate the Triton kernel output, not the Triton kernel itself - some tensor (likely position_ids or the rotary embedding module) is not moved to CUDA before this call. Exit code 1.

2. bx_quant.py --check: FAILS with ModuleNotFoundError: No module named 'quant_utils', as predicted from static analysis before running (see prior cell). Exit code 1.

Conclusion: Attention-speedup and throughput numbers (Claim 4) cannot currently be verified for correctness on this clone of the repo - both the low-rank+RoPE kernel check and the quantized kernel check fail before producing any comparison numbers. Any speed benchmarks run without --check passing first would be numbers without a correctness guarantee. Recording this as a valid negative finding per CLAUDE.md guidance rather than attempting to patch the upstream repo (which we are instructed not to modify).
