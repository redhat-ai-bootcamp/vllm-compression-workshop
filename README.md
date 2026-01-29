# vLLM + TensorRT-LLM Notebooks

This repo contains Jupyter notebooks for quantizing Llama Instruct models to 4-bit with vLLM compression tools, and for benchmarking full-precision throughput between vLLM and TensorRT-LLM.

## Files
- `gptq_quantization.ipynb` – W4A16 weight-only quantization with GPTQ (llm-compressor one-shot flow).
- `awq_quantization.ipynb` – W4A16 Activation-Aware Quantization (AWQ) flow.
- `throughput_comparison.ipynb` – Full-precision Llama-3-8B throughput comparison: TensorRT-LLM vs vLLM (tokens/sec).

## Provision a GPU VM (example: Ubuntu + CUDA 12.x)
1) Create a VM with an NVIDIA GPU (Ampere+), at least 80 GB disk, and Ubuntu 22.04.
2) Install drivers/CUDA if not preinstalled. Verify with `nvidia-smi` (look for CUDA >= 12.x).
3) Install system packages (Python venv + dev headers for Triton, build tools, Git):
   ```bash
   sudo apt update
   sudo apt install -y python3-venv python3-dev build-essential git
   ```
4) Clone this repo and create a venv:
   ```bash
   git clone <your-fork-or-path> vllm-compression-workshop
   cd vllm-compression-workshop
   python3 -m venv .venv
   source .venv/bin/activate
   ```
5) Install Python dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
6) If models are gated, export your HF token (or set it in the notebook cell):
   ```bash
   export HUGGINGFACE_HUB_TOKEN=<token>
   ```

## Provision a GPU VM (RHEL 8/9 + CUDA)
1) Create a VM with an NVIDIA GPU (Ampere+), at least 80 GB disk, and RHEL 8 or 9.
2) Install Git + Python 3 + pip:
   ```bash
   sudo dnf update -y
   sudo dnf install -y git python3 python3-pip
   ```
3) Install NVIDIA driver (RHEL 8/9, network repo):
   ```bash
   # Run only the block that matches your RHEL version
   # RHEL 8
   sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
   sudo dnf install -y dnf-plugins-core
   sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

   # RHEL 9
   sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm
   sudo dnf install -y dnf-plugins-core
   sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

   # Common
   sudo dnf install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)
   sudo dnf clean expire-cache
   sudo dnf module install -y nvidia-driver:latest-dkms
   sudo reboot
   ```
4) Verify the driver:
   ```bash
   nvidia-smi
   ```
5) Create a venv:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

## VS Code + SSH to the VM
1) Install the “Remote - SSH” extension in VS Code.
2) Add your VM to `~/.ssh/config`, e.g.:
   ```
   Host gpu-vm
     HostName <vm-public-ip>
     User ubuntu
     IdentityFile ~/.ssh/<your-key>
   ```
3) In VS Code: `F1` → “Remote-SSH: Connect to Host…” → select `gpu-vm`.
4) Open the repo folder. VS Code will prompt to use the venv; if not, activate manually in the integrated terminal: `source .venv/bin/activate`.
5) Launch Jupyter from the terminal:
   ```bash
   pip install jupyterlab
   jupyter lab --no-browser --port 8888
   ```
   Then, in VS Code, use the Jupyter server URL it prints.

## Using the notebooks
1) Open a notebook (`gptq_quantization.ipynb`, `awq_quantization.ipynb`, or `throughput_comparison.ipynb`).
2) Run the first cell to install dependencies (or install `requirements.txt` in the venv).
3) Adjust model IDs and calibration/benchmark settings for your hardware.
4) Execute cells top-to-bottom.

### Quantization notebooks (GPTQ/AWQ)
- Default model: `meta-llama/Llama-3.2-1B-Instruct`.
- Produces a local 4-bit checkpoint (`llama-gptq-w4a16/` or `llama-awq-w4a16/`).
- Includes a 100-sample `tweet_eval/sentiment` accuracy probe.
- Serve with vLLM (example):
  ```bash
  vllm serve ./llama-gptq-w4a16 --max-model-len 4096 --tensor-parallel-size 1 --port 8000 --api-key dummy
  ```

### Serving scripts
Use the helper scripts in `scripts/` to start a vLLM OpenAI-compatible server.
```bash
# Quantized (AWQ, compressed-tensors)
python scripts/serve_quantized.py --port 8000 --api-key dummy --quantization none

# If you see a GPU free-memory error, lower utilization:
python scripts/serve_quantized.py --port 8000 --api-key dummy --quantization none --gpu-memory-utilization 0.8

# If you see CUDA OOM during warmup, reduce concurrency and context length:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/serve_quantized.py --port 8000 --api-key dummy --quantization none \
  --gpu-memory-utilization 0.7 --max-model-len 2048 --max-num-seqs 32

# Base model
python scripts/serve_unquantized.py --port 8001 --api-key dummy

# Custom model/path
python scripts/serve_vllm.py --model /path/to/model --quantization none --port 8002 --api-key dummy
```

### Batch evaluation + comparison (standard dataset + metric)
Standard evaluation uses the XSum validation set with Rouge-L (F1). The `--model` flag must match
the served model name reported by `curl http://localhost:8000/v1/models`.
```bash
python scripts/run_batch.py --base-url http://localhost:8000/v1 --model llama-awq-w4a16 \
  --task xsum --tokenizer llama-awq-w4a16 --max-context-tokens 2048 --context-buffer 128 \
  --output results/awq.jsonl --max-samples 1000

python scripts/run_batch.py --base-url http://localhost:8001/v1 --model base-llama \
  --task xsum --tokenizer meta-llama/Llama-3.2-1B-Instruct --max-context-tokens 2048 --context-buffer 128 \
  --output results/base.jsonl --max-samples 1000
```
If you see a 400 context-length error, lower `--max-context-tokens` or `--max-tokens` (or
reduce inputs with `--max-input-chars` or increase `--context-buffer`).

Score Rouge-L locally:
```bash
python scripts/score_rouge.py --awq results/awq.jsonl --base results/base.jsonl
```
Then use the comparison cell at the bottom of `awq_quantization.ipynb` to compute Rouge-L.

### Throughput comparison (TensorRT-LLM vs vLLM)
- Default model: `meta-llama/Meta-Llama-3-8B-Instruct`.
- Build a TensorRT-LLM engine via `trtllm-build` (CUDA 12.x, Ampere+). The engine is saved to `trtllm_engine/`.
- Benchmarks tokens/sec for TensorRT-LLM and vLLM on identical prompts, then saves `throughput_results.json`.
- Tune `MAX_NEW_TOKENS`, prompt batch size, and tensor parallel size to match your GPU memory.

## Tips
- Keep the chat template when building calibration data; avoid adding extra BOS tokens.
- Increase `NUM_CALIBRATION_SAMPLES`/`MAX_SEQUENCE_LENGTH` if you have headroom for better quality.
- If TensorRT-LLM build fails, try `--enable_fp16` or reduce max lengths; ensure CUDA 12.x.
- Use `--tensor-parallel-size` (vLLM) or `--tp_size` (trtllm-build) to span multiple GPUs.
