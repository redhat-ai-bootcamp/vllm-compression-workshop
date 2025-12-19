# vLLM + TensorRT-LLM Notebooks

This repo contains Jupyter notebooks for quantizing Llama Instruct models to 4-bit with vLLM compression tools, and for benchmarking full-precision throughput between vLLM and TensorRT-LLM.

## Files
- `gptq_quantization.ipynb` – W4A16 weight-only quantization with GPTQ (llm-compressor one-shot flow).
- `awq_quantization.ipynb` – W4A16 Activation-Aware Quantization (AWQ) flow.
- `throughput_comparison.ipynb` – Full-precision Llama-3-8B throughput comparison: TensorRT-LLM vs vLLM (tokens/sec).

## Provision a GPU VM (example: Ubuntu + CUDA 12.x)
1) Create a VM with an NVIDIA GPU (Ampere+), at least 80 GB disk, and Ubuntu 22.04.
2) Install drivers/CUDA if not preinstalled. Verify with `nvidia-smi` (look for CUDA >= 12.x).
3) Install Python 3.10+ and Git:
   ```bash
   sudo apt update && sudo apt install -y python3.10-venv git
   ```
4) Clone this repo and create a venv:
   ```bash
   git clone <your-fork-or-path> vllm-compression-workshop
   cd vllm-compression-workshop
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   
5) If models are gated, export your HF token:
   ```bash
   export HUGGINGFACE_HUB_TOKEN=<token>
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
2) Run the first cell to install dependencies (this may take time and disk space).
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
