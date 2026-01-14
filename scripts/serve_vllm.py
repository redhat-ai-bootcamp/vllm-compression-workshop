#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


DEFAULT_MODELS = {
    "awq": "llama-awq-w4a16",
    "gptq": "llama-gptq-w4a16",
}


def resolve_model(model: str) -> tuple[str, str | None]:
    model_key = model.lower()
    if model_key in DEFAULT_MODELS:
        return DEFAULT_MODELS[model_key], model_key
    return model, None


def main(argv: list[str] | None = None, default_model: str | None = None, default_quant: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve a model with vLLM's OpenAI-compatible API.")
    parser.add_argument("--model", default=default_model, required=default_model is None)
    parser.add_argument("--quantization", choices=["awq", "gptq", "none"], default=default_quant)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    args = parser.parse_args(argv)

    model, inferred_quant = resolve_model(args.model)
    quant = args.quantization or inferred_quant
    if quant == "none":
        quant = None

    served_name = args.served_model_name
    if not served_name:
        served_name = Path(model).name if os.path.exists(model) else model

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--api-key",
        args.api_key,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
        "--served-model-name",
        served_name,
    ]
    if quant:
        cmd.extend(["--quantization", quant])
    if args.gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(args.max_num_seqs)])

    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
