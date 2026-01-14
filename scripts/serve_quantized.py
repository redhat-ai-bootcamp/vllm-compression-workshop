#!/usr/bin/env python3
from serve_vllm import main


if __name__ == "__main__":
    # Defaults to the AWQ directory; override with --model gptq or a custom path.
    # Use quantization=none because llmcompressor saves a compressed-tensors config.
    main(default_model="awq", default_quant="none")
