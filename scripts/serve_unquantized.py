#!/usr/bin/env python3
from serve_vllm import main


if __name__ == "__main__":
    main(
        default_model="meta-llama/Llama-3.2-1B-Instruct",
        default_quant="none",
    )
