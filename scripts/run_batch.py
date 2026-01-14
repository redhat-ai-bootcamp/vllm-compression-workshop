#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer


TASKS = {
    "xsum": {
        "dataset": ("xsum", None),
        "text": "document",
        "reference": "summary",
        "prompt": "Summarize the following article in 1-2 sentences:\n\n{input}\n\nSummary:",
    },
    "cnn_dailymail": {
        "dataset": ("cnn_dailymail", "3.0.0"),
        "text": "article",
        "reference": "highlights",
        "prompt": "Summarize the following article in 2-3 sentences:\n\n{input}\n\nSummary:",
    },
    "squad": {
        "dataset": ("squad", None),
        "text": "context",
        "reference": "answers",
        "prompt": "Answer the question using the context.\n\nContext:\n{input}\n\nQuestion: {question}\n\nAnswer:",
    },
}


def load_tokenizer(tokenizer_id: str | None):
    if not tokenizer_id:
        return None
    try:
        return AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    except Exception as exc:
        print(f"Warning: could not load tokenizer '{tokenizer_id}': {exc}")
        return None


def build_prompt(task_name: str, row: dict, tokenizer, max_user_tokens: int | None, max_input_chars: int):
    task = TASKS[task_name]
    text = row[task["text"]]
    if task_name == "squad":
        reference = row[task["reference"]]["text"][0]
        question = row["question"]
    else:
        reference = row[task["reference"]]
        question = None

    template = task["prompt"]
    if tokenizer and max_user_tokens is not None:
        prefix, suffix = template.split("{input}")
        if question is not None:
            suffix = suffix.format(question=question)
        prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
        suffix_tokens = len(tokenizer.encode(suffix, add_special_tokens=False))
        available = max(max_user_tokens - prefix_tokens - suffix_tokens, 0)
        input_tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(input_tokens) > available:
            text = tokenizer.decode(input_tokens[:available], skip_special_tokens=True)
        prompt = prefix + text + suffix
    else:
        if max_input_chars and len(text) > max_input_chars:
            text = text[:max_input_chars]
        if question is not None:
            prompt = template.format(input=text, question=question)
        else:
            prompt = template.format(input=text)

    return prompt, reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch job against a served vLLM model.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--model", required=True, help="Model name served by vLLM (see /v1/models).")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path/ID used for prompt truncation.")
    parser.add_argument("--task", choices=sorted(TASKS.keys()), default="xsum")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument("--max-input-chars", type=int, default=4000)
    parser.add_argument("--context-buffer", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", default="results/batch_results.jsonl")
    parser.add_argument("--system-prompt", default="You are a concise assistant.")
    args = parser.parse_args()

    dataset_name, dataset_config = TASKS[args.task]["dataset"]
    ds = load_dataset(dataset_name, dataset_config, split=args.split)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer or args.model)
    if tokenizer:
        system_tokens = len(tokenizer.encode(args.system_prompt, add_special_tokens=False))
        max_user_tokens = max(
            args.max_context_tokens - args.max_tokens - system_tokens - args.context_buffer,
            0,
        )
    else:
        max_user_tokens = None

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    rows_written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in tqdm(enumerate(ds), total=min(args.max_samples, len(ds))):
            if rows_written >= args.max_samples:
                break
            prompt, reference = build_prompt(
                args.task,
                row,
                tokenizer,
                max_user_tokens,
                args.max_input_chars,
            )
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            prediction = resp.choices[0].message.content or ""
            record = {
                "id": idx,
                "task": args.task,
                "prompt": prompt,
                "reference": reference,
                "prediction": prediction,
                "model": args.model,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows_written += 1

    print(f"Wrote {rows_written} rows to {output_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
