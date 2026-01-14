#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from statistics import mean

from rouge_score import rouge_scorer


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rouge_l(rows):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(r["reference"], r["prediction"])["rougeL"].fmeasure
        for r in rows
    ]
    return mean(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Rouge-L over result JSONL files.")
    parser.add_argument("--input", type=Path, help="Single results JSONL file.")
    parser.add_argument("--awq", type=Path, help="AWQ results JSONL file.")
    parser.add_argument("--base", type=Path, help="Base results JSONL file.")
    args = parser.parse_args()

    if args.input:
        rows = load_rows(args.input)
        score = rouge_l(rows)
        print(f"{args.input}: Rouge-L {score:.4f} ({len(rows)} samples)")
        return

    if not (args.awq and args.base):
        raise SystemExit("Provide --input or both --awq and --base.")

    awq_rows = load_rows(args.awq)
    base_rows = load_rows(args.base)
    awq_score = rouge_l(awq_rows)
    base_score = rouge_l(base_rows)
    print(f"AWQ Rouge-L:  {awq_score:.4f} ({len(awq_rows)} samples)")
    print(f"Base Rouge-L: {base_score:.4f} ({len(base_rows)} samples)")
    print(f"Delta (AWQ - Base): {awq_score - base_score:+.4f}")


if __name__ == "__main__":
    main()
