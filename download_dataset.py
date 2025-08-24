from datasets import load_dataset
import json

dataset = load_dataset("shaofish/qwen3_for_tp1")

dataset["train"].to_json("data/train.jsonl", lines=True, force_ascii=False)
dataset["validation"].to_json("data/val.jsonl", lines=True, force_ascii=False)

