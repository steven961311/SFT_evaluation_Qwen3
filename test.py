from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

train_dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
eval_dataset = load_dataset("json", data_files="data/val.jsonl", split="train")

config = SFTConfig(
    output_dir = "outputs/qwen3-0.6b",
    per_device_train_batch_size=4,
    max_length = 512,
    num_train_epochs=1
)

trainer = SFTTrainer(
    model="./models/Qwen3-0.6B",
    train_dataset= train_dataset,
    eval_dataset= eval_dataset,
    args=config
)
trainer.train()