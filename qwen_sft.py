from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

train_dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
eval_dataset = load_dataset("json", data_files="data/val.jsonl", split="train")

config = SFTConfig(
    completion_only_loss=True,
        max_length=24546,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=5,
        bf16=True,
        warmup_ratio=0.05,
        eval_strategy="no",
        logging_steps=1,
        save_strategy="no",
        lr_scheduler_type="cosine",
        learning_rate=1e-5,
        weight_decay=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        output_dir="output/qwen3-30B-A3B",
        save_only_model=True
)

trainer = SFTTrainer(
    model="./models/Qwen3-0.6B",
    train_dataset= train_dataset,
    eval_dataset= eval_dataset,
    args=config
)
trainer.train()