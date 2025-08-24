from huggingface_hub import snapshot_download

# Example: download Qwen3-0.6B
snapshot_download(repo_id="Qwen/Qwen3-30B-A3B", local_dir="./models/Qwen3-30B-A3B")
