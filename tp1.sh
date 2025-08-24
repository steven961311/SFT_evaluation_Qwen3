#!/bin/bash

#Batch Job Paremeters
#SBATCH --account=GOV113121
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1             # one torchrun per node https://stackoverflow.com/a/65897194
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=200G
#SBATCH --gpus-per-node=4
#SBATCH --distribution=cyclic:cyclic,NoPack
#SBATCH --time=02:00:00                 # total run time limit (HH:MM:SS)

# net
export UCX_NET_DEVICES=mlx5_0:1
export UCX_IB_GPU_DIRECT_RDMA=1

# enable NCCL log
# export NCCL_DEBUG=INFO
# export NCCL_TIMEOUT=3600

# setup master address and port
# https://iservice.nchc.org.tw/nchc_service/nchc_service_news_content.php?contentId=1000254&type=all_content&newsId=58494
# HPC容器環境打包技術與效能案例實作.pdf
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# prevent OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load cuda/12.4
cd /home/paslab504llm/silicon_mind/trl_env
source .venv/bin/activate
which python

# Set training parameters
uid="$(date +%Y%m%d_%H%M%S)"
run_name="v9.5_post-rl-sft-hard"

CMD="nsys profile -t cuda,nvtx,osrt,cublas,cudnn,ucx --force-overwrite=true \
    -o traces/sft-$run_name \
    accelerate launch \
    --config_file /home/paslab504llm/silicon_mind/slurm/grpo_deepspeed_config.yaml \
    --num_processes 8 \
    --num_machines 1 \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend c10d \
    /home/paslab504llm/silicon_mind/models/qwen3_sft.py \
    --model-path /home/paslab504llm/qwen3-30B-A3B \
    --train-data-path ntu-paslab-llm/qwen3-30B-A3B_train \
    --output-dir /home/paslab504llm/qwen3-30B-A3B_sft/$run_name"

# https://discuss.pytorch.org/t/distributed-training-on-slurm-cluster/150417/8
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist = " $(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "Number of nodes = " $SLURM_JOB_NUM_NODES
echo "GPUs per node = " $SLURM_GPUS_PER_NODE
echo "Ntasks per node = "  $SLURM_NTASKS_PER_NODE
echo "CPUs per GPU = " $SLURM_CPUS_PER_GPU
echo "Master =" $MASTER_ADDR $MASTER_PORT
echo "srun" $CMD
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

srun $CMD