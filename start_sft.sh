export PYTHONPATH=/njfs/train-nlp/zhouyi9/zz/open-r1/src:$PYTHONPATH

exp_name="open-qwen1.5b-bespoke17k"
export WANDB_BASE_URL="http://wandb.wml.weibo.com"
export WANDB_API_KEY="local-42b9064b9346fadf444a6c60b5a4cb121d8eebdf"
export WANDB_PROJECT="open-r1"
export WANDB_NOTES=${exp_name}
export WANDB_WATCH="all"
export WANDB_MODE="online"
export WANDB_DIR=${exp_name}



export GIT_SSH_COMMAND="ssh -F /njfs/train-nlp/zhouyi9/.ssh/config"


accelerate launch --debug  --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --wandb_entity zhouyi9 \
    --wandb_project open-r1 \
    --model_name_or_path /njfs/train-nlp/zhouyi9/base_models/Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir /njfs/train-nlp/zhouyi9/zz/open-r1/data/Qwen2.5-1.5B-Open-R1-Distill


# torchrun \
#     --nnodes=1 \
#     --nproc_per_node=8 \  # 根据实际 GPU 数量调整
#     --master_port=9901 \
#     src/open_r1/sft.py \
#     --model_name_or_path /njfs/train-nlp/zhouyi9/base_models/Qwen/Qwen2.5-1.5B-Instruct \
#     --dataset_name /njfs/train-nlp/zhouyi9/datasets/cache/bespokelabs___bespoke-stratos-17k \
#     --learning_rate 2.0e-5 \
#     --num_train_epochs 1 \
#     --packing \
#     --max_seq_length 4096 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --gradient_checkpointing \
#     --bf16 \
#     --output_dir data/Qwen2.5-1.5B-Open-R1-Distill \
#     --deepspeed recipes/accelerate_configs/zero3.yaml  # 需确保代码支持 DeepSpeed

