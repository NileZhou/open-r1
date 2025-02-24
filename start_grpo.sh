export PYTHONPATH=/njfs/train-nlp/zhouyi9/zz/open-r1/src:$PYTHONPATH

# http://10.136.0.191:10890
export http_proxy=
export https_proxy=

exp_name="open-qwen1_5b-OpenR1-Math-220k-grpo"
export WANDB_BASE_URL="http://wandb.wml.weibo.com"
export WANDB_API_KEY="local-42b9064b9346fadf444a6c60b5a4cb121d8eebdf"
export WANDB_PROJECT="open-r1"
export WANDB_NOTES=${exp_name}
export WANDB_WATCH="all"
export WANDB_MODE="online"
export WANDB_DIR=${exp_name}



export GIT_SSH_COMMAND="ssh -F /njfs/train-nlp/zhouyi9/.ssh/config"


ACCELERATE_LOG_LEVEL=info \
accelerate launch --debug --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml