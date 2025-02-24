MODEL=/njfs/train-nlp/zhouyi9/base_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8"
OUTPUT_DIR=data/evals/DeepSeek-R1-Distill-Qwen-7B

# AIME 2024
TASK=aime24
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR