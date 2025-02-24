from datasets import load_dataset

# 指定目标目录（例如 /data/datasets/numina_math）
custom_dir = "/njfs/train-nlp/zhouyi9/datasets"

# 下载数据集到固定目录
ds = load_dataset("AI-MO/NuminaMath-TIR", cache_dir=custom_dir)