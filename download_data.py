import os
import datasets

os.environ['http_proxy'] = 'http://10.136.0.191:10890'
os.environ['https_proxy'] = 'http://10.136.0.191:10890'
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-tiny-512")



# #数据集下载
# from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('/njfs/train-nlp/zhouyi9/datasets/cache/bespokelabs___bespoke-stratos-17k/default-45716a10dbb21a2b/0.0.0/master', subset_name='default')
# #您可按需配置 subset_name、split，参照“快速使用”示例代码
# print(len(ds))


dataset = datasets.load_dataset("HuggingFaceH4/aime_2024")
print(len(dataset))
# dataset = datasets.load_dataset("open-r1/aime_2025_1")
print(len(dataset))
dataset = datasets.load_dataset("HuggingFaceH4/MATH-500")
print(len(dataset))
dataset = datasets.load_dataset("Idavidrein/gpqa")
print(len(dataset))

