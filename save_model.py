import torch

from transformers import  AutoTokenizer
from transformers import AutoModelForSequenceClassification

checkpoint = "results/checkpoint-3000"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)  # 三分类

torch.save(model.state_dict(), "results/bert_params.pth")
print("模型保存成功")
