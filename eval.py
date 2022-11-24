import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel
from transformers import  AutoTokenizer, pipeline
from transformers import AutoModelForSequenceClassification
df_test = pd.read_csv("data/test.csv")
dict_key = {-1:2, 0:0, 1:1}  # 标签为-1的改为2
df_test["label"] = df_test["label"].apply(lambda x: dict_key[x])

checkpoint = "results/checkpoint-3000"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)  # 三分类

class_names = [0, 1, 2]
features = Features({'txt': Value('string'), 'label': ClassLabel(names=class_names)})

test_dataset = Dataset.from_pandas(df_test, split="test", features=features)

model = model.cpu()
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

import time
time0 = time.time()
print(test_dataset[0]['txt'])
result = classifier(test_dataset[0]['txt'])
time1 = time.time()
print(result, "use time:", time1-time0)
