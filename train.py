#!/usr/bin/env python
# coding: utf-8

# ## 读取本地数据

import pandas as pd

df_train = pd.read_csv("data/train.csv")
df_dev = pd.read_csv("data/dev.csv")
df_test = pd.read_csv("data/test.csv")

dict_key = {-1:2, 0:0, 1:1}  # 标签为-1的改为2
df_train["label"] = df_train["label"].apply(lambda x: dict_key[x])
df_dev["label"] = df_dev["label"].apply(lambda x: dict_key[x])
df_test["label"] = df_test["label"].apply(lambda x: dict_key[x])
df_dev.head()

# ### 加载需要的模块

import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from transformers import pipeline, DataCollatorWithPadding
from datasets import load_dataset, Dataset, Features, Value, ClassLabel

# 构造dataset格式数据集

torch.cuda.is_available()

class_names = [0, 1, 2]
features = Features({'txt': Value('string'), 'label': ClassLabel(names=class_names)})
features

train_dataset = Dataset.from_pandas(df_train, split="train", features=features)
dev_dataset = Dataset.from_pandas(df_test, split="dev", features=features)
test_dataset = Dataset.from_pandas(df_test, split="test", features=features)
train_dataset[0]

train_dataset.features

checkpoint = "bert-base-chinese"
num_labels = train_dataset.features["label"].num_classes
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)  # 三分类

# 为了放进pytorch模型训练，还要再声明格式和使用的字段

def tokenize_function(example):
    return tokenizer(example["txt"], truncation=True)

tokenized_datasets_train = train_dataset.map(tokenize_function, batched=True)
tokenized_datasets_dev = dev_dataset.map(tokenize_function, batched=True)
tokenized_datasets_test = test_dataset.map(tokenize_function, batched=True)

tokenized_datasets_train.features

tokenized_datasets_train[0]

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  #动态填充，即将每个批次的输入序列填充到一样的长度

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# 指定训练参数，使用trainer直接训练


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=5,              # total number of training epochs
    logging_dir='./logs',            # directory for storing logs
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch"
)


trainer = Trainer(
    model=model,                         # the instantiated   Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=tokenized_datasets_train,         # training dataset
    eval_dataset=tokenized_datasets_dev,            # evaluation dataset
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


train_out = trainer.train()


