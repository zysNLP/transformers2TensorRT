# Transformers文本分类微调和TensorRT推理加速

博客教程：https://blog.csdn.net/weixin_35977125/article/details/128008689

## 开发环境简介

### 宿主机环境

```
Linux version:  Ubuntu20.04
Docker version: 20.10.20
Python version: 3.8.5 (3.6-3.9)
NVIDIA version: 1080Ti
Cuda version: cuda11.6 (>11.1)
Driver version: 510.47.03 (>460)
```

Python packages: 

```shell
transformers           4.24.0
torch                  1.13.0+cu116
scikit-learn           1.1.3
pandas                 1.5.1
numpy                  1.23.5
```

### 容器内环境

容器由TensorRT DockerFile构建，将在后续进行介绍，这里只介绍python环境

```
Python version: 3.6.9
```

Python packages: 

```shell
numpy               1.19.5
pycuda              2022.1
torch               1.10.2+cu111
transformers        4.18.0
```

## 一. 准备数据

### 1.数据样例

```shell
(base) ys@ys:~/Documents/my_trt/data$ head -10 dev.csv 
label,txt
-1,一个月都有点卡了，
1,手机很不错，玩游戏很流畅，快递小哥态度特别特别特别好，重要的事情说三遍?
1,初步用了一下，开始拿到手觉得真**大，玩了一会以后 真香！玩游戏很爽，双扬声器看片也很震撼～拍照的话现在还凑活看后期更新优化吧！这款手机比较适合影音娱乐用，一般人不建议用。
-1,续航不行 玩吃鸡卡
1,物流配送速度很快，三天左右就到了，画面清晰，音质还挺好的，总之超喜欢的
0,着好看，都挺不错的。   但是京东的保价，双十一便宜100元，联系客服，说退100，这都10多天了，仍然没有解决。
1,手机是正品，质量很好，速度很快，价格不比别家便宜，还送膜、送耳机、送自拍神器。谢谢商家！
1,机很流畅，买的很好的一次，下次还要购买，一点都不卡。
0,不能再相信它了
```

标注为`1`表示情感积极，标注为`0`表示情感中立，标注为`-1`表示情感消极

```shell
(base) ys@ys:~/Documents/my_trt/data$ tree
.
├── dev.csv
├── test.csv
└── train.csv
```

### 2.数据处理

```python
#!/usr/bin/env python
# coding: utf-8

# ## 读取本地数据

import pandas as pd

df_train = pd.read_csv("data/train.csv")
df_dev   = pd.read_csv("data/dev.csv")
df_test  = pd.read_csv("data/test.csv")

dict_key = {-1:2, 0:0, 1:1}  # 标签为-1的改为2，其他保持不变；标签-1在Transformers中读取会报错
df_train["label"] = df_train["label"].apply(lambda x: dict_key[x])
df_dev["label"]   = df_dev["label"].apply(lambda x: dict_key[x])
df_test["label"]  = df_test["label"].apply(lambda x: dict_key[x])
```

### 3.构造DataSets格式数据

```python
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from transformers import pipeline, DataCollatorWithPadding
from datasets import load_dataset, Dataset, Features, Value, ClassLabel

# 构造dataset格式数据集

class_names = [0, 1, 2]
features = Features({'txt': Value('string'), 'label': ClassLabel(names=class_names)})

train_dataset = Dataset.from_pandas(df_train, split="train", features=features)
dev_dataset = Dataset.from_pandas(df_test, split="dev", features=features)
test_dataset = Dataset.from_pandas(df_test, split="test", features=features)
```

## 二.下载和加载预训练模型

### 1.下载bert预训练模型参数

前往https://huggingface.co/models，搜索`bert-base-chinese`，在`Files and versions`中下载三个文件

```shell
(base) ys@ys:~/Documents/my_trt/bert-base-chinese$ tree
.
├── config.json
├── pytorch_model.bin
└── vocab.txt
```

### 2.加载bert预训练模型参数

```python
checkpoint = "bert-base-chinese"
num_labels = train_dataset.features["label"].num_classes
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)  # 三分类
```

### 3.声明模型使用的字段

```python
# 为了放进pytorch模型训练，还要再声明格式和使用的字段
def tokenize_function(example):
    return tokenizer(example["txt"], truncation=True)

tokenized_datasets_train = train_dataset.map(tokenize_function, batched=True)
tokenized_datasets_dev = dev_dataset.map(tokenize_function, batched=True)
tokenized_datasets_test = test_dataset.map(tokenize_function, batched=True)
```

### 4.设置动态填充

```python
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  #动态填充，即将每个批次的输入序列填充到一样的长度
```

## 三.设置参数进行训练

### 1.定义模型Metrics

```python
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
```

### 2.指定训练参数

```python
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=5,              # total number of training epochs
    logging_dir='./logs',            # directory for storing logs
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch"
)
```

### 3.开始训练

训练过程没有设置收敛停止条件，模型将在训练num_train_epochs=5后停止训练，后续可以优化设置早停，防止过拟合

```python
trainer = Trainer(
    model=model,              # the instantiated   Transformers model to be trained
    args=training_args,       # training arguments, defined above
    train_dataset=tokenized_datasets_train,         # training dataset
    eval_dataset=tokenized_datasets_dev,            # evaluation dataset
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

train_out = trainer.train()
```

训练完成后，会产生checkpoint文件，用于直接进行推理和后续转pth格式文件

```shell
(base) ys@ys:~/Documents/my_trt/results/checkpoint-3000$ tree 
.
├── config.json
├── optimizer.pt
├── pytorch_model.bin
├── rng_state.pth
├── scheduler.pt
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
├── trainer_state.json
├── training_args.bin
└── vocab.txt

0 directories, 11 files
```



## 四. 单条数据推理验证

这里新建了一个`eval.py`，重新导包，读取的dev数据

训练完成后，取F1值最高的`checkpoint-3000`里的参数作为预测的模型参数

```python
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
input_text = test_dataset[0]['txt']
print("Input Text:", input_text)
result = classifier(input_text)
time1 = time.time()
print(result, "use time:", time1-time0)
```

```shell
(base) ys@ys:~/Documents/my_trt$ python eval.py 
Input Text: 荣耀8都比这强 # 注意到，这句是贬义，预测得分0.91，比较准确
[{'label': 'LABEL_2', 'score': 0.9167261719703674}] use time: 0.020716428756713867
```

## 五. TensorRT模型推理加速

### 1.模型重新保存为pth格式

这里，新建了一个`save_model.py`文件

保存格式使用`model.state_dict()`方法，仅保存模型参数(推荐使用)，不保存整个模型

注意，这一步可以在模型训练完后直接保存，即第三节3中`train_out = trainer.train()`完成后，直接加载`torch.save(model.state_dict(), "results/bert_params.pth")`

```python
import torch
from transformers import  AutoTokenizer
from transformers import AutoModelForSequenceClassification

checkpoint = "results/checkpoint-3000"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

torch.save(model.state_dict(), "results/bert_params.pth")
print("模型保存成功")
```

### 2.pth转Onnx

这里，使用 Pytorch 自带的` torch.onnx.export`方法：新建一个`pth2onnx.py`文件

```python
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from transformers import  AutoTokenizer
from transformers import AutoModelForSequenceClassification

def export_static_onnx_model(text, model, tokenizer, static_onnx_model_path):
    # example_tensor
    inputs = tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True)
    device = torch.device("cpu")
    inputs = inputs.to(device)
    print(inputs)

    with torch.no_grad():
        torch.onnx.export(model,            # model being run
                    (inputs['input_ids'],   # model input (or a tuple for multiple inputs)
                    inputs['attention_mask']),
                    static_onnx_model_path, # where to save the model (can be a file or file-like object)
                    verbose=True,
                    opset_version=11,       # the ONNX version to export the model to
                    input_names=['input_ids',    # the model's input names
                                'input_mask'],
                    output_names=['output'])     # the model's output names
        print("ONNX Model exported to {0}".format(static_onnx_model_path))

if __name__ == '__main__':
    torch_model = torch.load("results/bert_params.pth")  # pytorch模型加载
    # 模型多次调用了tokenizer和model，待优化
    checkpoint = "results/checkpoint-3000"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)  # 三分类
    model.load_state_dict(torch_model)

    text = "这个手机挺好的"
    static_onnx_model_path = "results/bert_static.onnx"
    export_static_onnx_model(text, model, tokenizer, static_onnx_model_path)
```

顺利转换完成后，会在results文件夹下生成一个bert_static.onnx文件

### 3.安装TensorRT

首先在宿主机上安装docker

#### (1) 编写DockerFile

```dockerfile
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG CUDA_VERSION=11.1.1
ARG CUDNN_VERSION=8
ARG OS_VERSION=18.04

# 从nvidia 官方镜像库拉取基础镜像
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${OS_VERSION}
LABEL maintainer="yuesheng"

ENV TRT_VERSION 7.2.3.4
# ENV TRT_VERSION 8.2.1.8
SHELL ["/bin/bash", "-c"]

# 安装必要的库
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    vim \
    zlib1g-dev \
    git \
    pkg-config \
    sudo \
    ssh \
    libssl-dev \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    devscripts \
    lintian \
    fakeroot \
    dh-make \
    build-essential \
    libgl1-mesa-glx


# 安装 python3 环境
RUN apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-wheel &&\
    cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip;

# 安装 TensorRT
RUN cd /tmp && sudo apt-get update

RUN version="8.2.1-1+cuda11.4" && \
    sudo apt-get install libnvinfer8=${version} libnvonnxparsers8=${version} libnvparsers8=${version} libnvinfer-plugin8=${version} libnvinfer-dev=${version} libnvonnxparsers-dev=${version} libnvparsers-dev=${version} libnvinfer-plugin-dev=${version} python3-libnvinfer=${version} &&\
    sudo apt-mark hold libnvinfer8 libnvonnxparsers8 libnvparsers8 libnvinfer-plugin8 libnvinfer-dev libnvonnxparsers-dev libnvparsers-dev libnvinfer-plugin-dev python3-libnvinfer

# 升级 pip 并切换成国内豆瓣源
RUN python3 -m pip install -i https://pypi.douban.com/simple/ --upgrade pip
RUN pip3 config set global.index-url https://pypi.douban.com/simple/
RUN pip3 install setuptools>=41.0.0

# 设置环境变量和工作路径
ENV TRT_LIBPATH /usr/lib/x86_64-linux-gnu
ENV TRT_OSSPATH /workspace/TensorRT
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"
WORKDIR /workspace

# 设置语言环境为中文，防止 print 中文报错
ENV LANG C.UTF-8

RUN ["/bin/bash"]
```

#### (2) 构建镜像

```shell
docker build -t tensorrt-7234-1080ti:v1 -f build/docker/Dockerfile .
```

#### (3) 创建容器

```shell
 docker run -itd --gpus all -v path/to/my_trt:/workspace/ --name my_trt tensorrt-7234-1080ti:v1
```

#### (4) 进入容器

```shell
docker exec -it my_trt /bin/bash
```

#### (5) 下载TensorRT压缩包

在英伟达官网注册账号后，打开https://developer.nvidia.com/nvidia-tensorrt-8x-download，找到[TensorRT 8.0.3 GA for Linux x86_64 and CUDA 11.3 TAR package](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.0.3/tars/tensorrt-8.0.3.4.linux.x86_64-gnu.cuda-11.3.cudnn8.2.tar.gz)，下载压缩包`TensorRT-8.0.3.4.Linux.x86_64-gnu.cuda-11.3.cudnn8.2.tar.gz`

容器内解压

```shell
tar -zxvf TensorRT-8.0.3.4.Linux.x86_64-gnu.cuda-11.3.cudnn8.2.tar.gz
export TRT_LIBPATH=`pwd`/TensorRT-8.0.3.4
```

### 4.Onnx转TensorRT engine

#### (1) 容器内，找到trtexec执行文件

```shell
cd /workspace/TensorRT-8.0.3.4
```

#### (2) 编写转换命令

这里新建了一个shell脚本：`vim run.sh`

因为提示报错，因此没有指定`--minShapes`，`--optShapes`，`--maxShapes`三个参数

```bash
./trtexec \
  --onnx=/workspace/results/bert_static.onnx \
  --saveEngine=/workspace/results/bert_static.trt \
  --workspace=1024 \
  --fp16
```

#### (3) 执行转换命令

```shell
bash run.sh
```

### 5.构建TensorRT Runtime加速推理

构建单条数据的推理流程，新建`preds.py`文件

#### (1) 获取 engine，建立上下文

```python
import numpy as np
from transformers import BertTokenizer
import tensorrt as trt
import common

"""
a、获取 engine，建立上下文
"""
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine

engine_model_path = "/workspace/results/bert_static.trt"
# Build a TensorRT engine.
engine = get_engine(engine_model_path)
# Contexts are used to perform inference.
context = engine.create_execution_context()
```

#### (2) 从 engine 中获取 inputs, outputs, bindings, stream 的格式以及分配缓存

```python
"""
b、从engine中获取inputs, outputs, bindings, stream 的格式以及分配缓存
"""
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

sentence = "荣耀8都比这强"
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
inputs = tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=True)

tokens_id =  to_numpy(inputs['input_ids'].int())
attention_mask = to_numpy(inputs['attention_mask'].int())

context.active_optimization_profile = 0
origin_inputshape = context.get_binding_shape(0)                # (1,-1) 
origin_inputshape[0],origin_inputshape[1] = tokens_id.shape     # (batch_size, max_sequence_length)
context.set_binding_shape(0, (origin_inputshape))               
context.set_binding_shape(1, (origin_inputshape))
```

#### (3) 输入数据填充

```python
"""
c、输入数据填充
"""
inputs, outputs, bindings, stream = common.allocate_buffers_v2(engine, context)
inputs[0].host = tokens_id
inputs[1].host = attention_mask
```

#### (4) TensorRT推理

```python
"""
d、tensorrt推理
"""
import time
time1 = time.time()
trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
preds = np.argmax(trt_outputs, axis=1)
time2 = time.time()
print("====sentence=:", sentence)
print("====preds====:", preds)
print("====use time:=", time2-time1)
```

#### (5) 输出结果

```shell
root@4aa888259e98:/workspace# python preds.py 
Reading engine from file /workspace/results/bert_static.trt
[TensorRT] WARNING: TensorRT was linked against cuBLAS/cuBLAS LT 11.6.3 but loaded cuBLAS/cuBLAS LT 11.3.0
[TensorRT] WARNING: TensorRT was linked against cuDNN 8.2.1 but loaded cuDNN 8.0.5
[TensorRT] WARNING: TensorRT was linked against cuBLAS/cuBLAS LT 11.6.3 but loaded cuBLAS/cuBLAS LT 11.3.0
[TensorRT] WARNING: TensorRT was linked against cuDNN 8.2.1 but loaded cuDNN 8.0.5
====sentence=: 荣耀8都比这强
====preds====: [2]
====use time:= 0.002056598663330078
```

