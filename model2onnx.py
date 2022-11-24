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
        torch.onnx.export(model,                                        # model being run
                    (inputs['input_ids'],                               # model input (or a tuple for multiple inputs)
                    inputs['attention_mask']),
                    static_onnx_model_path,                             # where to save the model (can be a file or file-like object)
                    verbose=True,
                    opset_version=11,                                   # the ONNX version to export the model to
                    input_names=['input_ids',                           # the model's input names
                                'input_mask'],                           
                    output_names=['output'])                            # the model's output names
        print("ONNX Model exported to {0}".format(static_onnx_model_path))

def export_dynamic_onnx_model(text, model, tokenizer, dynamic_onnx_model_path):
    # example_tensor
    inputs = tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True)
    device = torch.device("cpu")
    inputs = inputs.to(device)
    print("======inputs======:", inputs)

    dynamic_ax = {'input_ids': [1], "input_mask": [1]}
    with torch.no_grad():
        torch.onnx.export(model,                                        # model being run
                    (inputs['input_ids'],                               # model input (or a tuple for multiple inputs)
                    inputs['attention_mask']),
                    dynamic_onnx_model_path,                            # where to save the model (can be a file or file-like object)
                    verbose=True,
                    opset_version=11,                                   # the ONNX version to export the model to
                    do_constant_folding=True,                           # whether to execute constant folding for optimization
                    input_names=['input_ids',                           # the model's input names
                                'input_mask'],
                    output_names=['output'],                            # the model's output names
                    dynamic_axes=dynamic_ax)
        print("ONNX Model exported to {0}".format(dynamic_onnx_model_path))

if __name__ == '__main__':
    torch_model = torch.load("results/bert_params.pth")  # pytorch模型加载
    checkpoint = "results/checkpoint-3000"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)  # 三分类
    model.load_state_dict(torch_model)

    text = "这个手机挺好的"
    static_onnx_model_path = "results/bert_static.onnx"
    dynamic_onnx_model_path = "results/bert_dynamic.onnx"
    export_static_onnx_model(text, model, tokenizer, static_onnx_model_path)
    export_dynamic_onnx_model(text, model, tokenizer, dynamic_onnx_model_path)



