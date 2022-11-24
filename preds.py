import numpy as np
from transformers import AutoTokenizer
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

"""
c、输入数据填充
"""
inputs, outputs, bindings, stream = common.allocate_buffers_v2(engine, context)
inputs[0].host = tokens_id
inputs[1].host = attention_mask

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
