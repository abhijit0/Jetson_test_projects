import argparse
from onnx import ModelProto
import tensorrt as trt
import engine as eng 
 

def save_engine(engine_path="model/plan/resnet50.plan", onnx_path = "models/onnx/resnet50.onnx", batch_size = 1):
    model = ModelProto()
    with open(onnx_path, "rb") as f:
        model.ParseFromString(f.read())

    d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
    shape = [batch_size , d0, d1 ,d2]
    print("plan model ",shape)
    engine = eng.build_engine(onnx_path, shape= shape)
    eng.save_engine(engine, engine_path)

def load_engine(plan_path = "model/plan/resnet50.plan", trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))):
    print(plan_path)
    engine = eng.load_engine(trt_runtime, plan_path)
    return engine