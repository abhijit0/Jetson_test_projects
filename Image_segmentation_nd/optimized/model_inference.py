import tensorrt as trt
import engine_ops as eng
import inference as inf
import pycuda.autoinit 


def initialize(engine_path, data_set, batch_size):
    engine = eng.load_engine(engine_path)
    h_input, d_input, h_output, d_output, stream = inf .allocate_buffers(engine, batch_size, trt.float32)
    return engine, h_input, d_input, h_output, d_output, stream

def inference(engine_path, data_set, batch_size):
    engine, h_input, d_input, h_output, d_output, stream = initialize(engine_path, data_set, batch_size)
    out = inf.do_inference(engine, data_set, h_input, d_input, h_output, d_output, stream, batch_size)
    return out

def inference_winit(engine, h_input, d_input, h_output, d_output, stream, data_set, batch_size):
    out = inf.do_inference(engine, data_set, h_input, d_input, h_output, d_output, stream, batch_size)
    return out