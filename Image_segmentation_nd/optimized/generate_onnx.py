import tensorflow.keras as keras
from keras2onnx import convert_keras
import tensorflow as tf
import tf2onnx
import onnxruntime as rt
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
HEIGHT = 512
WIDTH = 1024
    
semantic_model = keras.models.load_model('../models')
semantic_model.summary()
#tf.keras.models.save_model(semantic_model, '../models')
spec = (tf.TensorSpec((None, 3, HEIGHT, WIDTH), tf.float32, name="input"),)
output_path = './models/onnx/semantic_model' + ".onnx"

model_proto, _ = tf2onnx.convert.from_keras(semantic_model, input_signature=spec, opset=13, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]
print(output_names)        
#if __name__=="__main__":
#    keras_to_onnx(semantic_model, './models/semantic_segmentation.onnx')