# Jetson_test_projects
Medium to high complexity Machine learning projects to compare the inference times between optimized and non optimized nueral networks

# Image Classification using Resnet50 
# Required Libraries:
```python
python == 3.6 
tensorrt == 8.0.1.6 
pycuda == 2021.1 
numpy == 1.19.4 
opencv == 4.1.1 #(preinstalled with jetpack) .
argparse == 1.1 
onnx == 1.10.1 
tensorflow == 2.5.0 
```

#### useful links for some of the important libraries:
Tensorflow : 
1) https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html
2) https://www.youtube.com/watch?v=ynK-X5IPu1A

(Make sure to specify right jetpack version if you are following the tutorial from the link 2)

onnx (both onnxruntime and tf2onnx):
1) https://github.com/onnx/tensorflow-onnx

Tensorrt engine:
1) https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html


### Very Important Note: As both unoptimized and optimized versions are running on GPUs make sure tensorflow is using GPU 
by using the following command in python environment:

```python
import tensorflow as tf
print(len(tf.config.list_physical_devices('GPU')))
# The output should be >= 1 .
```

## Instructions to Run the tests:
### place the data into ```data/```:
dataset can be found at : https://www.kaggle.com/alessiocorrado99/animals10?select=raw-img

After placing the data : the directory should look like : data/raw-img

Download the plan and onnx files from : https://drive.google.com/drive/u/2/folders/1yYDD2_teYR9wPLwCjV3OMMtrjWTyOBu0

### Place the plan and onnx file in the respective paths:
onnx : optimized/models/onnx/resnet50.onnx

plan : optimized/models/plan/resnet50.plan

## Run the following commands in the project directory
### unoptimized model on gpu:
To show the info of the python program (Helps to know if all the libraries are installed correctly): 
```bash
$ bash run_keras.sh -debug 
```
Without info (Only the intended output):
```bash
$ bash run_keras.sh -no-debug 
```

### For optimized Program,

To show the info of the python program: 
```bash
$ bash run_optimized.sh -debug 
```
Without info:
```bash
$ bash run_optimized.sh -no-debug 
```
