# Jetson Test Projects
Medium to high complexity Machine learning projects to compare the inference times between optimized and non optimized nueral networks

# 1) Image Classification using Resnet50 
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

Download the onnx file from : https://drive.google.com/drive/u/2/folders/1yYDD2_teYR9wPLwCjV3OMMtrjWTyOBu0

### Place the onnx file in the respective paths:
onnx : optimized/models/onnx/resnet50.onnx

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

# 2) Image Segmentation
Download the onnx file and the hd5 file from the following link: https://drive.google.com/drive/folders/1NypCjzfGBJEQ6Gkj4k7n4Ipwcl9nix42?usp=sharing

Place the onnx file in ```Image_segmentation_nd/optimized/models/onnx```

Place the hd5 file in ```Image_segmentation_nd/```

Place the any one of the folders (e.g cane) from the data set downloaded for Image Classification Example in the ```Image_segmentation_nd/data/```

### Running test scripts
Tensorflow : ``` bash run_tf.sh -d no -b 1 ``` (set the flag -b to yes to show the entire output)

Tensorrt : ``` bash run_optimized.sh -d no -b 1 ``` (set the flag -b to yes to show the entire output)
