# Jetson_test_projects
Medium to high complexity Machine learning projects to compare the inference times between optimized and non optimized nueral networks

Required Libraries:
python == 3.6
tensorrt == 8.0.1.6
pycuda == 2021.1
numpy == 1.19.4
opencv == 4.1.1 (preinstalled with jetpack)
argparse == 1.1
onnx == 1.10.1
tensorflow == 2.5.0

Very Important Note: As both unoptimized and optimized versions are running on GPUs make sure tensorflow is using GPU 
by using the following command in python environment:
$ len(tf.config.list_physical_devices('GPU'))
The output should be >= 1

Instructions:
place the data into data/ directory:
dataset can be found at : https://www.kaggle.com/alessiocorrado99/animals10?select=raw-img
After placing the data : the directory should look like : data/raw-img

Place the plan and onnx file in the respective paths:
onnx : optimized/models/onnx/resnet50.onnx
plan : optimized/models/plan/resnet50.plan
