#!/usr/bin/python
import model_inference as mi
import engine_ops as eop
from  preprosses import *
from os.path import isfile, join
import os
import time
import logging
import numpy as np
from os import listdir
import sys

 
HEIGHT = 512
WIDTH = 1024

if __name__ == "__main__":
    #Loading the logger
    logging.basicConfig(filename="optimized/logs_trt.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
    
    logger=logging.getLogger()
  
    logger.setLevel(logging.DEBUG)
    
    try:
        #Setting the engine path based on wehter the engine file is present or not
        engine_path = join(os.getcwd(),"optimized/models/plan")
        if(len(os.listdir(engine_path)) == 0 or 'semantic_model.plan' not in os.listdir(engine_path)): #if engine plan file is not present then generate from onnx file
            onnx_path = join(os.getcwd(), 'optimized/models/onnx/semantic_model.onnx')
            engine_path = join(os.getcwd(),"optimized/models/plan/semantic_model.plan")  
            eop.save_engine(engine_path, onnx_path)
        else:
            engine_path = join(os.getcwd(),"optimized/models/plan/semantic_model.plan")
    
        logger.debug("TRT engine loaded")
        logger.debug("Loading the test dataset")
        data_path = "./data"
        
        #reading the batch_size from command line argument
        batch_size = int(sys.argv[1])
        data_dir = './data'
        input_dir = join(data_dir, [f for f in listdir(data_dir) if(f != 'out')][0]) #Setting the input data directory
        f_data_dir = file_names_dir(input_dir)
        
        file_names = file_names(input_dir)
        out_dir = './data/out/' #output directory 
        start = time.time()
        logger.debug("Starting inference")
        for i in range(0, len(f_data_dir), batch_size): 
            if(i+batch_size > len(f_data_dir)): # Setting the batch size according to the location pointer i
                new_batch_size =  len(f_data_dir) - i 
            else:
                new_batch_size = batch_size
            data = return_processed(data_path,i,new_batch_size) # preporcessing the data and loading it
            out = mi.inference(engine_path,  data, new_batch_size) #TRT inference
            out_reshaped = out.reshape([new_batch_size,20,512,1024])
            for j in range(len(out_reshaped)): #processing inputs in each batch
                output = color_map(out_reshaped[j])
                colorImage_k = Image.fromarray(output.astype(np.uint8))
                colorImage_k.save(join(out_dir, file_names[i+j]))
        end = time.time()
        
        logger.debug("Total time taken for inference for " + str(len(f_data_dir)) + "images is " + str(np.round((end-start),2)) + " seconds")
    except BaseException as err:
        logger.error(err)
        raise err
        