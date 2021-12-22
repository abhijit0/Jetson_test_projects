import model_inference as mi
import engine_ops as eop
import preprosses as pre
from os.path import isfile, join
import os
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import time
import logging
import numpy as np
 


if __name__ == "__main__":
    logging.basicConfig(filename="optimized/logs_trt.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
    
    #Creating an object
    logger=logging.getLogger()
  
    #Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)
    
    #Test messages
    ##save engine
    try:
        engine_path = join(os.getcwd(),"optimized/models/plan")
        if(len(os.listdir(engine_path)) == 0 or os.listdir(engine_path) != 'resnet50.plan'):
            onnx_path = join(os.getcwd(), 'optimized/models/onnx/resnet50.onnx')
            engine_path = join(os.getcwd(),"optimized/models/plan/resnet50.plan")  
            eop.save_engine(engine_path, onnx_path)
        else:
            engine_path = join(os.getcwd(),"optimized/models/plan/resnet50.plan")
    
        logger.debug("TRT engine loaded")
        logger.debug("Loading the test dataset")
        #os.path.dirname(os.getcwd()) to go one level up
        data_path = join(os.getcwd(),"data/raw-img")
        data_set = pre.return_dataset(data_path)
        data_set = data_set.reshape(data_set.shape[1],data_set.shape[2], data_set.shape[3], data_set.shape[4])
        logger.debug("Starting inference")
        start = time.time()
        out = mi.inference(engine_path, data_set, data_set.shape[0])
        end = time.time()
        logger.debug("Total time taken for inference for " + str(data_set.shape[0]) + "images is " + str(np.round((end-start),2)) + " seconds")
        print('Keras Predicted:', decode_predictions(out, top=15)[0])
        logger.info("Keras Predicted: " + str(decode_predictions(out, top=15)[0]))
    except BaseException as err:
        logger.error(err)
        raise err
        