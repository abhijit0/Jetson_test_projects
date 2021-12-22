import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import onnxruntime
from preprosses import return_dataset
import time
import logging
tf.debugging.set_log_device_placement(True)

if __name__=="__main__":
    logging.basicConfig(filename="logs_keras.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
    logger=logging.getLogger()
  
    #Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)
    try:
        model = ResNet50(weights='imagenet')
        logger.debug("Loading Dataset")
        data_set = return_dataset()
        data_set = data_set.reshape(data_set.shape[1], data_set.shape[2], data_set.shape[3], data_set.shape[4])
        logger.debug("Starting inference")
        start = time.time()
        preds = model.predict(data_set)
        end = time.time()
        logger.debug("Total time for inference for "+str(data_set.shape[0])+" images "+str(np.round((end-start), 2))+" seconds")
        logger.debug("Keras Predicted: " +str(decode_predictions(preds, top=10)[0]))
    except BaseException as err:
        logger.error(err)
        raise err