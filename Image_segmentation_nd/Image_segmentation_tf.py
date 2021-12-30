import tensorflow.keras as keras
import numpy as np
from optimized.preprosses import *
import time 
import sys
import logging

HEIGHT = 512
WIDTH = 1024


if __name__ == "__main__":
    logging.basicConfig(filename="logs_tf.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
    
    
    logger=logging.getLogger()
  
    logger.setLevel(logging.DEBUG)
    try:
        data_path = "./data"
        semantic_model = keras.models.load_model('semantic_segmentation.hdf5')
        batch_size = int(sys.argv[1])
        
        data_dir = './data'
        input_dir = join(data_dir, [f for f in listdir(data_dir) if(f != 'out')][0])
        f_data_dir = file_names_dir(input_dir)
        file_names = file_names(input_dir)
        out_dir = './data/out/'
        logger.debug("Starting inference")
        start = time.time()
        for i in range(0, len(f_data_dir), batch_size):
            if(i+batch_size > len(f_data_dir)):
                new_batch_size =  len(f_data_dir) - i 
            else:
                new_batch_size = batch_size
            data = return_processed(data_path,i,new_batch_size)
            output = semantic_model.predict(data.reshape(-1, 3, HEIGHT, WIDTH))
            out = output.reshape([new_batch_size,20,512,1024])
            for j in range(len(out)):
                output = color_map(out[j])
                colorImage_k = Image.fromarray(output.astype(np.uint8))
                colorImage_k.save(join(out_dir, file_names[i+j]))
        end = time.time()
        
        logger.debug("Total time taken for inference for " + str(len(f_data_dir)) + "images is " + str(np.round((end-start),2)) + " seconds")
    except BaseException as err:
        logger.error(err)
        raise err