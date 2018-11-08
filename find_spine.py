#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
import random
import math
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import skimage.io
import train_spine
import utils
import model as modellib
import visualize




if __name__ == '__main__':
       
    
   
    # data_path = args.dataset
    data_path = './bin_img'  

    # img_save_dir = args.save
    output_path = './result_spine'

    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    
    
    
    # Configurations
    
    class InferenceConfig(train_spine.EngineConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display()
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    
    
    model_path = './model/mask_rcnn_spine.h5'
    print(model_path)
    # Load weights trained on MS-COCO
    model.load_weights(model_path, by_name=True)
    
    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['background', 'spine']
    
    


    
    total_dir = os.listdir(data_path)

    for each_dir in total_dir:
        print(each_dir)
        each_dir_path = data_path+'/'+ each_dir
        save_dir_path =output_path+'/'+ each_dir
        if not os.path.isdir(save_dir_path):
            os.mkdir(save_dir_path)


        total_file = os.listdir(each_dir_path)
        total_file = filter(lambda x: x.endswith('png'), total_file) #只抓png檔

        for file_name in total_file:
            image = skimage.io.imread(os.path.join(each_dir_path, file_name))
            #image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
            # Run detection
            results = model.detect([image], verbose=1)

            # Visualize results
            r = results[0]
            img_save_path = os.path.join(save_dir_path, 'res_'+file_name[4:])
            visualize.save_instances(img_save_path, image, r['rois'], r['masks'], r['class_ids'], 
                                        class_names)

    
