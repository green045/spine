#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Mask R-CNN
Configurations and data loading code for the synthetic engines dataset.
This is a duplicate of the code in the noteobook train_engines.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
import os
import sys
import cv2
import yaml
import math
import random
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from PIL import Image
from config import Config
import utils
from model import log
import model as modellib
import visualize
import time

# Root directory of the project
ROOT_DIR = os.getcwd()


# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

object_name = ['spine']

# object_name = ['AuxiliaryRadiator', 'Battery', 'BrakeFluid', 'Engine', 'FuseBox', 'OilCap', 'OilLevelRod', 'WiperWater','Point']
#object_name = ['AuxiliaryRadiator', 'Battery', 'BrakeFluid', 'Engine', 'OilCap', 'OilLevelRod', 'WiperWater']
class EngineConfig(Config):
    """Configuration for training on the toy engines dataset.
    Derives from the base Config class and overrides values specific
    to the toy engines dataset.
    """
    # Give the configuration a recognizable name
    NAME = "spine"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1#3  # background + 8 engines

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)#(4, 8, 16, 32, 64)#(8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.9

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3
    
    # Max number of final detections
    DETECTION_MAX_INSTANCES = 18


class EngineDataset(utils.Dataset):
    """Generates the engines synthetic dataset. The dataset consists of simple
    engines (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    #得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n
        
    #解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self,image_id):
        info=self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp=yaml.load(f.read())
            labels=temp['label_names']
            del labels[0]
        return labels   
        
    #重新写draw_mask
    def draw_mask(self,image_id, num_obj, mask, image):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] =1
        return mask
        
    #重新写load_shapes，里面包含自己的自己的类别（我的是box、column、package、fruit四类）
    #并在self.image_info信息中添加了path、mask_path 、yaml_path
    def load_engines(self, count, height, width, img_floder, mask_floder,yaml_floder, imglist,dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        #self.add_class("engines", 0, "background")
        for idx in range(len(object_name)):
            self.add_class("verticles", (idx+1), object_name[idx])
        for i in range(count):
            filestr = imglist[i][:-4]
            mask_path = mask_floder + "/" + filestr + "_mask.png"
            yaml_path = yaml_floder + "/" + filestr + "_info.yaml"
            img = cv2.imread(img_floder + "/" + imglist[i], 0)
            height, width = img.shape[:2]
            self.add_image("verticles", image_id=i, path=img_floder + "/" + imglist[i],
                           width=width, height=height, mask_path=mask_path,yaml_path=yaml_path)
    #重写load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(image_id,num_obj, mask, img)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels=[]
        labels=self.from_yaml_get_class(image_id)
        labels_form=[]
        for i in range(len(labels)):
            for obj in object_name:
                if labels[i].find(obj)!=-1:
                    #print (obj)
                    labels_form.append(obj)
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)    
    

    
    #old
    '''
    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['engines']:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def image_reference(self, image_id):
        """Return the engines data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "engines":
            return info["engines"]
        else:
            super(self.__class__).image_reference(self, image_id)
    

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'square':
            image = cv2.rectangle(image, (x - s, y - s),
                                  (x + s, y + s), color, -1)
        elif shape == "circle":
            image = cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array([[(x, y - s),
                                (x - s / math.sin(math.radians(60)), y + s),
                                (x + s / math.sin(math.radians(60)), y + s),
                                ]], dtype=np.int32)
            image = cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height // 4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple engines.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random engines and record their
        # bounding boxes
        engines = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            engines.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y - s, x - s, y + s, x + s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # engines covering each other
        keep_ixs = utils.non_max_suppression(
            np.array(boxes), np.arange(N), 0.3)
        engines = [s for i, s in enumerate(engines) if i in keep_ixs]
        return bg_color, engines
    '''
    
def evaluate_engine(model, dataset,inference_config, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick images from the dataset
    image_ids = np.random.choice(dataset.image_ids, 100)

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

   
    t_prediction = 0
    t_start = time.time()

    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
        
    print("mAP: ", np.mean(APs))
    print("Total time: ", time.time() - t_start)
    

    
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=True,
                        default="last",metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=100,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
                        
    parser.add_argument('--save', required=False,
                        default="./ans.jpg",
                        metavar="<image save path>",
                        help='Images path to save')
    args = parser.parse_args()
    
    
    dataset_root_path= args.dataset
    img_save_path= args.save
    # Configurations
    if args.command == "train":
        config = EngineConfig()
    else:
        class InferenceConfig(EngineConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()
    
    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
    
    
    model_path=""
    # Select weights file to load 
    # Train a new model starting from pre-trained COCO weights
    if args.model.lower() == "coco":
        model_path = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(model_path, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
        
        
    # Continue training the last model you trained. This will find
    # the last trained weights in the model directory.
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
        model.load_weights(model_path, by_name=True)
    # Train a new model starting from ImageNet weights
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
        model.load_weights(model_path, by_name=True)
    # Continue training a model that you had trained earlier
    else:
        model_path = args.model
        model.load_weights(model_path, by_name=True)

    # Load weights
    print("Loading weights ", model_path)
    
    
    
    
    
    
    #基础设置
    
    img_floder = dataset_root_path+"img"
    mask_floder = dataset_root_path+"mask"
    yaml_floder = dataset_root_path+"info"
    imglist = os.listdir(img_floder)
    count = len(imglist)
    width = 100
    height = 100
    
    
    if args.command == "train":
        #train与val数据集准备
        dataset_train = EngineDataset()
        dataset_train.load_engines(count, height, width, img_floder, mask_floder,yaml_floder, imglist,dataset_root_path)
        dataset_train.prepare()

        dataset_val = EngineDataset()
        dataset_val.load_engines(count, height, width, img_floder, mask_floder,yaml_floder, imglist,dataset_root_path)
        dataset_val.prepare()
        
        # *** This training schedule is an example. Update to your needs ***
        t_start = time.time()
        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=30,
                    layers='heads')
        
        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='4+')
        
        
        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=100,
                    layers='all')
        print("Total time: ", time.time() - t_start)
    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = EngineDataset()
        engines = dataset_val.load_engines(count, height, width, img_floder, mask_floder,yaml_floder, imglist,dataset_root_path)
        dataset_val.prepare()
        print("Running evaluation on {} images.".format(args.limit))
        evaluate_engine(model, dataset_val,config, "bbox", limit=int(args.limit))
    elif args.command == "test":
        # Test on a random image
        dataset_val = EngineDataset()
        engines = dataset_val.load_engines(count, height, width, img_floder, mask_floder,yaml_floder, imglist,dataset_root_path)
        dataset_val.prepare()
        for i in range(20):
            image_id = random.choice(dataset_val.image_ids)
            original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset_val, config, 
                                       image_id, use_mini_mask=False)
            
            log("original_image", original_image)
            log("image_meta", image_meta)
            log("gt_class_id", gt_class_id)
            log("gt_bbox", gt_bbox)
            log("gt_mask", gt_mask)
            
            visualize.save_instances((img_save_path+ str(i) +".jpg"),original_image, gt_bbox, gt_mask, gt_class_id, 
                                        dataset_val.class_names, figsize=(8, 8))
        