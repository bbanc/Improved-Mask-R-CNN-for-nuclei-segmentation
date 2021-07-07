import os
import sys
import time
import numpy as np
import albumentations as A
import skimage.draw
import skimage.color
import scipy.io
import tensorflow as tf


import zipfile
import urllib.request
import shutil

ROOT_DIR = os.path.abspath("..\\..\\")
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib, utils


class MonusegDISTConfig(Config):
    NAME = "Monuseg_DIST_"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    BACKBONE = "resnet50"

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    RPN_NMS_THRESHOLD = 0.8
    DETECTION_MAX_INSTANCES = 400
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # BG + 4 nuclei classes

    # Use small images for faster training. Set the limits of the smal
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    IMAGE_CHANNEL_COUNT = 3# 4

    MEAN_PIXEL = np.array([176.31886506, 115.74123431, 155.43802069])#,  15.18166894])

    # Use smaller anchors because our image and objects are small
    #RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64,128) 
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    #BACKBONE_STRIDES = [2, 4, 8, 16, 32]
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    MASK_SHAPE = [28, 28]

    ROI_POSITIVE_RATIO = 0.4
    DETECTION_NMS_THRESHOLD = 0.3

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 600

    USE_MINI_MASK = True

    IMAGE_RESIZE_MODE = "crop"

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 400


    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 60

    POST_NMS_ROIS_INFERENCE = 2000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 0

    LEARNING_RATE = 0.0001
    TRAIN_BN = False

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

class MonusegDIST_HConfig(Config):
    NAME = "Monuseg_DIST_"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    BACKBONE = "resnet50"

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    RPN_NMS_THRESHOLD = 0.8
    DETECTION_MAX_INSTANCES = 400
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # BG + 4 nuclei classes

    # Use small images for faster training. Set the limits of the smal
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    IMAGE_CHANNEL_COUNT = 5

    MEAN_PIXEL = np.array([176.31886506, 115.74123431, 155.43802069,  15.18166894, 123.75910])

    # Use smaller anchors because our image and objects are small
    #RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64,128) 
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    #BACKBONE_STRIDES = [2, 4, 8, 16, 32]
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    MASK_SHAPE = [28, 28]

    ROI_POSITIVE_RATIO = 0.4
    DETECTION_NMS_THRESHOLD = 0.3

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 600

    USE_MINI_MASK = True

    IMAGE_RESIZE_MODE = "crop"

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 400


    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 60

    POST_NMS_ROIS_INFERENCE = 2000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 0

    LEARNING_RATE = 0.0001
    TRAIN_BN = False

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
    
class MonusegDISTInferenceConfig(MonusegDISTConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "pad64"
    RPN_NMS_THRESHOLD = 0.7
    DETECTION_MAX_INSTANCES = 2000

class MonusegDIST_H_InferenceConfig(MonusegDIST_HConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "pad64"
    RPN_NMS_THRESHOLD = 0.7
    DETECTION_MAX_INSTANCES = 2000
    
    
class MonusegDISTDataset(utils.Dataset): 
    
    def __init__(self, source = "GT", *args, **kwargs):
        self.source = source
        super(MonusegDISTDataset, self).__init__(*args, **kwargs)
        
    def load_mask(self, image_id):
        # The field set in add_image (Contains: Source Name, ID, Path)
        info = self.image_info[image_id]
        
        path = info["path"]
        p = os.path.join(os.path.dirname(os.path.dirname(path)), "mask_binary")
        name = info["id"]
        npz_path = os.path.join(p, name + ".npz")
        if not os.path.exists(npz_path):
            raise Exception("Did not find .npz file in " + npz_path)
         
        mask = np.load(npz_path)["masks"].astype(np.bool)
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)
        
    def load_image(self, image_id):
        path = self.image_info[image_id]['path']
        name = self.image_info[image_id]["id"]
        # RGB
        image = skimage.io.imread(path)
        if self.source == "GT":
            # Dist: Go up one from RGB image path, and into distance_maps         
            dist_path = os.path.join(os.path.dirname(os.path.dirname(path)), "distance_maps", name + "_gt.png")
            dist = skimage.io.imread(dist_path)  
            dist = dist / np.max(dist) * 255                    
        elif self.source == "UNET":
            dist_path = os.path.join(os.path.dirname(os.path.dirname(path)), "distance_maps", name + "_dis.npy")        
            d = np.load(dist_path)
            dist = np.zeros(d.shape, dtype = np.uint8)
            dist = d / np.max(d) * 255          
        elif self.source == "ZEROS":
            dist = np.zeros((image.shape[0], image.shape[1]))
        
        else:
            raise Exception("Unsupported Source argument in Kumar_dist.load_image")
       
        shape = (image.shape[0], image.shape[1], 5)
        
        
        h_stain_path = os.path.join(os.path.dirname(os.path.dirname(path)), "H_stain", name + ".npy")
        img_h = np.load(h_stain_path)

        
        # Compose return
        ret = np.zeros(shape, dtype = np.uint8)
        ret[:,:,0:3] = image
        ret[:,:,3] = dist
        ret[:,:,4] = img_h
        return ret

        
    def load_labelled_mask(self, image_id):
        info = self.image_info[image_id]
        
        path = info["path"]
        p = os.path.join(os.path.dirname(os.path.dirname(path)), "labelled_masks")
        name = info["id"]
        np_path = os.path.join(p, name + ".npy")
        if not os.path.exists(np_path):
            raise Exception("Did not find .npy file in " + np_path)
         
        mask = np.load(np_path)
        return mask
                 
         
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id) 
            
    
def calc_and_save_detect(model, dataset, model_path):
    results_path = os.path.join(model_path, "Detections")
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    for _id in dataset.image_ids:
        image = dataset.load_image(_id)
        info = dataset.image_info[_id]
        n = info["id"]
        print(n)
        # detect works on a list of images, returns a dict with rois, masks, class ids, and scores
        print("Running detection...")
        r = model.detect([image], verbose = 0, binarize = False)
        r = r[0]
        print("Found {} instances".format(r["masks"].shape[-1]))
        print("Saving...")
        score_path = os.path.join(results_path, n + "_scores.npy")
        mask_path = os.path.join(results_path, n + "_masks.npz")
        label_path = os.path.join(results_path, n + "_label.npy")
        masks = r["masks"]
        scores = r["scores"]
        label_mask = utils.flatten_mask(masks > 0.7)
        np.save(score_path, scores)
        np.savez_compressed(mask_path, masks)
        np.save(label_path, label_mask)