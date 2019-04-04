#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN - Inspect Deforestation Trained Model

# In[27]:


#ROOT_DIR = os.path.abspath("D:/MRCNN/")

# Import Mask RCNN
#r = sys.path.append(ROOT_DIR) 
#print(r)


# In[28]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv

# Root directory of the project
#ROOT_DIR = os.path.abspath("D:/MRCNN/Mask_RCNN/")

# Import Mask RCNN
ROOT_DIR = os.path.abspath("D:/MRCNN/Mask_RCNN/")
sys.path.append(ROOT_DIR)  # To find local version of the library
from samples.mrcnn import utils
#from mrcnn import visualize
from samples.mrcnn import visualize
from samples.mrcnn.visualize import display_instances
from samples.mrcnn.visualize import save_image
import samples.mrcnn.model as modellib
from samples.mrcnn.model import log

from samples.forest import forest1

#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
BALLON_WEIGHTS_PATH = "D:/MRCNN/Mask_RCNN/samples/forest/mask_rcnn_forest_0004.h5"  # TODO: update this path


# ## Configurations

# In[29]:


config = forest1.CustomConfig()
BALLOON_DIR = "D:/MRCNN/Mask_RCNN/samples/forest/dataset"


# In[30]:


# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Notebook Preferences

# In[31]:


# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


# In[32]:


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# ## Load Validation Dataset

# In[33]:


# Load validation dataset
dataset = forest1.CustomDataset()
dataset.load_custom(BALLOON_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))


# ## Load Model

# In[34]:


# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)


# In[35]:


# Set path to balloon weights file

# Download file from the Releases page and set its path
# https://github.com/matterport/Mask_RCNN/releases
# weights_path = "/path/to/mask_rcnn_balloon.h5"

# Or, load the last model you trained
weights_path = "D:/MRCNN/Mask_RCNN/samples/forest/mask_rcnn_forest_0003.h5" 

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)


# ## Run Detection

# In[36]:


image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))
print(info["id"])
# Run object detection
results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)


# In[37]:


visualize.save_image(image, image_id, r['rois'], r['masks'],
    r['class_ids'],r['scores'],dataset.class_names,
    filter_classs_names=['forest', 'deforest'],scores_thresh=0.9,mode=0)


# In[38]:


visualize.detect(info["id"])


# In[ ]:




