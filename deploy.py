import base64
import numpy as np
import io
from PIL import Image
import keras
import mrcnn.model as modellib
from keras import backend as K
from flask import request
from flask import jsonify
from flask import Flask
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
from forest import forest
from mrcnn import visualize
'''
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from samples.forest import forest1
'''

app = Flask(__name__)

def get_model():
	global model
	global graph
	graph = tf.get_default_graph()
	global config
	global dataset
	config = forest.CustomConfig()
	class InferenceConfig(config.__class__):
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1
	DEVICE = "/cpu:0"
	TEST_MODE = "inference"
	config = InferenceConfig()
	config.display()
	print("Config loaded")
	with tf.device(DEVICE):
		model = modellib.MaskRCNN(mode="inference", model_dir="D:/Techathon/flask", config=config)
	print('Model loaded')
	print("Loading weights ")
	model.load_weights("mask_rcnn_forest_0001.h5", by_name=True)
	print("Weights loaded")

def Result(image):
	# Load validation dataset
	dataset = forest.CustomDataset()
	dataset.load_custom("D:/Techathon/flask/forest/dataset", "val")

# Must call before using the dataset
	dataset.prepare()

	print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
	image_id = random.choice(dataset.image_ids)
	image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    	modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
	#info = dataset.image_info[image_id]
	#print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
    #                                  dataset.image_reference(image_id)))
	print([image],"IMAGEEEEEE")
# Run object detection
	with graph.as_default():
		results = model.detect([image], verbose=1)
		ax = get_ax(1)
		r = results[0]
		#print(r[0])
		f = list(results[0].values())
		print(f)
		print("saving image")
		visualize.save_image(image, image_id, r['rois'], r['masks'],
    r['class_ids'],r['scores'],dataset.class_names,
    filter_classs_names=['forest', 'deforest'],scores_thresh=0.9,mode=0)
		print("doneS")
		return f 


# Display results

	

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


print('Loading model')
get_model()




@app.route('/predict',methods=['POST'])
def predict():
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))
	r = Result(image)
	print(r[1][0],"output")
	classid = str(r[0][0])
	print(classid)
	scores = str(r[1][0])
	print(scores)
	response = {
		'classid' : classid,
		'scores' : scores
	}
	return jsonify(response)

if __name__ == "__main__":
    app.run(debug=False)
