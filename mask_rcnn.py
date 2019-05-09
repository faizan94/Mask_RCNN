import os
import sys
import random
import math
import numpy as np
import cv2
import time

# Import Mask RCNN
#sys.path.append(ROOT_DIR)  # To find local version of the library
ROOT_DIR = "./"

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


COCO_MODEL_PATH = os.path.join(ROOT_DIR, "brain/mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    NAME = "coco"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80

config = InferenceConfig()
config.display()

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class MaskRcnn:
    modelmskrcnn = None

    @staticmethod
    def filter_class_ids(r):

        rr = dict()
        rr['class_ids'] = np.empty((0,),np.int32)
        rr['scores'] = np.empty((0,))
        rr['rois'] = np.empty((0,r['rois'].shape[1]))
        rr['masks'] = np.empty((r['masks'].shape[0], r['masks'].shape[1], 0))

        for idx, id in enumerate(r['class_ids']):
            if class_names[id] in ['person', 'car']:
                rr['class_ids'] = np.hstack((rr['class_ids'], id))
                rr['scores'] = np.hstack((rr['scores'],r['scores'][idx]))
                rr['rois'] = np.vstack((rr['rois'], r['rois'][idx, :]))
                rr['masks'] = np.append(rr['masks'], np.expand_dims(r['masks'][:, :, idx],axis=-1),axis=-1)
        return rr

    @staticmethod
    def get_colored_frame(batch_frames):
        if MaskRcnn.modelmskrcnn is None:
            # Create model object in inference mode.
            MaskRcnn.modelmskrcnn = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
            # Load weights trained on MS-COCO
            MaskRcnn.modelmskrcnn.load_weights(COCO_MODEL_PATH, by_name=True)

        results = MaskRcnn.modelmskrcnn.detect(batch_frames, verbose=0)
        results = [MaskRcnn.filter_class_ids(res) for res in results]

        res_frames = []
        for res, frame in zip(results, batch_frames):
            res_frame = visualize.display_instances(frame, res['rois'], res['masks'], res['class_ids'],
                        class_names, res['scores'])
            res_frames.append(res_frame)
        return res_frames

