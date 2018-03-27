#!/usr/bin/python3

import numpy as np
from config import Config
import coco
from visualize import random_colors


class CocoDataset:
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

    cmap = np.zeros((len(class_names), 3))
    colors = random_colors(len(class_names))
    for i in range(len(class_names)):
        cmap[i, :] = colors[i]

    instanceids = {}
    labels = {}
    for i in range(len(class_names)):
        labels[i] = class_names[i]

    # instanceids = {
    #     "person": 93,
    #     "rider": 97,
    #     "car": 101,
    #     "truck": 105,
    #     "bus": 109,
    #     "caravan": 113,
    #     "trailer": 117,
    #     "train": 121,
    #     "motorcycle": 125,
    #     "bicycle": 128,
    # }

class CocoConfig(coco.CocoConfig):
    NAME = "coco"

    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
