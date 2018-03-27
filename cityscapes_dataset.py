#!/usr/bin/python3

import numpy as np
from config import Config


class CityscapesDataset:
    # cityscape colormap
    cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0),
                     (111, 74,  0), ( 81,  0, 81), (128, 64,128), (244, 35,232), (250,170,160),
                     (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153), (180,165,180),
                     (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30),
                     (220,220,  0), (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60),
                     (255,  0,  0), (  0,  0,142), (  0,  0, 70), (  0, 60,100), (  0,  0, 90),
                     (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                    dtype=np.uint8)

    # key=list(c.keys())[list(c.values()).index(a)]] ---- to get key from value
    labels = {
        0: "filler",
        1: "ego vehicle",
        2: "rectification border",
        3: "out of roi",
        4: "static",
        5: "dynamic",
        6: "ground",
        7: "road",
        8: "sidewalk",
        9: "parking",
        10: "rail track",
        11: "building",
        12: "wall",
        13: "fence",
        14: "guard rail",
        15: "bridge",
        16: "tunnel",
        17: "pole",
        18: "polegroup",
        19: "traffic light",
        20: "traffic sign",
        21: "vegetation",
        22: "terrain",
        23: "sky",
        24: "person",
        25: "rider",
        26: "car",
        27: "truck",
        28: "bus",
        29: "caravan",
        30: "trailer",
        31: "train",
        32: "motorcycle",
        33: "bicycle",
        -1: "license plate",
    }

    instanceids = {
        "person": 93,
        "rider": 97,
        "car": 101,
        "truck": 105,
        "bus": 109,
        "caravan": 113,
        "trailer": 117,
        "train": 121,
        "motorcycle": 125,
        "bicycle": 128,
    }

    class_names = list(labels.values())

    # Temporary for testing with the coco dataset
    # print("Using the temporary coco names!!")
    # self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    #                     'bus', 'train', 'truck', 'boat', 'traffic light',
    #                     'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    #                     'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    #                     'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    #                     'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    #                     'kite', 'baseball bat', 'baseball glove', 'skateboard',
    #                     'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    #                     'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    #                     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    #                     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    #                     'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    #                     'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    #                     'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    #                     'teddy bear', 'hair drier', 'toothbrush']


class CityscapesConfig(Config):
    """Configuration for training on the Cityscapes dataset.
    Derives from the base Config class and overrides values specific
    to the Cityscapes shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cityscapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 35  # background + 34 labels

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor size in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 2000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 15
