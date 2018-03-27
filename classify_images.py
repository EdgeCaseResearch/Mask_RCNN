#!/usr/bin/python3
# coding: utf-8

# Mask R-CNN Demo
#
# A quick script to test image classification using Mas

import os
import numpy as np
import skimage.io

import warnings
from tqdm import tqdm

import coco
import utils
import model as modellib
import visualize
from cityscapes_dataset import CityscapesConfig, CityscapesDataset
from coco_dataset import CocoDataset

# Root directory of the project
# ROOT_DIR = os.getcwd()
ROOT_DIR = "/home/gfoil/data/"

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "tensorflow_logs")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "pittsburgh/strip_district/greyscale")

# Output directory
OUTPUT_DIR = os.path.join(ROOT_DIR, "pittsburgh/strip_district/output")

# ------------ COCO -------------
# Test with Coco weights:
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()

MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")
dataset = CocoDataset()

# Download COCO trained weights from Releases if needed
if not os.path.exists(MODEL_PATH):
    utils.download_trained_weights(MODEL_PATH)


# ------------ CITYSCAPES -------------
# Overwrite and use Cityscapes weights instead if desired
if 1:
    MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_cityscapes.2.h5")
    config = CityscapesConfig()
    config.display()
    dataset = CityscapesDataset()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights
print("loading weights: ", MODEL_PATH)
model.load_weights(MODEL_PATH, by_name=True)

class_names = dataset.class_names
print("Classes: {}".format(class_names))

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]

for f in tqdm(file_names):
    if f.endswith(".jpg") or f.endswith(".png"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skimage.io.imsave

        image = skimage.io.imread(os.path.join(IMAGE_DIR, f))
        # image = cv2.resize(im, (0,0), fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)

        # Run detection
        # Returns a list of dicts, one dict per image. The dict contains:
        # rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        # class_ids: [N] int class IDs
        # scores: [N] float probability scores for the class IDs
        # masks: [H, W, N] instance binary masks
        results = model.detect([image], verbose=0)

        # Visualize results
        r = results[0]

        base_name = os.path.join(OUTPUT_DIR, os.path.splitext(f)[0])
        output_name = base_name + "_roi.png"

        colored_im = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                                 class_names, r['scores'], save_name=output_name)

        skimage.io.imsave(base_name + "_viz.jpg", colored_im)

        colored_label_im = np.zeros((image.shape[0], image.shape[1], 3), dtype='uint8')
        label_im = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
        instance_im = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')

        for i, class_id, roi, score in zip(np.arange(len(r['class_ids'])), r['class_ids'], r['rois'], r['scores']):
            mask = r['masks'][:, :, i]
            inds = (mask > 0)

            key = None
            if class_names[class_id] in dataset.labels.values():
                key = list(dataset.labels.keys())[list(dataset.labels.values()).index(class_names[class_id])]

            if key is not None:
                label_im[inds] = key
                colored_label_im[inds] = dataset.cmap[key]

            if class_names[class_id] in dataset.instanceids.keys():
                instance_im[inds] = dataset.instanceids[class_names[class_id]]

            skimage.io.imsave(base_name + "_color.png", colored_label_im)
            skimage.io.imsave(base_name + "_labelIds.png", label_im)
            skimage.io.imsave(base_name + "_instanceIds.png", instance_im)

            #     print(class_names[class_id], "has an instance id")
            # # print(dataset.labels[class_id])
            # print(dataset.cmap[class_id])
            # print(mask.shape,"\n")

