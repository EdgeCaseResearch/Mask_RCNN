#!/usr/bin/python3
# coding: utf-8

# Mask R-CNN Demo
#
# A quick script to test image classification using Mas

import os
import numpy as np
import skimage.io
import argparse

import warnings
from tqdm import tqdm

import coco
import utils
import model as modellib
import visualize
from classifier import classifier
from cityscapes_dataset import CityscapesConfig, CityscapesDataset
from coco_dataset import CocoDataset, CocoConfig

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "tensorflow_logs")

# Output directory
OUTPUT_DIR = os.path.join(ROOT_DIR, "pittsburgh/strip_district/output")

# Handle args
parser = argparse.ArgumentParser(description='Classify all images in a given directory')
parser.add_argument("input_dir",
                    metavar="<input_dir>",
                    type=str,
                    help="input directory")
# parser.add_argument("output_dir",
#                     metavar="<output_directory>",
#                     type=str,
#                     # default=OUTPUT_DIR,
#                     help="output directory")
parser.add_argument('classifier',
                    metavar="<classifier [city|coco]>",
                    choices=["city", "coco"],
                    help='Classifier to use')

args = parser.parse_args()

classify_dataset = None

if args.classifier == 'city':
    classify_dataset = classifier()
    classify_dataset.setInferenceConfig(CityscapesConfig())
    classify_dataset.setDataset(CityscapesDataset())

elif args.classifier == 'coco':
    classify_dataset = classifier(modelname="coco")
    classify_dataset.setInferenceConfig(CocoConfig())
    classify_dataset.setDataset(CocoDataset())

classify_dataset.setDirectories(MODEL_DIR)
classify_dataset.createModel(mode="inference")

file_names = next(os.walk(args.input_dir))[2]

for f in tqdm(file_names):
    if f.endswith(".jpg") or f.endswith(".png"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skimage.io.imsave

        image = skimage.io.imread(os.path.join(args.input_dir, f))

        base_name = os.path.join(OUTPUT_DIR, os.path.splitext(f)[0] + ("_{}".format(args.classifier)))
        output_name = base_name + "_roi.png"

        [colored_im, colored_label_im, label_im, instance_im, rois] = classify_dataset.classifyImage(image, output_name)

        skimage.io.imsave(base_name + "_viz.jpg", colored_im)
        skimage.io.imsave(base_name + "_color.png", colored_label_im)
        skimage.io.imsave(base_name + "_labelIds.png", label_im)
        skimage.io.imsave(base_name + "_instanceIds.png", instance_im)
