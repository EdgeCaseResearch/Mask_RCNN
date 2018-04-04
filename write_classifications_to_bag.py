import numpy as np
import sys
import os
from tqdm import tqdm

import argparse

import roslib
import rosbag
import rospy
from sensor_msgs.msg import Image, RegionOfInterest
from custom_messages.msg import roi_list

from PIL import Image as pil

from classifier import classifier
from cityscapes_dataset import CityscapesDataset, CityscapesConfig
from coco_dataset import CocoDataset, CocoConfig
from ros_utils import *


# rois: [N, (y1, x1, y2, x2)] detection bounding boxes
def createRois(results, timestamp):
    roi_msg = roi_list()

    for r in results['rois']:
        roi = RegionOfInterest()
        roi.y_offset = r[0]
        roi.x_offset = r[1]
        roi.height = r[2] - r[0]
        roi.width = r[3] - r[1]
        roi_msg.rois.append(roi)
    roi_msg.header.stamp = timestamp

    return roi_msg


parser = argparse.ArgumentParser(
    description='Classify all instances of an image topic and add classifications to the given bag')
parser.add_argument("bagfile", 
                    metavar="<bagfile>",
                    type=str,
                    help="input bagfile")
parser.add_argument('classifier',
                    metavar="<classifier [city|coco]>",
                    choices=["city", "coco"],
                    help='Classifier to use')
parser.add_argument('--outbag',
                    type=str,
                    metavar="<outbag>",
                    help='Optionally save to a new bag')
parser.add_argument('--image_topic',
                    type=str,
                    metavar="<image_topic>",
                    help='Optional custom image topic [default: /monocular_camera/image]')
parser.add_argument('--skip',
                    type=int,
                    default=1,
                    metavar="<skip to every nth image",
                    help="Skip to every nth image")
# parser.add_argument('--outdir',
#                     type=str,
#                     metavar="<output_dir>",
#                     help="Save images to output directory")

args = parser.parse_args()

if args.outbag:
    print("Loading", args.bagfile, "and saving to", args.outbag)
    bag = rosbag.Bag(args.bagfile, 'r')
    outbag = rosbag.Bag(args.outbag, 'w')
else:
    print("Loading", args.bagfile, "and saving to", args.bagfile)
    bag = rosbag.Bag(args.bagfile, 'a')
    outbag = bag

if args.image_topic:
    image_topic = args.image_topic
else:
    image_topic = "/monocular_camera/image"

# Count the number of images in this bag. Takes a few seconds but it's worth it
im_count = 0
for topic, msg, t in bag.read_messages(topics=[image_topic]):
    im_count += 1

im_count = int(im_count/args.skip)
print(im_count, "total images to classify")

classify_dataset = None

if args.classifier == 'city':
    classify_dataset = classifier()
    classify_dataset.setInferenceConfig(CityscapesConfig())
    classify_dataset.setDataset(CityscapesDataset())

elif args.classifier == 'coco':
    classify_dataset = classifier(modelname="coco")
    classify_dataset.setInferenceConfig(CocoConfig())
    classify_dataset.setDataset(CocoDataset())

classify_dataset.setDirectories("/home/gfoil/data/tensorflow_logs/")
classify_dataset.createModel(mode="inference")

topic_num = 0

pbar = tqdm(total=int(im_count))

for topic, msg, t in bag.read_messages(topics=[image_topic]):
    topic_num += 1

    if topic_num % args.skip == 0:
        pbar.update(1)

        # If we're creating a new bag, also store the original image
        if args.outbag:
            outbag.write(image_topic, msg, t)

        np_arr = np.fromstring(msg.data, np.uint8)

        [image, timestamp] = convertImgMsgToNumpy(msg)

        # `killall display` to close all pil windows:
        # im = pil.fromarray(image)
        # im.show()

        [colored_im, colored_label_im, label_im, instance_im, results] = classify_dataset.classifyImage(image, verbose=0)
        # im = pil.fromarray(colored_im)
        # im.show()

        roi_msg = createRois(results, msg.header.stamp)

        colored_img = convertNumpyToImgMsg(colored_im, msg.header.stamp)
        label_img = convertNumpyToImgMsg(label_im, msg.header.stamp)
        instance_img = convertNumpyToImgMsg(instance_im, msg.header.stamp)

        # print(msg.header.stamp, instance_img.header.stamp)

        if args.classifier == 'city':
            outbag.write("/cityscapes_debug", colored_img, t)
            outbag.write("/cityscapes_label", label_img, t)
            outbag.write("/cityscapes_instance", instance_img, t)
            outbag.write("/cityscapes_rois", roi_msg, t)
        else:
            outbag.write("/coco_debug", colored_img, t)
            outbag.write("/coco_label", label_img, t)
            outbag.write("/coco_instance", instance_img, t)
            outbag.write("/coco_rois", roi_msg, t)


pbar.close()
bag.close()
if args.outbag:
    outbag.close()
