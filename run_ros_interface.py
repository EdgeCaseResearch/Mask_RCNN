# ROS actionlib interface for classifying images using MaskRCNN
# Copyright 2018 Edge Case Research, LLC

import os

from PIL import Image as pil

import rospy
import actionlib
import tut_common_msgs.msg
from sensor_msgs.msg import Image

from classifier import classifier
from coco_dataset import CocoDataset, CocoConfig
from ros_utils import *


class MaskRCNNAction(object):
    # _feedback = tut_common_msgs.msg.CheckForObjectsFeedback()
    _result = tut_common_msgs.msg.CheckForObjectsResult()

    _classifier = []

    def __init__(self, name):
        self._action_name = name

        self.init_models()

        self._as = actionlib.SimpleActionServer(self._action_name, tut_common_msgs.msg.CheckForObjectsAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

        self._image_pub = rospy.Publisher('/sut/detection_image', Image, queue_size=1)

    def init_models(self):
        # Root directory of the project
        ROOT_DIR = os.getcwd()

        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "tensorflow_logs")

        self._classifier = classifier(modelname="coco")
        self._classifier.setInferenceConfig(CocoConfig())
        self._classifier.setDataset(CocoDataset())
        self._classifier.setDirectories(MODEL_DIR)
        self._classifier.createModel(mode="inference")

        rospy.loginfo('%s: CNN initialized' % self._action_name)

    def createBBox(self, results, timestamp):
        bbox_msg = tut_common_msgs.msg.BoundingBoxes()
        bbox_msg.header.stamp = timestamp

        for r, classname, prob in zip(results['rois'], results['classnames'], results['scores']):
            bbox = tut_common_msgs.msg.BoundingBox()
            bbox.Class = classname
            bbox.probability = prob
            bbox.ymin = r[0]
            bbox.ymax = r[2]

            bbox.xmin = r[1]
            bbox.xmax = r[3]

            bbox_msg.boundingBoxes.append(bbox)

        return bbox_msg

    def classify_image(self, img_msg):
        [im, timestamp] = convertImgMsgToNumpy(img_msg)

        # print(im.shape)
        # pim = pil.fromarray(im)
        # pim.show()

        [colored_im, results] = self._classifier.classifySimple(im, output_name='tmp.jpg', verbose=0, suppress_display=True)
        # im = pil.fromarray(colored_im)
        # im.show()

        bbox_msg = self.createBBox(results, timestamp)

        colored_img = convertNumpyToImgMsg(colored_im, timestamp)
        # label_img = convertNumpyToImgMsg(label_im, timestamp)
        # instance_img = convertNumpyToImgMsg(instance_im, timestamp)

        return [colored_img, bbox_msg]

    def execute_cb(self, goal):
        # helper variables
        success = True

        rospy.loginfo("{}: Received image number {}. Processing...".format(self._action_name, goal.id))

        # check that preempt has not been requested by the client. Probably not needed for this module
        if self._as.is_preempt_requested():
            rospy.loginfo('%s: Preempted' % self._action_name)
            self._as.set_preempted()
            success = False
        else:
            [result_img_msg, bbox_msg] = self.classify_image(goal.image)

        if success:
            self._result.image = result_img_msg
            self._result.boundingBoxes = bbox_msg
            self._result.id = goal.id

            rospy.loginfo('%s: Succeeded' % self._action_name)
            self._as.set_succeeded(self._result)

            self._image_pub.publish(result_img_msg)


if __name__ == '__main__':
    rospy.init_node('sut_actionlib_server')
    server = MaskRCNNAction('/sut/check_for_objects')
    rospy.spin()
