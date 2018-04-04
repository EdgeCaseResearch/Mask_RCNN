import argparse

import rospy
import rosbag
import actionlib

from PIL import Image as pil

import sut_actionlib_msgs.msg
from ros_utils import *

parser = argparse.ArgumentParser(
    description='Read from a bag and receive classifications over an actionlib test interface')
parser.add_argument("bagfile",
                    metavar="<bagfile>",
                    type=str,
                    help="input bagfile")
parser.add_argument('--image_topic',
                    type=str,
                    metavar="<image_topic>",
                    help='Optional custom image topic [default: /monocular_camera/image]')

args = parser.parse_args()

bag = rosbag.Bag(args.bagfile, 'r')

if args.image_topic:
    image_topic = args.image_topic
else:
    image_topic = "/monocular_camera/image"


rospy.init_node("actionlib_client_test")

client = actionlib.SimpleActionClient('ros_mask_rcnn_interface', sut_actionlib_msgs.msg.CheckForObjectsAction)
client.wait_for_server()

message_num = 0

for topic, msg, t in bag.read_messages(topics=[image_topic]):
    # [colored_im, _] = convertImgMsgToNumpy(msg)
    # im = pil.fromarray(colored_im)
    # im.show()

    print("Sending message {}".format(message_num))

    goal = sut_actionlib_msgs.msg.CheckForObjectsGoal()
    goal.id = message_num
    goal.image = msg

    client.send_goal(goal)

    client.wait_for_result()

    r = client.get_result()

    print(r.boundingBoxes)
    # print(r.image.header)
    # [colored_im, _] = convertImgMsgToNumpy(r.image)
    # im = pil.fromarray(colored_im)
    # im.show()
    # exit()

    message_num += 1
