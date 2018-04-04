import numpy as np
from PIL import Image as pil
from sensor_msgs.msg import Image


# Parts taken from https://github.com/ros-perception/vision_opencv/blob/kinetic/cv_bridge/python/cv_bridge/core.py
# Kinetic's cv_bridge doesn't support python3, so we have to break it out manually
def convertNumpyToImgMsg(img, timestamp):
    img_msg = Image()
    img_msg.height = int(img.shape[0])
    img_msg.width = int(img.shape[1])

    if len(img.shape) < 3:
        cv_type = "mono8"
    else:
        cv_type = "rgb8"
    img_msg.encoding = cv_type
    # print(img.dtype.byteorder)
    if img.dtype.byteorder == '>':
        img_msg.is_bigendian = True

    img_msg.data = img.tostring()
    # print(len(img_msg.data), img_msg.height, img.shape, int(len(img_msg.data) / img_msg.height))
    img_msg.step = int(len(img_msg.data) / img_msg.height)

    img_msg.header.stamp = timestamp
    return img_msg


def convertImgMsgToNumpy(msg):
    image = []
    np_arr = np.fromstring(msg.data, np.uint8)

    if msg.encoding == 'rgb8' or msg.encoding == 'bgr8':
        image = np.resize(np_arr, (msg.height, msg.width, 3))
    else:
        image = np.resize(np_arr, (msg.height, msg.width))
        image = np.stack((image,)*3, axis=-1)

    return [image, msg.header.stamp]
