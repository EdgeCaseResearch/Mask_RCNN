import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def InvertColorsAugmentation(image):
    """
    @brief    Augments the image by inverting its colors
    
    @param    image a 3-dimensional numpy array with channels-last ordering
    
    @return   The augmented image
    """
    return image[:,:,::-1]