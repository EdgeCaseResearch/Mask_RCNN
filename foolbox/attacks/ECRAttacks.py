from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from collections import Iterable
import logging
import abc

from .base import Attack
from .base import call_decorator
from ..criteria import Misclassification


class ECRTestAttack(Attack):
    """Testing if 'Attack' class can be used for arbitrary code/augmentations"""

    def __init__(self, model=None, criterion=Misclassification()):
        super(ECRTestAttack, self).__init__(model=model, criterion=criterion)

    def __call__(self, input_img):
        print("'Attacking' image now!!!!!11!1!11!")
        return input_img[:,:,::-1]