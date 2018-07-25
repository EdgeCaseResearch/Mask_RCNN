from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from collections import Iterable
import logging
import abc
import inspect

from .base import Attack
from .base import call_decorator
from ..criteria import Misclassification


class ECRAugmentAttack(Attack):
    """Testing if 'Attack' class can be used for arbitrary code/augmentations"""

    def __init__(self, model=None, criterion=Misclassification()):
        super(ECRAugmentAttack, self).__init__(model=model, criterion=criterion)
        self.model = model

    def __call__(self, img, augmentation=None, conversion_fn=None, prediction_fn=None):
        """
        @brief Run the image through a chosen augmentation and return the results
        
        @param img           The image to be augmented and run by the neural network.
                             Must be a 3-dimensional channels-last ordered numpy array
        @param augmentation  Function that takes in a numpy array and returns a numpy array
                             Can be used for arbitrary augmentations to the input image. 
                             If unspecified, no augmentation will be performed.
        @param conversion_fn Function that takes in a numpy array and outputs an input to 
                             the neural network this Attack object was initialized with. 
                             If unspecified, the neural network will just be given the image 
                             numpy array as input
        @param prediction_fn Function that takes in the input to the neural network and returns 
                             the output. Used to specify how to run the neural network if additional 
                             processing steps are required. If unspecified, a model-specific implementation
                             of predict_on_inputs will be called.
                             
        @return              The output of the neural network or (in the case of prediction_fn being specified)
                             the output of prediction_fn.
        """
        
        augmented_img = img if augmentation is None else augmentation(img)
        
        model_input = augmented_img if conversion_fn is None else conversion_fn(augmented_img)
        
        assert prediction_fn is not None or hasattr(self.model, 'predict_on_inputs') or inspect.ismethod(getattr(self.model, 'predict_on_inputs')), "A prediction_fn must be specified!"
        
        if prediction_fn is not None:
            predictions = prediction_fn(model_input)
        else:
            predictions = self.model.predict_on_inputs(model_input)
#         print("Predictions = {}".format(predictions))
        return predictions

