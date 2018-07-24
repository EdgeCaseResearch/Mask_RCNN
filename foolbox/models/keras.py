from __future__ import absolute_import
import numpy as np
import logging

from .base import DifferentiableModel


class KerasModel(DifferentiableModel):
    """Creates a :class:`Model` instance from a `Keras` model.

    Parameters
    ----------
    model : `keras.models.Model`
        The `Keras` model that should be attacked.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first subtract the first
        element of preprocessing from the input and then divide the input by
        the second element.
    predicts : str
        Specifies whether the `Keras` model predicts logits or probabilities.
        Logits are preferred, but probabilities are the default.

    """

    def __init__(
            self,
            model,
            bounds,
            channel_axis=3,
            preprocessing=(0, 1),
            predicts='probabilities',
            image_input_idx=None,
            output_idx=None,
            num_classes=None):

        super(KerasModel, self).__init__(bounds=bounds,
                                         channel_axis=channel_axis,
                                         preprocessing=preprocessing)

        from keras import backend as K
        import keras
        from pkg_resources import parse_version

        assert parse_version(keras.__version__) >= parse_version('2.0.7'), 'Keras version needs to be 2.0.7 or newer'  # noqa: E501

        if predicts == 'probs':
            predicts = 'probabilities'
        assert predicts in ['probabilities', 'logits']

        images_input = model.input
        assert not (isinstance(images_input, list) and image_input_idx is None), "If there are multiple inputs, the image input must be specified"
        aux_inputs = []
        if isinstance(images_input, list):
            for i, input_tensor in enumerate(images_input):
                if i != image_input_idx:
                    aux_inputs.append(K.placeholder(shape=input_tensor.shape))
            images_input = images_input[image_input_idx]
            
        label_input = K.placeholder(shape=(1,))
        predictions = model.output
        assert not (isinstance(predictions, list) and output_idx is None), "If there are multiple outputs, the output with detections must be specified"
        if isinstance(predictions, list):
            predictions = predictions[output_idx]

        if num_classes is None:
            shape = K.int_shape(predictions)
            print("Shape of predictions is {}".format(shape))
            num_classes = None
            for i, dim in enumerate(shape):
                if i == len(shape) - 1:
                    num_classes = dim
            assert num_classes is not None
            
        print("Number of classes: {}".format(num_classes))
        self._num_classes = num_classes

        if predicts == 'probabilities':
            if K.backend() == 'tensorflow':
                predictions = predictions.op.inputs[0]
                print("Predictions_shape: {}".format(predictions.shape))
#                 if output_idx is not None:
#                     predictions = K.expand_dims(predictions[0,0], axis=0)
                print("label_input: {}".format(label_input))
                print("predictions: {}".format(predictions))
                loss = K.sparse_categorical_crossentropy(
                    label_input, predictions, from_logits=True)
            else:
                logging.warning('relying on numerically unstable conversion'
                                ' from probabilities to softmax')
                loss = K.sparse_categorical_crossentropy(
                    label_input, predictions, from_logits=False)

                # transform the probability predictions into logits, so that
                # the rest of this code can assume predictions to be logits
                predictions = self._to_logits(predictions)
        elif predicts == 'logits':
            loss = K.sparse_categorical_crossentropy(
                label_input, predictions, from_logits=True)

        # sparse_categorical_crossentropy returns 1-dim tensor,
        # gradients wants 0-dim tensor (for some backends)
        loss = K.squeeze(loss, axis=0)
        grads = K.gradients(loss, images_input)

        grad_loss_output = K.placeholder(shape=(num_classes, 1))
        external_loss = K.dot(predictions, grad_loss_output)
        # remove batch dimension of predictions
        external_loss = K.squeeze(external_loss, axis=0)
        # remove singleton dimension of grad_loss_output
        external_loss = K.squeeze(external_loss, axis=0)

        grads_loss_input = K.gradients(external_loss, images_input)

        if K.backend() == 'tensorflow':
            # tensorflow backend returns a list with the gradient
            # as the only element, even if loss is a single scalar
            # tensor;
            # theano always returns the gradient itself (and requires
            # that loss is a single scalar tensor)
            assert isinstance(grads, list)
            assert len(grads) == 1
            grad = grads[0]

            assert isinstance(grads_loss_input, list)
            assert len(grads_loss_input) == 1
            grad_loss_input = grads_loss_input[0]
        elif K.backend() == 'cntk':  # pragma: no cover
            assert isinstance(grads, list)
            assert len(grads) == 1
            grad = grads[0]
            grad = K.reshape(grad, (1,) + grad.shape)

            assert isinstance(grads_loss_input, list)
            assert len(grads_loss_input) == 1
            grad_loss_input = grads_loss_input[0]
            grad_loss_input = K.reshape(grad_loss_input, (1,) + grad_loss_input.shape)  # noqa: E501
        else:
            assert not isinstance(grads, list)
            grad = grads

            grad_loss_input = grads_loss_input

        self._loss_fn = K.function(
            [images_input, label_input],
            [loss])
        self._batch_pred_fn = K.function(
            [images_input], [predictions])
        self._pred_grad_fn = K.function(
            [images_input, label_input],
            [predictions, grad])
        self._bw_grad_fn = K.function(
            [grad_loss_output, images_input],
            [grad_loss_input])

    def _to_logits(self, predictions):
        from keras import backend as K
        eps = 10e-8
        predictions = K.clip(predictions, eps, 1 - eps)
        predictions = K.log(predictions)
        return predictions

    def num_classes(self):
        return self._num_classes

    def batch_predictions(self, images):
        px, _ = self._process_input(images)
        print("px: {}".format(px.shape))
        predictions = self._batch_pred_fn([px])
        assert len(predictions) == 1
        predictions = predictions[0]
        if len(predictions) > 1:
            predictions = predictions[0].reshape((1,predictions.shape[1]))
        assert predictions.shape == (images.shape[0], self.num_classes())
        return predictions

    def predictions_and_gradient(self, image, label):
        print("predictions and gradient function")
        px, dpdx = self._process_input(image)
        print("image shape: {}, label shape: {}".format(image.shape, label))
        predictions, gradient = self._pred_grad_fn([
            px[np.newaxis],
            np.array([label])])
        predictions = np.squeeze(predictions, axis=0)
        gradient = np.squeeze(gradient, axis=0)
        gradient = self._process_gradient(dpdx, gradient)
        assert predictions.shape == (self.num_classes(),)
        assert gradient.shape == image.shape
        return predictions, gradient

    def backward(self, gradient, image):
        assert gradient.ndim == 1
        gradient = np.reshape(gradient, (-1, 1))
        px, dpdx = self._process_input(image)
        gradient = self._bw_grad_fn([
            gradient,
            px[np.newaxis],
        ])
        gradient = np.squeeze(gradient, axis=0)
        gradient = self._process_gradient(dpdx, gradient)
        assert gradient.shape == image.shape
        return gradient
