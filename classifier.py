import os

import numpy as np
import skimage.io

from tqdm import tqdm

import utils
import model as modellib
import visualize


class classifier():
    model_dir = ''
    root_dir = ''
    model_path = ''
    model_name = ''

    image_dir = ''
    output_dir = ''

    config = None
    model = None
    dataset = None

    def __init__(self, modelname="cityscapes"):
        self.model_name = modelname

        # Root directory of the project
        self.root_dir = os.getcwd()

        # Directory to save logs and trained model
        self.model_dir = os.path.join(self.root_dir, "logs")

        # Local path to trained weights file
        if modelname == "cityscapes":
            self.model_path = os.path.join(self.model_dir, "mask_rcnn_cityscapes.h5")
        else:
            self.model_path = os.path.join(self.model_dir, "mask_rcnn_coco.h5")

    # Directory of images to run detection on or save output to
    def setImageDirs(self, imdir, outdir):
        self.image_dir = imdir
        self.output_dir = outdir

    def setInferenceConfig(self, config):
        self.config = config
        self.config.display()

    def setDirectories(self, model):
        self.model_dir = model

    def createModel(self, mode="inference"):
        # Download COCO trained weights from Releases if needed
        if self.model_name == "coco" and not os.path.exists(self.model_path):
            utils.download_trained_weights(self.model_path)

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode=mode, model_dir=self.model_dir, config=self.config)

        # Load weights
        self.model.load_weights(self.model_path, by_name=True)

    def setDataset(self, dataset):
        self.dataset = dataset

    def classifySimple(self, image, output_name=None, verbose=0):
        # Run detection
        # Returns a list of dicts, one dict per image. The dict contains:
        # rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        # class_ids: [N] int class IDs
        # scores: [N] float probability scores for the class IDs
        # masks: [H, W, N] instance binary masks
        results = self.model.detect([image], verbose=verbose)

        # Visualize results
        r = results[0]

        colored_im = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                                 self.dataset.class_names, r['scores'], save_name=output_name)

        # Also include class names
        classnames = []
        for id in r['class_ids']:
            classnames.append(self.dataset.class_names[id])
        r['classnames'] = classnames

        return [colored_im, r]

    def classifyImage(self, image, output_name=None, verbose=0):
        [colored_im, r] = self.classifySimple(image, output_name=output_name, verbose=verbose)

        colored_label_im = np.zeros((image.shape[0], image.shape[1], 3), dtype='uint8')
        label_im = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
        instance_im = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')

        for i, class_id, roi, score in zip(np.arange(len(r['class_ids'])), r['class_ids'], r['rois'], r['scores']):
            mask = r['masks'][:, :, i]
            inds = (mask > 0)
            # print(i, self.dataset.class_names[class_id])

            key = None
            if self.dataset.class_names[class_id] in self.dataset.labels.values():
                key = list(self.dataset.labels.keys())[list(self.dataset.labels.values()).index(self.dataset.class_names[class_id])]

            if key is not None:
                label_im[inds] = key
                colored_label_im[inds] = self.dataset.cmap[key]

            if self.dataset.class_names[class_id] in self.dataset.instanceids.keys():
                instance_im[inds] = self.dataset.instanceids[self.dataset.class_names[class_id]]

        return [colored_im, colored_label_im, label_im, instance_im, r]

    # Run classification on a directory of images
    def classify(self):
        file_names = next(os.walk(self.image_dir))[2]
        for f in tqdm(file_names):
            if f.endswith(".jpg") or f.endswith(".png"):
                image = skimage.io.imread(os.path.join(self.image_dir, f))

                base_name = os.path.join(self.output_dir, os.path.splitext(f)[0])
                output_name = base_name + "_roi.jpg"

                [colored_im, colored_label_im, label_im, instance_im, rois] = self.classifyImage(image, output_name)

                skimage.io.imsave(base_name + "_viz.jpg", colored_im)
                skimage.io.imsave(base_name + "_color.png", colored_label_im)
                skimage.io.imsave(base_name + "_labelIds.png", label_im)
                skimage.io.imsave(base_name + "_instanceIds.png", instance_im)

            exit()
