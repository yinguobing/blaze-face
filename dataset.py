"""Provide training/testing data."""

import os

import cv2
import numpy as np
import tensorflow as tf

from anchors import Anchors, Boxes


class DetectionSample(object):

    def __init__(self, image_file, boxes):
        """Construct an object detection sample.

        Args:
            image: image file path.
            boxes: numpy array of bounding boxes [[x, y, w, h], ...]

        """
        self.image_file = image_file
        self.boxes = boxes

    def read_image(self, format="BGR"):
        """Read in image as numpy array in format of BGR by defult, else RGB.

        Args:
            format: the channel order, default BGR.

        Returns:
            a numpy array.
        """
        img = cv2.imread(self.image_file)
        if format != "BGR":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img


class WiderFace(object):

    def __init__(self, dataset_path, mode="train"):
        """Construct the WiderFace dataset.

        Dataset file structure:

            wider  <-- path to here
            ├── wider_face_split
            │   ├── wider_face_train_bbx_gt.txt
            │   ├── wider_face_train.mat
            │   ├── ...
            ├── WIDER_train
            │   ├── 0--Parade
            │   ├── 10--People_Marching
            │   ├── ...
            └── WIDER_val
                ├── 0--Parade
                ├── 10--People_Marching
                ├── ...


        Args:
            dataset_path: path to the dataset directory.
        """
        # Find the label files.
        label_file_train = os.path.join(
            dataset_path, "wider_face_split", "wider_face_train_bbx_gt.txt")
        label_file_val = os.path.join(
            dataset_path, "wider_face_split", "wider_face_val_bbx_gt.txt")
        label_file_test = os.path.join(
            dataset_path, "wider_face_split", "wider_face_test_filelist.txt")

        # Parse the label files to get image file path and bounding boxes.
        def _parse(label_file, img_dir):
            samples = []
            with open(label_file, "r") as fid:
                while(True):
                    # Find out which image file to be processed.
                    line = fid.readline()
                    if line == "":
                        break
                    line = line.rstrip('\n').rstrip()
                    assert line.endswith(".jpg"), "Failed to read next label."
                    img_file = os.path.join(dataset_path, img_dir, line)

                    # Read the bounding boxes.
                    n_boxes = int(fid.readline().rstrip('\n').rstrip())
                    if n_boxes == 0:
                        fid.readline()
                        continue
                    lines = [fid.readline().rstrip('\n').rstrip().split(' ')
                             for _ in range(n_boxes)]

                    boxes = np.array(lines, dtype=np.int)[:, :4]

                    # Accumulate the results.
                    samples.append(DetectionSample(img_file, boxes))
            return samples

        if mode == 'train':
            self.dataset = _parse(label_file_train, "WIDER_train")
        elif mode == 'val':
            self.dataset = _parse(label_file_val, "WIDER_val")
        elif mode == 'test':
            # There is no bounding boxes in test dataset.
            self.dataset = []
            with open(label_file_test, "r") as fid:
                for img_file in fid:
                    self.dataset.append(DetectionSample(img_file, None))
        else:
            raise ValueError(
                'Mode {} not supported, check again.'.format(mode))

        # Set index for iterator.
        self.index = 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self.dataset):
            raise StopIteration
        sample = self.dataset[self.index]
        self.index += 1
        return sample
