import pandas as pd
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform
import numpy as np

class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each

class GridDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=True):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        dataset_info = pd.read_csv(os.path.join(data_folder, self.split + '_dataset_info.csv'))
        self.images = list(dataset_info.img)
        self.boxes = list(zip(dataset_info.x_min, dataset_info.y_min, dataset_info.x_max, dataset_info.y_max))
        self.labels = list(dataset_info.label)


        assert len(self.images) == len(self.boxes)
        assert len(self.images) == len(self.labels)

    def __getitem__(self, i):
        # Read image
        image = Image.open(os.path.join(self.data_folder, self.images[i]), mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        boxes = torch.FloatTensor(self.boxes[i]).reshape(1, -1)  # (n_objects, 4)
        labels = torch.LongTensor([self.labels[i]])  # (n_objects)

        # Apply transformations
        image, boxes, labels, _ = transform(image, boxes, labels, None, split=self.split)

        return image, boxes, labels

    def __len__(self):
        return len(self.images)

class PointDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=True):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        dataset_info = pd.read_csv(os.path.join(data_folder, self.split + '_dataset_info.csv'))
        self.images = list(dataset_info.img)
        self.boxes_you = list(
            zip(dataset_info.x_min_you, dataset_info.y_min_you,
                dataset_info.x_max_you, dataset_info.y_max_you))
        self.boxes_red = list(
            zip(dataset_info.x_min_red, dataset_info.y_min_red,
                dataset_info.x_max_red, dataset_info.y_max_red))
        self.boxes_green = list(
            zip(dataset_info.x_min_green, dataset_info.y_min_green,
                dataset_info.x_max_green, dataset_info.y_max_green))
        self.boxes_orange = list(
            zip(dataset_info.x_min_orange, dataset_info.y_min_orange,
                dataset_info.x_max_orange, dataset_info.y_max_orange))
        self.boxes_purple = list(
            zip(dataset_info.x_min_purple, dataset_info.y_min_purple,
                dataset_info.x_max_purple, dataset_info.y_max_purple))
        self.boxes_blue = list(
            zip(dataset_info.x_min_blue, dataset_info.y_min_blue,
                dataset_info.x_max_blue, dataset_info.y_max_blue))

        self.boxes_red = [np.array(l) for l in self.boxes_red]
        self.boxes_green = [np.array(l) for l in self.boxes_green]
        self.boxes_blue = [np.array(l) for l in self.boxes_blue]
        self.boxes_orange = [np.array(l) for l in self.boxes_orange]
        self.boxes_purple = [np.array(l) for l in self.boxes_purple]
        self.boxes_you = [np.array(l) for l in self.boxes_you]

        self.boxes = np.array(list(zip(self.boxes_red,
                                  self.boxes_green,
                                  self.boxes_blue,
                                  self.boxes_orange,
                                  self.boxes_purple,
                                  self.boxes_you)))

        self.labels = np.array(list(zip(dataset_info.label_red,
                               dataset_info.label_green,
                               dataset_info.label_blue,
                               dataset_info.label_orange,
                               dataset_info.label_purple,
                               dataset_info.label_you)))


        assert len(self.images) == len(self.boxes)
        assert len(self.images) == len(self.labels)

    def __getitem__(self, i):
        # Read image
        image = Image.open(os.path.join(self.data_folder, self.images[i]), mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        boxes = torch.FloatTensor(self.boxes[i])  # (n_objects, 4)
        labels = torch.LongTensor([self.labels[i]])  # (n_objects)

        # Apply transformations
        image, boxes, labels, _ = transform(image, boxes, labels, None, split=self.split)

        return image, boxes, labels

    def __len__(self):
        return len(self.images)
