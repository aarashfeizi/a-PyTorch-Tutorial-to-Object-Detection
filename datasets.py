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
        self.boxes_YOU = list(
            zip(dataset_info.x_min_YOU, dataset_info.y_min_YOU,
                dataset_info.x_max_YOU, dataset_info.y_max_YOU))
        self.boxes_NDP = list(
            zip(dataset_info.x_min_NDP, dataset_info.y_min_NDP,
                dataset_info.x_max_NDP, dataset_info.y_max_NDP))
        self.boxes_GPC = list(
            zip(dataset_info.x_min_GPC, dataset_info.y_min_GPC,
                dataset_info.x_max_GPC, dataset_info.y_max_GPC))
        self.boxes_LPC = list(
            zip(dataset_info.x_min_LPC, dataset_info.y_min_LPC,
                dataset_info.x_max_LPC, dataset_info.y_max_LPC))
        self.boxes_PPC = list(
            zip(dataset_info.x_min_PPC, dataset_info.y_min_PPC,
                dataset_info.x_max_PPC, dataset_info.y_max_PPC))
        self.boxes_CPC = list(
            zip(dataset_info.x_min_CPC, dataset_info.y_min_CPC,
                dataset_info.x_max_CPC, dataset_info.y_max_CPC))
        self.boxes_BQ = list(
            zip(dataset_info.x_min_BQ, dataset_info.y_min_BQ,
                dataset_info.x_max_BQ, dataset_info.y_max_BQ))

        self.boxes_NDP = [np.array(l) for l in self.boxes_NDP]
        self.boxes_GPC = [np.array(l) for l in self.boxes_GPC]
        self.boxes_CPC = [np.array(l) for l in self.boxes_CPC]
        self.boxes_LPC = [np.array(l) for l in self.boxes_LPC]
        self.boxes_PPC = [np.array(l) for l in self.boxes_PPC]
        self.boxes_YOU = [np.array(l) for l in self.boxes_YOU]
        self.boxes_BQ = [np.array(l) for l in self.boxes_BQ]

        self.boxes = np.array(list(zip(self.boxes_NDP,
                                       self.boxes_GPC,
                                       self.boxes_CPC,
                                       self.boxes_BQ,
                                       self.boxes_LPC,
                                       self.boxes_PPC,
                                       self.boxes_YOU)))

        self.labels = np.array(list(zip(dataset_info.label_NDP,
                               dataset_info.label_GPC,
                               dataset_info.label_CPC,
                               dataset_info.label_BQ,
                               dataset_info.label_LPC,
                               dataset_info.label_PPC,
                               dataset_info.label_YOU)))


        assert len(self.images) == len(self.boxes)
        assert len(self.images) == len(self.labels)

    def __getitem__(self, i):
        # Read image
        image = Image.open(os.path.join(self.data_folder, self.images[i]), mode='r')
        image = image.convert('RGB')

        labels = self.labels[i]
        boxes = self.boxes[i]
        for i in range(len(labels)):
            if labels[i] is None:
                labels = labels[:i] + labels[i + 1:]
                boxes = boxes[:i] + boxes[i + 1:]
                break

        # Read objects in this image (bounding boxes, labels, difficulties)
        boxes = torch.FloatTensor(boxes)  # (n_objects, 4)
        labels = torch.LongTensor([labels])[0]  # (n_objects)

        for i in range(len(labels)):
            if labels[i] is None:
                labels = labels[:i] + labels[i + 1:]
                boxes = boxes[:i] + boxes[i + 1:]
                break # there's at most 1 None value

        # Apply transformations
        image, boxes, labels, _ = transform(image, boxes, labels, None, split=self.split)

        return image, boxes, labels

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

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels  # tensor (N, 3, 300, 300), 3 lists of N tensors each

    def __len__(self):
        return len(self.images)

