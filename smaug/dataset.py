from torch.utils.data import Dataset

import cv2
import random
import imutils
import numpy as np


def read_image(path, target_size=(96, 96), grayscale=False):
    arr = cv2.imread(path)
    if arr is None:
        raise RuntimeError('Cannot read image at %s' % path)
    if grayscale:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    arr = cv2.resize(arr, target_size)
    arr = arr / 255.
    if len(arr.shape) < 3:
        arr = np.expand_dims(arr, -1)

    return arr


def rotate_img(img, angle):
    size = img.shape

    _img = imutils.rotate_bound(img, angle)
    return cv2.resize(_img, size[:2])


def random_crop_img(img, ratio=(0.9, 0.9)):
    size = img.shape

    xlen = size[0] * ratio[0]
    ylen = size[1] * ratio[1]
    xmax = size[0] - xlen
    ymax = size[1] - ylen
    xstart = random.randint(0, int(xmax))
    ystart = random.randint(0, int(ymax))

    new_img = img[xstart:int(xlen), ystart:int(ylen)]
    return cv2.resize(new_img, size[:2])


class SingleAugmentDataset(Dataset):
    """
    Generates a set of 3 different images of a single class at each index.
    """
    def __init__(self, files, labels, img_size=(96, 96), augment=True,
                 random_crop=0.8, random_crop_ratio=(0.9, 0.9),
                 rotate=0.5, rotate_angles=(-15, -10, -5, 0, 5, 10, 15),
                 channels_first=True, grayscale=False):
        super().__init__()
        self.img_size = img_size
        self.augment = augment
        self.rotate = rotate
        self.rotate_angles = rotate_angles
        self.random_crop = random_crop
        self.random_crop_ratio = random_crop_ratio
        self.channels_first = channels_first
        self.grayscale = grayscale

        self._files = files
        self._labels = labels

    def augment_images(self, images):
        new_images = []

        for img in images:
            _img = img
            if random.random() < self.random_crop:
                _img = random_crop_img(_img, self.random_crop_ratio)
            if random.random() < self.rotate:
                angle = random.sample(self.rotate_angles, 1)[0]
                _img = rotate_img(_img, angle)
            if len(_img.shape) < 3:
                _img = np.expand_dims(_img, -1)
            new_images.append(_img)

        return new_images

    def _random_triple(self, index):
        id3 = index
        f3, lbl = self._files[id3], self._labels[id3]
        id2, id1 = random.sample(list(filter(lambda i: self._labels[i] == lbl and i != id3,
                                      range(len(self._files)))), k=2)
        f2, f1 = self._files[id2], self._files[id1]
        return (f1, f2, f3), lbl

    def __getitem__(self, index):
        f, label = self._random_triple(index)
        f1, f2, f3 = f
        im1 = read_image(f1, grayscale=self.grayscale)
        im2 = read_image(f2, grayscale=self.grayscale)
        im3 = read_image(f3, grayscale=self.grayscale)

        if self.augment:
            im1, im2, im3 = self.augment_images([im1, im2, im3])

        if self.channels_first:
            im1 = np.transpose(im1, (2, 0, 1))
            im2 = np.transpose(im2, (2, 0, 1))
            im3 = np.transpose(im3, (2, 0, 1))
        return (im1, im2, im3), np.asarray([label], dtype=np.int)

    def __len__(self):
        return len(self._files)


class MultiAugmentDataset(SingleAugmentDataset):
    def __init__(self, files, labels, **kwargs):
        super().__init__(files, labels, **kwargs)
        self._mapping = {}
        for fn, lb in zip(files, labels):
            if lb not in self._mapping.keys():
                self._mapping[lb] = [fn]
            else:
                self._mapping[lb].append(fn)

    def _random_triple(self, index, lbl=None):
        id3 = index
        f3 = self._mapping[lbl][id3 % len(self._mapping[lbl])]
        f2, f1 = random.sample(list(filter(lambda f: f != f3,
                               self._mapping[lbl])), k=2)
        return f1, f2, f3

    def __getitem__(self, index):
        im1s, im2s, im3s = [], [], []
        labels = []

        for label in self._mapping.keys():
            f = self._random_triple(index, lbl=label)
            f1, f2, f3 = f
            im1 = read_image(f1, grayscale=self.grayscale)
            im2 = read_image(f2, grayscale=self.grayscale)
            im3 = read_image(f3, grayscale=self.grayscale)

            if self.augment:
                im1, im2, im3 = self.augment_images([im1, im2, im3])

            if self.channels_first:
                im1 = np.transpose(im1, (2, 0, 1))
                im2 = np.transpose(im2, (2, 0, 1))
                im3 = np.transpose(im3, (2, 0, 1))

            im1s.append(im1)
            im2s.append(im2)
            im3s.append(im3)
            labels.append(np.asarray(label, dtype=np.int))

        return (im1s, im2s, im3s), labels

    def __len__(self):
        lengths = map(lambda x: len(x), self._mapping.values())
        return max(lengths)
