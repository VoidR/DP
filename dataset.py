import torch
import cv2
import numpy as np
from tqdm import tqdm
from albumentations import Compose, RandomBrightnessContrast, ShiftScaleRotate
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
import pandas as pd
import os

import glob
import cv2
import numpy as np

def crop_image_from_gray(img, tol=7):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
    
    NOTE: This was used to generate the pre-processed dataset
    """

    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def circle_crop(img):
    """
    Create circular crop around image centre
    
    NOTE: This was used to generate the pre-processed dataset
    """

    img = crop_image_from_gray(img)

    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)

    return img

# if __name__ == '__main__':
#     data = pd.read_csv('../dataset/blindness/train.csv')

#     for idx in range(len(data)):
#         imgn = os.path.join('../dataset/blindness/train_images', data.loc[idx, 'id_code'] + '.png')
#         print (imgn)
#         img_cut = circle_crop(cv2.imread(imgn))
#         img_cut = np.array(cv2.resize(img_cut, (224, 224)))
#         # print (img_cut.shape, img_cut.max(), img_cut.min())
#         cv2.imwrite('../dataset/blindness/train_images_cut/' + data.loc[idx, 'id_code'] + '.png', img_cut)


class RetinopathyDatasetTrain(Dataset):
    def __init__(self, csv_file, transform=None, split=(-1,-1), test=False):
        # self.data = pd.read_csv(csv_file)
        self.data = np.load(csv_file, allow_pickle=True).item()
        self.transform = transform
        self.split = 0 if split[0] < 0 else split[0]
        self.total = 1 if split[1] < 0 else split[1]
        # self.start = int(len(self.data) // self.total) * self.split
        self.start = int(len(self.data['label']) // self.total) * self.split

        self.test = test
        # print ('???', split)
    def __len__(self):
        # return int(len(self.data['label']) // self.total) - 1
        return int(len(self.data['label']) // self.total) 



    def __getitem__(self, idx):
        # img_name = os.path.join('../dataset/ISIC_2018//ISIC2018_Task3_Training_Input/', self.data.loc[self.start + idx, 'image'] + '.jpg') # typo
        # img_name = self.data.loc[self.start + idx, 'image']
        img_name, target = self.data['img_path'][self.start + idx], self.data['label'][self.start + idx]
        im = cv2.imread(img_name)
        # print (im)
        # exit()
        if self.test:
            # label = torch.tensor(np.argmax(self.data.loc[self.start + idx].values[1:]))
            label = torch.tensor(np.argmax(target))
        else:
            # label = torch.tensor(np.array(self.data.loc[self.start + idx].values[1:]).astype(np.float))
            label = torch.tensor(target)
        if self.transform:
            augmented = self.transform(image=im)
            im = augmented['image']
        return im, label

