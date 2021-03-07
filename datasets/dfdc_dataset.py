import math
import os
import random
import sys
import traceback

import cv2
import numpy as np
import pandas as pd
import skimage.draw
from albumentations import ImageCompression, OneOf, GaussianBlur, Blur
from albumentations.augmentations.functional import image_compression, rot90
from albumentations.pytorch.functional import img_to_tensor
from scipy.ndimage import binary_erosion, binary_dilation
from skimage import measure
from torch.utils.data import Dataset
import dlib

def change_padding(image, part=5):
    h, w = image.shape[:2]
    # original padding was done with 1/3 from each side, too much
    pad_h = int(((3 / 5) * h) / part)
    pad_w = int(((3 / 5) * w) / part)
    image = image[h // 5 - pad_h:-h // 5 + pad_h, w // 5 - pad_w:-w // 5 + pad_w]
    return image


class DFDCDataset(Dataset):

    def __init__(self,
                 data_path="/mnt/sota/datasets/deepfake",
                 fold=0,
                 label_smoothing=0.01,
                 padding_part=3,
                 hardcore=True,
                 crops_dir="crops",
                 folds_csv="folds.csv",
                 normalize={"mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]},
                 mode='train',
                 reduce_val=True,
                 oversample_real=True,
                 transforms=None,
                 test=11
                 ):
        super().__init__()
        self.data_root = data_path
        self.folds_csv = folds_csv
        self.padding_part = padding_part
        self.hardcore = hardcore
        self.crops_dir = crops_dir
        self.label_smoothing = label_smoothing
        self.normalize = normalize
        self.transforms = transforms
        self.df = pd.read_csv(self.folds_csv)
        self.oversample_real = oversample_real
        self.reduce_val = reduce_val
        self.mode = mode
        self.fold = fold
        self.test = test

    def __getitem__(self, index: int):

        while True:
            video, img_file, label, ori_video, frame, fold = self.data[index]
            try:
                if self.mode == 'train':
                    label = np.clip(label, self.label_smoothing, 1 - self.label_smoothing)
                img_path = os.path.join(self.data_root, self.crops_dir, video, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                diff_path = os.path.join(self.data_root, "diffs", video, img_file[:-4] + "_diff.png")

                try:
                    msk = cv2.imread(diff_path, cv2.IMREAD_GRAYSCALE)
                    if msk is not None:
                        mask = msk
                except:
                    print("not found mask", diff_path)
                    pass

                landmark_path = os.path.join(self.data_root, "landmarks", ori_video, img_file[:-4] + ".npy")
                if os.path.exists(landmark_path):
                    landmarks = np.load(landmark_path)
                else:
                    landmarks = None


                if self.mode == "train" and self.padding_part > 3:
                    image = change_padding(image, self.padding_part)

                #valid_label = np.count_nonzero(mask[mask > 20]) > 32 or label < 0.5
                #valid_label = 1 if valid_label else 0

                if self.transforms:
                    data = self.transforms(image=image, mask=mask, landmarks=landmarks, label=label)
                    image = data["image"]
                    #mask = data["mask"]

                image = img_to_tensor(image, self.normalize)
                return {"image": image, "labels": np.array((label,)), "path": os.path.join(self.data_root, video, img_file), }

            except Exception as e:
                traceback.print_exc(file=sys.stdout)
                print("Broken image", os.path.join(self.data_root, self.crops_dir, video, img_file))
                index = random.randint(0, len(self.data) - 1)

    def reset(self, epoch, seed):
        self.data = self._prepare_data(epoch, seed)

    def __len__(self) -> int:
        return len(self.data)

    def _prepare_data(self, epoch, seed):
        df = self.df
        if self.mode == "train":
            rows = df[(df["fold"] != self.fold) & (df["fold"] != self.test)]
        elif self.mode == "val":
            rows = df[df["fold"] == self.fold]
        else:
            rows = df[df["fold"] == self.test]

        seed = (epoch + 1) * seed
        if self.oversample_real:
            rows = self._oversample(rows, seed)
        if (self.mode == "val" or self.mode == 'test') and self.reduce_val:
            # every 2nd frame, to speed up validation
            rows = rows[rows["frame"] % 30 == 0]
            # another option is to use public validation set
            #rows = rows[rows["video"].isin(PUBLIC_SET)]

        print(
            "real {} fakes {} mode {}".format(len(rows[rows["label"] == 0]), len(rows[rows["label"] == 1]), self.mode))
        data = rows.values

        np.random.seed(seed)
        np.random.shuffle(data)
        return data

    def _oversample(self, rows: pd.DataFrame, seed):
        real = rows[rows["label"] == 0]
        fakes = rows[rows["label"] == 1]
        num_real = real["video"].count()
        #Modified: removed -> if self.mode == "train":
        fakes = fakes.sample(n=num_real, replace=False, random_state=seed)
        return pd.concat([real, fakes])
