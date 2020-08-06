from training.transforms.albu import IsotropicResize
import numpy as np
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur
import cv2
from torchvision import transforms


class AverageMeter(object):
    '''
        Code taken from torchreid repository
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.



class DataAugmentationTransforms:


    @staticmethod
    def create_train_transforms(size=300):
        transformations = [
                ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                GaussNoise(p=0.1),
                GaussianBlur(blur_limit=3, p=0.05),
                HorizontalFlip(),
                OneOf([
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
                ], p=1),
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
                ToGray(p=0.2),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            ]


        return Compose(transformations)



    @staticmethod
    def create_val_transforms(size=300):

        transformations = [
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            ]

        return Compose(transformations)
