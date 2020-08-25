import cv2
import numpy as np

from torchvision import transforms
from datasets.ff_dataset import get_loader, read_dataset, CompositeDataset
from datasets.transforms_bank import IsotropicResize, RemoveLandmark, BlackoutConvexHull, RandomCutout, ColorJitter, \
    RandomPerspective
from albumentations import Compose, RandomBrightnessContrast, VerticalFlip, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, RandomResizedCrop, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, NoOp


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

def read_training_dataset(data_dir, transform, normalization, filter = lambda x: 'original' in x or 'neural' in x, max_images_per_video=10, max_videos=10000, window_size=1, splits_path='ff_splits'):
    '''

     This function was taken from the TUM repository. In order to have control over the datasets (i.e. original, face2face,
     faceswaps, etc.). I have added the filter parameter that takes in a function to filter the keays. Default is all
     originals & NeuralTextures at all levels of compression (raw, c23, c40).

    '''
    datasets = read_dataset(data_dir, normalization = normalization, transform=transform, max_images_per_video=max_images_per_video, max_videos=max_videos,
                            window_size=window_size, splits_path=splits_path)

    # only neural textures and original
    datasets = {
        k: v for k, v in datasets.items()
        if filter(k)
    }
    print('Using training data: ')
    print('\n'.join(sorted(datasets.keys())))

    trains, vals, tests = [], [], []
    for data_dir_name, dataset in datasets.items():
        train, val, test = dataset
        # repeat original data multiple times to balance out training data
        compression = data_dir_name.split('_')[-1]
        num_tampered_with_same_compression = len({x for x in datasets.keys() if compression in x}) - 1
        count = 1 if 'original' not in data_dir_name else num_tampered_with_same_compression
        for _ in range(count):
            trains.append(train)
        vals.append(val)
        tests.append(test)
    return CompositeDataset(*trains), CompositeDataset(*vals), CompositeDataset(*tests)



class DataAugmentationTransforms:

    @staticmethod
    def create_train_transforms(size=300, shift_limit=0.1, scale_limit=0.2, rotate_limit = 10, landmarks_blackout=True,
                                convex_hull_blackout=True, cutout=True):

        '''

        :param size:
        :param shif_limit:
        :param rotate_limit:
        :param shiftscale_limit:
        :param landmarks_blackout:
        :param convex_hull_blackout:
        :param cutout:
        :return: Composition of transformations via albumentations
        '''

        # TODO: The following must go on a OneOf. Set p = 0 in case input argument is set to False

        rm_ld = RemoveLandmark() if landmarks_blackout else NoOp()
        bo_ch = BlackoutConvexHull() if convex_hull_blackout else NoOp()
        co = RandomCutout() if cutout else NoOp()

        trans = [OneOf([rm_ld, bo_ch, co], p=1)]

        trans = trans + [
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
                ShiftScaleRotate(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            ]

        return Compose(trans)




    @staticmethod
    def create_val_transforms(size=300):

        transformations = [
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            ]

        return Compose(transformations)


def create_train_transforms(size=300, option='no_da', shift_limit=0.1, scale_limit=0.2, rotate_limit = 10, dataset='faceforensics'):

    assert option in ('no_da', 'simple_da', 'occlusions_da', 'cutout_da'),\
        "option must be one of ('no_da', 'simple_da', 'occlusions_da', 'cutout_da')"

    assert dataset in ('faceforensics', 'dfdc', 'other', 'other_dfdc'), "Dataset not recognized"


    trans = []


    if option == 'no_da' and dataset in ('faceforensics', 'dfdc'):

        trans.append(IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC))
        trans.append(PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT))

    elif dataset in ('faceforensics', 'dfdc'):
        trans.append(HorizontalFlip())
        trans.append(OneOf([
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
                ], p=1))

        trans.append(PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT))
        trans.append(OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.5))
        trans.append(ToGray(p=0.2))
        trans.append(ShiftScaleRotate(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit, border_mode=cv2.BORDER_CONSTANT, p=0.5))

        if dataset == 'dfdc':
            trans.insert(0, GaussianBlur(blur_limit=3, p=0.05))
            trans.insert(0, GaussNoise(p=0.1))
            trans.insert(0, ImageCompression(quality_lower=60, quality_upper=100, p=0.5))

    if option != 'no_da':
        rm_ld = RemoveLandmark() if (option == 'occlusions_da' or option == 'cutout_da') else NoOp()
        bo_ch = BlackoutConvexHull() if (option == 'occlusions_da' or option == 'cutout_da') else NoOp()
        co = RandomCutout() if option == 'cutout_da' else NoOp()

        total_black_out = OneOf([rm_ld, bo_ch, co], p=1)

        trans.insert(0, total_black_out)

    if 'other' in dataset:
        if 'dfdc' in dataset:
            trans.insert(0, GaussianBlur(blur_limit=3, p=0.05))
            trans.insert(0, GaussNoise(p=0.1))
            trans.insert(0, ImageCompression(quality_lower=60, quality_upper=100, p=0.5))

        trans.append(HorizontalFlip())
        trans.append(VerticalFlip())
        trans.append(ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=60, border_mode=cv2.BORDER_CONSTANT, p=0.5))
        trans.append(RandomResizedCrop(size, size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2))
        trans.append(RandomPerspective())
        trans.append(ColorJitter())

    return Compose(trans)

