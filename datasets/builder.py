import cv2
import os
import torch

from .dfdc_dataset import DFDCDataset
from .ff_dataset import read_dataset, CompositeDataset

from datasets.transforms_bank import IsotropicResize, RemoveLandmark, BlackoutConvexHull, RandomCutout, ColorJitter, \
    RandomPerspective
from albumentations import Compose, RandomBrightnessContrast, VerticalFlip, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, RandomResizedCrop, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, NoOp


def create_val_transforms(size=300):

    transformations = [
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        ]

    return Compose(transformations)


def create_train_transforms(size=300, option='policy_1', shift_limit=0.1, scale_limit=0.2, rotate_limit=10, dataset='faceforensics'):

    assert option in [f'policy_{i}' for i in range(1, 6)],\
        f"option must be one of ('policy_0', 'policy_1', 'policy_2', 'policy_3') NOT {option}"

    assert dataset in ('faceforensics', 'dfdc', ), "Dataset not recognized"

    trans = []

    if option == 'policy_1' and dataset in ('faceforensics', 'dfdc'):

        trans.append(IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC))
        trans.append(PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT))

    elif dataset in ('faceforensics', 'dfdc') and option != 'policy_5':


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

    if not option in ('policy_0', 'policy_5'):
        rm_ld = RemoveLandmark() if (option == 'policy_3' or option == 'policy_4') else NoOp()
        bo_ch = BlackoutConvexHull() if (option == 'policy_3' or option == 'policy_4') else NoOp()
        co = RandomCutout() if option == 'policy_4' else NoOp()

        total_black_out = OneOf([rm_ld, bo_ch, co], p=1)

        trans.insert(0, total_black_out)

    if option == 'policy_5':
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


def read_training_dataset(data_dir, transform, normalization, filter = lambda x: 'original' in x or 'neural' in x, max_images_per_video=10, max_videos=10000, window_size=1, splits_path='datasets/ff_splits'):
    '''

     This function was taken from the TUM repository. In order to have control over the datasets (i.e. original, face2face,
     faceswaps, etc.). I have added the filter parameter that takes in a function to filter the keays. Default is all
     originals & NeuralTextures at all levels of compression (raw, c23, c40).

    '''
    datasets = read_dataset(data_dir, normalization=normalization, transform=transform, max_videos=max_videos,
                            max_images_per_video=max_images_per_video, window_size=window_size,
                            splits_path=splits_path)

    # only neural textures and original
    datasets = {k: v for k, v in datasets.items() if filter(k)}
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


def create_train_dataset(dataset='dfdc', data_root_path='', normalization=None, size=220, da_policy='policy_0',
                   filter_ = lambda x: ('original' in x or 'neural' in x), seed=100):

    # Use original and neural_textures (could be modified) when using FF

    assert dataset in ('dfdc', 'faceforensics'), "The dataset must be either 'dfdc' or 'ff'"
    #assert mode in ('train', 'val'), "The modeis either train or val"

    if dataset =='dfdc':

        transforms = create_train_transforms(size, option=da_policy, dataset=dataset)
        res = DFDCDataset(crops_dir='crops', data_path=data_root_path, hardcore=True, fold=10,
                                          normalize=normalization, folds_csv=os.path.join(data_root_path, 'folds.csv'),
                                          transforms=transforms)
        res.reset(0, seed)

    elif dataset == 'faceforensics':

        transforms = create_train_transforms(size, option=da_policy, dataset=dataset)
        res, _, _ = read_training_dataset(data_root_path, filter=filter_, transform=transforms,
                                              normalization=normalization,
                                              splits_path=os.path.join(data_root_path, '../ff_splits'))

    return res


def create_val_dataset(dataset='dfdc', data_root_path='', normalization=None, size=220,
                       filter_=lambda x: ('original' in x or 'neural' in x)):

    val_datsets = {}

    transforms = create_val_transforms(size)
    if dataset == 'dfdc':
        ds = DFDCDataset(crops_dir='crops', data_path=data_root_path, hardcore=True, fold=10,
                      normalize=normalization, folds_csv=os.path.join(data_root_path, 'folds.csv'), mode='val',
                      transforms=transforms)
        ds.reset(0, 0)
        val_datsets['dfdc_val'] = ds

    elif dataset == 'faceforensics':
        raw_filter = lambda x: filter_(x) and 'raw' in x
        _, raw_val, _ = read_training_dataset(data_root_path, transforms, normalization=normalization,
                                              filter=raw_filter, splits_path=os.path.join(data_root_path, '../ff_splits'))
        del _
        val_datsets['raw'] = raw_val

        # validation set specific for c23
        c23_filter = lambda x: filter_(x) and 'c23' in x
        _, c23_val, _ = read_training_dataset(data_root_path, transforms, normalization=normalization,
                                              filter=c23_filter, splits_path=os.path.join(data_root_path, '../ff_splits'))
        del _
        val_datsets['c23'] = c23_val

        # validation set specific for c40
        c40_filter = lambda x: filter_(x) and 'c40' in x
        _, c40_val, _ = read_training_dataset(data_root_path, transforms, normalization=normalization,
                                              filter=c40_filter, splits_path=os.path.join(data_root_path, '../ff_splits'))
        del _
        val_datsets['c40'] = c40_val


    return val_datsets