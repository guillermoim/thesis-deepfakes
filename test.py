import os
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
from models import load_saved_model
from utils import DataAugmentationTransforms as DAT
from utils import read_training_dataset
from albumentations.pytorch.transforms import img_to_tensor

def get_loader(dataset, normalization, transform):

    if dataset == 'ff_c23':
        filter_ = lambda x: ('neural' in x or 'original' in x) and 'c23' in x
        _, _, test_dataset = read_training_dataset('../datasets/mtcnn', transform, normalization, filter=filter_,
                                                   max_images_per_video=10, max_videos=10000, window_size=1, splits_path='ff_splits')
    if dataset == 'ff_c40':
        filter_ = lambda x: ('neural' in x or 'original' in x) and 'c40' in x
        _, _, test_dataset = read_training_dataset('../datasets/mtcnn', transform, normalization, filter=filter_,
                                                   max_images_per_video=10, max_videos=10000, window_size=1, splits_path='ff_splits')
    if dataset == 'ff_raw':
        filter_ = lambda x: ('neural' in x or 'original' in x) and 'raw' in x
        _, _, test_dataset = read_training_dataset('../datasets/mtcnn', transform, normalization, filter=filter_,
                                                   max_images_per_video=10, max_videos=10000, window_size=1, splits_path='ff_splits')

    loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
    return loader


def main():
    parser = argparse.ArgumentParser(description='Train a model with the parameters specified.')

    parser.add_argument('--model', type=str, required=True, help='Specify path to model')
    parser.add_argument('--mode', type=str, choices=['image', 'dir', 'dataset'], help='Specify data entry mode')
    parser.add_argument('--path', type=str, default='.', help='Specify path to image (or dir if selected)')
    parser.add_argument('--cpu', action='store_true', help='If present, then send the model to cpu')
    parser.add_argument('--dataset', type=str, default='', help='Introduce dataset.')
    parser.add_argument('--save_path', type=str, default='test.csv', help='Path to save .csv file')

    args = parser.parse_args()

    path_to_model = os.path.abspath(args.model)
    path_to_data = os.path.abspath(args.path)
    save_path = os.path.abspath(args.save_path)
    mode = args.mode
    dataset = args.dataset
    cpu= args.cpu

    assert os.path.exists(path_to_data), 'Path not valid.'


    if mode == 'image':
        assert (path.endswith('.png') or path.endswith('.jpeg')), \
            'If image mode is selected then path should point to an image in png or JPEG'

    if mode == 'dir':
        assert mode == 'dir' and (os.path.isdir(path_to_data)), \
            'If directory mode is selected then path should point to a proper directory'
    if mode == 'dataset':

        assert dataset in ('ff_c23', 'ff_c40', 'ff_raw', 'dfdc'), 'Dataset should be one of ff_c23, ff_c40, ff_raw, dfdc'

    model, resize, model_name, loss, epoch, normalization = load_saved_model(path_to_model)
    transforms = DAT.create_val_transforms(resize)

    device = torch.device('cpu' if cpu else 'cuda')

    model = model.to(device)

    with torch.no_grad():

        if mode == 'image':

            image = np.asarray(Image.open(path_to_data))
            img = transforms(image=image)
            t = img_to_tensor(img['image'], normalization).to(device).unsqueeze(0)
            output = model(t)

            p = torch.sigmoid(output).to(device).item()

            print(f'Probaility picture being fake: {p}')

        if mode == 'dataset':

            loader = get_loader(dataset, normalization, transforms)

            pairs_prob_and_target = []

            for data in tqdm(loader, desc=f'Testing dataset {dataset}...'):
                video_ids, frame_ids, images, targets = data
                labels = targets.to(device)
                outputs = model(images)
                p = torch.sigmoid(outputs).to(device)
                probs_and_targets = torch.stack((p,labels)).squeeze(2)
                pairs_prob_and_target.append(probs_and_targets)


            ppt = torch.cat(pairs_prob_and_target, dim=1)
            basename = os.path.basename(save_path)
            os.makedirs(basename)
            pd.DataFrame(ppt.transpose(1, 0).numpy(), columns=['prob', 'target']).to_csv(save_path, index=False)

        else:
            # TODO: Implement read a directory and predict all images in it.
            pass

if __name__ == '__main__':

    main()
