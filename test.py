import os
import torch
import argparse
import numpy as np

from PIL import Image
from models import load_saved_model
from utils import DataAugmentationTransforms as DAT
from albumentations.pytorch.transforms import img_to_tensor

def main():
    parser = argparse.ArgumentParser(description='Train a model with the parameters specified.')

    parser.add_argument('--model', type=str, required=True, help='Specify path to model')
    parser.add_argument('--image', type=str, default='', help='Specify path to image')
    parser.add_argument('--dir', type=str, help='Specify path to dir of images')
    parser.add_argument('--cuda', action='store_false', help='If present, then send the model to cpu')

    args = parser.parse_args()

    path_to_model = os.path.abspath(args.model)
    path_to_image = os.path.abspath(args.image)
    cuda= args.cuda

    model, resize, model_name, loss, epoch, normalization = load_saved_model(path_to_model)
    transforms = DAT.create_val_transforms(resize)

    device = torch.device('cuda' if cuda else 'cpu')

    model = model.to(device)

    if path_to_image != '':

        image = np.asarray(Image.open(path_to_image))
        img = transforms(image=image)
        t = img_to_tensor(img['image'], normalization).to(device).unsqueeze(0)
        output = model(t)

        p = torch.sigmoid(output).cpu().item()

        print(f'Probaility picture being fake: {p}')
    else:
        pass



if __name__ == '__main__':

    main()
