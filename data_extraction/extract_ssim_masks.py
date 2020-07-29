import os
import cv2
import numpy as np
import pandas as pd

from PIL import Image
from skimage.io import imread
from skimage.metrics import structural_similarity as compare_ssim


def extract_ssim_masks(directory, file_name='dataset.csv'):

    os.makedirs(f'{directory}/diffs', exist_ok=True)
    diffs_dir = f'{directory}/diffs'

    df = pd.read_csv(f'{directory}/{file_name}')
    data = [list(x) for x in df.values]

    fakes = [row for row in data if row[-1] == 1]

    for fake_face in fakes:

        path_to_fake = os.path.join(directory, 'faces', fake_face[2])

        face_id = os.path.basename(fake_face[2])

        original_video_name = fake_face[-2][:-4]
        fake_video_name = fake_face[1][:-4]

        path_to_original = os.path.join(directory, 'faces', f'{original_video_name}/{face_id}')

        if os.path.exists(path_to_fake) and os.path.exists(path_to_original):
            os.makedirs(f'{diffs_dir}/{fake_video_name}', exist_ok=True)
            img_fake = imread(path_to_fake)
            img_ori = imread(path_to_original)
            d, a = compare_ssim(img_fake, img_ori, multichannel=True, full=True)
            a = 1 - a
            diff = (a * 255).astype(np.uint8)
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f'{diffs_dir}/{fake_video_name}/{face_id}', diff)

        else:
            continue


if __name__ == '__main__':

    print(extract_ssim_masks('sample/sample'))