import os
import json
import torch
import shutil
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN
from torchvision.io import read_video

def __init__():
    pass

def overlapping_1d(low_1, high_1, low_2, high_2):
    return max(0, min(high_1, high_2) - max(low_1, low_2))

def check_boxes_match(in_array):
    '''
    :param in_array: An array of numpy.ndarrays
    :return: Whether the number of boxes detected in all fractions are the same or not
    '''

    shapes = map(lambda x: len(x.shape) < 3, in_array)
    if any(shapes):
        return False
    else:
        n_faces_detected = list(map(lambda x: x.shape[1], in_array))

        max_ = max(n_faces_detected)
        min_ = min(n_faces_detected)

        return min_ == max_


def detect_sequence(in_array):
    '''
    :param in_array: Numpy.ndarry of shape Frames x N_Faces_detected x 4
    :return: The correct sequence of frames in case there is a mismatch on input
    '''

    assert in_array.shape[1] > 1, "The input array must contain more than one face"
    num_faces = in_array.shape[1]

    tracking = {}

    last_set_boxes = [None for _ in range(num_faces)]

    for ii, set_of_boxes in enumerate(in_array):
        for ij, box in enumerate(set_of_boxes):
            box = list(map(int, box))

            box_area = abs(box[0] - box[2]) * abs(box[1] - box[3])
            indx = ij
            if ii > 0:
                overlappings_x = [overlapping_1d(box[0], box[2], aux[0], aux[2]) for aux in last_set_boxes if aux is not None]
                overlappings_y = [overlapping_1d(box[1], box[3], aux[1], aux[3]) for aux in last_set_boxes if aux is not None]
                overlapped_areas = [x * y / (1.0 * box_area) for x, y in zip(overlappings_x, overlappings_y)]
                indx = np.argmax(overlapped_areas)

            last_set_boxes[indx] = box

        tracking.update({ii: {i:j for i, j in enumerate(last_set_boxes)}})

    return tracking


def process_fail(in_list):
    not_none_boxes = {}

    total = len(in_list)

    for i, chunk in enumerate(in_list):
        for j, set_of_boxes in enumerate(chunk):
            if set_of_boxes is not None:
                not_none_boxes[i + (j * total)] = {z : list(map(int, box)) for z, box in enumerate(set_of_boxes)}

    return not_none_boxes


def extract_original_boxes(src_dir, dst_dir=None, thresholds = [.85, .98, .99], verbose=False):

    assert os.path.exists(src_dir), f'Source dir {src_dir} does not exist.'

    if not os.path.exists(dst_dir):
        print('Destination directory does not exist. \nCreating destination directory.')
    else:
        print('Destination directory exists.')
        if not len(os.listdir(dst_dir)) == 0:
            print('It\'s not empty.')
            print('Deleting its content to avoid problems.')
            shutil.rmtree(dst_dir)

    os.makedirs(dst_dir, exist_ok=True);
    print('Destination directory created successfully')


    boxes_out = {}

    videos = json.load(open(f'{src_dir}/metadata.json'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mtcnn = MTCNN(thresholds=thresholds, device=device, keep_all=True)
    mtcnn.eval()

    real_videos = {video:videos[video] for video in videos if videos[video]['label'] == 'REAL'}

    for idx, video in enumerate(tqdm(real_videos, desc=f'Extracting boxes from real videos in {src_dir}', total=len(real_videos))):

        video_pth = os.path.join(src_dir, video)
        frames, _ , _ = read_video(video_pth, pts_unit='sec')

        with torch.no_grad():
            boxes = []
            for i in range(15):
                input = frames[i*20:(i*20)+20, :, :, :]
                aux, _ = mtcnn.detect(input.numpy())
                boxes.append(aux)

        if check_boxes_match(boxes):
            boxes = np.vstack(boxes)
            if boxes.shape[1] > 1:
                tracking = detect_sequence(boxes)
                boxes_out[video] = {'faces':tracking, 'sequential' : 1}
            else:
                boxes_out[video] = {'faces': {b:{0:list(map(int, box[0]))} for b, box in enumerate(boxes)}, 'sequential':1}
        else:
            if verbose:
                print(video_pth, 'has a mismatch in the number of faces detected during the whole video.')
            processed_fail = process_fail(boxes)
            boxes_out[video] = {'faces':processed_fail, 'sequential': 0}


    with open(f'{dst_dir}/boxes.json', 'w+') as f:
        json.dump(boxes_out, f)


if __name__ == '__main__':
    extract_original_boxes('../downloads/sample', dst_dir='sample/sample', thresholds=[.95, .98, .99], verbose=True)