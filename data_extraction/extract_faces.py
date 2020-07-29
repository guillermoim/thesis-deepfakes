import os
import random
import json
import torchvision
import pandas as pd

from tqdm import tqdm
from torchvision.transforms import ToPILImage


to_img = ToPILImage()

def get_original_fakes_dict(path_to_metadata):
    videos = json.load(open(path_to_metadata))

    res_videos = {video: [] for video in videos if videos[video]['label'] == 'REAL'}

    for video in videos:
        if videos[video]['label'] == 'FAKE':
            original = videos[video]['original']
            if not original in res_videos:
                continue
            res_videos.update({original: res_videos[original] + [video]})


    return res_videos


def process_video(video_path, bounding_boxes, video_name, dst_dir, idxs=None, slide=10):

    #TODO: Maybe it is better to put frames in folders with the video's name
    vframes, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
    n_frames = vframes.shape[0]

    if idxs is None:

        if len(bounding_boxes) > 299:
            idxs = range(0, n_frames, slide)
        else:
            aux = min(len(bounding_boxes), 30)
            idxs = random.sample(list(bounding_boxes), aux)

    imgs_saved = []

    for i in idxs:

        frame = vframes[int(i),:,:,:]

        max_x = frame.shape[0]
        max_y = frame.shape[1]

        for j, box in enumerate(bounding_boxes[str(i)]):

            # get the bounding box
            y0, x0, y1, x1 = bounding_boxes[str(i)][box]

            h = abs(x1 - x0)
            w = abs(y1 - y0)

            scale_y = w // (3)
            scale_x = h // (3)

            # Scale the box
            x_min, x_max, y_min, y_max = max(0, x0-scale_x), min(x1+scale_x, max_x), max(0,y0-scale_y), min(max_y, y1+scale_y)

            face = frame[x_min:x_max, y_min:y_max, :]
            path_to_save_img = f'{video_name}/{int(i):03d}_{j}.png'
            #append path_to_imgs, frame, n_face
            imgs_saved.append((f'{path_to_save_img}', i, j))
            face_img = to_img(face.permute(2,0,1))
            face_img.save(f'{dst_dir}/{path_to_save_img}')

    return idxs, imgs_saved

def extract_faces_from_dir(src_dir, dst_dir, file='boxes.json'):

    folder_name = os.path.basename(src_dir)

    assert os.path.exists(f'{dst_dir}'), 'Destination directory does not exist.'
    assert os.path.exists(f'{dst_dir}/{file}'), 'There is no such file in the destination directory'
    os.makedirs(f'{dst_dir}/faces', exist_ok=True)
    faces_dir = f'{dst_dir}/faces'

    videos = json.load(open(f'{src_dir}/metadata.json'))
    boxes = json.load(open(f'{dst_dir}/{file}'))

    real_videos = {video : videos[video] for video in videos if videos[video]['label'] == 'REAL' and video in boxes}

    fakes_by_original = get_original_fakes_dict(f'{src_dir}/metadata.json')

    pbar = tqdm(fakes_by_original, total=len(real_videos))

    rows_csv_file = []

    for video_k in pbar:

        if not video_k in ('chviwxsfhg.mp4', 'ehccixxzoe.mp4'):
            continue


        pbar.set_description(f'Extracting faces from original video {video_k}')
        video_name = video_k.split('.')[0]
        video_path = os.path.join(src_dir, video_k)

        if video_k not in boxes:
            continue

        os.makedirs(f'{dst_dir}/faces/{video_name}', exist_ok=True)

        bboxes = boxes[video_k]['faces']

        idxs, imgs_saved = process_video(video_path, bboxes, video_name, faces_dir)

        rows = [(folder_name, video_k, img_path, n_frame, n_face, '', 0) for (img_path, n_frame, n_face) in imgs_saved]

        rows_csv_file.extend(rows)

        for fake_video_k in fakes_by_original[video_k]:
            video_name = fake_video_k.split('.')[0]
            os.makedirs(f'{dst_dir}/faces/{video_name}', exist_ok=True)
            video_path = os.path.join(src_dir, fake_video_k)
            _, imgs_saved = process_video(video_path, bboxes, video_name, faces_dir, idxs=idxs)
            rows = [(folder_name, fake_video_k, img_path, n_frame, n_face, video_k, 1) for (img_path, n_frame, n_face) in imgs_saved]
            rows_csv_file.extend(rows)

        df = pd.DataFrame(rows_csv_file, columns=('folder', 'video', 'path', 'frame', 'n_face', 'original', 'label'))
        df.to_csv(f'{dst_dir}/dataset.csv', index=False)


def main():
    src_dir = '../downloads/sample'
    dst_dir = 'sample/test1'
    extract_faces_from_dir(src_dir, dst_dir, file='boxes.json')
    #print(get_original_fakes_dict('../downloads/dfdc_train_part_25/metadata.json'))

if __name__ == '__main__':

    main()




