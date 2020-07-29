import os
import json
import numpy as np
import pandas as pd
import face_alignment

from tqdm import tqdm
from PIL import Image

def extract_landmarks(directory, file_name='dataset.csv', good_file_name='finals_dataset.csv'):
    # Read the boxes.json
    # Retrieve and read the face from /faces/
    # perform landmarks detection and save them in landmarks.json with

    #{'original_video_name': {{'{frame_id}.png' : coordinates}, ...}}}

    assert os.path.exists(os.path.join(directory, file_name)), str(file_name)+' does not exist.'

    #videos = json.load(open(f'{directory}/{file_name}'))

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    df = pd.read_csv(f'{directory}/{file_name}')
    data = [list(x) for x in df.values]

    originals = [row for row in data if row[-1] == 0]

    landmarks_out = {}

    rows_with_landmarks = []

    for (folder, video, path, frame, n_face, original, label) in tqdm(originals):

        #read_image and extract landmarks
        id_ = os.path.basename(path)

        face = Image.open(f'{directory}/faces/{path}')

        landmarks = fa.get_landmarks(np.array(face))
        if landmarks is None:
            print(video, path)
            continue

        landmarks = landmarks[0].tolist()

        if video in landmarks_out:
            landmarks_out[video].update({id_ : landmarks})
        else:
            landmarks_out.update({video:{id_: landmarks}})

        rows_with_landmarks.append((folder, video, path, frame, n_face, original, label))

    json.dump(landmarks_out, open(f'{directory}/landmarks.json', 'w+'))
    df_out = pd.DataFrame(rows_with_landmarks, columns=('folder', 'video', 'path', 'frame', 'n_face', 'original', 'label'))
    df_out.to_csv(f'{directory}/{good_file_name}', index=False)


if __name__ == '__main__':
    extract_landmarks('sample/test1')
