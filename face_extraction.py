import os, json, torch, torchvision, shutil
from facenet_pytorch import MTCNN
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def overlapping(low_1, high_1, low_2, high_2):
    return max(0, min(high_1, high_2) - max(low_1, low_2))
    

def extract_faces_from_dir(src_dir, dst_dir):
    
    # Init face detection net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    mtcnn = MTCNN(image_size=224, thresholds=[.98, .98, .99], margin=30, post_process=False, select_largest=True, keep_all=True, device=device)
    
    to_pil_im = torchvision.transforms.ToPILImage()
    resize = torchvision.transforms.Resize((224, 224))   

    assert os.path.exists(src_dir), f'Source dir {src_dir} does not exist'
    if os.path.exists(dst_dir):
        print('Destination directory exists, removing it to avoid problems.')
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True); print('Destination directory created successfully')

    src_fp = os.path.abspath(src_dir)
    dst_fp = os.path.abspath(dst_dir)

    metadata = open(f'{src_fp}/metadata.json')
    videos = json.load(metadata)
    
    metadata_out = {}

    for video in tqdm(videos):

        video_name = video.split('.')[0]
        label = 0 if videos[video]['label'] == 'REAL' else 1

        video_path = f'{src_fp}/{video}'
        vframes, _, info = torchvision.io.read_video(video_path, start_pts=2, end_pts=3.02, pts_unit='sec')

        # First detected surrounding boxes as well as probabilities and landmarks points.
        with torch.no_grad():
            all_boxes, probs, all_points = mtcnn.detect(vframes.numpy(), landmarks=True)
        
        all_boxes = np.array([v for v in all_boxes if v is not None])
        all_points = np.array([p for p in all_points if p is not None])
        
        num_faces = max([v.shape[0] for v in all_boxes if v is not None] + [0])

        # Remove None elements and calculate the max number of faces detected in frames.       
        if len(all_boxes.shape) < 3 or all_boxes.shape[0] < 30 or num_faces == 0:
            continue

        # Scale factor
        scale_factor = 0.25

        # x and y lengths
        max_x = vframes[0].size()[0]
        max_y = vframes[0].size()[1]

        # create the dict that stores the images
        faces_in_video = {x: [] for x in range(num_faces)}

        # auxiliary list for box ordering
        last_set_boxes = [None for _ in range(num_faces)]

        # iterate over the frames, extract sequences of images and save.
        for ii, (boxes, points, frame) in enumerate(zip(all_boxes, all_points, vframes)):  
                        
            for ij, (box, point) in enumerate(zip(boxes, points)):

                box = list(map(int, box))
                landmarks = [(int(p[0] - box[0]), int(p[1] - box[1])) for p in point]

                len_x = abs(box[1] - box[3])
                len_y = abs(box[0] - box[2])

                scale_x = int(len_x * scale_factor // 2)
                scale_y = int(len_y * scale_factor // 2)

                face = frame[max(0, box[1]-scale_x):min(box[3]+scale_x, max_x), max(0,box[0]-scale_y):min(max_y, box[2]+scale_y), :]
                face = to_pil_im(face.permute(2, 0, 1))
                face = resize(face)

                box_area = abs(box[0] - box[2]) * abs(box[1] - box[3])

                # Do the tracking to correcly construct the sequences            
                indx = ij
                if ii > 0:

                    overlappings_x = [overlapping(box[0], box[2], b[0], b[2]) for b in last_set_boxes if b is not None]
                    overlappings_y = [overlapping(box[1], box[3], b[1], b[3]) for b in last_set_boxes if b is not None]

                    overlapped_areas = [x*y / (1.0*box_area) for x,y in zip(overlappings_x, overlappings_y)]

                    indx = np.argmax(overlapped_areas)

                last_set_boxes[indx] = box
                faces_in_video[indx].append(face)

        cnt = 0

        out_pic_list = {}

        for key in faces_in_video:
            if len(faces_in_video[key]) < 30:
                continue
            else:
                aux_list = []
                for idx, face in enumerate(faces_in_video[key]):
                    file_name = f'{video_name}_{cnt}_{idx}.png'
                    aux_list.append(file_name)
                    face.save(f'{dst_fp}/{file_name}')
                out_pic_list[cnt] = aux_list
                cnt+=1
                
        
        metadata_out[video] = {'label':label, 'num_faces':cnt, 'faces':out_pic_list}
                
    json.dump(metadata_out, open(f'{dst_dir}/metadata.json', 'w+'))                

    del mtcnn

if __name__ == '__main__':
    
    for i in range(0,10):
        extract_faces_from_dir(f'data/dfdc_train_part_{i}', f'data/faces/chunk{i}')
