import torch, os, math
import random
from PIL import Image
import torchvision.transforms as transforms
import csv
import torchvision
import random
from PIL import Image, ImageDraw
import dlib
import numpy as np
from skimage.metrics import structural_similarity as ssim
from albumentations import *
import cv2


regions = {  'left-face'     :   [0, 1, 2, 3, 4, 5, 6, 7, 8, 30, 29, 28, 27, 21, 20, 19, 18, 17],
             'right-face'    :   [16, 15, 14, 13, 12, 11, 10, 9, 8, 30, 29, 28, 27, 22, 23, 24, 25, 26],
             'bottom-face'   :   [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
             'nose'          :   [31, 32, 33, 34, 35, 23, 21],
             'eyes'          :   None,
             'mouth'         :   None,
             'top'           :   None,
             'bottom'        :   None,
             'right'         :   None,
             'left'          :   None,

         }

T = torchvision.transforms.ToTensor()


class ValidationDataset(torch.utils.data.Dataset):

    def __init__(self, path, path_data):
        super(ValidationDataset).__init__()
        f_p = os.path.abspath(path)
        assert os.path.exists(f_p), "The directory does not exist"
        f_p_data = os.path.abspath(path_data)
        assert os.path.exists(f_p_data), "The directory does not exist"
        self.f_p = f_p
        self.f_p_data = f_p_data
        self.augment = create_train_transforms()

        with open(self.f_p) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            rows = [row for row in reader][1:]
            self.rows = rows
    
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        video, frame, original_frame, n_face, order, lands, label = self.rows[idx]
        frame = Image.open(frame)
        return T(frame), torch.tensor(int(label))
    

class AugmentedDataset(torch.utils.data.Dataset):

    def __init__(self, path, path_data):
        super(AugmentedDataset).__init__()
        f_p = os.path.abspath(path)
        assert os.path.exists(f_p), "The directory does not exist"
        f_p_data = os.path.abspath(path_data)
        assert os.path.exists(f_p_data), "The directory does not exist"
        self.f_p = f_p
        self.f_p_data = f_p_data
        self.augment = create_train_transforms()
        
        with open(self.f_p) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            rows = [row for row in reader][1:]
            self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        video, frame, original_frame, n_face, order, lands, label = self.rows[idx]
        res = oclude_frame(f'{self.f_p_data}/{frame}', f'{self.f_p_data}/{original_frame}', float(label), eval(lands))
        if random.random() > 0.5:
            res = self.augment(image=res)['image']
        return T(res), torch.tensor(int(label))
    
    
def oclude_frame(frame, frame_original, label, lands, p=0.5):
    
    if label > 0.5:
        if random.random() > 0.5:
            res = np.array(make_occlusion_fake(frame, frame_original, lands))
        else:
            res = center_image(np.array(Image.open(frame)), lands)
    else:
        if random.random() > 0.5:
            res = np.array(make_occlusion_original(frame, lands))
        else:
            res = center_image(np.array(Image.open(frame)), lands)
    
    return res
    
def create_train_transforms(size=224):
    # Augmentations with albumentations
    return Compose([
       ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
       GaussNoise(p=0.1),
       GaussianBlur(blur_limit=3, p=0.05),
       HorizontalFlip(),
       #IsotropicResize(max_side=size),
       PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
       OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
       ToGray(p=0.2),
       ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ])

def oclude(lands, size=224):
    
    part = random.choice(list(regions.keys()))
    
    poly = None
    
    if part == 'eyes':
        poly = [(lands[36][0]-10, lands[36][1] + 20), (lands[36][0]-10, lands[36][1] - 20), (lands[45][0]+10, lands[45][1] - 20), (lands[45][0]+10, lands[45][1] + 20)]
    elif part == 'mouth':
        poly = [(lands[48][0]-5, lands[48][1] + 15), (lands[48][0]-5, lands[48][1] - 15), (lands[54][0]+5, lands[54][1] - 15), (lands[54][0]+5, lands[54][1] + 15)]
    elif part == 'top':
        y = max((lands[29][1], lands[41][1], lands[46][1]))
        poly = [(0, 0), (size, 0), (size, y), (0, y)]
    elif part == 'bottom':
        y = max((lands[29][1], lands[41][1], lands[46][1]))
        poly = [(0, y), (size, y), (size, size), (0, size)]
    elif part == 'left':
        x = lands[30][0]
        poly = [(0, 0), (x, 0), (x, size), (0, size)]
    elif part == 'right':
        x = lands[30][0]
        poly = [(x, 0), (size, 0), (size, size), (x, size)]
    else:
        poly = [tuple(lands[v]) for v in regions[part]]
        
    return part, poly
    

def make_occlusion_fake(frame, original, lands):
    
    av = True
    done = []
    

    forged = Image.open(frame)
    pristine = Image.open(original)
    
    pristine_np = np.array(pristine)
    forged_np = np.array(forged)
    
    score, score_mask = ssim(pristine_np, forged_np, multichannel=True, full=True)
    diff = ((1 - score_mask) * 255).astype(np.uint8)

    poly_ = None
    
    while av:
        
        diff_im = Image.fromarray(diff)
        draw = ImageDraw.Draw(diff_im)
    
        part, poly = oclude(lands)
        
        if part in done:
            continue
        done.append(part)
        if len(done) == len(regions): av = False
        
        draw.polygon(poly, fill='black')
        
        mask_edited = np.array(diff_im)

        mean_original = np.mean(diff)
        mean_edited = np.mean(mask_edited)
                
        if mean_edited < mean_original * 0.75:
            #refuse change
            continue
        else:
            poly_ = poly
            av = False
    
    if poly_ is not None:
        draw = ImageDraw.Draw(forged)
        draw.polygon(poly_, fill='black')
        
    pic = T(forged) * 255
    pic = pic.permute(1,2,0).to(torch.uint8).numpy()
    
    return center_image(pic, lands)

def make_occlusion_original(frame, lands):
    
    img = Image.open(frame)
    
    draw = ImageDraw.Draw(img)

    _, poly = oclude(lands)
        
    draw.polygon(poly, fill='black')
    pic = T(img) * 255
    pic = pic.permute(1,2,0).to(torch.uint8).numpy()
    
    return center_image(pic, lands)
    
    
def center_image(image, lands):
    '''Input: Image in numpy and points'''
    box = dlib.rectangle(0,0,224,224)
    f_o_ds = dlib.full_object_detections()
    points = [dlib.point(np.array(point)) for point in lands]
    f_o_d = dlib.full_object_detection(box, points)
    f_o_ds.append(f_o_d)
    images = dlib.get_face_chips(image, f_o_ds, size=224, padding=0.4)
    return images[0]       
    

def format_row(item, f_p_data):
    img = Image.open(f'{f_p_data}/{item[-2]}')
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
    # transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    transform = transforms.ToTensor()
    # TODO: Should normalization be removed?
    # If needed for regression the negative label can be changed here instead of in the middle of the running code
    label = int(item[-3])
    return transform(img), torch.tensor(int(label))