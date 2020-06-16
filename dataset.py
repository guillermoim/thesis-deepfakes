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


regions = {  'left'     :   [0, 1, 2, 3, 4, 5, 6, 7, 8, 30, 29, 28, 27, 21, 20, 19, 18, 17],
             'right'    :   [16, 15, 14, 13, 12, 11, 10, 9, 8, 30, 29, 28, 27, 22, 23, 24, 25, 26],
             'bottom'   :   [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
             'nose'     :   [31, 32, 33, 34, 35, 23, 21],
             'eyes'     :   None,
             'mouth'    :   None,
             'top'      :   None,
         }

T = torchvision.transforms.ToTensor()

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        elif self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class RandomDataset(torch.utils.data.IterableDataset):

    def __init__(self, total):
        super(RandomDataset).__init__()
        assert total > 0, ' The total number must be greater than 0.'
        self.start = 0
        self.end = total

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator in a worker process
            iter_start = self.start
            iter_end = self.end
        else:
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        return iter([(torch.rand((3, 224, 224)), torch.randint(low=0, high=2, size=(1,)).squeeze(0).long()) for _ in range(iter_start, iter_end)])


class BalancedClusterDataset(torch.utils.data.IterableDataset):

    def __init__(self, path):
        super(BalancedClusterDataset).__init__()
        f_p = os.path.abspath(path)
        assert os.path.exists(f_p), "The directory does not exist"
        self.path = f_p

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        with open(self.path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            faces = [row for row in reader][1:]
            
        if worker_info is None:
            iter_start = 0
            iter_end = len(faces) - 1
        else:
            per_worker = int(math.ceil((len(faces) - 0) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(faces) - 1)

        return map(lambda x: format_row(x), faces[iter_start:iter_end])
    

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
        return T(res), torch.tensor(float(label))
    
    
def oclude_frame(frame, frame_original, label, lands, p=0.5):
    res = None
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
    
        part = random.choice(list(regions.keys()))
        
        if part in done:
            continue
        done.append(part)
        if len(done) == len(regions): av = False
        # TODO: Wrap this in another function shared with original one!
        if part == 'eyes':
            poly = [(lands[36][0]-10, lands[36][1] + 20), (lands[36][0]-10, lands[36][1] - 20), (lands[45][0]+10, lands[45][1] - 20), (lands[45][0]+10, lands[45][1] + 20)]
        elif part == 'mouth':
            poly = [(lands[48][0]-5, lands[48][1] + 15), (lands[48][0]-5, lands[48][1] - 15), (lands[54][0]+5, lands[54][1] - 15), (lands[54][0]+5, lands[54][1] + 15)]
        elif part == 'top':
            y = max((lands[29][1], lands[41][1], lands[46][1]))
            poly = [(0, 0), (224, 0), (224, y), (0, y)]
        else:
            poly = [tuple(lands[v]) for v in regions[part]]
        
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

    part = random.choice(list(regions.keys()))
    
    if part == 'eyes':
        poly = [(lands[36][0]-10, lands[36][1] + 20), (lands[36][0]-10, lands[36][1] - 20), (lands[45][0]+10, lands[45][1] - 20), (lands[45][0]+10, lands[45][1] + 20)]
    elif part == 'mouth':
        poly = [(lands[48][0]-5, lands[48][1] + 15), (lands[48][0]-5, lands[48][1] - 15), (lands[54][0]+5, lands[54][1] - 15), (lands[54][0]+5, lands[54][1] + 15)]
    elif part == 'top':
        y = max((lands[29][1], lands[41][1], lands[46][1]))
        poly = [(0, 0), (224, 0), (224, y), (0, y)]
    else:
        poly = [tuple(lands[v]) for v in regions[part]]
        
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