import math
import random
import cv2
import dlib
import numpy as np
import skimage.draw

from skimage import measure
from albumentations import DualTransform, ImageOnlyTransform
from scipy.ndimage import binary_erosion, binary_dilation

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('libs/shape_predictor_68_face_landmarks.dat')


def prepare_bit_masks(mask):
    h, w = mask.shape
    mid_w = w // 2
    mid_h = w // 2
    masks = []
    ones = np.ones_like(mask)
    ones[:mid_h] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[mid_h:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, :mid_w] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, :mid_w] = 0
    ones[mid_h:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, mid_w:] = 0
    ones[mid_h:, :mid_w] = 0
    masks.append(ones)
    return masks


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def remove_eyes(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_nose(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2]
    x3, y3 = landmarks[2]
    mask = np.zeros_like(image[..., 0])
    x4 = int((x1 + x2) / 2)
    y4 = int((y1 + y2) / 2)
    line = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_mouth(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[-2:]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 3)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_landmark(image, landmarks):
    if random.random() > 0.5:
        image = remove_eyes(image, landmarks)
    elif random.random() > 0.5:
        image = remove_mouth(image, landmarks)
    elif random.random() > 0.5:
        image = remove_nose(image, landmarks)
    return image

def blackout_convex_hull(image):

    img = image.copy()

    try:
        rect = detector(img)[0]
        sp = predictor(img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])
        outline = landmarks[[*range(17), *range(26, 16, -1)]]
        Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])
        cropped_img = np.zeros(img.shape[:2], dtype=np.uint8)
        cropped_img[Y, X] = 1

        y, x = measure.centroid(cropped_img)
        y = int(y)
        x = int(x)
        first = random.random() > 0.5
        if random.random() > 0.5:
            if first:
                cropped_img[:y, :] = 0
            else:
                cropped_img[y:, :] = 0
        else:
            if first:
                cropped_img[:, :x] = 0
            else:
                cropped_img[:, x:] = 0

        img[cropped_img > 0] = 0

    except Exception as e:
        img = image
        pass

    return img

def prepare_bit_masks(mask):
    h, w = mask.shape
    mid_w = w // 2
    mid_h = w // 2
    masks = []
    ones = np.ones_like(mask)
    ones[:mid_h] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[mid_h:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, :mid_w] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, :mid_w] = 0
    ones[mid_h:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, mid_w:] = 0
    ones[mid_h:, :mid_w] = 0
    masks.append(ones)
    return masks

def blackout_cutout(img, mask, label):

    image = img.copy()

    binary_mask = mask > 0.4 * 255

    masks = prepare_bit_masks((binary_mask * 1).astype(np.uint8))
    tries = 6
    current_try = 1
    while current_try < tries:
        bitmap_msk = random.choice(masks)
        if label < 0.5 or np.count_nonzero(mask * bitmap_msk) > 20:
            mask *= bitmap_msk
            image *= np.expand_dims(bitmap_msk, axis=-1)
            break
        current_try += 1

    return image

class RemoveLandmark(ImageOnlyTransform):

    def __init__(self, always_apply=False, p=0.7):
        super().__init__(always_apply, p)

    def apply(self, image, landmarks, **params):
        if landmarks is not None:
            return remove_landmark(image, landmarks)
        else:
            return image

    @property
    def targets_as_params(self):
        return ['image', 'landmarks']

    def get_params_dependent_on_targets(self, params):
        lands = params['landmarks']

        return {'landmarks': lands}

    def get_transform_init_args_names(self):
        return ()


class BlackoutConvexHull(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.2):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        return blackout_convex_hull(img)

    def get_transform_init_args_names(self):
        return ()


class RandomCutout(ImageOnlyTransform):

    def __init__(self, always_apply=False, p=0.1):
        super().__init__(always_apply, p)

    def apply(self, image, mask, label, **params):
        return blackout_cutout(image, mask, label)

    @property
    def targets_as_params(self):
        return ['image', 'mask', 'label']

    def get_params_dependent_on_targets(self, params):
        mask = params['mask']
        label = params['label']

        return {'mask': mask, 'label': label}

    def get_transform_init_args_names(self):
        return ()


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down, interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")