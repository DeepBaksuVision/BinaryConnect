import json
import cv2
from matplotlib import pyplot as plt
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)


def aug(config:{}, p=0.5):
    return Compose([
        HorizontalFlip(True),
        RandomRotate90(True),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=config["MotionBlur"]),
            MedianBlur(blur_limit=config["blur_limit"], p=0.1),
            Blur(blur_limit=config["blur_limit"], p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=config["shift_limit"], scale_limit=config['scale_limit'],
                         rotate_limit=config["rotate_limit"], p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=config["clip_limit"]),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=config["HueSaturationValue"]),
    ], p=p)


def show_img(img, figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(img)
    plt.imshow(img)


# with open('aug_config.json') as json_file:
#     json_data = json.load(json_file)
# print(json_data['MotionBlur'])

if __name__ == "__main__":
    image = cv2.imread('dog.12473.jpg')
    config = {
      "MotionBlur": 0.2,
      "blur_limit": 3,
      "MedianBlur": 0.1,
      "shift_limit": 0.0625,
      "scale_limit": 0.2,
      "rotate_limit": 45,
      "clip_limit": 2,
      "HueSaturationValue": 0.3
        }
    augmentation = aug(config)
    data = {'image': image}
    augmented = augmentation(**data)
    image = augmented['image']
    show_img(image)