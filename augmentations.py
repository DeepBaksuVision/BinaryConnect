import numpy as np
import cv2
from matplotlib import pyplot as plt
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)


def aug(p:dict):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=p["p0"]),
        OneOf([
            MotionBlur(p=p["MotionBlur"]),
            MedianBlur(blur_limit=3, p=p["MedianBlur"]),
            Blur(blur_limit=3, p=p["Blur"]),
        ], p=p["p1"]),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=p["ShiftScaleRotate"]),
        OneOf([
            OpticalDistortion(p=p["OpticalDistortion"]),
            GridDistortion(p=p["GridDistortion"]),
            IAAPiecewiseAffine(p=p["IAAPiecewiseAffine"]),
        ], p=p["p2"]),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=p["p3"]),
        HueSaturationValue(p=p["HueSaturationValue"]),
    ], p=p['Compose'])


def show_img(img, figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(img)
    plt.imshow(img)


if __name__ == "__main__":

    image = cv2.imread('images.jpeg')
    augmentation = aug({"p0": 0.2, "MotionBlur": 0.2, "MedianBlur": 0.1, "Blur": 0.1, "p1": 0.2, "ShiftScaleRotate": 0.2,
                        "OpticalDistortion": 0.3, "GridDistortion": 0.1, "IAAPiecewiseAffine": 0.3, "p2": 0.2, "p3": 0.3,
                        "HueSaturationValue": 0.3, "Compose": 0.5})
    data = {'image': image}
    augmented = augmentation(**data)
    image = augmented['image']
    show_img(image)