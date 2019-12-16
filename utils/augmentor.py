import numpy as np
from PIL import Image


class Augmenter:

    def __init__(self, seq):
        self.seq = seq

    def __call__(self, img):
        seq_det = self.seq.to_deterministic()

        image = img
        image = np.array(image)
        image_aug = seq_det.augment_images([image])[0]
        image_aug = Image.fromarray(image_aug)

        return image_aug
