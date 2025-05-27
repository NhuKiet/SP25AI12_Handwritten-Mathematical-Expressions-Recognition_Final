import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomAugmentation:
    def __init__(self, is_train: bool):
        def erode(image, **kwargs):
            kernel = np.ones((3, 3), np.uint8)
            return cv2.erode(image, kernel, iterations=1)

        def dilate(image, **kwargs):
            kernel = np.ones((3, 3), np.uint8)
            return cv2.dilate(image, kernel, iterations=1)

        if is_train:
            self.transform = A.Compose([
                A.Resize(height=256, width=256, p=1.0),

                A.Solarize(p=0.5),

                A.OneOf([
                    A.Lambda(image=erode, p=1),
                    A.Lambda(image=dilate, p=1),
                ], p=0.5),

                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=256, width=256, p=1.0),
                ToTensorV2(),
            ])

    def __call__(self, img):
        img = np.array(img)
        transformed = self.transform(image=img)
        return transformed['image']