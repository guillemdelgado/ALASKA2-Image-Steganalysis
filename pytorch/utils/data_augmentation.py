from albumentations import VerticalFlip, HorizontalFlip, Compose
from albumentations.pytorch.transforms import ToTensorV2

def get_transforms():
    AUGMENTATIONS_TRAIN = Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        ToTensorV2(p=1.0)
    ], p=1)


    AUGMENTATIONS_TEST = Compose([
        ToTensorV2(p=1.0)
    ], p=1)

    return AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST
