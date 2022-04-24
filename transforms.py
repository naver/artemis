#############################################
## Artemis                                 ##
## Copyright (c) 2022-present NAVER Corp.  ##
## CC BY-NC-SA 4.0                         ##
#############################################
import torchvision.transforms as transforms
from PIL import ImageOps

def get_transform(opt, phase):
    # first standard transformations
    t_list = [PadSquare(), transforms.Resize(256)]
    # phase-specific transformations
    if phase == 'train':
        # include data augmentation
        t_list += [transforms.RandomHorizontalFlip(), transforms.RandomCrop(opt.crop_size)]
    else: # evaluate
        t_list += [transforms.CenterCrop(opt.crop_size)]
    # standard transformations
    t_list += [transforms.ToTensor(), Normalizer()]
    return MyTransforms(t_list)


class PadSquare(object):
    """
    Pad the image with white pixel until it has the form of a square. The amount
    of added white pixels is the same at the left & right, and at the top &
    bottom.
    
    Input & Output: PIL image.
    """
    def __call__(self, img):

        w, h = img.size
        if w > h:
            delta = w - h
            padding = (0, delta//2, 0, delta - delta//2)
            img = ImageOps.expand(img, padding, (255, 255, 255))
        elif w < h:
            delta = h - w
            padding = (delta//2, 0, delta - delta//2, 0)
            img = ImageOps.expand(img, padding, (255, 255, 255))
        return img


def Normalizer():
    """
    Normalize pixels of a PIL Image according to the mean and std of the
    ImageNet pixels.
    """
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

class MyTransforms(object):

    def __init__(self, trfs_list):
        self.transform = transforms.Compose(trfs_list)

    def __call__(self, x):
        y = self.transform(x)
        return y