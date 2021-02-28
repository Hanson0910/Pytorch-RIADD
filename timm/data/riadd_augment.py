import albumentations
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def crop_maskImg(image, sigmaX=10):
    image = crop_image_from_gray(image)
    #image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image

def get_riadd_train_transforms(args):
    image_size = args.img_size
    transforms_train = albumentations.Compose([
        #albumentations.RandomResizedCrop(image_size, image_size, scale=(0.85, 1), p=1), 
        albumentations.Resize(image_size, image_size), 
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.MedianBlur(blur_limit = 7, p=0.3),
        albumentations.IAAAdditiveGaussianNoise(scale = (0,0.15*255), p = 0.5),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
        albumentations.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.3),
        albumentations.Cutout(max_h_size=20, max_w_size=20, num_holes=5, p=0.5),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    return transforms_train


def get_riadd_valid_transforms(args):
    image_size = args.img_size
    valid_transforms = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    return valid_transforms


def get_riadd_test_transforms(args):
    image_size = args['img_size']
    test_transforms = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
        albumentations.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.5),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    return test_transforms


# if __name__ == '__main__':
#     img = cv2.imread('/media/ExtDiskB/Hanson/datasets/wheat/RIADD/valid/1.png')
#     img1 = preprocessing(img)
#     # result=color_seperate(hsv_img, thresh_image)
#     cv2.imwrite('1222.png',img1)