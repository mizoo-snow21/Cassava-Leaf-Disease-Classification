# Cassava-Leaf-Disease-Classification

https://www.kaggle.com/c/cassava-leaf-disease-classification/leaderboard   
We got 26th prize.  
This is summary and codes.

# Summary
![image](https://user-images.githubusercontent.com/55850618/124774601-e7070a80-df78-11eb-8571-72bd3a390282.png)

# Model
## vit_base_patch16_384
・img_size = 384 x 384  
・4x TTA  

## efficientnet_b4_ns
・img_size = 512 x 512  
・4x TTA  

## resnest50d_4s2x40d
・img_size = 512 x 512  
・4x TTA  

## Weighted_Averaging
・Pulic Score 0.9063  
・Private Score 0.9010  

# Some Settings
・5fold StratifiedKFold  
・Using 2020 data  

## Augmentaion
```
def get_train_transforms():
    return Compose([
            RandomResizedCrop(CFG['img_size'], CFG['img_size']),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_valid_transforms():
    return Compose([
            CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
            Resize(CFG['img_size'], CFG['img_size']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)
```
## Loss
TaylorCrossEntropyLoss

## LR Scheduler
CosineAnnealingWarmRestarts


