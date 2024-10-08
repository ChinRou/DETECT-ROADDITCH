import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils

def multiClassToVisualize(image):
    output = np.array(image.shape)
    output = np.argmax(image, 0)
    return output

class Dataset(BaseDataset):
    CLASSES = ['green', 'yellow', 'blue', 'orange', 'red']#改

    def __init__(
           self,
           images_dir,
           masks_dir,
           classes=None,
           augmentation=None,
           preprocessing=None,
    ):
       self.ids = os.listdir(images_dir)
       if '.DS_Store' in self.ids: self.ids.remove('.DS_Store')
       self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
       self.masks_fps = [os.path.join(masks_dir, image_id.split('.')[0] + '.png') for image_id in self.ids]

       # convert str names to class values on masks
       self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
       self.augmentation = augmentation
       self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # h, w = image.shape[:2]
        # image = cv2.resize(image, (int(h/4), int(w/4)), interpolation=cv2.INTER_NEAREST)
        mask = cv2.imread(self.masks_fps[i], 0)
        # mask = cv2.resize(mask, (int(h/4), int(w/4)), interpolation=cv2.INTER_NEAREST)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask ==  v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentation
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask
    
    def __len__(self):
        return len(self.ids)

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        # albu.PadIfNeeded(min_height=960, min_width=960, always_apply=True, border_mode=0),
        albu.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, always_apply=True, border_mode=0),
        # albu.RandomCrop(height=960, width=960, always_apply=True),

         albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        )

    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [albu.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, always_apply=True)]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # force_cudnn_initialization()
    torch.cuda.empty_cache()
    # torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    DATA_DIR = './dataset'
    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')
    
    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')
    
    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'testannot')

    ENCODER = 'resnext101_32x4d'
    ENCODER_WEIGHTS = 'swsl'
    CLASSES = ['green', 'yellow', 'blue', 'orange', 'red']# 改
    ACTIVATION = 'softmax2d'
    DEVICE = 'cuda'

    model = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    loss = utils.losses.CrossEntropyLoss()
    metrics = [
        utils.metrics.IoU(threshold=0.5)
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    train_epoch = utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )


    valid_epoch = utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    max_score = 0
    epoch = 50
    train_iou_score = []
    train_loss = []
    val_iou_score = []
    val_loss = []
    for i in range(0, epoch):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_iou_score.append(train_logs['iou_score'])
        train_loss.append(train_logs['cross_entropy_loss'])
        val_iou_score.append(valid_logs['iou_score'])
        val_loss.append(valid_logs['cross_entropy_loss'])
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')
