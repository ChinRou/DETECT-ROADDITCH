import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import torch
from smp import Dataset, get_validation_augmentation, get_preprocessing
import segmentation_models_pytorch as smp
from matplotlib import colors
from segmentation_models_pytorch import utils

def visualize(file_name=None, **images):
    """Plot images in one row."""
    cmap = colors.ListedColormap(['red', 'green'])
    n = len(images)
    plt.figure(figsize=(7, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap=cmap)
    if file_name != None:
        plt.savefig(f'{file_name}')
    plt.show()

def multiClassToVisualize(image):
    output = np.array(image.shape)
    output = np.argmax(image, 0)
    return output

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.backends.cudnn.benchmark = False

    DATA_DIR = './dataset'
     
    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'testannot')


    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['green', 'yellow', 'blue', 'orange', 'red']
    ACTIVATION = 'sigmoid'
    DEVICE = 'cuda'

    loss = utils.losses.CrossEntropyLoss()
    metrics = [
        utils.metrics.IoU(threshold=0.5)
    ]

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    best_model = torch.load('best_model.pth')

    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    test_dataloader = DataLoader(test_dataset, shuffle=False)

    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    test_dataset_vis = Dataset(
        x_test_dir, y_test_dir,
        classes=CLASSES,
    )
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(test_dataset)):
        # n = np.random.choice(len(test_dataset))
        image_vis = test_dataset_vis[i][0].astype('uint8')
        image, gt_mask = test_dataset[i]
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        #
        predicted_mask = multiClassToVisualize(pr_mask)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(test_dataset.images_fps[i]).split('.')[0] + '.png'), predicted_mask)
        """
        visualize(
            file_name=os.path.join(output_dir, os.path.basename(test_dataset.images_fps[i]).split('.')[0] + '.png'),
            image=image_vis,
            ground_truth_mask=multiClassToVisualize(gt_mask),
            prdicted_mask=multiClassToVisualize(pr_mask)
        )
        """
