from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import torch
import numpy as np
from torchvision.datasets import ImageFolder
import cv2
import matplotlib.pyplot as plt

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class TinyImageNet():
    def __init__(self, root="~/data", transform=None, device='cpu'):
        dataset = ImageFolder(root=root, transform=transform)
        self.imgs = dataset.imgs
        self.targets = dataset.targets
        self.class_to_idx = dataset.class_to_idx
        self.classes = dataset.classes
        self.samples = dataset.samples
        self.transform = transform
        self.device = device
        #self.target_transform = target_transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image_filepath, label = self.imgs[index]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"].to(self.device)
        label = torch.tensor(label).to(self.device)
        return image, label


def get_dataloader(data, cuda, sampler, batch_size=128, num_workers=4, pin_memory=True):

    dataloader_args = dict(sampler=sampler, batch_size=batch_size, num_workers=num_workers,
                           pin_memory=pin_memory) if cuda else dict(shuffle=True, batch_size=64)
    dataloader = torch.utils.data.DataLoader(data, ** dataloader_args)

    return dataloader


def get_transforms(type, norm_mean, norm_std):
    '''
    get the train and test transform by albumentations
    '''
    if type == 'train':
        return A.Compose([
            # A.PadIfNeeded(min_height=64, min_width=64,
            #               value=[0, 0, 0], always_apply=True),
            # A.RandomResizedCrop(height=64, width=64, always_apply=True),
            # A.Flip(0.5),
            # A.Cutout(num_holes=1, max_h_size=8, max_w_size=8,
            #          fill_value=0, always_apply=False, p=0.5),
            A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=255.0,),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=norm_mean, std=norm_std),
                          ToTensorV2()
                          ])


def plot_sample_images(dataloader, classes=None, ncols=5, nrows=5, fig_size=(3, 3)):
    images, targets = next(iter(dataloader))
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=fig_size)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('Sample Images in Data')
    for ax, image, target in zip(axes.flatten(), images, targets):
        ax.imshow(np.uint8(torch.Tensor.cpu(image.permute(2, 1, 0))))
        #ax.imshow()
        ax.set(title='{t}'.format(
            t=classes[target.item()]))
        ax.axis('off')
