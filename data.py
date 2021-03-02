from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import torch
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

class TinyImageNet():
    def __init__(self, root="~/data", transform=None):
        dataset = ImageFolder(root=root, transform=transform)
        self.imgs = dataset.imgs
        self.targets = dataset.targets
        self.class_to_idx = dataset.class_to_idx
        self.classes = dataset.classes
        self.samples = dataset.samples
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image_path, label = self.imgs[index][0], self.targets[index]
        pillow_image = Image.open(image_path)
        image = np.array(pillow_image)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label


def get_dataloader(data, shuffle=True, batch_size=128, num_workers=4, pin_memory=True):
    cuda = torch.cuda.is_available()

    dataloader_args = dict(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                           pin_memory=pin_memory) if cuda else dict(shuffle=True, batch_size=64)
    dataloader = torch.utils.data.DataLoader(data, ** dataloader_args)

    return dataloader


def get_transforms(norm_mean, norm_std):
    '''
    get the train and test transform by albumentations
    '''
    train_transform = A.Compose([
        A.PadIfNeeded(min_height=36, min_width=36, border_mode=4,
                      value=[0, 0, 0], always_apply=True),
        A.RandomResizedCrop(height=32, width=32, always_apply=True),
        A.Flip(0.5),
        A.Cutout(num_holes=1, max_h_size=8, max_w_size=8,
                 fill_value=0, always_apply=False, p=0.5),
        A.Normalize(mean=norm_mean, std=norm_std),
        ToTensorV2()
    ])
    test_transform = A.Compose([A.Normalize(mean=norm_mean, std=norm_std),
                                ToTensorV2()
                                ])
    return train_transform, test_transform



#def get_class(idx):
