import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from utils.augment import RandAugment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

_normalize = transforms.Normalize(
    mean=imagenet_mean, std=imagenet_std)
_inv_normalize = transforms.Normalize(
    mean= [-m/s for m, s in zip(imagenet_mean, imagenet_std)],
    std= [1/s for s in imagenet_std])
def normalize(x):
    return _normalize(x)
def inv_normalize(x):
    return _inv_normalize(x)

def get_loaders(data_directory, batch_size, image_size, augment=True, N=2, M=9): # only support imagenet-size image
    print('==> Preparing dataset..')
    # move normalize into model, don't normalize here, 
    # is better for classic adversarial attacks
    train_transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(imagenet_mean, imagenet_std), 
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    
    if augment:
        # Add RandAugment with N, M(hyperparameter)
        train_transform.transforms.insert(0, RandAugment(N, M))

    train_dataset = datasets.ImageFolder(root=data_directory+'/train', \
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,\
        shuffle=True, drop_last=True, num_workers=12, pin_memory=True)
    test_dataset = datasets.ImageFolder(root=data_directory+'/val', \
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,\
        shuffle=True, drop_last=True, num_workers=12, pin_memory=True)
    return train_loader, test_loader

class Normalize_tops(nn.Module):
    def __init__(self, mean=imagenet_mean, std=imagenet_std):
        super(Normalize_tops, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
    
    def forward(self, x):
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (x-mean) / std
