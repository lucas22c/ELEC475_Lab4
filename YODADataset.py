import torch
import os
from torch.utils.data import Dataset
from PIL import Image

mean = torch.tensor([0.3656, 0.3844, 0.3725])
std = torch.tensor([0.4194, 0.4075, 0.4239])


def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


class YODADataset(Dataset):
    def __init__(self, dir, training=True, transform=None):
        self.dir = dir
        self.training = training
        if self.training == True:
            self.dir = os.path.join(dir, 'train')
        else:
            self.dir = os.path.join(dir, 'test')

        self.label_dir = os.path.join(self.dir, 'labels.txt')

        self.transform = transform
        self.data = []

        with open(self.label_dir, 'r') as f:
            for line in f:
                filename, label, _ = line.strip().split(' ')
                full_path = os.path.join(self.dir, filename)
                self.data.append((full_path, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_path, label = self.data[idx]
        image = Image.open(full_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
