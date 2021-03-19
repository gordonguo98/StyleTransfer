import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None


class TransferDataset(Dataset):
    def __init__(self, content_dir):
        super(TransferDataset, self).__init__()
        self.content_dir = content_dir
        self.content_name_list = self.get_name_list(self.content_dir)
        self.transforms = self.transform()

    def get_name_list(self, name):
        name_list = os.listdir(name)
        name_list = [os.path.join(name, i) for i in name_list]
        np.random.shuffle(name_list)
        return name_list

    def transform(self):
        data_transform = transforms.Compose([
            # transforms.RandomRotation(15),
            # transforms.RandomResizedCrop(size=512, scale=(0.5, 1.0)),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        return data_transform

    def __len__(self):
        a = len(self.content_name_list)
        return a

    def __getitem__(self, item):
        img = Image.open(self.content_name_list[item]).convert('RGB')
        img_out = self.transforms(img)
        return img_out
