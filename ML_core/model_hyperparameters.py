import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import cv2

IMAGE_PATH = ''
MASK_PATH = ''

class UltraPhotoDataSet(Dataset):
    def __init__(self, img_path, mask_path, data,
                 augmentation,
                 patching):
        self.img_path = img_path
        self.mask_path = mask_path
        self.data = data
        self.patching = patching
        self.augmentation = augmentation
        self.transformation = T.Compose([T.ToTensor(),
                                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = cv2.imread(self.img_path + str(self.data[index]) + '.jpeg')
        mask = np.load(self.mask_path + str(self.data[index]) + '.npz')

        if self.augmentation is not None:
            a = self.augmentation(image=img, mask=mask)
            img = a['image']
            mask = a['mask']
        else:
            img = self.transformation(img)

        mask = torch.from_numpy(mask).long()

        if self.patching:
            img, mask = self.patch(img, mask)
        
        return img, mask

    def patch(self, img, mask):
        return img, mask