import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import cv2
import albumentations as albu
from PIL import Image

IMAGE_PATH = 'C:/Users/tolik/information_technology/third_year/practice_project/CoreAnalysis-ML/data_for_study/photos/'
ULTRA_MASK_PATH = 'C:/Users/tolik/information_technology/third_year/practice_project/CoreAnalysis-ML/data_for_study/labels/ultraviolet/label_'
DAY_MASK_PATH = 'C:/Users/tolik/information_technology/third_year/practice_project/CoreAnalysis-ML/data_for_study/labels/daylight/label_'

########################################################
# Datasets
class CoreDataset(Dataset):
    def __init__(self, img_path, mask_path, data,
                 augmentation=None,
                 patching=True):
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
        h, w, c = img.shape
        h = h // 256 * 256
        w = w // 256 * 256
        mask = np.load(self.mask_path + str(self.data[index]) + '.npz')['x']

        if self.augmentation is not None:
            a = self.augmentation(image=img, mask=mask)
            img = a['image']
            mask = np.asarray(a['mask'])
        img = Image.fromarray(img)

        img = T.Resize([h, w])(img)
        mask = cv2.resize(mask, (w, h))
        img = self.transformation(img)
        mask = torch.from_numpy(mask).long()

        if self.patching:
            img, mask = self.patch(img, mask)
        return img, mask

    def patch(self, img, mask, s=256):
        # input: resized img (3, h, w) and mask (h,w)

        img_patches = img.unfold(1, s, step=s).unfold(2, s, step=s)
        img_patches = img_patches.contiguous().view(3, -1, s, s)
        img_patches = img_patches.contiguous().permute(1, 0, 2, 3)

        mask = mask.unsqueeze(0)
        mask_patches = mask.unfold(1, s, step=s).unfold(2, s, step=s)
        mask_patches = mask_patches.contiguous().view(-1, s, s)

        # output: img_patch (N-patches * 3 * s * s) and mask-patch (N-patches * s * s)
        return img_patches, mask_patches


class UltravioletDataset(CoreDataset):
    classes = ['Отсутствует, Карбонатное, Насыщенное']


class DaylightDataset(CoreDataset):
    classes = ['Переслаивание пород', 'Алевролит глинистый', 'Песчаник',
               'Аргиллит', 'Разлом', 'Проба', 'Песчаник глинистый']
########################################################


########################################################
#

########################################################