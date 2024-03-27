import os
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import albumentations as A
import numpy as np
import time
import chardet
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 933120000


class PracticalTestset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(annotations_file) as f:
            lines = f.readlines()
        self.annotations = [{'image': line.strip().split(" ")[0], 'label': line.strip().split(" ")[1:]} for line in lines]
        self.albu_default = A.Compose([A.PadIfNeeded(448, 448), A.CenterCrop(448, 448)],
                                  p=1)
        self.label_transform = lambda data: torch.tensor(data, dtype=torch.long)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations[idx]['image'])
        image_o = np.asarray(Image.open(img_path).convert('RGB'))
        label = self.annotations[idx]['label']
        if self.transform:
            image = self.albu_default(image=image_o)["image"]
            image = self.transform(image)
                #b = time.time()
                #print(b-a)
            #a = time.time()
            #b = time.time()
            #print(b-a)
            #raise Exception()
        return image, self.label_transform(list(map(int, label)))

class LASTED_Load(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None, max_split=2, aug=False):
        self.root_dir = root_dir
        self.transform = transform
        self.aug = aug
        with open(annotations_file) as f:
            lines = f.readlines()
        self.annotations = [{'image': line.strip().rsplit(",", max_split)[0], 'label': line.strip().rsplit(",", max_split)[1:]} for line in lines]
        self.albu_default = A.Compose([A.PadIfNeeded(448, 448), A.CenterCrop(448, 448)],
                                  p=1)
        self.label_transform = lambda data: torch.tensor(data, dtype=torch.long)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations[idx]['image'])
        image = np.asarray(Image.open(img_path).convert('RGB'))
        label = self.annotations[idx]['label']
        if self.transform:
            if self.aug:
                h = random.randint(256, max(min(min(image.shape[0:2]), 1000), 256))
                interp = 4
                resize =  A.Compose([
                    A.PadIfNeeded(h, h),
                    A.RandomCrop(h,h),
                    A.Resize(256, 256, interpolation=interp),
                    A.OneOf([
                        A.ImageCompression(quality_lower=60, quality_upper=100, compression_type=A.ImageCompression.ImageCompressionType.JPEG, p=1),
                        A.ImageCompression(quality_lower=60, quality_upper=100, compression_type=A.ImageCompression.ImageCompressionType.WEBP, p=1),
                    ], p=0.75),
                ], p=1)
                image = resize(image=image)["image"]
            image = self.albu_default(image=image)["image"]
            image = self.transform(image)
                #b = time.time()
                #print(b-a)
            #a = time.time()
            #b = time.time()
            #print(b-a)
            #raise Exception()
        return image, self.label_transform(list(map(int, label)))

class ImagiNet_Testset(Dataset):
    def __init__(self, root_dir, annotations_file, transform, track="all"):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        with open(annotations_file) as f:
            lines = f.readlines()
        tracks = ["origin", "model", "all"]
        if track not in tracks:
            raise Exception("Not valid track")
        ann = [{'image': line.strip().split(",")[0], 'label': line.strip().split(",")[1:]} for line in lines]
        if track == tracks[0]:
            self.annotations = [{'image': a["image"], 'label': a["label"][0]} for a in ann]
        elif track == tracks[1]:
            self.annotations = [{'image': a["image"], 'label': a["label"][2]} for a in ann if int(a["label"][0]) == 1]
        else:
            self.annotations = [{'image': a["image"], 'label': a["label"]} for a in ann]
        self.label_transform = lambda data: torch.tensor(data, dtype=torch.long)
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations[idx]['image'])
        image_o = Image.open(img_path).convert('RGB')
        label = self.annotations[idx]['label']
        image = self.transform(image_o)
        return image, self.label_transform(list(map(int, label)))