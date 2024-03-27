import os
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import albumentations as A
import numpy as np
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 933120000

    
class ImagiNet(Dataset):
    def __init__(self, root_dir, annotations_file, track="all", train=True, resize=False, default_aug_album=None, anchor=False,  test_aug=False, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.anchor = anchor
        self.resize = resize
        self.test_aug = test_aug
        with open(annotations_file) as f:
            lines = f.readlines()
        tracks = ["origin", "content_type", "model", "specific_model", "all"]
        if track not in tracks:
            raise Exception("Not valid track")
        ann = [{'image': line.strip().split(",")[0], 'label': line.strip().split(",")[1:]} for line in lines]
        if track == tracks[0]:
            self.annotations = [{'image': a["image"], 'label': a["label"][0]} for a in ann]
        elif track == tracks[2]:
            self.annotations = [{'image': a["image"], 'label': a["label"][1]} for a in ann if int(a["label"][0]) == 1]
        elif track == tracks[2]:
            self.annotations = [{'image': a["image"], 'label': a["label"][2]} for a in ann if int(a["label"][0]) == 1]
        elif track == tracks[3]:
            self.annotations = [{'image': a["image"], 'label': a["label"][3]} for a in ann if int(a["label"][0]) == 1]
        else:
            self.annotations = [{'image': a["image"], 'label': a["label"]} for a in ann]
        self.albu_transform = A.Compose([
            A.PadIfNeeded(256, 256),
            A.RandomCrop(256, 256),
            A.OneOf([
                A.OneOf([
                    A.ImageCompression(quality_lower=50, quality_upper=95, compression_type=A.ImageCompression.ImageCompressionType.JPEG, p=1),
                    A.ImageCompression(quality_lower=50, quality_upper=95, compression_type=A.ImageCompression.ImageCompressionType.WEBP, p=1),
                ], p=1),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(var_limit=(3.0, 10.0), p=1.0),
            ], p=0.5),
            A.RandomRotate90(p=0.33),
            A.Flip(p=0.33),
        ], p=1.0)
        default_transform = [A.PadIfNeeded(256, 256), A.CenterCrop(256, 256)]
        self.albu_default = A.Compose(default_transform, p=1) if default_aug_album is None else default_aug_album
        self.label_transform = lambda data: torch.tensor(data, dtype=torch.long)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations[idx]['image'])
        image_o = np.asarray(Image.open(img_path).convert('RGB'))
        label = self.annotations[idx]['label']
        image = None
        if self.transform:
            if self.train == True:
                if self.resize:
                    h = random.randint(160, max(min(image_o.shape[0:2]), 160))
                    interp = 4 if h > 256 else 2
                    resize =  A.Compose([
                            A.PadIfNeeded(h, h),
                            A.RandomCrop(h,h),
                            A.Resize(256, 256, interpolation=interp)], p=0.5)
                    image_o = resize(image=image_o)["image"]
                image = self.albu_transform(image=image_o)["image"]
                image = self.transform(image)
            else:
                if self.anchor:
                    h = random.randint(160, max(min(image_o.shape[0:2]), 160))
                    interp = 4 if h > 256 else 2
                    res = A.Compose([
                                A.PadIfNeeded(256, 256),
                                A.RandomCrop(256, 256),
                                A.OneOf([
                                    A.OneOf([
                                        A.ImageCompression(quality_lower=50, quality_upper=95, compression_type=A.ImageCompression.ImageCompressionType.JPEG, p=1),
                                        A.ImageCompression(quality_lower=50, quality_upper=95, compression_type=A.ImageCompression.ImageCompressionType.WEBP, p=1),
                                    ], p=1),
                                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                                    A.GaussNoise(var_limit=(3.0, 10.0), p=1.0)
                                ], p=1)
                            ], p=0.5)
                    if self.resize:
                        res =  A.OneOf([
                            A.Compose([
                                A.PadIfNeeded(h, h),
                                A.RandomCrop(h,h),
                                A.Resize(256, 256, interpolation=interp),
                                A.OneOf([
                                    A.ImageCompression(quality_lower=50, quality_upper=95, compression_type=A.ImageCompression.ImageCompressionType.JPEG, p=1),
                                    A.ImageCompression(quality_lower=50, quality_upper=95, compression_type=A.ImageCompression.ImageCompressionType.WEBP, p=1),
                                ], p=1),
                            ], p=1),
                            A.Compose([
                                A.PadIfNeeded(256, 256),
                                A.RandomCrop(256, 256),
                                A.OneOf([
                                    A.OneOf([
                                        A.ImageCompression(quality_lower=50, quality_upper=95, compression_type=A.ImageCompression.ImageCompressionType.JPEG, p=1),
                                        A.ImageCompression(quality_lower=50, quality_upper=95, compression_type=A.ImageCompression.ImageCompressionType.WEBP, p=1),
                                    ], p=1),
                                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                                    A.GaussNoise(var_limit=(3.0, 10.0), p=1.0)
                                ], p=1)
                            ], p=1),
                        ], p=0.5)
                    anchor = A.Compose([
                    res,
                    A.PadIfNeeded(256, 256),
                    A.RandomCrop(256, 256),
                    A.RandomRotate90(p=0.33),
                    A.Flip(p=0.33),
                    ], p=1.0)
                    image = anchor(image=image_o)["image"]
                else:
                    if self.test_aug:
                        h = random.randint(256, max(min(min(image_o.shape[0:2]), 1000), 256))
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
                        image = resize(image=image_o)["image"]
                    else:
                        image = self.albu_default(image=image_o)["image"]
                image = self.transform(image)
        return image, self.label_transform(list(map(int, label)))
class ImagiNet_Calibration(Dataset):
    def __init__(self, root_dir, annotations_file, model_track=False, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        with open(annotations_file) as f:
            lines = f.readlines()
        if model_track:
            self.annotations = []
            for line in lines:
                if int(line.strip().split(",")[2]) != 0:
                    self.annotations.append({'image': line.strip().split(",")[0], 'label': str(int(line.strip().split(",")[2])-1)})
        else:
            self.annotations = [{'image': line.strip().split(",")[0], 'label': [line.strip().split(",")[1]]} for line in lines]
        self.albu_transform = A.Compose([
            A.PadIfNeeded(256, 256),
            A.RandomCrop(256, 256),
            A.OneOf([
                A.ImageCompression(quality_lower=50, quality_upper=95, compression_type=0, p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(var_limit=(3.0, 10.0), p=1.0),
                A.ToGray(p=1.0),
            ], p=0.5),
            A.RandomRotate90(p=0.33),
            A.Flip(p=0.33),
        ], p=1.0)
        self.albu_default = A.Compose([
            A.PadIfNeeded(256, 256),
            A.CenterCrop(256, 256)],
                                  p=1)
        self.label_transform = lambda data: torch.tensor(data, dtype=torch.long)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations[idx]['image'])
        image = np.asarray(Image.open(img_path).convert('RGB'))
        label = self.annotations[idx]['label']
        if self.transform:
            if self.train == True:
                image = self.albu_transform(image=image)["image"]
                #image2 = self.albu_transform(image=image)["image"]
                image = self.transform(image)
                #image2 = self.transform(image2)
            else:
                #a = time.time()
                image = self.albu_default(image=image)["image"]
                image = self.transform(image)
                #b = time.time()
                #print(b-a)
            #a = time.time()
            #b = time.time()
            #print(b-a)
            #raise Exception()
        return image, self.label_transform(list(map(int, label)))
