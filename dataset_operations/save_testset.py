from ..training_utils.datasets import ImagiNet
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
import random
import numpy as np

random.seed(42)
torch.cuda.manual_seed(42)
torch.manual_seed(42)
np.random.seed(42)
models = ["gan", "sd", "midjourney", "dalle3"]
categories = ["photos", "paintings", "faces", "uncategorized"]
writers = [open("./test_gan.txt", "w"), open("./test_sd.txt", "w"), open("./test_midjourney.txt", "w"), open("./test_dalle3.txt", "w")]
def save_images_from_dataloader(dataloader, output_directory, model, k):
    for images_tensor, labels_tensor in tqdm(dataloader):
        label_value = labels_tensor
        images_numpy = images_tensor.numpy()
        for i in range(images_numpy.shape[0]):
            image_data = images_numpy[i].transpose((1, 2, 0))
            pil_image = Image.fromarray((image_data * 255).astype('uint8'))
            label_info = "real" if label_value[i, 0].item() == 0 else "synthetic"
            category_info = categories[label_value[i, 1].item()]
            main_folder = f"./testset/{category_info}/{label_info}"
            w_idx = label_value[i, 2]
            if label_value[i,0].item() != 0:
                model_info = models[label_value[i, 2].item()]
                main_folder += f"/{model_info}"
            else:
                w_idx = model
            if not os.path.isdir(os.path.join(output_directory, main_folder)):
                os.makedirs(os.path.join(output_directory, main_folder))
            image_filename = f'{main_folder}/{str(k).rjust(5, "0")}.webp'
            writers[w_idx].write(f"{image_filename},{label_value[i, 0].item()},{label_value[i, 1].item()},{label_value[i, 2].item()},{label_value[i, 3].item()}\n")
            writers[w_idx].flush()
            pil_image.save(os.path.join(output_directory, image_filename), 'WEBP', lossless=True)
            k += 1
    return k

files = ["../annotations/test_annotations_gan.txt", "../annotations/test_annotations_sd.txt", "../annotations/test_annotations_midjourney.txt", "../annotations/test_annotations_dalle3.txt"]

output_folder = "./" # REPLACE WITH FOLDER WHERE YOU WANT TO SAVE YOUR TESTSET
os.mkdir(os.path.join(output_folder, "testset"))
k_t = 0
for i, a in enumerate(files):
    dataset = ImagiNet("/home/delyan/BIG_DATASET_DONT_DELETE/", f"{a}", train=False, test_aug=True, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=20, num_workers=8, shuffle=False) # DO NOT CHANGE THE BATCH SIZE AND THE WORKERS (IT WON'T BE REPRODUCIBLE)
    k_t = save_images_from_dataloader(dataloader, output_folder, i, k_t)
