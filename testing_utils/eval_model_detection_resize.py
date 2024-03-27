import os
import sys
repo_path = os.path.abspath("../training_utils") 
sys.path.insert(0, repo_path)

import torch
import torch.nn as nn
from torchvision import transforms
from datasets import ImagiNet
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
from networks.resnet_big import ConResNet
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import random
import pandas as pd
import seaborn as sns
import cv2

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
norm = transforms.Normalize(mean, std)
transform =  transforms.Compose([transforms.ToTensor(), norm])
device = torch.device("cuda")

model = ConResNet(selfcon_pos=[False, True, False], selfcon_size="fc", dataset="imaginet")
state = torch.load("./testing/required_libs/imaginet_weights.pt", map_location="cpu")
model.load_state_dict(state["model"])
model = model.to("cuda")
model = model.eval()
class L2Norm(nn.Module):
    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=1, p=2)
classifier = nn.Sequential(
                L2Norm(),
                nn.Linear(2048, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 4)
            )
classifier.load_state_dict(state["classifier"])
classifier = classifier.to("cuda")
classifier.eval()



files = ["test_annotations_sd.txt", "test_annotations_gan.txt", "test_annotations_midjourney.txt", "test_annotations_dalle3.txt"]
models = ["GAN", "SD", "Midjourney", "DALLE"]
first = False
l_all = []
l1_all = []
acc_avg = []
resize_aug =  lambda x: A.Compose([A.PadIfNeeded(x, x), A.CenterCrop(x, x), A.Resize(256, 256, interpolation=cv2.INTER_LANCZOS4)])
x = []
y = []
for i in tqdm(range(2, 12+1, 1)):
    
    x.append(12/i)
    acc_avg = []
    for k, a in enumerate(files):
        l = []
        l1 = []
        dataset = ImagiNet("", f"./{a}", train=False, test_aug=False, default_aug_album=resize_aug(int(256*(12/(i)))), track="model", transform=transform)
        dataloader = DataLoader(dataset, batch_size=20, num_workers=8, shuffle=True)
        with torch.no_grad():
            for image, label in dataloader:
                out = classifier(model.encoder(image.cuda())[1])
                l.append(out)
                l1.append(label.cuda())
        l = torch.softmax(torch.concatenate(l, dim=0), dim=1).cpu().numpy()
        l1 = torch.concatenate(l1, dim=0).cpu().numpy()
        acc = accuracy_score(l1[:, 0], np.argmax(l, axis=1))
        acc_avg.append(acc)
    y.append(sum(acc_avg)/len(acc_avg))
df = pd.DataFrame({'Resize Fraction': x,
                   'Accuracy (%)': y,
                   })

sns.lineplot(x="Resize Fraction", y='Accuracy (%)', data=df)
plt.title("Accuracy under Resizing")
plt.ylim((0, 1))
plt.tight_layout()
plt.savefig("curve_acc_resize.pdf", dpi=400)
