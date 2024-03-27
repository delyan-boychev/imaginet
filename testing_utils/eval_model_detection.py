import os
import sys
repo_path = os.path.abspath("../training_utils") 
sys.path.insert(0, repo_path)

import torch
import torch.nn as nn
from torchvision import transforms
from datasets import ImagiNet
import numpy as np
from networks.resnet_big import ConResNet
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import random

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
state = torch.load("/home/delyan/con-synthimg-detection/linear_imaginet_save_model_calib/model_5.pt", map_location="cpu")
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



files = ["test_annotations_gan.txt", "test_annotations_sd.txt", "test_annotations_midjourney.txt",  "test_annotations_dalle3.txt"]
models = ["GAN", "SD", "Midjourney", "DALLE"]
first = False
l_all = []
l1_all = []
acc_avg = []
x = []
y = []
print("ACC")
for k, a in enumerate(files):
    l = []
    l1 = []
    dataset = ImagiNet("./", f"../annotations/{a}", train=False, test_aug=False, track="model", transform=transform)
    dataloader = DataLoader(dataset, batch_size=20, num_workers=8, shuffle=True)
    with torch.no_grad():
        for image, label in dataloader:
            out = classifier(model.encoder(image.cuda())[1])
            l.append(out)
            l1.append(label.cuda())
    l = torch.softmax(torch.concatenate(l, dim=0), dim=1).cpu().numpy()
    l1 = torch.concatenate(l1, dim=0).cpu().numpy()
    l1_all.append(l1[:, 0])
    l_all.append(l)
    acc = balanced_accuracy_score(l1[:, 0], np.argmax(l, axis=1))
    print(f"{models[k]}: {acc:.4f}")
l_all = np.concatenate(l_all, axis=0)
l1_all = np.concatenate(l1_all, axis=0) 
encoded_arr = np.zeros((l1_all.size, l1_all.max()+1), dtype=int)
encoded_arr[np.arange(l1_all.size),l1_all] = 1
auc =  roc_auc_score(encoded_arr, l_all, multi_class="ovr")
print(f"AUC: {auc:.4f}")
