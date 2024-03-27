import os
import sys

repo_path = os.path.abspath("../../training_utils") 
sys.path.insert(0, repo_path)

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from components.datasets import ImagiNet_Testset
import numpy as np
from networks.resnet_big import ConResNet
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.special import expit
import csv

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
norm = transforms.Normalize(mean, std)
transform =  transforms.Compose([transforms.ToTensor(), norm])
device = torch.device("cuda")

dataset = ImagiNet_Testset("./", "../../../annotations/test_calibration.txt", transform=transform)
dataloader = DataLoader(dataset, batch_size=20, num_workers=8, shuffle=True)


model = ConResNet(selfcon_pos=[False, True, False], selfcon_size="fc", dataset="imaginet")
state = torch.load("../required_libs/imaginet_weights.pt", map_location="cpu")
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
                nn.Linear(1024, 1)
            )
classifier.load_state_dict(state["classifier"])
classifier = classifier.to("cuda")
classifier.eval()

l = []
l1 =[]
k = 0

with torch.no_grad():
    for image, label in tqdm(dataloader):
        out =  classifier(model.encoder(image.cuda())[1])
        l.append(out)
        l1.append(label.cuda())
l = torch.concatenate(l, dim=0)
l1 = torch.concatenate(l1, dim=0)

calib = LogisticRegression(max_iter=1000, C=1e10, solver="saga")
calib.fit(l.cpu().numpy(), l1.cpu().numpy()[:,0])
f = open("./imaginet_model2.csv", "w")
csvwriter =  csv.writer(f)
csvwriter.writerow(["Model", "ACC", "AUC"])


files = ["test_gan_2.txt", "test_sd_2.txt", "test_midjourney_2.txt", "test_dalle3_2.txt"]
models = ["GAN", "SD", "Midjourney", "DALLE3"]


average_acc = []
average_auc = []
prob_dist = []
labels_dist = []
for i, a in enumerate(files):
    print(a)
    l = []
    l1 = []
    dataset = ImagiNet_Testset("./", f"../../annotations/{a}", transform=transform)
    dataloader = DataLoader(dataset, batch_size=20, num_workers=8, shuffle=True)
    
    with torch.no_grad():
        for image, label in tqdm(dataloader):
            out = classifier(model.encoder(image.cuda())[1])
            l.append(out)
            l1.append(label.cuda())
    l = torch.concatenate(l, dim=0).cpu().numpy()
    l1 = torch.concatenate(l1, dim=0).cpu().numpy()
    l = expit(calib.decision_function(l))
    acc = balanced_accuracy_score(l1[:, 0], l >= 0.5)
    auc = roc_auc_score(l1[:, 0], l)
    average_acc.append(acc)
    average_auc.append(auc)
    csvwriter.writerow([models[i], f"{acc:.4f}", f"{auc:.4f}"])
    f.flush()
csvwriter.writerow(["Mean", f"{sum(average_acc)/len(average_acc):.4f}", f"{sum(average_auc)/len(average_auc):.4f}"])
