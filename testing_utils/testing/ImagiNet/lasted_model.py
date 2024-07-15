import sys
import os
repo_path = os.path.abspath("../required_libs/LASTED") 
sys.path.insert(0, repo_path)


import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from components.datasets import LASTED_Load
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F
from scipy.special import expit
from model import LASTED
import csv
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

dataset = LASTED_Load("./", "../../../annotations/lasted_anchor.txt", max_split=4, transform=transform)
dataloader_anchor = DataLoader(dataset, batch_size=20, num_workers=8, shuffle=True)

dataset = LASTED_Load("./", f"../../../annotations/test_calibration.txt", max_split=4, transform=transform)
dataloader_calib = DataLoader(dataset, batch_size=20, num_workers=8, shuffle=True)


model = nn.DataParallel(LASTED(num_class=4))
model.load_state_dict(torch.load("../required_libs/LASTED_pretrained.pt"))
model = model.module.cuda()
model.eval()
anchor_feats = []
for (images, labels) in tqdm(dataloader_anchor):
    images = images.cuda()
    with torch.no_grad():
        prob, feats = model(images, isTrain=False)
        anchor_feats.append(feats)
            
anchor_feats = torch.mean(torch.cat(anchor_feats, dim=0), dim=0, keepdim=True)

def validation_similarity(model, dataloader):
    
    data_loader =  dataloader
    sims = []
    labels_t = []
    with torch.no_grad():
        for (images, labels) in tqdm(data_loader):
            images = images.cuda()
            labels = labels.cuda()
            labels_t.append(labels)
            prob, feats = model(images, isTrain=False)
            sims.append(torch.mm(feats, torch.t(anchor_feats)))
    probs =  torch.concatenate(sims, dim=0)
    return probs.cpu().numpy(), torch.concatenate(labels_t, dim=0).cpu().numpy()
l, l1 = validation_similarity(model, dataloader_calib)

calib = LogisticRegression(max_iter=1000, C=1e10, solver="saga")
calib.fit(l, l1[:,0])
f = open("./lasted_model.csv", "w")
csvwriter =  csv.writer(f)
csvwriter.writerow(["Model", "ACC", "AUC"])


files = ["test_gan.txt", "test_sd.txt", "test_midjourney.txt", "test_dalle3.txt"]
models = ["GAN", "SD", "Midjourney", "DALLE3"]


average_acc = []
average_auc = []
for i, a in enumerate(files):
    print(a)
    l = []
    l1 = []
    dataset = LASTED_Load("./", f"../../../annotations/{a}", max_split=4, transform=transform)
    dataloader = DataLoader(dataset, batch_size=20, num_workers=8, shuffle=True)
    l, l1 = validation_similarity(model, dataloader)
    print(l1.shape)
    print(l.shape)
    l = expit(calib.decision_function(l))
    acc = balanced_accuracy_score(l1[:, 0], l >= 0.5)
    auc = roc_auc_score(l1[:, 0], l)
    average_acc.append(acc)
    average_auc.append(auc)
    csvwriter.writerow([models[i], f"{acc:.4f}", f"{auc:.4f}"])
    f.flush()
csvwriter.writerow(["Mean", f"{sum(average_acc)/len(average_acc):.4f}", f"{sum(average_auc)/len(average_auc):.4f}"])