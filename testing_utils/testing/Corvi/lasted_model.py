import sys
import os
repo_path = os.path.abspath("../required_libs/LASTED") 
sys.path.insert(0, repo_path)


import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from components.datasets import CorviTestset, LASTED_Load
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAccuracy, BinaryAUROC, MulticlassAUROC
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score
from torchmetrics import AveragePrecision
from torch.utils.data import TensorDataset
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import pickle
import torch.nn.functional as F
from scipy.special import softmax, expit
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

dataset = LASTED_Load("./TestSetCSV/", "../additional_annotations/anchor_lasted_corvi.txt", transform=transform)
dataloader_anchor = DataLoader(dataset, batch_size=20, num_workers=8, shuffle=True)
dataset_calib = LASTED_Load("./TestSetCSV/", "../additional_annotations/annotations_calib_corvi.txt", transform, max_split=2)
dataloader_calib = DataLoader(dataset_calib, batch_size=10, num_workers=4, shuffle=True)
dataset2 = LASTED_Load("./TestSetCSV/", "../additional_annotations/annotations_corvi.txtt", transform, max_split=2)
dataloader2 = DataLoader(dataset2, batch_size=10, num_workers=4, shuffle=True)

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
print(l.shape)
print(l1.shape)
l2, l3 = validation_similarity(model, dataloader2)
print(l2.shape)
print(l3.shape)

# Instantiate model, loss function, and optimizer
calib = LogisticRegression(max_iter=1000, C=1e10, solver="saga")
calib.fit(l, l1[:, 0:1])
l2 = expit(calib.decision_function(l2))
print(l2.shape)


print(np.mean((l2 >= 0.5) == l3[:, 0]))
print(roc_auc_score(l3[:, 0], l2))
f = open("./lasted_model.csv", "w")
csvwriter =  csv.writer(f)
csvwriter.writerow(["Model", "ACC", "AUC"])
l2_real = l2[l3[:, 0]==0]
l3_real = l3[l3[:, 0]==0, 0]
l2_fake = l2[l3[:, 0]==1]
l3_fake = l3[l3[:, 0]==1]
lst_vars = ["progan", "stylegan2", "stylegan3", "biggan", "eg3d", "taming_trans", "dalle_mini", "dalle2", "glide", "latent_diff", "stable_diff", "adm"]
average_acc = []
average_auc = []
average_ap = []
for i in range(12):
    l_m = l2_fake[l3_fake[:, 1]==i] 
    l1_m = l3_fake[l3_fake[:, 1]==i, 0]
    acc = balanced_accuracy_score(np.concatenate([l3_real, l1_m], axis=0), np.concatenate([l2_real >= 0.5, l_m >= 0.5], axis=0)) 
    ap = average_precision_score(np.concatenate([l3_real, l1_m], axis=0), np.concatenate([l2_real, l_m], axis=0), average='weighted')
    average_acc.append(acc)
    auc = roc_auc_score(np.concatenate([l3_real, l1_m], axis=0), np.concatenate([l2_real, l_m], axis=0), average='weighted')
    average_auc.append(auc)
    average_ap.append(ap)
    csvwriter.writerow([lst_vars[i], f"{acc:.4f}", f"{auc:.4f}"])
csvwriter.writerow(["Mean", f"{sum(average_acc)/len(average_acc):.4f}", f"{sum(average_auc)/len(average_auc):.4f}"])
