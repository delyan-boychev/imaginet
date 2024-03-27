import os
import sys

repo_path = os.path.abspath("../required_libs/GANimageDetection")
sys.path.insert(0, repo_path)

import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from components.datasets import CorviTestset
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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
import pickle
import torch.nn.functional as F
from scipy.special import softmax, expit
from resnet50nodown import resnet50nodown
import csv
import random


random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


model = resnet50nodown("cuda", "../required_libs/grag2021_stylegan2.pth")
model = model.eval()

norm = transforms.Normalize(mean, std)
transform =  transforms.Compose([transforms.ToTensor(), norm])
device = torch.device("cuda")
dataset = CorviTestset("./TestSetCSV/", "../additional_annotations/annotations_calib_corvi.txt", transform)
dataloader = DataLoader(dataset, batch_size=10, num_workers=4, shuffle=True)
dataset2 = CorviTestset("./TestSetCSV/", "../additional_annotations/annotations_corvi.txt", transform)
dataloader2 = DataLoader(dataset2, batch_size=10, num_workers=4, shuffle=True)



l = []
l1 = []
l2 = []
l3 = []

with torch.no_grad():
    for image, label in tqdm(dataloader):
        out = model(image.cuda()).mean((2, 3))
        l.append(out)
        l1.append(label.cuda())
l = torch.concatenate(l, dim=0)
l1 = torch.concatenate(l1, dim=0)
with torch.no_grad():
    for image, label in tqdm(dataloader2):
        out = model(image.cuda()).mean((2, 3))
        l2.append(out)
        l3.append(label.cuda())
l2 = torch.concatenate(l2, dim=0)
l3 = torch.concatenate(l3, dim=0)

# Instantiate model, loss function, and optimizer
tensdataset = TensorDataset(l, l1)
tensdataset2 = TensorDataset(l2, l3)
dataload = DataLoader(tensdataset, 100, shuffle=True)
dataload2 = DataLoader(tensdataset2, 100, shuffle=True)
calib = LogisticRegression(max_iter=1000, C=1e10, solver="saga")
calib.fit(l.cpu().numpy(), l1[:, 0].cpu().numpy())
l2 = expit(calib.decision_function(l2.cpu().numpy()))
l3 = l3.cpu().numpy()


print(np.mean((l2 >= 0.5) == l3[:, 0]))
print(roc_auc_score(l3[:, 0], l2))
f = open("./grag2021_model.csv", "w")
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
