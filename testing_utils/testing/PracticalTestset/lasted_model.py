import sys
import os
repo_path = os.path.abspath("../required_libs/LASTED") 
sys.path.insert(0, repo_path)


import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from components.datasets import LASTED_Load, PracticalTestset
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

dataset = PracticalTestset("./", "../additional_annotations/anchor_lasted_practical.txt", transform=transform)
dataloader_anchor = DataLoader(dataset, batch_size=20, num_workers=8, shuffle=True)

dataset_calib = PracticalTestset("./", "../additional_annotations/annotations_calib_practical.txt", transform)
dataloader_calib = DataLoader(dataset_calib, batch_size=20, num_workers=8, shuffle=True)

model = nn.DataParallel(LASTED(num_class=4))
model.load_state_dict(torch.load("../required_libs/LASTED_pretrained.pt"))
model = model.module.cuda()
model.eval()
anchor_feats = []
for (images, labels) in tqdm(dataloader_anchor):
    with torch.no_grad():
        images = images.cuda()
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
            sim = torch.mm(feats, torch.t(anchor_feats))
            sims.append(sim)
    probs =  torch.concatenate(sims, dim=0)
    return probs.cpu().numpy(), torch.concatenate(labels_t, dim=0).cpu().numpy()
l, l1 = validation_similarity(model, dataloader_calib)

calib = LogisticRegression(max_iter=1000, C=1e10, solver="saga")
calib.fit(l, l1[:,0])
f = open("./lasted_model.csv", "w")
csvwriter =  csv.writer(f)
csvwriter.writerow(["Model", "ACC", "AUC"])


files = ["Test_DreamBooth_num1052.txt", "Test_MidjourneyV4_num1354.txt", "Test_MidjourneyV5_num2000.txt", "Test_NightCafe_num1300.txt", "Test_StableAI_num1290.txt", "Test_YiJian_num796.txt"]
models = ["DreamBooth", "MidjoruneyV4", "MidjourneyV5", "NightCafe", "StableAI", "YiJian"]


average_acc = []
average_auc = []
for i, a in enumerate(files):
    print(a)
    l = []
    l1 = []
    dataset = PracticalTestset("./", f"../additional_annotations/{a}", transform)
    dataloader = DataLoader(dataset, batch_size=20, num_workers=8, shuffle=True)
    l, l1 = validation_similarity(model, dataloader)
    l = expit(calib.decision_function(l))
    acc = balanced_accuracy_score(l1[:, 0], l >= 0.5)
    auc = roc_auc_score(l1[:, 0], l)
    average_acc.append(acc)
    average_auc.append(auc)
    csvwriter.writerow([models[i], f"{acc:.4f}", f"{auc:.4f}"])
    f.flush()
csvwriter.writerow(["Mean", f"{sum(average_acc)/len(average_acc):.4f}", f"{sum(average_auc)/len(average_auc):.4f}"])