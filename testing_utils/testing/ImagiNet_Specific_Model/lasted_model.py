import sys
import os
repo_path = os.path.abspath("../required_libs/LASTED") 
sys.path.insert(0, repo_path)


import torch
import torch.nn as nn
from torchvision import transforms
from components.datasets import LASTED_Load
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
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

dataset = LASTED_Load("./",f"../../../annotations/lasted_anchor.txt", max_split=4, transform=transform)
dataloader_anchor = DataLoader(dataset, batch_size=20, num_workers=8, shuffle=True)

dataset = LASTED_Load("./",f"../../../annotations/test_calibration.txt", max_split=4, transform=transform)
dataloader_calib = DataLoader(dataset, batch_size=20, num_workers=8, shuffle=True)

dataset = LASTED_Load("./", f"../../../annotations/test_all.txt", max_split=4, transform=transform)
dataloader_test = DataLoader(dataset, batch_size=20, num_workers=8, shuffle=True)


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
l2, l3 = validation_similarity(model, dataloader_test)

calib = LogisticRegression(max_iter=1000, C=1e10, solver="saga")
calib.fit(l, l1[:,0])
f = open("./lasted_model.csv", "w")
csvwriter =  csv.writer(f)
csvwriter.writerow(["Model", "ACC", "AUC"])

l2 = expit(calib.decision_function(l2))



types = ["ProGAN", "StyleGAN-XL", "StyleGAN3", "SD v2.1", "SDXL v1.0", "Animagine XL", "Midjourney", "DALL·E 3"]


average_acc = []
average_auc = []
for i in range(8):
    probs, true =  (l2[l3[:, 3] == i], l3[l3[:, 3] == i, 0])
    r_p, r_t = (l2[l3[:, 0] == 0], l3[l3[:, 0] == 0, 0])
    p = np.random.permutation(len(r_p))
    probs_r, true_r = r_p[p], r_t[p]
    probs = np.concatenate([probs_r[:probs.shape[0]], probs], axis=0)
    true = np.concatenate([true_r[:true.shape[0]], true], axis=0)
    auc = roc_auc_score(true, probs)
    acc = balanced_accuracy_score(true, probs >= 0.5)
    average_auc.append(auc)
    average_acc.append(acc)
    csvwriter.writerow([types[i], f"{acc:.4f}", f"{auc:.4f}"])
csvwriter.writerow(["Mean", f"{sum(average_acc)/len(average_acc):.4f}", f"{sum(average_auc)/len(average_auc):.4f}"])