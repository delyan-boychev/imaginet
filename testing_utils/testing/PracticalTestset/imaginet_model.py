import os
import sys

repo_path = os.path.abspath("../../") 
sys.path.insert(0, repo_path)

repo_path = os.path.abspath("../required_libs/DMimageDetection")
sys.path.insert(0, repo_path)

import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from components.datasets import PracticalTestset
import matplotlib.pyplot as plt
import albumentations as A
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

dataset_calib = PracticalTestset("./", "../additional_annotations/annotations_calib_practical.txt", transform)
dataloader_calib = DataLoader(dataset_calib, batch_size=20, num_workers=8, shuffle=True)


model = ConResNet(name="resnet50nodown", selfcon_pos=[False, True, False], selfcon_size="fc", dataset="imaginet")
state = torch.load("../required_libs/imaginet_weights.pt", map_location="cpu")
model.load_state_dict(state["model"])
model = model.to("cuda")
model = model.eval()
classifier = nn.Sequential(nn.BatchNorm1d(2048), nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1))
classifier.load_state_dict(state["classifier"])
classifier = classifier.to("cuda")
classifier.eval()

l = []
l1 = []
with torch.no_grad():
    for image, label in tqdm(dataloader_calib):
        out = classifier(model.encoder(image.cuda())[1])
        l.append(out)
        l1.append(label.type(torch.LongTensor).cuda())
l = torch.concatenate(l, dim=0)
l1 = torch.concatenate(l1, dim=0)

calib = LogisticRegression(max_iter=1000, C=1e10, solver="saga")
calib.fit(l.cpu().numpy(), l1.cpu().numpy()[:, 0])
f = open("./imaginet_model.csv", "w")
csvwriter =  csv.writer(f)
csvwriter.writerow(["Model", "ACC", "AUC"])


files = ["Test_DreamBooth_num1052.txt", "Test_MidjourneyV4_num1354.txt", "Test_MidjourneyV5_num2000.txt", "Test_NightCafe_num1300.txt", "Test_StableAI_num1290.txt", "Test_YiJian_num796.txt"]
models = ["DreamBooth", "MidjoruneyV4", "MidjourneyV5", "NightCafe", "StableAI", "YiJian"]

average_acc = []
average_auc = []
for i, a in  enumerate(files):
    l = []
    l1 = []
    dataset = PracticalTestset("./", f"../additional_annotations/{a}", transform)
    dataloader = DataLoader(dataset, batch_size=20, num_workers=8, shuffle=True)
    with torch.no_grad():
        for image, label in tqdm(dataloader):
            out = classifier(model.encoder(image.cuda())[1])
            l.append(out)
            l1.append(label.type(torch.LongTensor).cuda())
    l = torch.concatenate(l, dim=0).cpu().numpy()
    l1 = torch.concatenate(l1, dim=0).cpu().numpy()
    l = expit(calib.decision_function(l))
    acc = balanced_accuracy_score(l1[:, 0], l >= 0.5)
    auc = roc_auc_score(l1[:, 0], l)
    average_acc.append(acc)
    average_auc.append(auc)
    csvwriter.writerow([models[i], f"{acc:.4f}", f"{auc:.4f}"])
csvwriter.writerow(["Mean", f"{sum(average_acc)/len(average_acc):.4f}", f"{sum(average_auc)/len(average_auc):.4f}"])