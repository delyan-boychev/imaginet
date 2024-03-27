import os
import sys

repo_path = os.path.abspath("../required_libs/GANimageDetection")
sys.path.insert(0, repo_path)

import torch
from torchvision import transforms
from components.datasets import ImagiNet_Testset
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from sklearn.linear_model import LogisticRegression
from resnet50nodown import resnet50nodown
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

dataset = ImagiNet_Testset("./",f"../../../annotations/test_calibration.txt", transform=transform)
dataloader = DataLoader(dataset, batch_size=20, num_workers=8, shuffle=True)

dataset2 = ImagiNet_Testset("./", f"../../../annotations/test_all.txt", transform=transform)
dataloader2 = DataLoader(dataset2, batch_size=20, num_workers=8, shuffle=True)


model = resnet50nodown("cuda", "../required_libs/grag2021_stylegan2.pth")
model = model.eval()


l = []
l1 =[]
k = 0

with torch.no_grad():
    for image, label in tqdm(dataloader):
        out = model(image.cuda()).squeeze((2,3))
        l.append(out)
        l1.append(label.cuda())
l = torch.concatenate(l, dim=0)
l1 = torch.concatenate(l1, dim=0)
print(l.shape)

l2 = []
l3 = []
with torch.no_grad():
    for image, label in tqdm(dataloader2):
        out = model(image.cuda()).squeeze((2,3))
        l2.append(out)
        l3.append(label.cuda())
l2 = torch.concatenate(l2, dim=0).cpu().numpy()
l3 = torch.concatenate(l3, dim=0).cpu().numpy()

calib = LogisticRegression(max_iter=1000, C=1e10, solver="saga")
calib.fit(l.cpu().numpy(), l1.cpu().numpy()[:,0])
f = open("./imaginet_model.csv", "w")
csvwriter =  csv.writer(f)
csvwriter.writerow(["Content Type", "ACC", "AUC"])

l2 = expit(calib.decision_function(l2))



types = ["Photos", "Paintings", "Faces", "Uncategorised"]


average_acc = []
average_auc = []
for i in range(4):
    probs, true =  (l2[l3[:, 1] == i], l3[l3[:, 1] == i])
    auc = roc_auc_score(true[:, 0], probs)
    acc = balanced_accuracy_score(true[:, 0], probs >= 0.5)
    average_auc.append(auc)
    average_acc.append(acc)
    csvwriter.writerow([types[i], f"{acc:.4f}", f"{auc:.4f}"])
csvwriter.writerow(["Mean", f"{sum(average_acc)/len(average_acc):.4f}", f"{sum(average_auc)/len(average_auc):.4f}"])
