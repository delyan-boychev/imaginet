import os
import sys

repo_path = os.path.abspath("../required_libs/DMimageDetection")
sys.path.insert(0, repo_path)

import torch
from torchvision import transforms
from components.datasets import ImagiNet_Testset
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from sklearn.linear_model import LogisticRegression
from get_method_here import get_method_here, def_model
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.special import softmax, expit
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

_, model_path, arch, norm_type, patch_size = get_method_here('Grag2021_latent', weights_path="./required_libs/weights")
device = 'cuda:0'
model = def_model(arch, model_path, localize=False)
model = model.to(device).eval()

_, model_path, arch, norm_type, patch_size = get_method_here('Grag2021_progan', weights_path="./required_libs/weights")
device = 'cuda:0'
model2 = def_model(arch, model_path, localize=False)
model2 = model2.to(device).eval()


l = []
l1 =[]
k = 0

with torch.no_grad():
    for image, label in tqdm(dataloader):
        out =  model(image.cuda())
        out2 =  model2(image.cuda())
        out = torch.mean((out + out2)/2.0, (1, 2, 3)).unsqueeze(1)
        l.append(out)
        l1.append(label.cuda())
l = torch.concatenate(l, dim=0)
l1 = torch.concatenate(l1, dim=0)

l2 = []
l3 = []
with torch.no_grad():
    for image, label in tqdm(dataloader2):
        out =  model(image.cuda())
        out2 =  model2(image.cuda())
        out = torch.mean((out + out2)/2.0, (1, 2, 3)).unsqueeze(1)
        l2.append(out)
        l3.append(label.cuda())
l2 = torch.concatenate(l2, dim=0).cpu().numpy()
l3 = torch.concatenate(l3, dim=0).cpu().numpy()

calib = LogisticRegression(max_iter=1000, C=1e10, solver="saga")
calib.fit(l.cpu().numpy(), l1.cpu().numpy()[:,0])
f = open("./corvi_model.csv", "w")
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
