import torch
import torch.nn as nn
from torchvision import transforms
from ..training_utils.datasets import ImagiNet
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
from ..training_utils.networks.resnet_big import ConResNet
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
import random
import pandas as pd
import seaborn as sns

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
state = torch.load("./testing/required_libs/imaginet_weights.pt", map_location="cpu")
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


files = ["test_annotations_sd.txt", "test_annotations_gan.txt", "test_annotations_dalle3.txt", "test_annotations_midjourney.txt"]
models = ["GAN", "SD", "DALLE", "Midjourney"]
first = False
l_all = []
l1_all = []
acc_avg = []
jpeg_aug =  lambda x: A.Compose([A.PadIfNeeded(256, 256), A.CenterCrop(256, 256), A.ImageCompression(quality_lower=x, quality_upper=x)])
x = []
y = []
for i in tqdm(range(40, 100+1, 5)):
    x.append(i)
    avg_acc = []
    for k, a in enumerate(files):
        l = []
        l1 = []
        dataset = ImagiNet("./", f"../annotations/{a}", train=False, test_aug=False, default_aug_album=jpeg_aug(i), track="model", transform=transform)
        dataloader = DataLoader(dataset, batch_size=20, num_workers=8, shuffle=True)
        with torch.no_grad():
            for image, label in dataloader:
                out = classifier(model.encoder(image.cuda())[1])
                l.append(out)
                l1.append(label.cuda())
        l = torch.softmax(torch.concatenate(l, dim=0), dim=1).cpu().numpy()
        l1 = torch.concatenate(l1, dim=0).cpu().numpy()
        acc = balanced_accuracy_score(l1[:, 0], np.argmax(l, axis=1))
        avg_acc.append(acc)
    y.append(sum(avg_acc)/len(avg_acc))
df = pd.DataFrame({'JPEG Quality': x,
                   'Accuracy (%)': y
                   })

sns.lineplot(x="JPEG Quality", y='Accuracy (%)', data=df)
plt.title("Accuracy under JPEG Compression")
plt.ylim((0.9, 1))
plt.tight_layout()
plt.savefig("curve_acc_jpeg.pdf", dpi=400)
