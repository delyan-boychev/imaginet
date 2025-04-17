import torch
import os
from torchvision import transforms
from torcheval.metrics import BinaryAccuracy, BinaryAUROC
from torchmetrics import AveragePrecision
from torch.utils.tensorboard import SummaryWriter
from networks.resnet_big import ConResNet
from .datasets import ImagiNet
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random

np.random.seed(100)
random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed(100)

model = ConResNet(name="resnet50nodown", selfcon_pos=[False, True, False], selfcon_arch="resnet", selfcon_size="fc", dataset="imaginet")
state = torch.load("./checkpoint_of_the_selfcon_model.pth", map_location="cpu")
model.load_state_dict(state["model"])
model.eval()
for module in model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.train()
classifier = nn.Sequential(nn.BatchNorm1d(2048), nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1))
classifier = classifier.to("cuda")

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
normalize = transforms.Normalize(mean=mean, std=std)
train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
train_dataset = ImagiNet("./", annotations_file="../annotations/calibration.txt", train=False, anchor=True, resize=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=200, shuffle=True,
        num_workers=16, pin_memory=True)
test_dataset = ImagiNet("./", annotations_file="../annotations/test_annotations.txt", train=False, transform=train_transform)
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=200,
        num_workers=16, pin_memory=True)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-3)
def train_epoch(loader, model, cls, optimizer, criterion, epoch):
    acc = BinaryAccuracy()
    auc = BinaryAUROC()
    acc = acc.to("cuda", non_blocking=True)
    auc = auc.to("cuda", non_blocking=True)
    l_mean = 0
    iterat = tqdm(loader)
    iterat.set_description(f"Epoch {str(epoch)}")
    k = 0
    for (images, labels_base) in iterat:
        k += 1
        images =  images.to("cuda", non_blocking=True)
        labels_base = labels_base.type(torch.FloatTensor).to("cuda", non_blocking=True)[:, 0]
        with torch.no_grad():
            _ , t = model.encoder(images)
        out = cls(t.detach())
        sig_out = torch.sigmoid(out).squeeze(1)
        loss = criterion(out.squeeze(1), labels_base)
        acc.update((sig_out >= 0.5).int(), labels_base)
        auc.update(sig_out, labels_base)
        l_mean += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iterat.set_postfix(loss=l_mean/k, auc=auc.compute().item(), acc=acc.compute().item())
    return l_mean/k, acc.compute().item(), auc.compute().item()
def eval_model(loader, model, criterion):
    acc = BinaryAccuracy()
    auc = BinaryAUROC()
    mAP = AveragePrecision(task="binary")
    acc = acc.to("cuda")
    auc = auc.to("cuda")
    mAP = mAP.to("cuda")
    l_mean = 0
    iterat = tqdm(loader)
    iterat.set_description(f"Eval")
    k = 0
    model.eval()
    with torch.no_grad():
        for (images, labels_base) in iterat:
            k += 1
            images = images.to("cuda", non_blocking=True)
            labels_base = labels_base.type(torch.FloatTensor).to("cuda", non_blocking=True)[:, 0]
            _, out = model.encoder(images)
            out = classifier(torch.nn.functional.normalize(out, p=2, dim=1))
            sig_out = torch.sigmoid(out).squeeze(1)
            loss = criterion(sig_out, labels_base)
            acc.update((sig_out >= 0.5).int(), labels_base)
            auc.update(sig_out, labels_base)
            mAP.update(sig_out, labels_base)
            l_mean += loss.item()
            iterat.set_postfix(loss=l_mean/k, auc=auc.compute().item(), acc=acc.compute().item(), mAP=mAP.compute().item())
        return l_mean/k, acc.compute().item(), auc.compute().item()
def save_model(model, classifier, optimizer, epoch, save_file):
    print('==> Saving...')
    state = {
        'model': model.state_dict(),
        'classifier': classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
dir_tb = "./linear_imaginet_tb_origin_calib"
dir_save = "./linear_imaginet_save_origin_calib"
if not os.path.exists(dir_tb):
    os.mkdir(dir_tb)
if not os.path.exists(dir_save):
    os.mkdir(dir_save)
summwrt = SummaryWriter(dir_tb, flush_secs=30)
for i in range(1, 5+1):
    loss, acc, auc = train_epoch(train_loader, model, classifier, optimizer, criterion, i)
    summwrt.add_scalar("Train Loss", loss, i)
    summwrt.add_scalar("Train Acc", acc, i)
    summwrt.add_scalar("Train AUC", auc, i)
    save_model(model, classifier, optimizer, i, f"{dir_save}/model_{i}.pt")
