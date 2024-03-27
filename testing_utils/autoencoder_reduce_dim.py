import numpy as np
import matplotlib.pyplot as plt
import argparse
from ..training_utils.networks.resnet_big import ConResNet, LinResNet
from ..training_utils.datasets import ImagiNet
from torchvision import transforms
import torch
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import torch.multiprocessing
import random
from ..training_utils.utils.dim_red_rae import dim_red
torch.multiprocessing.set_sharing_strategy('file_system')

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title")
    parser.add_argument("--data_folder", type=str, default="./")
    parser.add_argument("--model_type", choices=["ce", "selfcon"], default="selfcon")
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--model_path")
    parser.add_argument("--annotations_path")
    parser.add_argument("--save_filename")
    args = parser.parse_args()
    if args.model_type == "ce":
        model = LinResNet("resnet50", args.num_classes)
        model.load_state_dict(torch.load(args.model_path, map_location="cpu")["model"])
    elif args.model_type == "selfcon":
        model = ConResNet(selfcon_arch="resnet", selfcon_size="fc", selfcon_pos=[False, True, False], dataset="imaginet")
        model.load_state_dict(torch.load(args.model_path, map_location="cpu")["model"])
    model = model.to("cuda")
    model.eval()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = ImagiNet(args.data_folder, annotations_file=args.annotations_path, train=False, test_aug=False, track="all", transform=train_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=125, shuffle=True,
        num_workers=4, sampler=None)

    X = []
    X_train = []
    labels = []
    labels_train = []
    n=0
    with torch.no_grad():
        for (images, labels_) in tqdm(dataloader, total=len(dataloader)):
            if n == 32:
                break
            _, out2 = model.encoder(normalize(images).to("cuda"))
            X.append(out2)
            labels.append(labels_)
            n+=1
    X = torch.concat(X, dim=0)
    labels = torch.concat(labels, dim=0).numpy()
    reduced_dim = dim_red(X, 2)
    df = pd.DataFrame()
    labels1 = ["Real", "Synthetic"]
    labels2 = ["photo", "painting", "face"]
    labels3 = ["GAN", "Stable Diffusion", "Midjourney", "DALL·E"]
    
    df["y"] = [ labels1[i] for i in labels[:, 0].tolist()]
    df["x-ax"] = reduced_dim[:,0]
    df["y-ax"] = reduced_dim[:,1]

    scatter_plot = sns.scatterplot(x="x-ax", y="y-ax", hue=df.y, style=df.y,
                    palette=sns.color_palette("hls", 2),
                    markers={'Real':'s', 'Synthetic':'o'},
                    data=df, legend=True)
    scatter_plot.set(xlabel=None)
    scatter_plot.set(ylabel=None)
    scatter_plot.set(title=args.title)
    plt.legend(title="Source", loc="lower right")
    plt.savefig(args.save_filename +"_origin.pdf", dpi=1000, bbox_inches='tight',
    pad_inches=0)
    plt.clf()
    
    df = pd.DataFrame()
    df["y"] = [labels3[i] for i in labels[labels[:, 0] == 1, 2].tolist()]
    df["x-ax"] = reduced_dim[labels[:, 0] == 1, 0]
    df["y-ax"] = reduced_dim[labels[:, 0] == 1, 1]

    scatter_plot = sns.scatterplot(x="x-ax", y="y-ax", hue=df.y, style=df.y,
                    palette=sns.color_palette("hls", 4),
                    markers={"GAN":"s", "Stable Diffusion":"o", "Midjourney":"^", "DALL·E":"X"},
                    data=df, legend=True)
    scatter_plot.set(xlabel=None)
    scatter_plot.set(ylabel=None)
    scatter_plot.set(title=args.title)
    plt.legend(title="Generator", loc="lower right")
    plt.savefig(args.save_filename + "_model.pdf", dpi=1000, bbox_inches='tight',
    pad_inches=0)
    plt.clf()
