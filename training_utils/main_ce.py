from __future__ import print_function

import os
import argparse
import time
import random
import numpy as np
import torch.nn as nn

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from RandAugment import RandAugment
from .datasets import ImagiNet
from torch.utils.tensorboard import SummaryWriter

from utils.util import *
from utils.tinyimagenet import TinyImageNet
from utils.imagenet import ImageNetSubset
from networks.resnet_big import LinResNet
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./save/CE')
    parser.add_argument('--resume', help='path of model checkpoint to resume', type=str, 
                        default='')
    parser.add_argument("--optimizer_name", type=str, default="sgd")
    parser.add_argument("--cuda_device", type=str, default="")

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tinyimagenet', 'imagenet', 'imagenet100', 'imaginet_origin', 'imaginet_model', 'imaginet'])
    parser.add_argument('--data_folder', type=str, default='datasets/')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)

    # model
    parser.add_argument('--model', type=str, choices=['resnet18', 'resnet50'], default='resnet50')
    # optimization
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--pretrained_imagenet', action='store_true')

    # other arguments
    parser.add_argument('--randaug', action='store_true', 
                        help='whether to add randaugment or not')
    parser.add_argument('--weakaug', action='store_true', 
                        help='whether to use weak augmentation or not')

    opt = parser.parse_args()
                
    # set the path according to the environment
    opt.model_path = '%s/%s_models' % (opt.save_dir, opt.dataset)
    opt.tb_path = '%s/%s_tensorboard' % (opt.save_dir, opt.dataset)
    opt.device = torch.device((f"cuda:{opt.cuda_device}" if opt.cuda_device != "" else "cuda") if torch.cuda.is_available() else "cpu")

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'tinyimagenet':
        opt.n_cls = 200
    elif opt.dataset == 'imagenet':
        opt.n_cls = 1000
    elif opt.dataset == 'imagenet100':
        opt.n_cls = 100
    elif opt.dataset == 'imaginet_origin':
        opt.n_cls = 2
    elif opt.dataset == 'imaginet_model':
        opt.n_cls = 4
    elif opt.dataset == 'imaginet':
        opt.n_cls = 5
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    
    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_seed_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay, opt.batch_size, opt.seed)

    # warm-up for large-batch training,
    if opt.batch_size >= 1024:
        opt.warm = True
            
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)            
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
    if opt.exp_name:
        opt.model_name = '{}_{}'.format(opt.model_name, opt.exp_name)
        
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    return opt


def set_loader(opt):
    # construct data loader
    transform = []
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size = 32
        transform = [transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)), transforms.RandomHorizontalFlip()]
        if not opt.weakaug:
            transform += [transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2)]
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        size = 32
        transform = [transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)), transforms.RandomHorizontalFlip()]
        if not opt.weakaug:
            transform += [transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2)]
    elif opt.dataset == 'tinyimagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        size = 64
        transform = [transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)), transforms.RandomHorizontalFlip()]
        if not opt.weakaug:
            transform += [transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2)]
    elif opt.dataset == 'imagenet' or opt.dataset == 'imagenet100':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        size = 224
        transform = [transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)), transforms.RandomHorizontalFlip()]
        if not opt.weakaug:
            transform += [transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2)]
    elif "imaginet" in opt.dataset:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        size = 256
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    normalize = transforms.Normalize(mean=mean, std=std)
    
    transform += [transforms.ToTensor(), normalize]
    train_transform = transforms.Compose(transform)

    if opt.dataset != "imaginet":
        if opt.randaug:
            train_transform.transforms.insert(0, RandAugment(2, 9))
            
    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
    elif opt.dataset == 'tinyimagenet':
        train_dataset = TinyImageNet(root=opt.data_folder,
                                     transform=train_transform,
                                     download=True)
    elif opt.dataset == 'imagenet':
        traindir = os.path.join(opt.data_folder, 'train')
        train_dataset = datasets.ImageFolder(root=traindir,
                                     transform=train_transform)
    elif opt.dataset == 'imagenet100':
        traindir = os.path.join(opt.data_folder, 'train')
        train_dataset = ImageNetSubset('./utils/imagenet100.txt',
                                       root=traindir,
                                       transform=train_transform)
    elif opt.dataset == 'imaginet_origin':
        train_dataset = ImagiNet(opt.data_folder, annotations_file="../annotations/train_annotations.txt", track="origin", train=True, transform=train_transform)
    elif opt.dataset == 'imaginet_model':
        train_dataset = ImagiNet(opt.data_folder, annotations_file="../annotations/train_annotations.txt", track="model", train=True, transform=train_transform)
    elif opt.dataset == 'imaginet':
        train_dataset = ImagiNet(opt.data_folder, annotations_file="../annotations/train_annotations.txt", track="all", train=True, transform=train_transform)
    else:
        raise ValueError(opt.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=None)

    return train_loader


def set_model(opt):
    n_cls = opt.n_cls
    criterion = nn.CrossEntropyLoss()
    if opt.n_cls is 2:
        n_cls = 1
        criterion = nn.BCEWithLogitsLoss()
    model = LinResNet(opt.model, n_cls)
    if opt.model == "resnet50" and opt.pretrained_imagenet:
            print("True")
            m_t = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            st = model.encoder.state_dict()
            st1 = m_t.state_dict()
            for name, param in st.items():
                if "selfcon" not in name:
                    st[name].copy_(st1[name.replace("shortcut", "downsample")])
            del st1, m_t
            model.encoder.load_state_dict(st)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(opt.device)
        criterion = criterion.to(opt.device)
        cudnn.benchmark = True
        
    return model, criterion, opt

    

def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    end = time.time()
    iterat = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
    iterat.set_description(f"Epoch {epoch}")
    for idx, (images, labels) in iterat:
        data_time.update(time.time() - end)

        bsz = labels.shape[0]
        if opt.n_cls == 2:
            labels = labels.type(torch.FloatTensor)
        else:
            if opt.dataset == "imaginet":
                labels = labels[:, 2]
            else:
                labels = labels[:, 0]
        if torch.cuda.is_available():
            images = images.to(opt.device, non_blocking=True)
            labels = labels.to(opt.device, non_blocking=True)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        losses.update(loss.item(), bsz)
        optimizer.step()
        if opt.n_cls == 2:
            acc_c = (((torch.sigmoid(out) >= 0.5) == labels).sum()/bsz)*100
        else:
            acc_c = accuracy(out, labels)[0].item()
        acc.update(acc_c)
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
           iterat.set_postfix_str('BT {batch_time.val:.3f} ({batch_time.avg:.3f}) DT {data_time.val:.3f} ({data_time.avg:.3f}) acc {acc.val:.2f} ({acc.avg:.2f})  loss {loss.val:.3f} ({loss.avg:.3f})'.format(
               batch_time=batch_time,
                data_time=data_time, loss=losses, acc=acc))

    return losses.avg

    
def main():
    opt = parse_option()
    
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
#     cudnn.deterministic = True
    
    # build model and criterion
    model, criterion, opt = set_model(opt)
        
        # build data loader
    train_loader = set_loader(opt)
        # build optimizer
    optimizer = set_optimizer(opt, model, optim_name=opt.optimizer_name)
    lr_sched = set_lr_sched(opt, optimizer)
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume, map_location="cpu")
            opt.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_sched.load_state_dict(checkpoint['lr_sched'])
            torch.set_rng_state(checkpoint["rng_states"][0])
            torch.cuda.set_rng_state(checkpoint["rng_states"][1])
            random.setstate(checkpoint["rng_states"][2])
            np.random.set_state(checkpoint["rng_states"][3])
            opt.start_epoch += 1
            print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    else:
            opt.start_epoch = 1
    writer = SummaryWriter(opt.tb_folder)   
                
        # training routine
    for epoch in range(opt.start_epoch, opt.epochs + 1):

        # train for one epoch
        #time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        #time2 = time.time()
        #print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        writer.add_scalar("loss", loss, epoch)
        writer.add_scalar('learning_rate', lr_sched.get_last_lr()[0], epoch)
        lr_sched.step()

        if opt.save_freq:
            if epoch % opt.save_freq == 0:
                save_file = os.path.join(
                    opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(model, optimizer, lr_sched, opt, epoch, save_file, rng_states=[torch.get_rng_state(), torch.cuda.get_rng_state(), random.getstate(), np.random.get_state()])

            # save the last model
        save_file = os.path.join(
                opt.save_folder, 'last.pth')
        save_model(model, optimizer, lr_sched, opt, epoch, save_file, rng_states=[torch.get_rng_state(), torch.cuda.get_rng_state(), random.getstate(), np.random.get_state()])

if __name__ == '__main__':
    main()
