from __future__ import print_function

from flash.core.optimizers import LARS
import os
import json
import pickle
import math
import numpy as np
from scipy import linalg
import torch 
from torch import nn
import torch.optim as optim

__all__ = ['AverageMeter', 'TwoCropTransform', 'accuracy', 'class_accuracy', 'set_optimizer', 'set_lr_sched', 'save_model', 'update_json', 'update_json_list', "LambdaModule"]


class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def compute_mean_cov(sum, cov_sum, num):
    mean = (sum / num).unsqueeze(0)
    cov_num = cov_sum - num * mean.t().mm(mean)
    cov = cov_num / (num - 1)
    return mean, cov
def compute_fid(mu1, sigma1, mu2, sigma2):
    r"""Compute adjusted version of `Fid Score`_.

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples

    Returns:
        Scalar value of the distance between sets.

    """
    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)

    return a + b - 2 * c

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #correct_k = correct[:k].reshape(-1, k).float().sum(1).sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def class_accuracy(output, target, cls, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        
        output = output[target == cls]
        target = target[target == cls]

        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #correct_k = correct[:k].reshape(-1, k).float().sum(1).sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, batch_size


def set_optimizer(opt, model, optim_name='sgd'):
    if optim_name == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
    elif optim_name == "lars":
        optimizer = LARS(model.parameters(), lr=opt.learning_rate,
                         momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
    elif optim_name == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=opt.learning_rate, 
                               weight_decay=opt.weight_decay)                               
    return optimizer
def set_lr_sched(opt, optimizer):
    lr_scheds = []
    if opt.warm:
        if opt.epochs > 10:
            lr_scheds.append(optim.lr_scheduler.LinearLR(optimizer, 1e-4, 1, 10))
    if opt.cosine:
        lr_scheds.append(optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs-(10 if opt.warm else 0), eta_min=opt.learning_rate*(0.1**3)))
    return optim.lr_scheduler.SequentialLR(optimizer, lr_scheds, milestones=[10])
def save_model(model, optimizer, lr_sched, opt, epoch, save_file, rng_states=None):
    print('==> Saving...')
    state = {
        'opt': opt,
        'rng_states': rng_states,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_sched': lr_sched.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
    
    
def update_json(exp_name, acc={}, path='./save/results.json'):
    for k, v in acc.items():
        acc[k] = [round(a, 2) for a in v]
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump({}, f)

    with open(path, 'r', encoding="UTF-8") as f:
        result_dict = json.load(f)
        result_dict[exp_name] = acc
    
    with open(path, 'w') as f:
        json.dump(result_dict, f)
        
    print('best accuracy: {}'.format(acc))
    print('results updated to %s' % path)
    

def update_json_list(exp_name, acc=[0., 0.], path='./save/results.json'):
    acc = [round(a, 2) for a in acc]
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump({}, f)

    with open(path, 'r', encoding="UTF-8") as f:
        result_dict = json.load(f)
        result_dict[exp_name] = acc
    
    with open(path, 'w') as f:
        json.dump(result_dict, f)
        
    print('best accuracy: {}'.format(acc))
    print('results updated to %s' % path)