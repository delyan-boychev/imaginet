import torch.nn as nn
import torch
from networks.autoencoder import Autoencoder
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def dim_red(X, encoding_dim):
    if len(X.shape) > 2:
        X = X.flatten(start_dim=1)
    input_dim = X.shape[1]
    def L(X, X_, t):
        if t == 'mse':
            l=nn.MSELoss()
        elif t == "bce":
            l=nn.BCEWithLogitsLoss()
        return l(X, X_)
    def R(X):
        return torch.mm(X, torch.t(X))

    def tau(X, t):
        return torch.where(X < t, X.float(), torch.zeros(X.shape).float().to("cuda"))

    def rae_loss(alpha, L_type='mse'):
        def rae(y_true, y_pred):
            return (1-alpha)*torch.sqrt(nn.MSELoss()(y_pred, y_true)) + alpha*torch.sqrt(nn.MSELoss()(torch.cdist(y_pred, y_pred, p=2), torch.cdist(y_true, y_true, p=2)))
        return rae
    def rae_loss2(alpha, t, L_type='mse'):
        def rae(y_true, y_pred):
            return (1 - alpha)*L(y_true, y_pred, L_type) + alpha*L(tau(R(y_true), t), tau(R(y_pred), t), L_type)
        return rae
    losses = np.zeros(48)
    dataset_vecs = torch.utils.data.TensorDataset(X)
    dataloader_vecs = torch.utils.data.DataLoader(dataset_vecs, batch_size=250, shuffle=True)
    dataloader_vecs2 = torch.utils.data.DataLoader(dataset_vecs, batch_size=250, shuffle=False)
    model_test = Autoencoder(in_shape=input_dim, enc_shape=encoding_dim).to("cuda")
    model_state_dict = model_test.state_dict()
    del model_test
    for idx, j in tqdm(enumerate(np.linspace(0, 1, 50)[1:-1], 0), total=48):
        model_test = Autoencoder(in_shape=input_dim, enc_shape=encoding_dim).to("cuda")
        model_test.load_state_dict(model_state_dict)
        criterion = rae_loss(j, 'mse')
        optimizer = optim.Adam(model_test.parameters(), lr=1e-3)
        lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)
        for epoch in range(10):
            l_mean = 0
            l_cos = 0
            k = 0
            for data in dataloader_vecs:
                optimizer.zero_grad()
                data[0] = nn.functional.normalize(data[0], p=2, dim=1)
                decoded = model_test(data[0])
                loss = criterion(data[0], decoded)
                loss.backward()
                l_mean += loss.item()
                l_cos += torch.abs(R(decoded) - R(data[0])).mean() + torch.nn.functional.l1_loss(decoded, data[0]).mean()
                optimizer.step()
                k+=1
            lr_sched.step()
        losses[idx] = l_cos/k
        del model_test
    model_e = Autoencoder(in_shape=input_dim, enc_shape=encoding_dim).to("cuda")
    model_e.load_state_dict(model_state_dict)
    criterion = rae_loss((np.linspace(0, 1, 50)[1:-1])[losses.argmin()], 'mse')
    optimizer = optim.Adam(model_e.parameters(), lr=1e-3)
    lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)
    tqdm_itr = tqdm(range(200))
    for epoch in tqdm_itr:
        l_mean = 0
        k = 0
        tqdm_itr.set_description(f"Epoch {epoch+1}")
        for data in dataloader_vecs:
            optimizer.zero_grad()
            data[0] = nn.functional.normalize(data[0], p=2, dim=1)
            decoded = model_e(data[0])
            loss = criterion(data[0], decoded)
            loss.backward()
            l_mean += loss.item()
            optimizer.step()
            k+=1
            tqdm_itr.set_postfix_str(f"loss {l_mean/k:.4f}")
        lr_sched.step()
    reduced_dim = []
    with torch.no_grad():
        for data in dataloader_vecs2:
                data[0] = nn.functional.normalize(data[0], p=2, dim=1)
                encoded = model_e.encode(data[0])
                decoded = model_e.decode(encoded)
                reduced_dim.append(encoded)
    reduced_dim = torch.concat(reduced_dim, dim=0).cpu()
    return reduced_dim