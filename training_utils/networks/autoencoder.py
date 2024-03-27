from torch import nn
class Autoencoder(nn.Module):
    def __init__(self, in_shape, enc_shape):
        super(Autoencoder, self).__init__()
            
        self.encode = nn.Sequential(
            nn.Linear(in_shape, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, enc_shape),
            nn.BatchNorm1d(enc_shape),
        )
            
        self.decode = nn.Sequential(
            nn.Linear(enc_shape, 64),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, in_shape),
        )
            
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x