import numpy as np
import time
import torch

class Softmax_layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        e = torch.exp(x - x.max(1, True)[0] )
        summ = e.sum(1, True)[0]
        return e / summ
    
class Flatten(torch.nn.Module):
    def forward(self, x):
        N = x.shape[0]
        return x.view(N, -1)

class GestureRecognizer():
    ###
    ### Not implemented yet
    ###
    def __init__(self):
        self.conv_net = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1),
                               torch.nn.ReLU(),
                               torch.nn.BatchNorm2d(8),
                               torch.nn.Dropout2d(0.03),
                               # 32x32x8
                               
                               torch.nn.Conv2d(8, 16, 3, stride=2, padding=1),
                               torch.nn.ReLU(),
                               torch.nn.BatchNorm2d(16),
                               torch.nn.Dropout2d(0.03),
                               # 16x16x16
                               
                               torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
                               torch.nn.ReLU(),
                               torch.nn.BatchNorm2d(32),
                               torch.nn.Dropout2d(0.03),
                               # 8x8x32
                               
                               torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
                               torch.nn.ReLU(),
                               torch.nn.BatchNorm2d(64),
                               torch.nn.Dropout2d(0.03),
                               # 4x4x64
                               
                               torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
                               torch.nn.ReLU(),
                               torch.nn.BatchNorm2d(128),
                               torch.nn.Dropout2d(0.03),
                               # 2x2x128
                               
                               Flatten(),
                               torch.nn.Linear(512, 10),
                               Softmax_layer())
        
        PATH = './my_network_1.pt'
        self.conv_net.load_state_dict(torch.load(PATH))
        print(self.conv_net)

    def predict(self, frame):
        ###
        ### Not implemented yet
        ###

        return np.random.randint(0, 10)
