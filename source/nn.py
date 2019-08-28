import numpy as np
import time
import torch
import torchsummary
# import keras
# from keras.models import load_model

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
    def __init__(self, device='cpu'):
        # self.model = load_model('./model_072accReal.h5')
        self.device =device
        self.conv_net = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),
                               torch.nn.ReLU(),
#                                torch.nn.Dropout2d(0.2),
                               
                               torch.nn.BatchNorm2d(64),
                               torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
                               torch.nn.ReLU(),
                               
#                                torch.nn.BatchNorm2d(128),
                               torch.nn.Conv2d(128, 128, 3, stride=2, padding=1),
                               torch.nn.ReLU(),

                               
                               torch.nn.BatchNorm2d(128),
                               torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),
                               torch.nn.ReLU(),
    
    
#                                torch.nn.BatchNorm2d(256),
                               torch.nn.Conv2d(256, 512, 3, stride=2, padding=1),
                               torch.nn.ReLU(),
#                                torch.nn.Dropout2d(0.2),
                               
                               
                               Flatten(),
                               torch.nn.Linear(512*2*2, 128),
                               torch.nn.ReLU(),
                               torch.nn.Linear(128, 6),
                               Softmax_layer()\
                              )
#         self.conv_net = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
#                                torch.nn.LeakyReLU(),
#                                torch.nn.BatchNorm2d(32),
# #                                torch.nn.Dropout2d(0.3),
                               
#                                torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
#                                torch.nn.LeakyReLU(),
#                                torch.nn.BatchNorm2d(64),
# #                                torch.nn.Dropout2d(0.3),
                               
# #                                torch.nn.MaxPool2d(3, 2, 1),
                               
#                                torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
#                                torch.nn.LeakyReLU(),
#                                torch.nn.BatchNorm2d(128),

                               
#                                torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),
#                                torch.nn.LeakyReLU(),
#                                torch.nn.BatchNorm2d(256),

                               
#                                torch.nn.Conv2d(256, 512, 3, stride=2, padding=1),
#                                torch.nn.LeakyReLU(),
# #                                torch.nn.BatchNorm2d(512),
                               
                               
#                                Flatten(),
#                                torch.nn.Linear(512*2*2, 10),
#                                Softmax_layer()\
#                               )
        
        print(torchsummary.summary(self.conv_net, (1, 64, 64)))
        PATH = 'trained_net_2dset_63per.pt'
        # PATH = 'trained_net_2dset_73.pt'
        self.conv_net.load_state_dict(torch.load(PATH))

    def predict(self, frame):
        ###
        ### Not implemented yet
        ###
        number = 0
        self.conv_net.eval()
        self.conv_net.to(self.device)

        # frame = np.transpose(frame, (2, 0, 1))
        frame = np.expand_dims(frame, 0)
        frame = np.expand_dims(frame, 0)
        frame = torch.FloatTensor(frame)
        frame = frame.to(self.device)

        # probs = self.model.predict(frame)
        # print(np.ravel(probs))
        # number = np.argmax(np.ravel(probs))
        # print(number)

        postures = ['A', 'B', 'C', 'Five', 'V', 'Point']

        with torch.no_grad():
            probs = self.conv_net(frame)[0]
            number = torch.argmax(probs).numpy()
            number = postures[number]
            print(torch.argmax(probs))
            print(probs)

        return number
