import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.line1=nn.Linear(in_features=28*28,out_features=64,bias=True)
        self.relu1=nn.ReLU(inplace=False)
        self.line2=nn.Linear(in_features=64,out_features=128,bias=True)
        self.relu2=nn.ReLU(inplace=False)
        self.line3=nn.Linear(in_features=128,out_features=10,bias=True)
        self.softmax1=nn.Softmax(dim=1)

    def forward(self,input):
        reshape= torch.reshape(input,shape=(-1,28*28))
        line1=self.line1(reshape)
        relu1=self.relu1(line1)
        line2=self.line2(relu1)
        relu2=self.relu2(line2)
        line3=self.line3(relu2)
        return self.softmax1(line3)
