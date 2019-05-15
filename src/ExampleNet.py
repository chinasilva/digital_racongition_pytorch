import torch
import torch.nn as nn


class ExampleNet(nn.Module):

    def __init__(self):
        super(ExampleNet, self).__init__()
        self.linear_5 = nn.Linear(
            in_features=28*28, out_features=64, bias=True)
        self.reLU_3 = nn.ReLU(inplace=False)
        self.linear_6 = nn.Linear(in_features=64, out_features=128, bias=True)
        self.reLU_7 = nn.ReLU(inplace=False)
        self.linear_10 = nn.Linear(in_features=128, out_features=10, bias=True)
        self.softmax_1=nn.Softmax(dim=None)

    def forward(self, x_para_1):
        x_reshape_4 = torch.reshape(x_para_1, shape=(-1, 28*28))
        x_linear_5 = self.linear_5(x_reshape_4)
        x_reLU_3 = self.reLU_3(x_linear_5)
        x_linear_6 = self.linear_6(x_reLU_3)
        x_reLU_7 = self.reLU_7(x_linear_6)
        x_linear_10 = self.linear_10(x_reLU_7)
        # print("x_linear_10",x_linear_10.shape)
        x_softmax_1=self.softmax_1(x_linear_10)
        return  x_softmax_1