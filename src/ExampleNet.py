import torch
import torch.nn as nn
import torch.nn.functional as F


class ExampleNet(nn.Module):

    def __init__(self,):
        super(ExampleNet, self).__init__()
        self.linear_5 = nn.Linear(
            in_features=28*28, out_features=128, bias=True)
            # nn.Conv2d()
        # self.reLU_3 =  nn.LeakyReLU(negative_slope=0.01,inplace=False)
        self.reLU_3 = nn.ReLU(inplace=False)
        # self.linear_6 = nn.Linear(in_features=196, out_features=128, bias=True)
        # # self.reLU_7 = nn.LeakyReLU(negative_slope=0.01,inplace=False)
        # self.reLU_7 = nn.ReLU(inplace=False)
        self.linear_10 = nn.Linear(in_features=128, out_features=64, bias=True)
        # self.reLU_8 =  nn.LeakyReLU(negative_slope=0.01,inplace=False)
        self.reLU_8 = nn.ReLU(inplace=False)
        self.linear_11 = nn.Linear(in_features=64, out_features=10, bias=True)
        # self.bn_input = nn.BatchNorm2d(1, momentum=0.5)   # 给 input 的 BN 归一化
        self.softmax_1=nn.Softmax(dim=1)

    def forward(self, x_para_1):

        # x_para_1=self.bn_input(x_para_1)
        x_reshape_4 = torch.reshape(x_para_1, shape=(-1, 28*28))
        # print("x_reshape_4:",x_reshape_4)
        x_linear_5 = self.linear_5(x_reshape_4)
        # x_linear_5=F.dropout(x_linear_5, p=0.5, training=self.training)
        # print("x_linear_5:",x_linear_5)
        x_reLU_3 = self.reLU_3(x_linear_5)
        # print("x_reLU_3:",x_reLU_3)

        # x_linear_6 = self.linear_6(x_reLU_3)
        # # x_linear_6=F.dropout(x_linear_6, p=0.5, training=self.training)
        # # print("x_linear_6:",x_linear_6)
        # x_reLU_7 = self.reLU_7(x_linear_6)
        x_linear_10 = self.linear_10(x_reLU_3)
        # print("x_linear_10:",x_linear_10)
        # x_linear_10=F.dropout(x_linear_10, p=0.5, training=self.training) # dropout 避免过拟合
        x_reLU_8 = self.reLU_8(x_linear_10)
        x_linear_11 = self.linear_11(x_reLU_8)
        x_softmax_1=self.softmax_1(x_linear_11)
        return  x_softmax_1 #x_softmax_1