import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from src.MyNet import MyNet

class MyMain():
    def __init__(self):
        pass
    
    def main(self):
        batch=10
        batchSize=64 # 每批次的数量
        
        net=MyNet()

        #交叉熵定义
        cel=nn.CrossEntropyLoss()

        # ADAM优化器
        optimizer=optim.Adam(net.parameters)

        # 图像转换
        transform = transforms.Compose([
            transforms.CenterCrop(10),
            transforms.ToTensor(),])
        # 提取训练集    
        trainDatasets =torch.utils.data.dataloader(
             datasets.MNIST(root='../datasets/',train=True,transform=transform),
             batch_size=batchSize,collate_fn=None,pin_memory=True)
        
        # 提取测试集
        testDatasets =torch.utils.data.dataloader(
            datasets.MNIST(root='../datasets/',train=False,transform=transform),
            batch_size=batchSize,collate_fn=None,pin_memory=True)
                
        loss=[]
        for i in range(batch):
            for i, input,target in enumerate(trainDatasets):
                # print(sample.inp.is_pinned())
                # print(sample.tgt.is_pinned())
                output=net(input) # 训练集输入参数传入网络，得到输出结果
                loss=cel(output,target)

                