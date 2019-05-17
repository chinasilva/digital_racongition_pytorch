import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import make_grid
import copy
import time

from src.MyMnistNet import MyMnistNet


def device_fun():
    device=torch.device("cuda:0" if torch.cuda.is_available() else"cpu")
    print(device)
    return device
# 可视化部分样本

def visualize_example(input_tensors):
    grid = make_grid(input_tensors[:64])
    # 转置，方便显示
    img = grid.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title('Training Examples', fontsize=24)
    plt.savefig('Training_Examples.jpg')
    plt.show()


def train_loop(epochs, model, optimizer, scheduler, criterion, device, dataloader):
    model = model.to(device)
    loss_hist, acc_hist = [], []
    best_acc = 0.
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        since = time.time()
        running_loss = 0.
        running_correct = 0
        scheduler.step()
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * images.size(0)
            running_correct += torch.sum(preds == labels.detach())
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_correct.item() / len(dataloader.dataset)
        loss_hist.append(epoch_loss)
        acc_hist.append(epoch_acc)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('Epoch: {} / {}, Loss: {:.4f}, Accuracy:{:.4f}, Time: {:.0f}m {:.0f}s'.format(
            epoch + 1, epochs, epoch_loss, epoch_acc, time_elapsed // 60, time_elapsed % 60))
    print('Best Accuracy: {:.4f}'.format(best_acc))
    return best_model_wts, loss_hist, acc_hist


def visualize_loss_acc(loss_hist, acc_hist):
    '''
    可视化损失和梯度
    '''
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(loss_hist)), loss_hist)
    plt.title('Loss', fontsize=16)
    plt.xlabel("epochs")
    plt.ylabel('loss')
    plt.subplot(1,2,2)
    plt.plot(np.arange(len(acc_hist)), acc_hist)
    plt.title('Accuracy', fontsize=16)
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.savefig('Loss and Accuracy.jpg')
    plt.show()




def eval_loop(model, device, dataloader):
    '''
    测试
    '''
    pass
    model.to(device)
    model.eval()
    result = None
    since = time.time()
    for images in dataloader:
        since = time.time()
        #visualize_example(images)
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            #print(preds.cpu().numpy()[:64])
            if result is None:
                result = preds.cpu().numpy().copy()
            else:
                result = np.hstack((result, preds.cpu().numpy()))
    time_elapsed = time.time() - since
    print('Time: {:.0f}m {:.0f}s {:.0f}ms'.format(
        time_elapsed // 60, time_elapsed % 60, time_elapsed * 1000 % 1000))
    return result


def main():
    epochs = 10
    batch_size = 512

    model = MyMnistNet()
    criterion=nn.CrossEntropyLoss()

    # 定义使用GPU
    device=device_fun()
    print(device)

    #调用cuda
    model.to(device)

    optimizer = optim.Adam(model.parameters(), weight_decay=0,
                           amsgrad=False, lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    transform = transforms.Compose([
        # transforms.Resize(28),
        transforms.ToTensor()
        # transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST("datasets/", train=True,
                             download=True, transform=transform)
    #测试集
    datasetTest = datasets.MNIST("datasets/", train=False,
                            download=True, transform=transform)

    # 定义 DataLoader
    train_loader = torch.utils.data.DataLoader(dataset, 
        batch_size=batch_size, shuffle=True, num_workers=2)
    # 不需要labels
    data = next(iter(train_loader))[0]
    visualize_example(data)



    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)


    model_state_dict, loss_hist, acc_hist = \
        train_loop(epochs, model, optimizer, scheduler, criterion, device, train_loader)

    # 损失及识别率显示
    visualize_loss_acc(loss_hist, acc_hist)

    # # 测试
    # model.load_state_dict(model_state_dict)
    # # 测试集搞什么 shuffle，吐血三升
    # test_loader = torch.utils.data.DataLoader(datasetTest, 
    #     batch_size=512, shuffle=False, num_workers=2)
        
    # test_loop(epochs, model, optimizer, scheduler, criterion, device, test_loader)


    # 保存模型
    save_model = torch.jit.trace(model,  torch.rand(1, 1, 28, 28).to(device))
    save_model.save("models/net.pth")


    
    
    # plt.ioff() # 画动态图
   
