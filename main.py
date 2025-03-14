'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import RandomErasing
import mlflow
import mlflow.pytorch
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import numpy as np

import os
import argparse

from models import *
from utils import progress_bar

# Alert: Parm num no more than 5 million
# !!! warning: check num_worker & batch size
num_w = 16
# Hyper-param Ablation Exp. Table:
epoch_num = 200
train_batch_size, test_batch_size = 128, 128  # 64 / 128 / 256
learning_rate = 0.01 # 0.1 ~ 0.01 for SGD; 1e-3 ~ 1e-4 for Adam
optimizer_type = 1 # 0 -> Adam; 1 -> SGD; 2 -> AdamW
w_decay = 5e-4 # range: 5e-4 ~ 1e-3; overfit -> 1e-3 , underfit -> 5e-4 / 0 ; 5e-4 for SGD 1e-4 for Adam
scheduler_type = 2
# 0 -> MultiStep; 1 -> Exponential; 2 -> Cosine_Annealing; 3 -> Plateau; 4 -> Warmup

mlflow.set_experiment("ResnetEnhanced")
mlflow.pytorch.autolog(log_models=False)
parser = argparse.ArgumentParser(description='ResnetECA')
parser.add_argument('--lr', default=learning_rate, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
input_schema = Schema([TensorSpec(np.dtype("float32"), (-1, 3, 32, 32))]) 
output_schema = Schema([TensorSpec(np.dtype("float32"), (-1, 10))]) 
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Preparing data..')
v = [0.042, 0.090, 0.268]
transform_train = transforms.Compose([
    # 1) Scale Transform
    transforms.RandomCrop(32, padding=4),
    # transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # no 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    # 2) Pixel Transform
    # test1:
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
    # test2:
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), 
    # transforms.RandomGrayscale(p=0.1),  
    # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # no 
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    RandomErasing(scale=(0.02, 0.1), ratio=(0.3, 3.3), value=v, inplace=False),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_w)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=test_batch_size, shuffle=False, num_workers=num_w)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Building model..')

#  FAST MARK -==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-============-=-=-=-=-=-=-=-=-=-

net = Resnet_Custom()
summary(net, input_size=(1, 3, 32, 32))
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
criterion = nn.CrossEntropyLoss()
if optimizer_type == 1:
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=w_decay)
elif optimizer_type==0:
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
else:
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=w_decay)

if scheduler_type == 0:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120], gamma=0.1)
elif scheduler_type == 1:
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
elif scheduler_type == 2:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=5e-4, T_max=epoch_num)
elif scheduler_type == 3:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, threshold=1e-5, min_lr=1e-4)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=5e-4)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer'])

# Training
def train(epoch):
    current_lr = optimizer.param_groups[0]['lr']
    print(f'\nEpoch: {epoch} | Learning Rate: {current_lr:.6f}') 
    mlflow.log_metric("learning_rate", current_lr, step=epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    avg_train_loss = train_loss / len(trainloader)
    train_accuracy = 100. * correct / total
    mlflow.log_metric("train_loss", avg_train_loss, step=epoch)  
    mlflow.log_metric("train_accuracy", train_accuracy, step=epoch) 

    print(f"Epoch {epoch} Summary: Train Loss: {train_loss/len(trainloader):.4f}, Train Acc: {100.*correct/total:.2f}%")
    
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    signal = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    mlflow.log_metric("test_loss", test_loss / len(testloader), step=epoch)
    mlflow.log_metric("test_accuracy", acc, step=epoch)

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        mlflow.log_metric("best_accuracy", acc, step=epoch)
        mlflow.log_artifact('./checkpoint/ckpt.pth', artifact_path="model_weights")
        
        input_example, _ = next(iter(testloader)) 
        input_example = input_example[:1].float() 
        input_example = input_example.to(device)
        input_example_numpy = input_example.cpu().numpy()
        
        mlflow.pytorch.log_model(net, "model", pip_requirements="requirements.txt", signature=signature)
        best_acc = acc
        signal = 1
    return acc, signal


def main():
    with mlflow.start_run():  
        mlflow.log_param("epochs", epoch_num)
        mlflow.log_param("train_batch_size", train_batch_size)
        mlflow.log_param("test_batch_size", test_batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("optimizer", "SGD" if optimizer_type == 1 else "Adam")
        mlflow.log_param("weight_decay", w_decay)
        mlflow.log_param("scheduler", ["MultiStep", "Exponential", "CosineAnnealing", "Plateau"][scheduler_type])

        early_stopping_threshold = 30
        no_improve_epochs = 0
        for Epoch in range(start_epoch, start_epoch + epoch_num):
            train(Epoch)
            accuracy, s = test(Epoch)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(accuracy)
            else:
                scheduler.step()
                
            if s:
                no_improve_epochs = 0 
            else:
                no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_threshold:
                print(f'Early stopping operated at epoch {Epoch}')
                break


if __name__ == '__main__':
    main()


