import numpy as np
import os
import random
import torch
import torchvision
from tqdm import tqdm
import torchvision.transforms as transforms
from prodigyopt import Prodigy
import schedulefree
from dog import DoG,LDoG,PolynomialDecayAverager
from parameterfree import COCOB
import dadaptation
from py import *
import json

def seed_everything(seed=1029):
    '''
    Set the seed for reproducibility.
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def load_data(dataset='cifar10', batch_size=128, num_workers=1):
    """
    Loads the required dataset
    :param dataset: Can be either 'cifar10' or 'cifar100'
    :param batch_size: The desired batch size
    :return: Tuple (train_loader, test_loader, num_classes)
    """
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset == 'cifar10':
        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = 10
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    elif dataset == 'cifar100':
        num_classes = 100
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError('Only cifar 10 and cifar 100 are supported')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader, num_classes


def accuracy_and_loss(net, dataloader, device, criterion,Meth,optimizer):
    net.eval()
    if Meth == 'schedulefree':
        optimizer.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).cpu().item() / len(dataloader)

    return 100 * correct / total, loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

N_train = 50000
batch_size = 256
trainloader, testloader, num_classes = load_data(batch_size=batch_size)
checkpoint = len(trainloader) // 3 + 1

n_epoch = 200
Dataname = 'cifar10_default'
seeds = [600]
Netname = 'ResNet18'
# Netname = 'ResNet50'
# Netname = 'VGG11'
# Netname = 'SimpleDLA'

for seed in seeds:

    seed_everything(seed)
    if Netname=='ResNet18':
        net = ResNet18()
    elif Netname=='ResNet50':
        net = ResNet50()
    elif Netname=='VGG11':
        net = VGG('VGG11')
    elif Netname=='SimpleDLA':
        net = SimpleDLA() 
    net.to(device)
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    ####opt####
    Meth = 'Prodigy'

    optimizer = Prodigy(net.parameters())
    ####opt####
    
    epoch_train_losses = []
    test_losses = []
    test_acc = []
    it_test = []
    ds = []
    dlrs = []
    for epoch in tqdm(range(n_epoch)):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad(set_to_none=True)
    
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
        avg_epoch_loss = running_loss / len(trainloader)
        epoch_train_losses.append(avg_epoch_loss)
        test_a, test_l = accuracy_and_loss(net, testloader, device, criterion, Meth, optimizer)
        test_acc.append(test_a)
        test_losses.append(test_l)
        lr = optimizer.param_groups[0]['lr']
        ds.append(optimizer.param_groups[0].get('d', 1))  # Default to 1 if 'd' is not present
        dlrs.append(lr * ds[-1])
        it_test.append(epoch + 1)
        net.train()
    
    results = {
        "epoch_train_losses": epoch_train_losses,
        "test_losses": test_losses,
        "test_acc": test_acc,
        "it_test": it_test,
        "ds": ds,
        "dlrs": dlrs,
    }
    
    filename = f"./outs/{Dataname}/{Meth}_{Netname}_{seed}_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

    seed_everything(seed)
    if Netname=='ResNet18':
        net = ResNet18()
    elif Netname=='ResNet50':
        net = ResNet50()
    elif Netname=='VGG11':
        net = VGG('VGG11')
    elif Netname=='SimpleDLA':
        net = SimpleDLA() 
    net.to(device)
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    ####opt####
    Meth = 'schedulefree'
    optimizer = schedulefree.SGDScheduleFree(net.parameters())
    optimizer.train()
    ####opt####
    
    epoch_train_losses = []
    test_losses = []
    test_acc = []
    it_test = []
    dlrs = []
    for epoch in tqdm(range(n_epoch)):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad(set_to_none=True)
    
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
        avg_epoch_loss = running_loss / len(trainloader)
        epoch_train_losses.append(avg_epoch_loss)
        test_a, test_l = accuracy_and_loss(net, testloader, device, criterion, Meth, optimizer)
        test_acc.append(test_a)
        test_losses.append(test_l)
        lr = optimizer.param_groups[0]['lr']
        ds.append(optimizer.param_groups[0].get('d', 1))  # Default to 1 if 'd' is not present
        dlrs.append(lr * ds[-1])
        it_test.append(epoch + 1)
        net.train()
        optimizer.train()
    
    results = {
        "epoch_train_losses": epoch_train_losses,
        "test_losses": test_losses,
        "test_acc": test_acc,
        "it_test": it_test,
        "dlrs": dlrs,
    }
    
    filename = f"./outs/{Dataname}/{Meth}_{Netname}_{seed}_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

    seed_everything(seed)
    if Netname=='ResNet18':
        net = ResNet18()
    elif Netname=='ResNet50':
        net = ResNet50()
    elif Netname=='VGG11':
        net = VGG('VGG11')
    elif Netname=='SimpleDLA':
        net = SimpleDLA() 
    net.to(device)
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    ####opt####
    Meth = 'Dog'
    optimizer = DoG(net.parameters(), weight_decay=0)
    averager = PolynomialDecayAverager(net,gamma=8)
    ####opt####
    
    epoch_train_losses = []
    test_losses = []
    test_acc = []
    it_test = []
    dlrs = []
    for epoch in tqdm(range(n_epoch)):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad(set_to_none=True)
    
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            averager.step()
    
            running_loss += loss.item()
    
        avg_epoch_loss = running_loss / len(trainloader)
        epoch_train_losses.append(avg_epoch_loss)
        test_a, test_l = accuracy_and_loss(net, testloader, device, criterion, Meth, optimizer)
        test_acc.append(test_a)
        test_losses.append(test_l)
        lr = optimizer.param_groups[0]['lr']
        ds.append(optimizer.param_groups[0].get('d', 1))  # Default to 1 if 'd' is not present
        dlrs.append(lr * ds[-1])
        it_test.append(epoch + 1)
        net.train()
        
    results = {
        "epoch_train_losses": epoch_train_losses,
        "test_losses": test_losses,
        "test_acc": test_acc,
        "it_test": it_test,
        "dlrs": dlrs,
    }
    
    filename = f"./outs/{Dataname}/{Meth}_{Netname}_{seed}_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

    seed_everything(seed)
    if Netname=='ResNet18':
        net = ResNet18()
    elif Netname=='ResNet50':
        net = ResNet50()
    elif Netname=='VGG11':
        net = VGG('VGG11')
    elif Netname=='SimpleDLA':
        net = SimpleDLA() 
    net.to(device)
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    ####opt####
    Meth = 'LDog'
    optimizer = LDoG(net.parameters(),weight_decay=0)
    averager = PolynomialDecayAverager(net,gamma=8)
    ####opt####
    
    epoch_train_losses = []
    test_losses = []
    test_acc = []
    it_test = []
    dlrs = []
    for epoch in tqdm(range(n_epoch)):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad(set_to_none=True)
    
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            averager.step()
    
            running_loss += loss.item()
    
        avg_epoch_loss = running_loss / len(trainloader)
        epoch_train_losses.append(avg_epoch_loss)
        test_a, test_l = accuracy_and_loss(net, testloader, device, criterion, Meth, optimizer)
        test_acc.append(test_a)
        test_losses.append(test_l)
        lr = optimizer.param_groups[0]['lr']
        ds.append(optimizer.param_groups[0].get('d', 1))  # Default to 1 if 'd' is not present
        dlrs.append(lr * ds[-1])
        it_test.append(epoch + 1)
        net.train()
        
    results = {
        "epoch_train_losses": epoch_train_losses,
        "test_losses": test_losses,
        "test_acc": test_acc,
        "it_test": it_test,
        "dlrs": dlrs,
    }
    
    filename = f"./outs/{Dataname}/{Meth}_{Netname}_{seed}_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

    seed_everything(seed)
    if Netname=='ResNet18':
        net = ResNet18()
    elif Netname=='ResNet50':
        net = ResNet50()
    elif Netname=='VGG11':
        net = VGG('VGG11')
    elif Netname=='SimpleDLA':
        net = SimpleDLA() 
    net.to(device)
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    ####opt####
    Meth = 'dadaptation_SGD'
    optimizer = dadaptation.DAdaptSGD(net.parameters())
    ####opt####
    
    epoch_train_losses = []
    test_losses = []
    test_acc = []
    it_test = []
    dlrs = []
    for epoch in tqdm(range(n_epoch)):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad(set_to_none=True)
    
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
        avg_epoch_loss = running_loss / len(trainloader)
        epoch_train_losses.append(avg_epoch_loss)
        test_a, test_l = accuracy_and_loss(net, testloader, device, criterion, Meth, optimizer)
        test_acc.append(test_a)
        test_losses.append(test_l)
        lr = optimizer.param_groups[0]['lr']
        ds.append(optimizer.param_groups[0].get('d', 1))  # Default to 1 if 'd' is not present
        dlrs.append(lr * ds[-1])
        it_test.append(epoch + 1)
        net.train()
        
    results = {
        "epoch_train_losses": epoch_train_losses,
        "test_losses": test_losses,
        "test_acc": test_acc,
        "it_test": it_test,
        "dlrs": dlrs,
    }
    
    filename = f"./outs/{Dataname}/{Meth}_{Netname}_{seed}_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

    seed_everything(seed)
    if Netname=='ResNet18':
        net = ResNet18()
    elif Netname=='ResNet50':
        net = ResNet50()
    elif Netname=='VGG11':
        net = VGG('VGG11')
    elif Netname=='SimpleDLA':
        net = SimpleDLA() 
    net.to(device)
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    ####opt####
    Meth = 'dadaptation_Adam'
    optimizer = dadaptation.DAdaptAdam(net.parameters())
    ####opt####
    
    epoch_train_losses = []
    test_losses = []
    test_acc = []
    it_test = []
    dlrs = []
    for epoch in tqdm(range(n_epoch)):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad(set_to_none=True)
    
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
        avg_epoch_loss = running_loss / len(trainloader)
        epoch_train_losses.append(avg_epoch_loss)
        test_a, test_l = accuracy_and_loss(net, testloader, device, criterion, Meth, optimizer)
        test_acc.append(test_a)
        test_losses.append(test_l)
        lr = optimizer.param_groups[0]['lr']
        ds.append(optimizer.param_groups[0].get('d', 1))  # Default to 1 if 'd' is not present
        dlrs.append(lr * ds[-1])
        it_test.append(epoch + 1)
        net.train()
    results = {
        "epoch_train_losses": epoch_train_losses,
        "test_losses": test_losses,
        "test_acc": test_acc,
        "it_test": it_test,
        "dlrs": dlrs,
    }
    
    filename = f"./outs/{Dataname}/{Meth}_{Netname}_{seed}_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

    seed_everything(seed)
    if Netname=='ResNet18':
        net = ResNet18()
    elif Netname=='ResNet50':
        net = ResNet50()
    elif Netname=='VGG11':
        net = VGG('VGG11')
    elif Netname=='SimpleDLA':
        net = SimpleDLA() 
    net.to(device)
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    ####opt####
    Meth = 'COCOB'
    optimizer = COCOB(net.parameters())
    ####opt####
    
    epoch_train_losses = []
    test_losses = []
    test_acc = []
    it_test = []
    dlrs = []
    for epoch in tqdm(range(n_epoch)):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad(set_to_none=True)
    
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
        avg_epoch_loss = running_loss / len(trainloader)
        epoch_train_losses.append(avg_epoch_loss)
        test_a, test_l = accuracy_and_loss(net, testloader, device, criterion, Meth, optimizer)
        test_acc.append(test_a)
        test_losses.append(test_l)
        it_test.append(epoch + 1)
        net.train()
    results = {
        "epoch_train_losses": epoch_train_losses,
        "test_losses": test_losses,
        "test_acc": test_acc,
        "it_test": it_test,
        "dlrs": dlrs,
    }
    
    filename = f"./outs/{Dataname}/{Meth}_{Netname}_{seed}_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

    seed_everything(seed)
    if Netname == 'ResNet18':
        net = ResNet18()
    elif Netname == 'ResNet50':
        net = ResNet50()
    elif Netname == 'VGG11':
        net = VGG('VGG11')
    elif Netname == 'SimpleDLA':
        net = SimpleDLA() 
    net.to(device)
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    ####opt####
    Meth = 'Adam'
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
    ####opt####
    
    epoch_train_losses = []
    test_losses = []
    test_acc = []
    it_test = []
    dlrs = []
    for epoch in tqdm(range(n_epoch)):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad(set_to_none=True)
    
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
        avg_epoch_loss = running_loss / len(trainloader)
        epoch_train_losses.append(avg_epoch_loss)
        test_a, test_l = accuracy_and_loss(net, testloader, device, criterion, Meth, optimizer)
        test_acc.append(test_a)
        test_losses.append(test_l)
        lr = optimizer.param_groups[0]['lr']
        dlrs.append(lr)
        it_test.append(epoch + 1)
        net.train()
    results = {
        "epoch_train_losses": epoch_train_losses,
        "test_losses": test_losses,
        "test_acc": test_acc,
        "it_test": it_test,
        "dlrs": dlrs,
    }
    
    filename = f"./outs/{Dataname}/{Meth}_{Netname}_{seed}_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

    seed_everything(seed)
    if Netname == 'ResNet18':
        net = ResNet18()
    elif Netname == 'ResNet50':
        net = ResNet50()
    elif Netname == 'VGG11':
        net = VGG('VGG11')
    elif Netname == 'SimpleDLA':
        net = SimpleDLA() 
    net.to(device)
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    ####opt####
    Meth = 'Adadelta' 
    optimizer = torch.optim.Adadelta(net.parameters(),lr=0.1)  # 使用 Adadelta 优化器
    ####opt####
    
    epoch_train_losses = []
    test_losses = []
    test_acc = []
    it_test = []
    dlrs = []
    for epoch in tqdm(range(n_epoch)):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad(set_to_none=True)
    
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
        avg_epoch_loss = running_loss / len(trainloader)
        epoch_train_losses.append(avg_epoch_loss)
        test_a, test_l = accuracy_and_loss(net, testloader, device, criterion, Meth, optimizer)
        test_acc.append(test_a)
        test_losses.append(test_l)
        lr = optimizer.param_groups[0]['lr']
        dlrs.append(lr)
        it_test.append(epoch + 1)
        net.train()
    results = {
        "epoch_train_losses": epoch_train_losses,
        "test_losses": test_losses,
        "test_acc": test_acc,
        "it_test": it_test,
        "dlrs": dlrs,
    }
    
    filename = f"./outs/{Dataname}/{Meth}_{Netname}_{seed}_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

    seed_everything(seed)
    if Netname == 'ResNet18':
        net = ResNet18()
    elif Netname == 'ResNet50':
        net = ResNet50()
    elif Netname == 'VGG11':
        net = VGG('VGG11')
    elif Netname == 'SimpleDLA':
        net = SimpleDLA() 
    net.to(device)
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    ####opt####
    Meth = 'SGD_StepLR'  # 方法名称反映优化器和调度器类型
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) 
    ####opt####
    
    epoch_train_losses = []
    test_losses = []
    test_acc = []
    it_test = []
    dlrs = []
    for epoch in tqdm(range(n_epoch)):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad(set_to_none=True)
    
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
        avg_epoch_loss = running_loss / len(trainloader)
        epoch_train_losses.append(avg_epoch_loss)
        test_a, test_l = accuracy_and_loss(net, testloader, device, criterion, Meth, optimizer)
        test_acc.append(test_a)
        test_losses.append(test_l)
        lr = optimizer.param_groups[0]['lr']
        dlrs.append(lr)
        it_test.append(epoch + 1)
    
        # 调整学习率
        scheduler.step()
        net.train()
    results = {
        "epoch_train_losses": epoch_train_losses,
        "test_losses": test_losses,
        "test_acc": test_acc,
        "it_test": it_test,
        "dlrs": dlrs,
    }
    
    filename = f"./outs/{Dataname}/{Meth}_{Netname}_{seed}_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

    seed_everything(seed)
    if Netname == 'ResNet18':
        net = ResNet18()
    elif Netname == 'ResNet50':
        net = ResNet50()
    elif Netname == 'VGG11':
        net = VGG('VGG11')
    elif Netname == 'SimpleDLA':
        net = SimpleDLA() 
    net.to(device)
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    ####opt####
    Meth = 'SGD_MultiStepLR'  # 方法名称反映优化器和调度器类型
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 125, 175], gamma=0.1)
    ####opt####
    
    epoch_train_losses = []
    test_losses = []
    test_acc = []
    it_test = []
    dlrs = []
    for epoch in tqdm(range(n_epoch)):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad(set_to_none=True)
    
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
        avg_epoch_loss = running_loss / len(trainloader)
        epoch_train_losses.append(avg_epoch_loss)
        test_a, test_l = accuracy_and_loss(net, testloader, device, criterion, Meth, optimizer)
        test_acc.append(test_a)
        test_losses.append(test_l)
        lr = optimizer.param_groups[0]['lr']
        dlrs.append(lr)
        it_test.append(epoch + 1)
    
        # 调整学习率
        scheduler.step()
        net.train()
    results = {
        "epoch_train_losses": epoch_train_losses,
        "test_losses": test_losses,
        "test_acc": test_acc,
        "it_test": it_test,
        "dlrs": dlrs,
    }
    
    filename = f"./outs/{Dataname}/{Meth}_{Netname}_{seed}_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

    seed_everything(seed)
    if Netname == 'ResNet18':
        net = ResNet18()
    elif Netname == 'ResNet50':
        net = ResNet50()
    elif Netname == 'VGG11':
        net = VGG('VGG11')
    elif Netname == 'SimpleDLA':
        net = SimpleDLA() 
    net.to(device)
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    ####opt####
    Meth = 'SGD_CosLR'  # 方法名称反映优化器和调度器类型
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
    ####opt####
    
    epoch_train_losses = []
    test_losses = []
    test_acc = []
    it_test = []
    dlrs = []
    for epoch in tqdm(range(n_epoch)):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad(set_to_none=True)
    
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
        avg_epoch_loss = running_loss / len(trainloader)
        epoch_train_losses.append(avg_epoch_loss)
        test_a, test_l = accuracy_and_loss(net, testloader, device, criterion, Meth, optimizer)
        test_acc.append(test_a)
        test_losses.append(test_l)
        lr = optimizer.param_groups[0]['lr']
        dlrs.append(lr)
        it_test.append(epoch + 1)
    
        # 调整学习率
        scheduler.step()
        net.train()
    results = {
        "epoch_train_losses": epoch_train_losses,
        "test_losses": test_losses,
        "test_acc": test_acc,
        "it_test": it_test,
        "dlrs": dlrs,
    }
    
    filename = f"./outs/{Dataname}/{Meth}_{Netname}_{seed}_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")