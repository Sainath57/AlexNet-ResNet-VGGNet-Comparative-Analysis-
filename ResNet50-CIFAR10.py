#This code implements the Restnet50 Model on the CIFAR-10 dataset
#Pytorch is used for implementation
#Code Reference:- https://discuss.pytorch.org/t/resnet50-torchvision-implementation-gives-low-accuracy-on-cifar-10/82046

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import os

#This code is a way to check if a CUDA GPU is available for computations, and if so, it sets the device to CUDA; otherwise, it falls back to using the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}")

# Hyperparameters are tuned according; we did many trial and errors, these work best for CIFAR-10 dataset
in_channel = 1
num_classes = 10
learning_rate =3e-4
batch_size = 512
num_epochs = 50
best_acc = 0  # best test accuracy
start_epoch = 0

# Define transformations for the dataset (Data Augmentation)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
    #For normalization values, referred https://github.com/kuangliu/pytorch-cifar/issues/19
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Download and load the CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False)

#Using the latest Imagenet weights to train the Restnet50 model
model = models.resnet50(weights="IMAGENET1K_V2")

#This line of code retrieves the number of input features to the fully connected layer before modifying it.
num_ftrs = model.fc.in_features
#By replacing the fully connected layer, you essentially redefine the architecture of the model to adapt it to your specific task.
model.fc = nn.Linear(num_ftrs, 10)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
#Using the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
#Using LR Conine Annealing for scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

#Move model to the device
model.to(device)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    loop = tqdm(train_loader)
    for batch_idx, (inputs, targets) in enumerate(loop):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(loss=train_loss/(batch_idx+1), acc=100.*correct/total)


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(test_loader)
        for batch_idx, (inputs, targets) in enumerate(loop):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=test_loss/(batch_idx+1), acc=100.*correct/total)

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


#We have implemented the training and testing set simultaenously to monitor the the learning progress and make changes in the parameters
for epoch in range(start_epoch, start_epoch+50):
    train(epoch)
    test(epoch)
    scheduler.step()