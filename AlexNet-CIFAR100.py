#This code implements the AlexNet Model on the CIFAR-100 dataset
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}")

# Hyperparameters
num_classes = 100  # Changed for CIFAR-100
learning_rate = 1e-3  # Adjusted learning rate
batch_size = 128  # Adjusted batch size
num_epochs = 100  # Adjusted number of epochs
best_acc = 0  # best test accuracy
start_epoch = 0

# Define transformations for the dataset
transform_train = transforms.Compose([
    transforms.Resize(224),  # Resize the image to 224x224
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),  # Resize the image to 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Download and load the CIFAR-100 dataset
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                         download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False)

# Load pre-trained AlexNet
alexnet = models.alexnet(pretrained=True)

# Modify the last layer to match the number of classes in CIFAR-100
num_features = alexnet.classifier[6].in_features
alexnet.classifier[6] = nn.Linear(num_features, num_classes)

# Move the model to the appropriate device
alexnet.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(alexnet.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    alexnet.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = alexnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(trainset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Validation loop
    alexnet.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = alexnet(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Save the model if it has the best accuracy
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(alexnet.state_dict(), 'alexnet_cifar100.pth')
        print("Model saved!")