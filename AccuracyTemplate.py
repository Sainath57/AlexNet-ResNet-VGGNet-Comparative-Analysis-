#This code is a template to find the precision, recall and f1 score of all the trained models
#It needs to be modified respectively

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import os
from sklearn.metrics import classification_report, accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}")

# Hyperparameters
num_classes = 10
batch_size = 512

# Define transformations for the dataset
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load the CIFAR-10 test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False)

# Load the trained VGG19 model
model = torchvision.models.vgg19(pretrained=False)
model.classifier[6] = nn.Linear(4096, num_classes)  # Change the output layer to match CIFAR-10
model.to(device)

# Load the trained weights
checkpoint_path = './checkpoint/ckpt.pth'
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from epoch {start_epoch} with test accuracy {best_acc}%")
else:
    print("Checkpoint not found!")

# Evaluation
model.eval()
test_loss = 0
correct = 0
total = 0
all_predicted = []
all_targets = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        all_predicted.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Calculate metrics for each class
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
print("Metrics for each class:")
class_report = classification_report(all_targets, all_predicted, target_names=class_names, output_dict=True)
for class_name in class_names:
    precision = class_report[class_name]['precision']
    recall = class_report[class_name]['recall']
    f1 = class_report[class_name]['f1-score']
    print(f"{class_name.capitalize()}: Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%, F1-Score: {f1 * 100:.2f}%")

# Calculate overall accuracy and F1 score
acc = accuracy_score(all_targets, all_predicted)
f1 = f1_score(all_targets, all_predicted, average='macro')
print(f"Overall Accuracy: {acc * 100:.2f}%")
print(f"Overall F1 Score: {f1 * 100:.2f}%")