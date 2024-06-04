import os

import torch
import torchvision.transforms as transforms
import torchvision
from torch import nn

from model import SimpleCNNModel, fgsm_attack


def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            fgsm = fgsm_attack(model, criterion, inputs, labels, epsilon=0.5)
            outputs = model(fgsm)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
model_path = os.path.join("model", "simple_model.pth")
model = SimpleCNNModel().to(device)
model.load_state_dict(torch.load(model_path))
evaluate_model(model, testloader)
