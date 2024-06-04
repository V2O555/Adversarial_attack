import torch
from torch import nn


class SimpleCNNModel(nn.Module):
    def __init__(self):
        super(SimpleCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def fgsm_attack(model, loss_fn, images, labels, epsilon):
    model.eval()
    with torch.enable_grad():
        images.requires_grad = True

        outputs = model(images)

        loss = loss_fn(outputs, labels)

        model.zero_grad()
        loss.backward()

        data_grad = images.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_image = images + epsilon * sign_data_grad

        perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image