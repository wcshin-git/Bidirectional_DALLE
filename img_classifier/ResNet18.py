import torch
import torch.nn as nn
import torchvision.models as models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.model = models.resnet18(pretrained=False, progress=False)  # randomly initialized model parameter
        self.model.fc = Identity()   # we don't need last fully connected layer

        # 2-head classifier
        self.color_classifier = nn.Linear(512, 4)
        self.number_classifier = nn.Linear(512, 10)

    def forward(self, x):
        # x (B, 3, 28, 28)
        output = self.model(x)   # (B, 512)
        color = self.color_classifier(output)  # (B, 4)
        number = self.number_classifier(output)  # (B, 10)
        return color, number

class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()

        self.model = models.resnet34(pretrained=True)
        self.model.fc = Identity()

        # 2-head classifier
        self.color_classifier = nn.Linear(512, 4)
        self.number_classifier = nn.Linear(512, 10)

    def forward(self, x):
        # x (B, 3, 28, 28)
        output = self.model(x)   # (B, 512)
        color = self.color_classifier(output)  # (B, 4)
        number = self.number_classifier(output)  # (B, 10)
        return color, number

class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()

        self.model = models.resnet50(pretrained=True)
        self.model.fc = Identity()

        # 2-head classifier
        self.color_classifier = nn.Linear(2048, 4)
        self.number_classifier = nn.Linear(2048, 10)

    def forward(self, x):
        # x (B, 3, 28, 28)
        output = self.model(x)   # (B, 512)
        color = self.color_classifier(output)  # (B, 4)
        number = self.number_classifier(output)  # (B, 10)
        return color, number