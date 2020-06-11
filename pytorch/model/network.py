import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        #self.model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)
        self.model = EfficientNet.from_pretrained('efficientnet-b2')
        self.model._fc = nn.Linear(in_features=1408, out_features=4, bias=True)
    def forward(self, x):
        return self.model.forward(x)