import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        network = config["train_config"]["network"]
        self.model = EfficientNet.from_pretrained(network)
        if network == "efficientnet-b7":
            self.model._fc = nn.Linear(in_features=2560, out_features=num_classes, bias=True)
        elif network == "efficientnet-b2":
            self.model._fc = nn.Linear(in_features=1408, out_features=num_classes, bias=True)
        else:
            print("Network {} not implemented".format(network))
            exit()
        frozen = True
        if "frozen_layer" in config["train_config"]:
            layer_frozen = config["train_config"]["frozen_layer"]

        for name, p in self.named_parameters():
            if layer_frozen in name:
                frozen = False
            if frozen:
                p.requires_grad = False
            else:
                p.requires_grad = True
            print("Layer: {} frozen={}".format(name, p.requires_grad))

    def forward(self, x):
        return self.model.forward(x)
