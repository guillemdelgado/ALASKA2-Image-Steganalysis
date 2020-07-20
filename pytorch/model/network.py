import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from model.attention import ProjectorBlock, LinearAttentionBlock


class AttentionNet(EfficientNet):
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__(blocks_args, global_params)
        #super().__init__()
        #self.model = EfficientNet.from_pretrained('efficientnet-b2')

    def build_attention(self):
        # This are my attention layers
        self.projector1 = ProjectorBlock(48, 1408)
        self.projector2 = ProjectorBlock(120, 1408)
        self.projector3 = ProjectorBlock(208, 1408)
        self.attn1 = LinearAttentionBlock(in_features=1408, normalize_attn=True)
        self.attn2 = LinearAttentionBlock(in_features=1408, normalize_attn=True)
        self.attn3 = LinearAttentionBlock(in_features=1408, normalize_attn=True)

        #This is my classification layer
        self.classification = nn.Linear(in_features=1408*3, out_features=4, bias=True)


    def forward(self, inputs):
        bs = inputs.size(0)
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            
            if idx == 7:
                l1 = x
            if idx == 15:
                l2 = x
            if idx == 20:
                l3 = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        x = self._avg_pooling(x)
        g = x
        x = x.view(bs, -1)
        c1, g1 = self.attn1(self.projector1(l1), g)
        c2, g2 = self.attn2(self.projector2(l2), g)
        c3, g3 = self.attn3(self.projector3(l3), g)
        g = torch.cat((g1, g2, g3), dim=1)  # batch_sizexC
        g = self._dropout(g)
        #x = self._fc(x)
        # l1 = Get features from self.inter_1
        x = self.classification(g)  # batch_sizexnum_classes

        return x

class Net(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()

        network = config["train_config"]["network"]

        if network == "attention-b2":
            self.model = AttentionNet.from_pretrained('efficientnet-b2')
            self.model.build_attention()
        else:
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
            print("Layer: {} Requires grad={}".format(name, p.requires_grad))


    def forward(self, x):
        return self.model.forward(x)