import torch
from torch import nn, Tensor
from torch.nn import functional as F
from semseg.models.modules.Convs import CustomDWConv,CustomPWConv
from semseg.models.modules.ChannelAttentionBlock import ChannelAttentionBlock
import torch.nn.init as init

class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.pwconv1 = CustomPWConv(c2)
        self.dwconv3 = CustomDWConv(c2, 3)
        self.dwconv5 = CustomDWConv(c2, 5)
        self.dwconv7 = CustomDWConv(c2, 7)
        self.pwconv2 = CustomPWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

        # Initialize fc1 layer with Kaiming initialization
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x: Tensor, H, W) -> Tensor:
        x = self.fc1(x)
        x = self.pwconv1(x, H, W)
        x1 = self.dwconv3(x, H, W)
        x2 = self.dwconv5(x, H, W)
        x3 = self.dwconv7(x, H, W)
        return self.fc2(F.gelu(self.pwconv2(x + x1 + x2 + x3, H, W)))


class FusionBlock(nn.Module):
    def __init__(self, channels, reduction=16, num_modals=2):
        super(FusionBlock, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.num_modals = num_modals

        self.liner_fusion_layers = nn.ModuleList([
            nn.Linear(self.channels[0]*self.num_modals, self.channels[0]),
            nn.Linear(self.channels[1]*self.num_modals, self.channels[1]),
            nn.Linear(self.channels[2]*self.num_modals, self.channels[2]),
            nn.Linear(self.channels[3]*self.num_modals, self.channels[3]),
        ])
        
        self.mix_ffn = nn.ModuleList([
            MixFFN(self.channels[0], self.channels[0]),
            MixFFN(self.channels[1], self.channels[1]),
            MixFFN(self.channels[2], self.channels[2]),
            MixFFN(self.channels[3], self.channels[3]),
        ])

        self.channel_attns = nn.ModuleList([
            ChannelAttentionBlock(self.channels[0]),
            ChannelAttentionBlock(self.channels[1]),
            ChannelAttentionBlock(self.channels[2]),
            ChannelAttentionBlock(self.channels[3]),
        ])

        # Initialize linear fusion layers with Kaiming initialization
        for linear_layer in self.liner_fusion_layers:
            init.kaiming_normal_(linear_layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, layer_idx):
        B, C, H, W = x[0].shape
        x = torch.cat(x, dim=1)
        x = x.flatten(2).transpose(1, 2)
        x_sum = self.liner_fusion_layers[layer_idx](x)
        x_sum = self.mix_ffn[layer_idx](x_sum, H, W) + self.channel_attns[layer_idx](x_sum, H, W)
        return x_sum.reshape(B, H, W, -1).permute(0, 3, 1, 2)