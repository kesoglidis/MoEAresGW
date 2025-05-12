from functools import partial
from typing import Callable, List, Optional, Type, Union

import torch
from torch import nn
from torch import Tensor, flatten
from torch.nn import functional as F

from modules.kan_convs import KALNConv1DLayer, BottleNeckKAGNConv1DLayer, MoEKALNConv1DLayer, MoEKAGNConv1DLayer

class ResBlockMoE(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck, num_experts, top_k, stride=1):
        super().__init__()
        # self.bottleneck = bottleneck
        if out_channels != in_channels or stride > 1:
            self.x_transform = KALNConv1DLayer(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.x_transform = nn.Identity()

        self.bottleneck = bottleneck

        if bottleneck:
            width = int(out_channels/4.0)
            self.conv1 = KALNConv1DLayer(in_channels,  width, kernel_size=1, stride=1, padding=0)
            self.conv2 = MoEKALNConv1DLayer(width, width, kernel_size=3, stride=stride, padding=1, 
                                            num_experts=num_experts, k=top_k)
            self.conv3 = KALNConv1DLayer(width, out_channels, kernel_size=1, stride=1, padding=0)
        
        else:
            self.conv1 = MoEKALNConv1DLayer(in_channels, out_channels, kernel_size=3, stride=stride, 
                                            padding=1, num_experts=num_experts, k=top_k)
            self.conv2 = KALNConv1DLayer(out_channels, out_channels, kernel_size=1,  stride=1, padding=0) # 1x1x1

    def forward(self, x: Tensor, train: bool = True) -> Tensor:
        if self.bottleneck:
            out = self.conv1(x)
            out, moe_loss = self.conv2(out, train=train)
            out = self.conv3(out)
        else:
            out, moe_loss = self.conv1(x, train=train)
            out = self.conv2(out)

        x = out + self.x_transform(x)
        # print(moe_loss)
        return x, moe_loss


class MoEResNet54KAN(nn.Module):
    def __init__(self, bottleneck, num_experts, top_k):
        super().__init__()
        self.loss = 0
        self.feature_extractor = nn.Sequential(
            ResBlockMoE(2 , 8 , bottleneck, num_experts, top_k),
            ResBlockMoE(8 , 8 , bottleneck, num_experts, top_k),
            ResBlockMoE(8 , 8 , bottleneck, num_experts, top_k),
            ResBlockMoE(8 , 8 , bottleneck, num_experts, top_k),
            ResBlockMoE(8 , 16, bottleneck, num_experts, top_k, stride=2),
            ResBlockMoE(16, 16, bottleneck, num_experts, top_k),
            ResBlockMoE(16, 16, bottleneck, num_experts, top_k),
            ResBlockMoE(16, 32, bottleneck, num_experts, top_k, stride=2),
            ResBlockMoE(32, 32, bottleneck, num_experts, top_k),
            ResBlockMoE(32, 32, bottleneck, num_experts, top_k),
            ResBlockMoE(32, 64, bottleneck, num_experts, top_k, stride=2),
            ResBlockMoE(64, 64, bottleneck, num_experts, top_k),
            ResBlockMoE(64, 64, bottleneck, num_experts, top_k),
            ResBlockMoE(64, 64, bottleneck, num_experts, top_k, stride=2),
            ResBlockMoE(64, 64, bottleneck, num_experts, top_k),
            ResBlockMoE(64, 64, bottleneck, num_experts, top_k),
            ResBlockMoE(64, 64, bottleneck, num_experts, top_k, stride=2),
            ResBlockMoE(64, 64, bottleneck, num_experts, top_k),
            ResBlockMoE(64, 64, bottleneck, num_experts, top_k),
            ResBlockMoE(64, 32, bottleneck, num_experts, top_k),
            ResBlockMoE(32, 32, bottleneck, num_experts, top_k),
            ResBlockMoE(32, 32, bottleneck, num_experts, top_k),
            ResBlockMoE(32, 32, bottleneck, num_experts, top_k),
            ResBlockMoE(32, 32, bottleneck, num_experts, top_k),
            ResBlockMoE(32, 16, bottleneck, num_experts, top_k),
            ResBlockMoE(16, 16, bottleneck, num_experts, top_k),
            ResBlockMoE(16, 16, bottleneck, num_experts, top_k) #16x64
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(16, 32, 64), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 2, 1), nn.Softmax(dim=1)
        )

    def forward(self, x, train: bool = True) -> Tensor:
        moe_loss = 0
        for block in self.feature_extractor:
            x, _moe_loss = block(x, train)
            moe_loss += _moe_loss

        # print('Final', moe_loss)
        # x, moe_loss = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2), moe_loss

class MoEResNet54DoubleKAN(nn.Module):
    def __init__(self, bottleneck, num_experts, top_k):
        super().__init__()
        self.loss = 0
        self.feature_extractor = nn.Sequential(
            ResBlockMoE(2  , 16 , bottleneck, num_experts, top_k),
            ResBlockMoE(16 , 16 , bottleneck, num_experts, top_k),
            ResBlockMoE(16 , 16 , bottleneck, num_experts, top_k),
            ResBlockMoE(16 , 16 , bottleneck, num_experts, top_k),
            ResBlockMoE(16 , 32 , bottleneck, num_experts, top_k, stride=2),
            ResBlockMoE(32 , 32 , bottleneck, num_experts, top_k),
            ResBlockMoE(32 , 32 , bottleneck, num_experts, top_k),
            ResBlockMoE(32 , 64 , bottleneck, num_experts, top_k, stride=2),
            ResBlockMoE(64 , 64 , bottleneck, num_experts, top_k),
            ResBlockMoE(64 , 64 , bottleneck, num_experts, top_k),
            ResBlockMoE(64 , 128, bottleneck, num_experts, top_k, stride=2),
            ResBlockMoE(128, 128, bottleneck, num_experts, top_k),
            ResBlockMoE(128, 128, bottleneck, num_experts, top_k),
            ResBlockMoE(128, 128, bottleneck, num_experts, top_k, stride=2),
            ResBlockMoE(128, 128, bottleneck, num_experts, top_k),
            ResBlockMoE(128, 128, bottleneck, num_experts, top_k),
            ResBlockMoE(128, 128, bottleneck, num_experts, top_k, stride=2),
            ResBlockMoE(128, 128, bottleneck, num_experts, top_k),
            ResBlockMoE(128, 128, bottleneck, num_experts, top_k),
            ResBlockMoE(128, 64 , bottleneck, num_experts, top_k),
            ResBlockMoE(64 , 64 , bottleneck, num_experts, top_k),
            ResBlockMoE(64 , 64 , bottleneck, num_experts, top_k),
            ResBlockMoE(64 , 64 , bottleneck, num_experts, top_k),
            ResBlockMoE(64 , 64 , bottleneck, num_experts, top_k),
            ResBlockMoE(64 , 32 , bottleneck, num_experts, top_k),
            ResBlockMoE(32 , 32 , bottleneck, num_experts, top_k),
            ResBlockMoE(32 , 32 , bottleneck, num_experts, top_k) #32x64
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(32, 64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 2, 1), nn.Softmax(dim=1)
        )


    def forward(self, x, train: bool = True) -> Tensor:
        moe_loss = 0
        for block in self.feature_extractor:
            x, _moe_loss = block(x, train)
            moe_loss += _moe_loss
        # print('Final', moe_loss)
        # print("what?")
        # print(self.feature_extractor[0].conv1.w_gate)
        # x, moe_loss = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2), moe_loss
    
    # def forward(self, x):
    #     x, moe_loss = self.feature_extractor(x)
    #     return self.cls_head(x).squeeze(2), moe_loss
    
