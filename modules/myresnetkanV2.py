import torch
from torch import nn
from torch.nn import functional as F
from modules.kan_convs.fast_kan_conv import FastKANConv1DLayer
from modules.kan_convs import BottleNeckKAGNConv1DLayer, ReLUKANConv1DLayer


class ResBlockKANv2(nn.Module):
    expansion: int = 1
    def __init__(self, in_channels, out_channels, basis, base, bottleneck, stride=1):
        super().__init__()
        if out_channels != in_channels or stride > 1:
            self.x_transform = basis(in_channels, out_channels, kernel_size=1, stride=stride, base_activation=base) 
        else:
            self.x_transform = nn.Identity()

        if bottleneck:
            width = int(out_channels/4)
            self.body = nn.Sequential(
                basis(in_channels,  width, kernel_size=1, stride=1, padding='same', base_activation=base),
                basis(width, width,        kernel_size=3, stride=stride, padding=1, base_activation=base),
                basis(width, out_channels, kernel_size=1, stride=1, padding='same', base_activation=base)
            )
        else:
            self.body = nn.Sequential(
                basis(in_channels,  out_channels, kernel_size=3, stride=1, padding='same', base_activation=base),
                basis(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, base_activation=base),
            )

    def forward(self, x):
        x = self.body(x) + self.x_transform(x)

        return x

class ResNet54KANv2(nn.Module):
    def __init__(self, basis, base, bottleneck):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlockKANv2(2,  8 , basis, base, bottleneck), #8x2048
            ResBlockKANv2(8,  8 , basis, base, bottleneck), 
            ResBlockKANv2(8,  8 , basis, base, bottleneck),
            ResBlockKANv2(8,  8 , basis, base, bottleneck),
            ResBlockKANv2(8,  16, basis, base, bottleneck, stride=2), #16x1024
            ResBlockKANv2(16, 16, basis, base, bottleneck),
            ResBlockKANv2(16, 16, basis, base, bottleneck),
            ResBlockKANv2(16, 32, basis, base, bottleneck, stride=2), #32x512
            ResBlockKANv2(32, 32, basis, base, bottleneck),
            ResBlockKANv2(32, 32, basis, base, bottleneck),
            ResBlockKANv2(32, 64, basis, base, bottleneck, stride=2), #64x256
            ResBlockKANv2(64, 64, basis, base, bottleneck),
            ResBlockKANv2(64, 64, basis, base, bottleneck),
            ResBlockKANv2(64, 64, basis, base, bottleneck, stride=2), #64x128
            ResBlockKANv2(64, 64, basis, base, bottleneck),
            ResBlockKANv2(64, 64, basis, base, bottleneck),
            ResBlockKANv2(64, 64, basis, base, bottleneck, stride=2), #64x64
            ResBlockKANv2(64, 64, basis, base, bottleneck),
            ResBlockKANv2(64, 64, basis, base, bottleneck),
            ResBlockKANv2(64, 32, basis, base, bottleneck), #32x64
            ResBlockKANv2(32, 32, basis, base, bottleneck),
            ResBlockKANv2(32, 32, basis, base, bottleneck),
            ResBlockKANv2(32, 32, basis, base, bottleneck),
            ResBlockKANv2(32, 32, basis, base, bottleneck),
            ResBlockKANv2(32, 16, basis, base, bottleneck), #16x64
            ResBlockKANv2(16, 16, basis, base, bottleneck),
            ResBlockKANv2(16, 16, basis, base, bottleneck),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(16, 32, 64), nn.BatchNorm1d(32), nn.ReLU(), #32x1
            nn.Conv1d(32, 2, 1), nn.Softmax(dim=1) #2x1
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)

class ResNet54DoubleKANv2(nn.Module):
    def __init__(self, basis, bottleneck):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlockKANv2(2,   16 , basis, bottleneck),
            ResBlockKANv2(16,  16 , basis, bottleneck),
            ResBlockKANv2(16,  16 , basis, bottleneck),
            ResBlockKANv2(16,  16 , basis, bottleneck),
            ResBlockKANv2(16,  32 , basis, bottleneck, stride=2),
            ResBlockKANv2(32,  32 , basis, bottleneck),
            ResBlockKANv2(32,  32 , basis, bottleneck),
            ResBlockKANv2(32,  64 , basis, bottleneck, stride=2),
            ResBlockKANv2(64,  64 , basis, bottleneck),
            ResBlockKANv2(64,  64 , basis, bottleneck),
            ResBlockKANv2(64,  128, basis, bottleneck, stride=2),
            ResBlockKANv2(128, 128, basis, bottleneck),
            ResBlockKANv2(128, 128, basis, bottleneck),
            ResBlockKANv2(128, 128, basis, bottleneck, stride=2),
            ResBlockKANv2(128, 128, basis, bottleneck),
            ResBlockKANv2(128, 128, basis, bottleneck),
            ResBlockKANv2(128, 128, basis, bottleneck, stride=2),
            ResBlockKANv2(128, 128, basis, bottleneck),
            ResBlockKANv2(128, 128, basis, bottleneck),
            ResBlockKANv2(128, 64 , basis, bottleneck),
            ResBlockKANv2(64,  64 , basis, bottleneck),
            ResBlockKANv2(64,  64 , basis, bottleneck),
            ResBlockKANv2(64,  64 , basis, bottleneck),
            ResBlockKANv2(64,  64 , basis, bottleneck),
            ResBlockKANv2(64,  32 , basis, bottleneck),
            ResBlockKANv2(32,  32 , basis, bottleneck),
            ResBlockKANv2(32,  32 , basis, bottleneck),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(32, 64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 2, 1), nn.Softmax(dim=1)
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)
    
    
class ResBlockKAXNv2(nn.Module):
    def __init__(self, in_channels, out_channels, basis, bottleneck, stride=1):
        super().__init__()
        if out_channels != in_channels or stride > 1:
            self.x_transform = basis(in_channels, out_channels, kernel_size=1, stride=stride) 
        else:
            self.x_transform = nn.Identity()
        if bottleneck:
            width = int(out_channels/4)
            self.body = nn.Sequential(
                basis(in_channels,  width, kernel_size=1, stride=1, padding='same'),
                basis(width,        width, kernel_size=3, stride=stride, padding=1),
                basis(width, out_channels, kernel_size=1, stride=1, padding='same')
            )
        else:
            self.body = nn.Sequential(
                basis(in_channels,  out_channels, kernel_size=3, stride=1, padding='same'),
                basis(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            )

    def forward(self, x):
        x = self.body(x) + self.x_transform(x)
        return x


class ResNet54KAXNv2(nn.Module):
    def __init__(self, basis, bottleneck):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlockKAXNv2(2,  8 , basis, bottleneck), #8x2048
            ResBlockKAXNv2(8,  8 , basis, bottleneck), 
            ResBlockKAXNv2(8,  8 , basis, bottleneck),
            ResBlockKAXNv2(8,  8 , basis, bottleneck),
            ResBlockKAXNv2(8,  16, basis, bottleneck, stride=2), #16x1024
            ResBlockKAXNv2(16, 16, basis, bottleneck),
            ResBlockKAXNv2(16, 16, basis, bottleneck),
            ResBlockKAXNv2(16, 32, basis, bottleneck, stride=2), #32x512
            ResBlockKAXNv2(32, 32, basis, bottleneck),
            ResBlockKAXNv2(32, 32, basis, bottleneck),
            ResBlockKAXNv2(32, 64, basis, bottleneck, stride=2), #64x256
            ResBlockKAXNv2(64, 64, basis, bottleneck),
            ResBlockKAXNv2(64, 64, basis, bottleneck),
            ResBlockKAXNv2(64, 64, basis, bottleneck, stride=2), #64x128
            ResBlockKAXNv2(64, 64, basis, bottleneck),
            ResBlockKAXNv2(64, 64, basis, bottleneck),
            ResBlockKAXNv2(64, 64, basis, bottleneck, stride=2), #64x64
            ResBlockKAXNv2(64, 64, basis, bottleneck),
            ResBlockKAXNv2(64, 64, basis, bottleneck),
            ResBlockKAXNv2(64, 32, basis, bottleneck), #32x64
            ResBlockKAXNv2(32, 32, basis, bottleneck),
            ResBlockKAXNv2(32, 32, basis, bottleneck),
            ResBlockKAXNv2(32, 32, basis, bottleneck),
            ResBlockKAXNv2(32, 32, basis, bottleneck),
            ResBlockKAXNv2(32, 16, basis, bottleneck), #16x64
            ResBlockKAXNv2(16, 16, basis, bottleneck),
            ResBlockKAXNv2(16, 16, basis, bottleneck),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(16, 32, 64), nn.BatchNorm1d(32), nn.ReLU(), #32x1
            nn.Conv1d(32, 2, 1), nn.Softmax(dim=1) #2x1
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)
    
class ResNet54DoubleKAXNv2(nn.Module):
    def __init__(self, basis, bottleneck):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlockKAXNv2(2,   16 , basis, bottleneck), #8x2048
            ResBlockKAXNv2(16,  16 , basis, bottleneck), 
            ResBlockKAXNv2(16,  16 , basis, bottleneck),
            ResBlockKAXNv2(16,  16 , basis, bottleneck),
            ResBlockKAXNv2(16,  32 , basis, bottleneck, stride=2), #16x1024
            ResBlockKAXNv2(32,  32 , basis, bottleneck),
            ResBlockKAXNv2(32,  32 , basis, bottleneck),
            ResBlockKAXNv2(32,  64 , basis, bottleneck, stride=2), #32x512
            ResBlockKAXNv2(64,  64 , basis, bottleneck),
            ResBlockKAXNv2(64,  64 , basis, bottleneck),
            ResBlockKAXNv2(64,  128, basis, bottleneck, stride=2), #64x256
            ResBlockKAXNv2(128, 128, basis, bottleneck),
            ResBlockKAXNv2(128, 128, basis, bottleneck),
            ResBlockKAXNv2(128, 128, basis, bottleneck, stride=2), #64x128
            ResBlockKAXNv2(128, 128, basis, bottleneck),
            ResBlockKAXNv2(128, 128, basis, bottleneck),
            ResBlockKAXNv2(128, 128, basis, bottleneck, stride=2), #64x64
            ResBlockKAXNv2(128, 128, basis, bottleneck),
            ResBlockKAXNv2(128, 128, basis, bottleneck),
            ResBlockKAXNv2(128, 64 , basis, bottleneck), #32x64
            ResBlockKAXNv2(64,  64 , basis, bottleneck),
            ResBlockKAXNv2(64,  64 , basis, bottleneck),
            ResBlockKAXNv2(64,  64 , basis, bottleneck),
            ResBlockKAXNv2(64,  64 , basis, bottleneck),
            ResBlockKAXNv2(64,  32 , basis, bottleneck), #16x64
            ResBlockKAXNv2(32,  32 , basis, bottleneck),
            ResBlockKAXNv2(32,  32 , basis, bottleneck),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(32, 64, 64), nn.BatchNorm1d(64), nn.ReLU(), #32x1
            nn.Conv1d(64, 2, 1), nn.Softmax(dim=1) #2x1
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)


