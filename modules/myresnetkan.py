import torch
from torch import nn
from torch.nn import functional as F
from modules.kan_convs.fast_kan_conv import FastKANConv1DLayer
from modules.kan_convs import BottleNeckKAGNConv1DLayer, ReLUKANConv1DLayer

class ResBlockKAN(nn.Module):
    def __init__(self, in_channels, out_channels, basis, base, stride=1):
        super().__init__()
        if out_channels != in_channels or stride > 1:
            self.x_transform = basis(in_channels, out_channels, kernel_size=1, stride=stride, base_activation=base) 
        else:
            self.x_transform = nn.Identity()

        self.body = nn.Sequential(
            basis(in_channels,  out_channels, kernel_size=3, stride=1, padding='same', base_activation=base),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            basis(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, base_activation=base),
            nn.BatchNorm1d(out_channels)
        )



    def forward(self, x):
        x = F.relu(self.body(x) + self.x_transform(x))

        # x = self.body(x) + self.x_transform(x)
        return x

class ResNet54KAN(nn.Module):
    def __init__(self, basis, base):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlockKAN(2,  8 , basis ,base), #8x2048 layer 1
            ResBlockKAN(8,  8 , basis ,base), 
            ResBlockKAN(8,  8 , basis ,base),
            ResBlockKAN(8,  8 , basis ,base),
            ResBlockKAN(8,  16, basis ,base, stride=2), #16x1024 layer 2
            ResBlockKAN(16, 16, basis ,base),
            ResBlockKAN(16, 16, basis ,base),
            ResBlockKAN(16, 32, basis ,base, stride=2), #32x512 layer 3
            ResBlockKAN(32, 32, basis ,base),
            ResBlockKAN(32, 32, basis ,base),
            ResBlockKAN(32, 64, basis ,base, stride=2), #64x256 layer 4
            ResBlockKAN(64, 64, basis ,base),
            ResBlockKAN(64, 64, basis ,base),
            ResBlockKAN(64, 64, basis ,base, stride=2), #64x128 layer 5
            ResBlockKAN(64, 64, basis ,base),
            ResBlockKAN(64, 64, basis ,base),
            ResBlockKAN(64, 64, basis ,base, stride=2), #64x64 layer 6
            ResBlockKAN(64, 64, basis ,base),
            ResBlockKAN(64, 64, basis ,base),
            ResBlockKAN(64, 32, basis ,base), #32x64 layer 7
            ResBlockKAN(32, 32, basis ,base),
            ResBlockKAN(32, 32, basis ,base),
            ResBlockKAN(32, 32, basis ,base),
            ResBlockKAN(32, 32, basis ,base),
            ResBlockKAN(32, 16, basis ,base), #16x64 layer 8
            ResBlockKAN(16, 16, basis ,base),
            ResBlockKAN(16, 16, basis ,base),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(16, 32, 64), nn.BatchNorm1d(32), nn.ReLU(), #32x1
            nn.Conv1d(32, 2, 1), nn.Softmax(dim=1) #2x1
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)

class ResNet54DoubleKAN(nn.Module):
    def __init__(self, basis, base):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlockKAN(2,   16 , basis, base),
            ResBlockKAN(16,  16 , basis, base),
            ResBlockKAN(16,  16 , basis, base),
            ResBlockKAN(16,  16 , basis, base),
            ResBlockKAN(16,  32 , basis, base, stride=2),
            ResBlockKAN(32,  32 , basis, base),
            ResBlockKAN(32,  32 , basis, base),
            ResBlockKAN(32,  64 , basis, base, stride=2),
            ResBlockKAN(64,  64 , basis, base),
            ResBlockKAN(64,  64 , basis, base),
            ResBlockKAN(64,  128, basis, base, stride=2),
            ResBlockKAN(128, 128, basis, base),
            ResBlockKAN(128, 128, basis, base),
            ResBlockKAN(128, 128, basis, base, stride=2),
            ResBlockKAN(128, 128, basis, base),
            ResBlockKAN(128, 128, basis, base),
            ResBlockKAN(128, 128, basis, base, stride=2),
            ResBlockKAN(128, 128, basis, base),
            ResBlockKAN(128, 128, basis, base),
            ResBlockKAN(128, 64 , basis, base),
            ResBlockKAN(64,  64 , basis, base),
            ResBlockKAN(64,  64 , basis, base),
            ResBlockKAN(64,  64 , basis, base),
            ResBlockKAN(64,  64 , basis, base),
            ResBlockKAN(64,  32 , basis, base),
            ResBlockKAN(32,  32 , basis, base),
            ResBlockKAN(32,  32 , basis, base),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(32, 64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 2, 1), nn.Softmax(dim=1)
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)
    

class ResBlockKAXN(nn.Module):
    def __init__(self, in_channels, out_channels, basis, stride=1):
        super().__init__()
        if out_channels != in_channels or stride > 1:
            self.x_transform = basis(in_channels, out_channels, kernel_size=1, stride=stride) 
        else:
            self.x_transform = nn.Identity()

        self.body = nn.Sequential(
            basis(in_channels,  out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            basis(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        x = F.relu(self.body(x) + self.x_transform(x))
        return x


class ResNet54KAXN(nn.Module):
    def __init__(self, basis):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlockKAXN(2,  8 , basis), #8x2048
            ResBlockKAXN(8,  8 , basis), 
            ResBlockKAXN(8,  8 , basis),
            ResBlockKAXN(8,  8 , basis),
            ResBlockKAXN(8,  16, basis, stride=2), #16x1024
            ResBlockKAXN(16, 16, basis),
            ResBlockKAXN(16, 16, basis),
            ResBlockKAXN(16, 32, basis, stride=2), #32x512
            ResBlockKAXN(32, 32, basis),
            ResBlockKAXN(32, 32, basis),
            ResBlockKAXN(32, 64, basis, stride=2), #64x256
            ResBlockKAXN(64, 64, basis),
            ResBlockKAXN(64, 64, basis),
            ResBlockKAXN(64, 64, basis, stride=2), #64x128
            ResBlockKAXN(64, 64, basis),
            ResBlockKAXN(64, 64, basis),
            ResBlockKAXN(64, 64, basis, stride=2), #64x64
            ResBlockKAXN(64, 64, basis),
            ResBlockKAXN(64, 64, basis),
            ResBlockKAXN(64, 32, basis), #32x64
            ResBlockKAXN(32, 32, basis),
            ResBlockKAXN(32, 32, basis),
            ResBlockKAXN(32, 32, basis),
            ResBlockKAXN(32, 32, basis),
            ResBlockKAXN(32, 16, basis), #16x64
            ResBlockKAXN(16, 16, basis),
            ResBlockKAXN(16, 16, basis),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(16, 32, 64), nn.BatchNorm1d(32), nn.ReLU(), #32x1
            nn.Conv1d(32, 2, 1), nn.Softmax(dim=1) #2x1
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)
    
class ResNet54DoubleKAXN(nn.Module):
    def __init__(self, basis):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlockKAXN(2,   16 , basis), #8x2048
            ResBlockKAXN(16,  16 , basis), 
            ResBlockKAXN(16,  16 , basis),
            ResBlockKAXN(16,  16 , basis),
            ResBlockKAXN(16,  32 , basis, stride=2), #16x1024
            ResBlockKAXN(32,  32 , basis),
            ResBlockKAXN(32,  32 , basis),
            ResBlockKAXN(32,  64 , basis, stride=2), #32x512
            ResBlockKAXN(64,  64 , basis),
            ResBlockKAXN(64,  64 , basis),
            ResBlockKAXN(64,  128, basis, stride=2), #64x256
            ResBlockKAXN(128, 128, basis),
            ResBlockKAXN(128, 128, basis),
            ResBlockKAXN(128, 128, basis, stride=2), #64x128
            ResBlockKAXN(128, 128, basis),
            ResBlockKAXN(128, 128, basis),
            ResBlockKAXN(128, 128, basis, stride=2), #64x64
            ResBlockKAXN(128, 128, basis),
            ResBlockKAXN(128, 128, basis),
            ResBlockKAXN(128, 64 , basis), #32x64
            ResBlockKAXN(64,  64 , basis),
            ResBlockKAXN(64,  64 , basis),
            ResBlockKAXN(64,  64 , basis),
            ResBlockKAXN(64,  64 , basis),
            ResBlockKAXN(64,  32 , basis), #16x64
            ResBlockKAXN(32,  32 , basis),
            ResBlockKAXN(32,  32 , basis),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(32, 64, 64), nn.BatchNorm1d(64), nn.ReLU(), #32x1
            nn.Conv1d(64, 2, 1), nn.Softmax(dim=1) #2x1
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)


