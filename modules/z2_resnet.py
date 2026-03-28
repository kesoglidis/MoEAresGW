import torch
import torch.nn as nn
import torch.nn.functional as F


class Z2LiftingConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x):
        # x: [B, C_in, T] -> [B, C_out, 2, T]
        out_e = F.conv1d(x, self.weight, self.bias, self.stride, self.padding)
        out_r = F.conv1d(x, self.weight.flip(-1), self.bias, self.stride, self.padding)
        return torch.stack([out_e, out_r], dim=2)


class Z2GroupConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.w0 = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.w1 = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding
        nn.init.kaiming_uniform_(self.w0, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.w1, nonlinearity='relu')

    def forward(self, x):
        # x: [B, C_in, 2, T] -> [B, C_out, 2, T]
        x_e = x[:, :, 0, :]
        x_r = x[:, :, 1, :]

        out_e = F.conv1d(x_e, self.w0, None, self.stride, self.padding) + \
                F.conv1d(x_r, self.w1, None, self.stride, self.padding)
        out_r = F.conv1d(x_e, self.w1.flip(-1), None, self.stride, self.padding) + \
                F.conv1d(x_r, self.w0.flip(-1), None, self.stride, self.padding)

        b = self.bias.view(1, -1, 1)
        return torch.stack([out_e + b, out_r + b], dim=2)


class Z2BatchNorm1d(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        B, C, G, T = x.shape
        return self.bn(x.reshape(B * G, C, T)).reshape(B, C, G, T)


class Z2ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if out_channels != in_channels or stride > 1:
            self.skip = Z2GroupConv1d(in_channels, out_channels, 1, stride=stride, padding=0)
        else:
            self.skip = nn.Identity()

        self.conv1 = Z2GroupConv1d(in_channels, out_channels, 3, stride=1, padding=1)
        self.bn1 = Z2BatchNorm1d(out_channels)
        self.conv2 = Z2GroupConv1d(out_channels, out_channels, 3, stride=stride, padding=1)
        self.bn2 = Z2BatchNorm1d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class Z2LiftingResBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Z2LiftingConv1d(in_channels, out_channels, 3, stride=1, padding=1)
        self.bn1 = Z2BatchNorm1d(out_channels)
        self.conv2 = Z2GroupConv1d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = Z2BatchNorm1d(out_channels)
        self.skip = Z2LiftingConv1d(in_channels, out_channels, 1, stride=1, padding=0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class Z2ResNet54(nn.Module):

    def __init__(self, bottleneck=False):
        super().__init__()
        self.lift = Z2LiftingResBlock(2, 8)
        self.feature_extractor = nn.Sequential(
            Z2ResBlock(8, 8),
            Z2ResBlock(8, 8),
            Z2ResBlock(8, 8),
            Z2ResBlock(8, 16, stride=2),
            Z2ResBlock(16, 16),
            Z2ResBlock(16, 16),
            Z2ResBlock(16, 32, stride=2),
            Z2ResBlock(32, 32),
            Z2ResBlock(32, 32),
            Z2ResBlock(32, 64, stride=2),
            Z2ResBlock(64, 64),
            Z2ResBlock(64, 64),
            Z2ResBlock(64, 64, stride=2),
            Z2ResBlock(64, 64),
            Z2ResBlock(64, 64),
            Z2ResBlock(64, 64, stride=2),
            Z2ResBlock(64, 64),
            Z2ResBlock(64, 64),
            Z2ResBlock(64, 32),
            Z2ResBlock(32, 32),
            Z2ResBlock(32, 32),
            Z2ResBlock(32, 32),
            Z2ResBlock(32, 32),
            Z2ResBlock(32, 16),
            Z2ResBlock(16, 16),
            Z2ResBlock(16, 16),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(16, 32, 64), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 2, 1), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.lift(x)
        x = self.feature_extractor(x)
        x = x.mean(dim=2)
        return self.cls_head(x).squeeze(2)


class Z2ResNet54Double(nn.Module):

    def __init__(self, bottleneck=False):
        super().__init__()
        self.lift = Z2LiftingResBlock(2, 16)
        self.feature_extractor = nn.Sequential(
            Z2ResBlock(16, 16),
            Z2ResBlock(16, 16),
            Z2ResBlock(16, 16),
            Z2ResBlock(16, 32, stride=2),
            Z2ResBlock(32, 32),
            Z2ResBlock(32, 32),
            Z2ResBlock(32, 64, stride=2),
            Z2ResBlock(64, 64),
            Z2ResBlock(64, 64),
            Z2ResBlock(64, 128, stride=2),
            Z2ResBlock(128, 128),
            Z2ResBlock(128, 128),
            Z2ResBlock(128, 128, stride=2),
            Z2ResBlock(128, 128),
            Z2ResBlock(128, 128),
            Z2ResBlock(128, 128, stride=2),
            Z2ResBlock(128, 128),
            Z2ResBlock(128, 128),
            Z2ResBlock(128, 64),
            Z2ResBlock(64, 64),
            Z2ResBlock(64, 64),
            Z2ResBlock(64, 64),
            Z2ResBlock(64, 64),
            Z2ResBlock(64, 32),
            Z2ResBlock(32, 32),
            Z2ResBlock(32, 32),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(32, 64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 2, 1), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.lift(x)
        x = self.feature_extractor(x)
        x = x.mean(dim=2)
        return self.cls_head(x).squeeze(2)
