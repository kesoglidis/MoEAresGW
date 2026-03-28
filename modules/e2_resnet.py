import torch
import torch.nn as nn
import torch.nn.functional as F

from e2cnn import gspaces
from e2cnn import nn as enn


class SpectrogramLayer(nn.Module):

    def __init__(self, n_fft=128, hop_length=16):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, x):
        B, C, T = x.shape
        specs = []
        for c in range(C):
            spec = torch.stft(x[:, c, :], self.n_fft, self.hop_length,
                              window=self.window, return_complex=True)
            specs.append(spec.abs())
        return torch.stack(specs, dim=1)


def _make_gspace(group_name='C4'):
    if group_name == 'C2':
        return gspaces.Rot2dOnR2(N=2)
    elif group_name == 'C4':
        return gspaces.Rot2dOnR2(N=4)
    elif group_name == 'C8':
        return gspaces.Rot2dOnR2(N=8)
    elif group_name == 'D4':
        return gspaces.FlipRot2dOnR2(N=4)
    elif group_name == 'trivial':
        return gspaces.TrivialOnR2()
    else:
        raise ValueError(f"Unknown group: {group_name}")


class E2ResBlock(enn.EquivariantModule):

    def __init__(self, in_type, out_type, stride=1):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self._need_skip = (stride > 1 or in_type.size != out_type.size)

        if self._need_skip:
            self.skip = enn.R2Conv(in_type, out_type, kernel_size=1, stride=stride, bias=False)

        self.conv1 = enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False)
        self.bn1 = enn.InnerBatchNorm(out_type)
        self.relu1 = enn.ReLU(out_type)
        self.conv2 = enn.R2Conv(out_type, out_type, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = enn.InnerBatchNorm(out_type)
        self.relu2 = enn.ReLU(out_type)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        skip = self.skip(x) if self._need_skip else x
        return self.relu2(enn.GeometricTensor(out.tensor + skip.tensor, self.out_type))

    def evaluate_output_shape(self, input_shape):
        return input_shape


class E2ResNet54(nn.Module):

    def __init__(self, bottleneck=False, group='C4', n_fft=128, hop_length=16):
        super().__init__()
        self.spec = SpectrogramLayer(n_fft, hop_length)
        r2_act = _make_gspace(group)
        self.r2_act = r2_act

        in_type = enn.FieldType(r2_act, 2 * [r2_act.trivial_repr])
        self.in_type = in_type

        def reg(n):
            return enn.FieldType(r2_act, n * [r2_act.regular_repr])

        t8 = reg(8)
        t16 = reg(16)
        t32 = reg(32)
        t64 = reg(64)

        self.lift = enn.SequentialModule(
            enn.R2Conv(in_type, t8, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(t8),
            enn.ReLU(t8),
        )

        self.feature_extractor = enn.SequentialModule(
            E2ResBlock(t8, t8),
            E2ResBlock(t8, t8),
            E2ResBlock(t8, t8),
            E2ResBlock(t8, t16, stride=2),
            E2ResBlock(t16, t16),
            E2ResBlock(t16, t16),
            E2ResBlock(t16, t32, stride=2),
            E2ResBlock(t32, t32),
            E2ResBlock(t32, t32),
            E2ResBlock(t32, t64, stride=2),
            E2ResBlock(t64, t64),
            E2ResBlock(t64, t64),
            E2ResBlock(t64, t64, stride=2),
            E2ResBlock(t64, t64),
            E2ResBlock(t64, t64),
            E2ResBlock(t64, t32),
            E2ResBlock(t32, t32),
            E2ResBlock(t32, t32),
            E2ResBlock(t32, t16),
            E2ResBlock(t16, t16),
            E2ResBlock(t16, t16),
        )

        self.pool = enn.GroupPooling(t16)
        pool_out = self.pool.out_type.size

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(pool_out, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.spec(x)
        x = enn.GeometricTensor(x, self.in_type)
        x = self.lift(x)
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.tensor
        return self.cls_head(x)


class E2ResNet54Double(nn.Module):

    def __init__(self, bottleneck=False, group='C4', n_fft=128, hop_length=16):
        super().__init__()
        self.spec = SpectrogramLayer(n_fft, hop_length)
        r2_act = _make_gspace(group)
        self.r2_act = r2_act

        in_type = enn.FieldType(r2_act, 2 * [r2_act.trivial_repr])
        self.in_type = in_type

        def reg(n):
            return enn.FieldType(r2_act, n * [r2_act.regular_repr])

        t16 = reg(16)
        t32 = reg(32)
        t64 = reg(64)
        t128 = reg(128)

        self.lift = enn.SequentialModule(
            enn.R2Conv(in_type, t16, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(t16),
            enn.ReLU(t16),
        )

        self.feature_extractor = enn.SequentialModule(
            E2ResBlock(t16, t16),
            E2ResBlock(t16, t16),
            E2ResBlock(t16, t16),
            E2ResBlock(t16, t32, stride=2),
            E2ResBlock(t32, t32),
            E2ResBlock(t32, t32),
            E2ResBlock(t32, t64, stride=2),
            E2ResBlock(t64, t64),
            E2ResBlock(t64, t64),
            E2ResBlock(t64, t128, stride=2),
            E2ResBlock(t128, t128),
            E2ResBlock(t128, t128),
            E2ResBlock(t128, t128, stride=2),
            E2ResBlock(t128, t128),
            E2ResBlock(t128, t128),
            E2ResBlock(t128, t64),
            E2ResBlock(t64, t64),
            E2ResBlock(t64, t64),
            E2ResBlock(t64, t32),
            E2ResBlock(t32, t32),
            E2ResBlock(t32, t32),
        )

        self.pool = enn.GroupPooling(t32)
        pool_out = self.pool.out_type.size

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(pool_out, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.spec(x)
        x = enn.GeometricTensor(x, self.in_type)
        x = self.lift(x)
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.tensor
        return self.cls_head(x)
