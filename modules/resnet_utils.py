from typing import Callable, List, Optional

import torch.nn as nn

from modules.kan_convs.reskanet import ResKANet, MoEResKANet
from modules.blocks import KANBasicBlock, FastKANBasicBlock, KALNBasicBlock, KACNBasicBlock, KAGNBasicBlock, \
KANBottleneck, FastKANBottleneck, KALNBottleneck, KACNBottleneck, KAGNBottleneck, \
BottleneckKAGNBottleneck, BottleneckKAGNBasicBlock, MoEKALNBottleneck, MoEKALNBasicBlock, MoEBottleneckKAGNBasicBlock

def reskanet_18x32p(input_channels, num_classes, groups: int = 1, spline_order: int = 3, grid_size: int = 5,
                    base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                    grid_range: List = [-1, 1], hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                    dropout_linear: float = 0.25, affine: bool = False):
    return ResKANet(KANBasicBlock, [2, 2, 2, 2],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    spline_order=spline_order, grid_size=grid_size, base_activation=base_activation,
                    grid_range=grid_range, hidden_layer_dim=hidden_layer_dim,
                    dropout_linear=dropout_linear,
                    dropout=dropout,
                    l1_decay=l1_decay,
                    affine=affine
                    )


def fast_reskanet_18x32p(input_channels, num_classes, groups: int = 1, grid_size: int = 5,
                         base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                         grid_range: List = [-1, 1], hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                         dropout_linear: float = 0.25, affine: bool = False):
    return ResKANet(FastKANBasicBlock, [2, 2, 2, 2],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    grid_size=grid_size, base_activation=base_activation,
                    grid_range=grid_range, hidden_layer_dim=hidden_layer_dim,
                    dropout_linear=dropout_linear,
                    dropout=dropout,
                    l1_decay=l1_decay,
                    affine=affine)


def reskalnet_18x32p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                     hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                     dropout_linear: float = 0.25, affine: bool = False):
    return ResKANet(KALNBasicBlock, [2, 2, 2, 2],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine
                    )


def reskagnet_18x32p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                     hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                     dropout_linear: float = 0.25, affine: bool = False):
    return ResKANet(KAGNBasicBlock, [2, 2, 2, 2],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine
                    )


def reskagnet18(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                dropout_linear: float = 0.25, affine: bool = False, norm_layer: nn.Module = nn.InstanceNorm2d):
    return ResKANet(KAGNBasicBlock, [2, 2, 2, 2],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=7, fcnv_stride=2, fcnv_padding=3,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale,
                    hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    norm_layer=norm_layer,
                    affine=affine
                    )


def reskalnet_18x64p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                     hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                     dropout_linear: float = 0.25, affine: bool = False):
    return ResKANet(KALNBasicBlock, [2, 2, 2, 2],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale,
                    hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine
                    )


def moe_reskalnet_18x64p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                         num_experts: int = 8, noisy_gating: bool = True, k: int = 2,
                         hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                         dropout_linear: float = 0.25, affine: bool = False):
    return MoEResKANet(MoEKALNBasicBlock, [2, 2, 2, 2],
                       input_channels=input_channels,
                       use_first_maxpool=True,
                       fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                       num_classes=num_classes,
                       groups=groups,
                       width_per_group=64,
                       degree=degree,
                       width_scale=width_scale,
                       num_experts=num_experts,
                       noisy_gating=noisy_gating,
                       k=k, hidden_layer_dim=hidden_layer_dim,
                       dropout=dropout,
                       dropout_linear=dropout_linear,
                       l1_decay=l1_decay,
                       affine=affine)


def reskacnet_18x32p(input_channels, num_classes, groups: int = 1, degree: int = 3,
                     hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                     dropout_linear: float = 0.25, affine: bool = False):
    return ResKANet(KACNBasicBlock, [2, 2, 2, 2],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine)


def reskalnet_50x64p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                     dropout: float = 0.15, dropout_linear: float = 0.25, l1_decay: float = 0.0,
                     hidden_layer_dim=None, affine: bool = False):
    return ResKANet(KALNBottleneck, [3, 4, 6, 3],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    hidden_layer_dim=hidden_layer_dim,
                    affine=affine)


def reskagnet50(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                dropout: float = 0.15, dropout_linear: float = 0.25, l1_decay: float = 0.0,
                hidden_layer_dim=None, affine: bool = True, norm_layer: nn.Module = nn.InstanceNorm2d):
    return ResKANet(KAGNBottleneck, [3, 4, 6, 3],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=7, fcnv_stride=2, fcnv_padding=3,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    hidden_layer_dim=hidden_layer_dim,
                    norm_layer=norm_layer,
                    affine=affine)


def reskagnet_bn50(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                   dropout: float = 0.15, dropout_linear: float = 0.25, l1_decay: float = 0.0,
                   hidden_layer_dim=None, affine: bool = True, norm_layer: nn.Module = nn.InstanceNorm2d):
    return ResKANet(BottleneckKAGNBottleneck, [3, 4, 6, 3],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=7, fcnv_stride=2, fcnv_padding=3,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    hidden_layer_dim=hidden_layer_dim,
                    norm_layer=norm_layer,
                    affine=affine)


def reskagnet101(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                 dropout: float = 0.15, dropout_linear: float = 0.25, l1_decay: float = 0.0,
                 hidden_layer_dim=None, affine: bool = True, norm_layer: nn.Module = nn.InstanceNorm2d):
    return ResKANet(KAGNBottleneck, [3, 4, 23, 3],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=7, fcnv_stride=2, fcnv_padding=3,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    hidden_layer_dim=hidden_layer_dim,
                    norm_layer=norm_layer,
                    affine=affine)


def reskagnet152(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                 dropout: float = 0.15, dropout_linear: float = 0.25, l1_decay: float = 0.0,
                 hidden_layer_dim=None, affine: bool = True, norm_layer: nn.Module = nn.InstanceNorm2d):
    return ResKANet(KAGNBottleneck, [3, 8, 36, 3],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=7, fcnv_stride=2, fcnv_padding=3,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    hidden_layer_dim=hidden_layer_dim,
                    norm_layer=norm_layer,
                    affine=affine)


def moe_reskalnet_50x64p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                         num_experts: int = 8, noisy_gating: bool = True, k: int = 2,
                         hidden_layer_dim=None, dropout: float = 0.15, dropout_linear: float = 0.25,
                         l1_decay: float = 0.0, affine: bool = False):
    return MoEResKANet(MoEKALNBottleneck, [3, 4, 6, 3],
                       input_channels=input_channels,
                       use_first_maxpool=True,
                       fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                       num_classes=num_classes,
                       groups=groups,
                       width_per_group=64,
                       degree=degree,
                       width_scale=width_scale,
                       num_experts=num_experts,
                       noisy_gating=noisy_gating,
                       k=k, hidden_layer_dim=hidden_layer_dim,
                       dropout=dropout,
                       dropout_linear=dropout_linear,
                       l1_decay=l1_decay,
                       affine=affine
                       )


def reskalnet_101x64p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                      hidden_layer_dim=None, dropout: float = 0.15, dropout_linear: float = 0.25,
                      l1_decay: float = 0.0, affine: bool = False):
    return ResKANet(KALNBottleneck, [3, 4, 23, 3],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine)


def reskagnet_101x64p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                      hidden_layer_dim=None, dropout: float = 0.15, dropout_linear: float = 0.25,
                      l1_decay: float = 0.0, affine: bool = False):
    return ResKANet(KAGNBottleneck, [3, 4, 23, 3],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine)


def reskalnet_101x32p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                      hidden_layer_dim=None, dropout: float = 0.15, dropout_linear: float = 0.25,
                      l1_decay: float = 0.0, affine: bool = False):
    return ResKANet(KALNBottleneck, [3, 4, 23, 3],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine)


def moe_reskalnet_101x64p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                          num_experts: int = 8, noisy_gating: bool = True, k: int = 2,
                          hidden_layer_dim=None, dropout: float = 0.15, dropout_linear: float = 0.25,
                          l1_decay: float = 0.0, affine: bool = False):
    return MoEResKANet(MoEKALNBottleneck, [3, 4, 23, 3],
                       input_channels=input_channels,
                       use_first_maxpool=True,
                       fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                       num_classes=num_classes,
                       groups=groups,
                       width_per_group=64,
                       degree=degree,
                       width_scale=width_scale,
                       num_experts=num_experts,
                       noisy_gating=noisy_gating,
                       k=k, hidden_layer_dim=hidden_layer_dim,
                       dropout=dropout,
                       dropout_linear=dropout_linear,
                       l1_decay=l1_decay,
                       affine=affine
                       )


def reskalnet_152x64p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                      hidden_layer_dim=None, dropout: float = 0.15, dropout_linear: float = 0.25,
                      l1_decay: float = 0.0, affine: bool = False):
    return ResKANet(KALNBottleneck, [3, 8, 36, 3],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine)


def reskalnet_152x32p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                      hidden_layer_dim=None, dropout: float = 0.15, dropout_linear: float = 0.25,
                      l1_decay: float = 0.0, affine: bool = False):
    return ResKANet(KALNBottleneck, [3, 8, 36, 3],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine)


def moe_reskalnet_152x64p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                          num_experts: int = 8, noisy_gating: bool = True, k: int = 2,
                          hidden_layer_dim=None, dropout: float = 0.15, dropout_linear: float = 0.25,
                          l1_decay: float = 0.0, affine: bool = False):
    return MoEResKANet(MoEKALNBottleneck, [3, 8, 36, 3],
                       input_channels=input_channels,
                       use_first_maxpool=True,
                       fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                       num_classes=num_classes,
                       groups=groups,
                       width_per_group=64,
                       degree=degree,
                       width_scale=width_scale,
                       num_experts=num_experts,
                       noisy_gating=noisy_gating,
                       k=k, hidden_layer_dim=hidden_layer_dim,
                       dropout=dropout,
                       dropout_linear=dropout_linear,
                       l1_decay=l1_decay,
                       affine=affine
                       )


def reskagnetbn_18x32p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                       hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                       dropout_linear: float = 0.25, affine: bool = False):
    return ResKANet(BottleneckKAGNBasicBlock, [2, 2, 2, 2],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine
                    )


def reskagnetbn_moe_18x32p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                           hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                           dropout_linear: float = 0.25, affine: bool = False,

                           num_experts: int = 8,
                           noisy_gating: bool = True,
                           k: int = 2
                           ):
    return MoEResKANet(MoEBottleneckKAGNBasicBlock, [2, 2, 2, 2],
                       input_channels=input_channels,
                       use_first_maxpool=False,
                       fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                       num_classes=num_classes,
                       groups=groups,
                       width_per_group=64,
                       degree=degree,
                       width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                       dropout=dropout,
                       dropout_linear=dropout_linear,
                       l1_decay=l1_decay,
                       affine=affine,
                       num_experts=num_experts,
                       noisy_gating=noisy_gating,
                       k=k
                       )




def reskagnetbn_34x32p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                       hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                       dropout_linear: float = 0.25, affine: bool = False):
    return ResKANet(BottleneckKAGNBasicBlock, [3, 4, 6, 3],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine
                    )


def reskagnetbn_moe_34x32p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                           hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                           dropout_linear: float = 0.25, affine: bool = False,

                           num_experts: int = 8,
                           noisy_gating: bool = True,
                           k: int = 2
                           ):
    return MoEResKANet(MoEBottleneckKAGNBasicBlock, [3, 4, 6, 3],
                       input_channels=input_channels,
                       use_first_maxpool=False,
                       fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                       num_classes=num_classes,
                       groups=groups,
                       width_per_group=64,
                       degree=degree,
                       width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                       dropout=dropout,
                       dropout_linear=dropout_linear,
                       l1_decay=l1_decay,
                       affine=affine,
                       num_experts=num_experts,
                       noisy_gating=noisy_gating,
                       k=k
                       )


def reskagnet_34x32p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                     hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                     dropout_linear: float = 0.25, affine: bool = False):
    return ResKANet(KAGNBasicBlock, [3, 4, 6, 3],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine
                    )