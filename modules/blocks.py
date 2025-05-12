from functools import partial
from typing import Callable, List, Optional, Type, Union

import torch.nn as nn
from torch import Tensor, flatten
from modules.model_utils import *

class BasicBlockTemplate(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            conv1x1x1_fun,
            conv3x3x3_fun,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
    ) -> None:
        super().__init__()
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3x3_fun(inplanes, planes, stride=stride, groups=groups)
        self.conv2 = conv1x1x1_fun(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity

        return out


class KANBasicBlock(BasicBlockTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 spline_order: int = 3,
                 grid_size: int = 5, base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                 grid_range: List = [-1, 1],
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs
                 ):
        conv1x1x1_fun = partial(kan_conv1x1, spline_order=spline_order, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range,
                                dropout=dropout, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(kan_conv3x3, spline_order=spline_order, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range,
                                dropout=dropout, l1_decay=l1_decay, **norm_kwargs)

        super(KANBasicBlock, self).__init__(conv1x1x1_fun,
                                            conv3x3x3_fun,
                                            inplanes=inplanes,
                                            planes=planes,
                                            stride=stride,
                                            downsample=downsample,
                                            groups=groups,
                                            base_width=base_width,
                                            dilation=dilation)


class FastKANBasicBlock(BasicBlockTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 grid_size: int = 5,
                 base_activation: Optional[Callable[..., nn.Module]] = nn.SiLU,
                 grid_range: List = [-1, 1],
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs):
        conv1x1x1_fun = partial(fast_kan_conv1x1, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range,
                                l1_decay=l1_decay, dropout=dropout, **norm_kwargs)
        conv3x3x3_fun = partial(fast_kan_conv3x3, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range,
                                l1_decay=l1_decay, dropout=dropout, **norm_kwargs)

        super(FastKANBasicBlock, self).__init__(conv1x1x1_fun,
                                                conv3x3x3_fun,
                                                inplanes=inplanes,
                                                planes=planes,
                                                stride=stride,
                                                downsample=downsample,
                                                groups=groups,
                                                base_width=base_width,
                                                dilation=dilation)


class KALNBasicBlock(BasicBlockTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs):
        conv1x1x1_fun = partial(kaln_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(kaln_conv3x3, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)

        super(KALNBasicBlock, self).__init__(conv1x1x1_fun,
                                             conv3x3x3_fun,
                                             inplanes=inplanes,
                                             planes=planes,
                                             stride=stride,
                                             downsample=downsample,
                                             groups=groups,
                                             base_width=base_width,
                                             dilation=dilation)


class KAGNBasicBlock(BasicBlockTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 norm_layer: nn.Module = nn.InstanceNorm2d,
                 **norm_kwargs):
        conv1x1x1_fun = partial(kagn_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay,
                                norm_layer=norm_layer, **norm_kwargs)
        conv3x3x3_fun = partial(kagn_conv3x3, degree=degree, dropout=dropout, l1_decay=l1_decay,
                                norm_layer=norm_layer, **norm_kwargs)

        super(KAGNBasicBlock, self).__init__(conv1x1x1_fun,
                                             conv3x3x3_fun,
                                             inplanes=inplanes,
                                             planes=planes,
                                             stride=stride,
                                             downsample=downsample,
                                             groups=groups,
                                             base_width=base_width,
                                             dilation=dilation)


class BottleneckKAGNBasicBlock(BasicBlockTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 norm_layer: nn.Module = nn.BatchNorm2d,
                 **norm_kwargs):
        conv1x1x1_fun = partial(bottleneck_kagn_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay,
                                norm_layer=norm_layer, **norm_kwargs)
        conv3x3x3_fun = partial(bottleneck_kagn_conv3x3, degree=degree, dropout=dropout, l1_decay=l1_decay,
                                norm_layer=norm_layer, **norm_kwargs)

        super(BottleneckKAGNBasicBlock, self).__init__(conv1x1x1_fun,
                                                       conv3x3x3_fun,
                                                       inplanes=inplanes,
                                                       planes=planes,
                                                       stride=stride,
                                                       downsample=downsample,
                                                       groups=groups,
                                                       base_width=base_width,
                                                       dilation=dilation)


class KACNBasicBlock(BasicBlockTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs):
        conv1x1x1_fun = partial(kacn_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(kacn_conv3x3, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)

        super(KACNBasicBlock, self).__init__(conv1x1x1_fun,
                                             conv3x3x3_fun,
                                             inplanes=inplanes,
                                             planes=planes,
                                             stride=stride,
                                             downsample=downsample,
                                             groups=groups,
                                             base_width=base_width,
                                             dilation=dilation)


class BottleneckTemplate(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            conv1x1x1_fun,
            conv3x3x3_fun,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1x1_fun(inplanes, width)
        # self.bn1 = norm_layer(width)
        self.conv2 = conv3x3x3_fun(width, width, stride=stride, groups=groups, dilation=dilation)
        # self.bn2 = norm_layer(width)
        self.conv3 = conv1x1x1_fun(width, planes * self.expansion)
        # self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return out


class KANBottleneck(BottleneckTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 spline_order: int = 3,
                 grid_size: int = 5, base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                 grid_range: List = [-1, 1],
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs
                 ):
        conv1x1x1_fun = partial(kan_conv1x1, spline_order=spline_order, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range, dropout=dropout,
                                l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(kan_conv3x3, spline_order=spline_order, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range, dropout=dropout,
                                l1_decay=l1_decay, **norm_kwargs)

        super(KANBottleneck, self).__init__(conv1x1x1_fun,
                                            conv3x3x3_fun,
                                            inplanes=inplanes,
                                            planes=planes,
                                            stride=stride,
                                            downsample=downsample,
                                            groups=groups,
                                            base_width=base_width,
                                            dilation=dilation)


class FastKANBottleneck(BottleneckTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 grid_size: int = 5,
                 base_activation: Optional[Callable[..., nn.Module]] = nn.SiLU,
                 grid_range: List = [-1, 1],
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs):
        conv1x1x1_fun = partial(fast_kan_conv1x1, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range, dropout=dropout,
                                l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(fast_kan_conv3x3, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range, dropout=dropout,
                                l1_decay=l1_decay, **norm_kwargs)

        super(FastKANBottleneck, self).__init__(conv1x1x1_fun,
                                                conv3x3x3_fun,
                                                inplanes=inplanes,
                                                planes=planes,
                                                stride=stride,
                                                downsample=downsample,
                                                groups=groups,
                                                base_width=base_width,
                                                dilation=dilation)


class KALNBottleneck(BottleneckTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs):
        conv1x1x1_fun = partial(kaln_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(kaln_conv3x3, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)

        super(KALNBottleneck, self).__init__(conv1x1x1_fun,
                                             conv3x3x3_fun,
                                             inplanes=inplanes,
                                             planes=planes,
                                             stride=stride,
                                             downsample=downsample,
                                             groups=groups,
                                             base_width=base_width,
                                             dilation=dilation)


class KAGNBottleneck(BottleneckTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 norm_layer: nn.Module = nn.InstanceNorm2d,
                 **norm_kwargs):
        conv1x1_fun = partial(kagn_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay,
                              norm_layer=norm_layer, **norm_kwargs)
        conv3x3_fun = partial(kagn_conv3x3, degree=degree, dropout=dropout, l1_decay=l1_decay,
                              norm_layer=norm_layer, **norm_kwargs)

        super(KAGNBottleneck, self).__init__(conv1x1_fun,
                                             conv3x3_fun,
                                             inplanes=inplanes,
                                             planes=planes,
                                             stride=stride,
                                             downsample=downsample,
                                             groups=groups,
                                             base_width=base_width,
                                             dilation=dilation)


class BottleneckKAGNBottleneck(BottleneckTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 norm_layer: nn.Module = nn.InstanceNorm2d,
                 **norm_kwargs):
        conv1x1_fun = partial(bottleneck_kagn_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay,
                              norm_layer=norm_layer, **norm_kwargs)
        conv3x3_fun = partial(bottleneck_kagn_conv3x3, degree=degree, dropout=dropout, l1_decay=l1_decay,
                              norm_layer=norm_layer, **norm_kwargs)

        super(BottleneckKAGNBottleneck, self).__init__(conv1x1_fun,
                                                       conv3x3_fun,
                                                       inplanes=inplanes,
                                                       planes=planes,
                                                       stride=stride,
                                                       downsample=downsample,
                                                       groups=groups,
                                                       base_width=base_width,
                                                       dilation=dilation)


class MoEKALNBottleneck(BottleneckTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 num_experts: int = 8,
                 noisy_gating: bool = True,
                 k: int = 2,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs
                 ):
        conv1x1x1_fun = partial(kaln_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(moe_kaln_conv3x3, degree=degree, num_experts=num_experts,
                                k=k, noisy_gating=noisy_gating, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)

        super(MoEKALNBottleneck, self).__init__(conv1x1x1_fun,
                                                conv3x3x3_fun,
                                                inplanes=inplanes,
                                                planes=planes,
                                                stride=stride,
                                                downsample=downsample,
                                                groups=groups,
                                                base_width=base_width,
                                                dilation=dilation)

    def forward(self, x: Tensor, train: bool = True) -> Tensor:
        identity = x

        out = self.conv1(x)

        out, moe_loss = self.conv2(out, train=train)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity

        return out, moe_loss


class MoEKALNBasicBlock(BasicBlockTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 num_experts: int = 8,
                 noisy_gating: bool = True,
                 k: int = 2,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs):
        conv1x1x1_fun = partial(kaln_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(moe_kaln_conv3x3, degree=degree, num_experts=num_experts,
                                k=k, noisy_gating=noisy_gating, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)

        super(MoEKALNBasicBlock, self).__init__(conv1x1x1_fun,
                                                conv3x3x3_fun,
                                                inplanes=inplanes,
                                                planes=planes,
                                                stride=stride,
                                                downsample=downsample,
                                                groups=groups,
                                                base_width=base_width,
                                                dilation=dilation)

    def forward(self, x: Tensor, train: bool = True) -> Tensor:
        identity = x

        out, moe_loss = self.conv1(x, train=train)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity

        return out, moe_loss


class KACNBottleneck(BottleneckTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs):
        conv1x1x1_fun = partial(kacn_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(kacn_conv3x3, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)

        super(KACNBottleneck, self).__init__(conv1x1x1_fun,
                                             conv3x3x3_fun,
                                             inplanes=inplanes,
                                             planes=planes,
                                             stride=stride,
                                             downsample=downsample,
                                             groups=groups,
                                             base_width=base_width,
                                             dilation=dilation)


class MoEBottleneckKAGNBasicBlock(BasicBlockTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 num_experts: int = 8,
                 noisy_gating: bool = True,
                 k: int = 2,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs):
        conv1x1x1_fun = partial(kaln_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(moe_bottleneck_kagn_conv3x3, degree=degree, num_experts=num_experts,
                                k=k, noisy_gating=noisy_gating, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)

        super(MoEBottleneckKAGNBasicBlock, self).__init__(conv1x1x1_fun,
                                                          conv3x3x3_fun,
                                                          inplanes=inplanes,
                                                          planes=planes,
                                                          stride=stride,
                                                          downsample=downsample,
                                                          groups=groups,
                                                          base_width=base_width,
                                                          dilation=dilation)

    def forward(self, x: Tensor, train: bool = True) -> Tensor:
        identity = x

        out, moe_loss = self.conv1(x, train=train)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity

        return out, moe_loss
