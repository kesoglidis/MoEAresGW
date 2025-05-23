from functools import partial
from typing import Callable, List, Optional, Type, Union

import torch.nn as nn
from torch import Tensor, flatten

from modules.kan_convs import KALNConv2DLayer, KANConv2DLayer, KACNConv2DLayer, FastKANConv2DLayer, KAGNConv2DLayer, \
    BottleNeckKAGNConv2DLayer
from modules.blocks import KANBasicBlock, FastKANBasicBlock, KALNBasicBlock, KACNBasicBlock, KAGNBasicBlock, \
KANBottleneck, FastKANBottleneck, KALNBottleneck, KACNBottleneck, KAGNBottleneck, \
BottleneckKAGNBottleneck, BottleneckKAGNBasicBlock, MoEKALNBottleneck, MoEKALNBasicBlock, MoEBottleneckKAGNBasicBlock
from modules.model_utils import kan_conv1x1, kagn_conv1x1, kacn_conv1x1, kaln_conv1x1, fast_kan_conv1x1, \
    bottleneck_kagn_conv1x1
# from modules.model_utils import kan_conv3x3, kagn_conv3x3, kacn_conv3x3, kaln_conv3x3, fast_kan_conv3x3, moe_kaln_conv3x3, \
#     bottleneck_kagn_conv3x3
# from modules.model_utils import moe_bottleneck_kagn_conv3x3

class ResKANet(nn.Module):
    def __init__(
            self,
            block: Type[Union[KANBasicBlock, FastKANBasicBlock, KALNBasicBlock, KACNBasicBlock, KAGNBasicBlock,
                              KANBottleneck, FastKANBottleneck, KALNBottleneck, KACNBottleneck, KAGNBottleneck,
                              BottleneckKAGNBottleneck, BottleneckKAGNBasicBlock]],
            layers: List[int],
            input_channels: int = 3,
            use_first_maxpool: bool = True,
            mp_kernel_size: int = 3, mp_stride: int = 2, mp_padding: int = 1,
            fcnv_kernel_size: int = 7, fcnv_stride: int = 2, fcnv_padding: int = 3,
            num_classes: int = 1000,
            groups: int = 1,
            width_per_group: int = 64,
            width_scale: int = 1,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            dropout_linear: float = 0.25,
            hidden_layer_dim: int = None,
            norm_layer: nn.Module = nn.BatchNorm2d,
            **kan_kwargs
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.inplanes = 8 * width_scale
        self.hidden_layer_dim = hidden_layer_dim
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.use_first_maxpool = use_first_maxpool

        self.hidden_layer = None

        kan_kwargs_clean = kan_kwargs.copy()
        kan_kwargs_clean.pop('l1_decay', None)
        kan_kwargs_clean.pop('groups', None)

        kan_kwargs_fc = kan_kwargs.copy()
        kan_kwargs_fc.pop('groups', None)
        kan_kwargs_fc.pop('dropout', None)
        kan_kwargs_fc['dropout'] = dropout_linear

        if hidden_layer_dim is not None:
            fc_layers = [64 * width_scale * block.expansion, hidden_layer_dim, num_classes]
        else:
            fc_layers = [64 * width_scale * block.expansion, num_classes]

        if block in (KANBasicBlock, KANBottleneck):
            self.conv1 = KANConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size, stride=fcnv_stride,
                                        padding=fcnv_padding, **kan_kwargs_clean)
            # self.fc = KAN(fc_layers, **kan_kwargs_fc)

        elif block in (FastKANBasicBlock, FastKANBottleneck):
            self.conv1 = FastKANConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                            stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
            # self.fc = FastKAN(fc_layers, **kan_kwargs_fc)

        elif block in (KALNBasicBlock, KALNBottleneck):
            self.conv1 = KALNConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                         stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
            # self.fc = KALN(fc_layers, **kan_kwargs_fc)
        elif block in (KAGNBasicBlock, KAGNBottleneck):
            self.conv1 = KAGNConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                         stride=fcnv_stride, padding=fcnv_padding,
                                         norm_layer=norm_layer, **kan_kwargs_clean)
            # self.fc = KAGN(fc_layers, **kan_kwargs_fc)
        elif block in (BottleneckKAGNBottleneck, BottleneckKAGNBasicBlock):
            self.conv1 = BottleNeckKAGNConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                                   stride=fcnv_stride, padding=fcnv_padding,
                                                   norm_layer=norm_layer, **kan_kwargs_clean)
            # self.fc = KAGN(fc_layers, **kan_kwargs_fc)
        elif block in (KACNBasicBlock, KACNBottleneck):
            self.conv1 = KACNConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                         stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
            # self.fc = KACN(fc_layers, **kan_kwargs_fc)
        else:
            raise TypeError(f"Block {type(block)} is not supported")
        self.maxpool = None
        if use_first_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride, padding=mp_padding)

        self.layer1 = self._make_layer(block, 8 * width_scale, layers[0],
                                       norm_layer=norm_layer, **kan_kwargs)
        self.layer2 = self._make_layer(block, 16 * width_scale, layers[1], stride=2, norm_layer=norm_layer,
                                       dilate=replace_stride_with_dilation[0],
                                       **kan_kwargs)
        self.layer3 = self._make_layer(block, 32 * width_scale, layers[2], stride=2, norm_layer=norm_layer,
                                       dilate=replace_stride_with_dilation[1],
                                       **kan_kwargs)
        self.layer4 = self._make_layer(block, 64 * width_scale, layers[3], stride=2, norm_layer=norm_layer,
                                       dilate=replace_stride_with_dilation[2],
                                       **kan_kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=dropout_linear)
        self.fc = nn.Linear(64 * width_scale * block.expansion if self.hidden_layer is None else hidden_layer_dim,
                            num_classes)

    def _make_layer(
            self,
            block: Type[Union[
                KANBasicBlock, FastKANBasicBlock, KALNBasicBlock, KACNBasicBlock, BottleneckKAGNBasicBlock, BottleneckKAGNBottleneck,
                KANBottleneck, FastKANBottleneck, KALNBottleneck, KACNBottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
            **kan_kwargs
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:

            if block in (KANBasicBlock, KANBottleneck):
                conv1x1 = partial(kan_conv1x1, **kan_kwargs)
            elif block in (FastKANBasicBlock, FastKANBottleneck):
                conv1x1 = partial(fast_kan_conv1x1, **kan_kwargs)
            elif block in (KALNBasicBlock, KALNBottleneck):
                conv1x1 = partial(kaln_conv1x1, **kan_kwargs)
            elif block in (KAGNBasicBlock, KAGNBottleneck):
                conv1x1 = partial(kagn_conv1x1, **kan_kwargs)
            elif block in (KAGNBasicBlock, KAGNBottleneck):
                conv1x1 = partial(kagn_conv1x1, **kan_kwargs)
            elif block in (KACNBasicBlock, KACNBottleneck):
                conv1x1 = partial(kacn_conv1x1, **kan_kwargs)
            elif block in (BottleneckKAGNBasicBlock, BottleneckKAGNBottleneck):
                conv1x1 = partial(bottleneck_kagn_conv1x1, **kan_kwargs)
            else:
                raise TypeError(f"Block {type(block)} is not supported")

            downsample = conv1x1(self.inplanes, planes * block.expansion, stride=stride)

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride=stride, downsample=downsample, groups=self.groups,
                base_width=self.base_width, dilation=previous_dilation, **kan_kwargs
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    **kan_kwargs
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        if self.use_first_maxpool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.hidden_layer is not None:
            x = self.hidden_layer(x)
        x = flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self._forward_impl(x)


class MoEResKANet(nn.Module):
    def __init__(
            self,
            block: Type[Union[MoEKALNBottleneck, MoEKALNBasicBlock, MoEBottleneckKAGNBasicBlock]],
            layers: List[int],
            input_channels: int = 3,
            use_first_maxpool: bool = True,
            mp_kernel_size: int = 3, mp_stride: int = 2, mp_padding: int = 1,
            fcnv_kernel_size: int = 7, fcnv_stride: int = 2, fcnv_padding: int = 3,
            num_classes: int = 1000,
            groups: int = 1,
            width_per_group: int = 64,
            width_scale: int = 1,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            num_experts: int = 8,
            noisy_gating: bool = True,
            k: int = 2,
            hidden_layer_dim: int = None,
            dropout_linear: float = 0.0,
            **kan_kwargs
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.inplanes = 16 * width_scale
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.use_first_maxpool = use_first_maxpool

        self.hidden_layer = None

        kan_kwargs_clean = kan_kwargs.copy()
        kan_kwargs_clean.pop('l1_decay', None)
        if block in (MoEKALNBottleneck, MoEKALNBasicBlock):
            self.conv1 = KALNConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                         stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
            if hidden_layer_dim is not None:
                self.hidden_layer = kaln_conv1x1(64 * width_scale * block.expansion, hidden_layer_dim, **kan_kwargs)
        elif block in (MoEBottleneckKAGNBasicBlock, ):
            self.conv1 = BottleNeckKAGNConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                         stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
            if hidden_layer_dim is not None:
                self.hidden_layer = bottleneck_kagn_conv1x1(64 * width_scale * block.expansion,
                                                            hidden_layer_dim, **kan_kwargs)
        else:
            raise TypeError(f"Block {type(block)} is not supported")
        self.maxpool = None
        if use_first_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride, padding=mp_padding)

        self.layer1 = self._make_layer(block, 8 * width_scale, layers[0],
                                       num_experts=num_experts, noisy_gating=noisy_gating, k=k, **kan_kwargs)
        self.layer2 = self._make_layer(block, 16 * width_scale, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       num_experts=num_experts, noisy_gating=noisy_gating, k=k,
                                       **kan_kwargs)
        self.layer3 = self._make_layer(block, 32 * width_scale, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       num_experts=num_experts, noisy_gating=noisy_gating, k=k,
                                       **kan_kwargs)
        self.layer4 = self._make_layer(block, 64 * width_scale, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       num_experts=num_experts, noisy_gating=noisy_gating, k=k,
                                       **kan_kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * width_scale * block.expansion if self.hidden_layer is None else hidden_layer_dim,
                            num_classes)
        self.drop = nn.Dropout(p=dropout_linear)

    def _make_layer(
            self,
            block: Type[Union[MoEKALNBottleneck,]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
            num_experts: int = 8,
            noisy_gating: bool = True,
            k: int = 2,
            **kan_kwargs
    ) -> nn.Module:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if block in (MoEKALNBottleneck, MoEKALNBasicBlock, MoEBottleneckKAGNBasicBlock):
                kan_kwargs.pop('num_experts', None)
                kan_kwargs.pop('noisy_gating', None)
                kan_kwargs.pop('k', None)
                conv1x1 = partial(kaln_conv1x1, **kan_kwargs)
            else:
                raise TypeError(f"Block {type(block)} is not supported")

            downsample = conv1x1(self.inplanes, planes * block.expansion, stride=stride)

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride=stride, downsample=downsample, groups=self.groups,
                base_width=self.base_width, dilation=previous_dilation, num_experts=num_experts,
                noisy_gating=noisy_gating, k=k, **kan_kwargs
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    num_experts=num_experts,
                    noisy_gating=noisy_gating,
                    k=k,
                    **kan_kwargs
                )
            )

        return nn.ModuleList(layers)

    def _forward_layer(self, layer, x, train):
        moe_loss = 0
        for block in layer:
            x, _moe_loss = block(x, train)
            moe_loss += _moe_loss
        return x, moe_loss

    def _forward_impl(self, x: Tensor, train: bool = True) -> Tensor:
        x = self.conv1(x)
        if self.use_first_maxpool:
            x = self.maxpool(x)

        x, moe_loss1 = self._forward_layer(self.layer1, x, train)
        x, moe_loss2 = self._forward_layer(self.layer2, x, train)
        x, moe_loss3 = self._forward_layer(self.layer3, x, train)
        x, moe_loss4 = self._forward_layer(self.layer4, x, train)

        x = self.avgpool(x)
        if self.hidden_layer is not None:
            x = self.hidden_layer(x)
        x = flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)

        return x, (moe_loss1 + moe_loss2 + moe_loss3 + moe_loss4) / 4

    def forward(self, x: Tensor, train: bool = True) -> Tensor:
        return self._forward_impl(x, train)


