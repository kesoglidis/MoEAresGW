import torch
from torch import nn
from torch.nn import functional as F

from modules.kan_convs import WavKANConv1DLayer

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class WavConvND(nn.Module):
    def __init__(self, conv_class, input_dim, output_dim, kernel_size,
                 padding=0, stride=1, dilation=1,
                 ndim: int = 2, wavelet_type='mexican_hat'):
        super(WavConvND, self).__init__()

        _shapes = (1, output_dim, input_dim) + tuple(1 for _ in range(ndim))

        self.scale = nn.Parameter(torch.ones(output_dim//2)*0.1)
        self.frequency = nn.Parameter(torch.randn(output_dim//2)*500)
        # self.scale = nn.Parameter(torch.ones(output_dim//2)*0.1)
        # self.frequency = nn.Parameter(torch.ones(output_dim//2)*600)
        # self.translation = nn.Parameter(torch.zeros(*_shapes))

        # self.t = torch.linspace(-1,1,2049).to('cuda:0')
        self.t = torch.linspace(-1,1,129).to('cuda:0')

        self.ndim = ndim
        self.wavelet_type = wavelet_type

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.counter = 100
    # @staticmethod
    # def _forward_morlet(x):
    #     omega0 = 5.0  # Central frequency
    #     real = torch.cos(omega0 * x)
    #     envelope = torch.exp(-0.5 * x ** 2)
    #     wavelet = envelope * real
    #     return wavelet

    def forward(self, x):
        batch_size = x.shape[0]

        wavelets = []
        for i in range(self.output_dim//2):
            s = self.scale[i]
            f = self.frequency[i]
            t = self.t
            g = torch.exp(-t**2 / (2 * s**2)) * torch.cos(2*np.pi*f * t/s)
            wavelets.append(g)
        
        # print(x.shape)
        wavelets = torch.stack(wavelets).unsqueeze(1)
        # print(wavelets.shape)
        # print(x[:,0:1,:].shape)
        x1 = F.conv1d(x[:, 0:1, :], wavelets, padding=64)  # Shape [128, 8, 2048]

        x2 = F.conv1d(x[:, 1:2, :], wavelets, padding=64)  # Shape [128, 8, 2048]

        coeffs = torch.cat((x1, x2), dim=1)  # Shape [128, 16, 2048]

        # print(self.scale)
        # print(self.frequency)      
        
        self.counter = self.counter +1

        if self.counter < 0:
            self.counter = 0

            fig = plt.figure(figsize=(24,15))
            fig.suptitle('Training loss & acc')

            gs = gridspec.GridSpec(4,2, figure=fig)
            
            s = self.scale[0]
            f = self.frequency[0]
            t = self.t
            g = torch.exp(-t**2 / (2 * s**2)) * torch.cos(2*np.pi*f * t / s)
            g = g.detach().cpu().numpy()
            axs00 = fig.add_subplot(gs[0,0])
            axs00.plot(g, label='raw')
            axs00.title.set_text('Raw')

            s = self.scale[1]
            f = self.frequency[1]
            t = self.t
            g = torch.exp(-t**2 / (2 * s**2)) * torch.cos(2*np.pi*f * t / s)
            g = g.detach().cpu().numpy()
            axs10 = fig.add_subplot(gs[1,0])
            axs10.plot(g, label='raw')
            axs10.title.set_text('Raw')

            s = self.scale[2]
            f = self.frequency[2]
            t = self.t
            g = torch.exp(-t**2 / (2 * s**2)) * torch.cos(2*np.pi*f * t/ s)
            g = g.detach().cpu().numpy()
            axs20 = fig.add_subplot(gs[2,0])
            axs20.plot(g, label='raw')
            axs20.title.set_text('Raw')
            
            s = self.scale[3]
            f = self.frequency[3]
            t = self.t
            g = torch.exp(-t**2 / (2 * s**2)) * torch.cos(2*np.pi*f * t/ s)
            g = g.detach().cpu().numpy()
            axs30 = fig.add_subplot(gs[3,0])
            axs30.plot(g, label='raw')
            axs30.title.set_text('Raw')


            s = self.scale[4]
            f = self.frequency[4]
            t = self.t
            g = torch.exp(-t**2 / (2 * s**2)) * torch.cos(2*np.pi*f * t/ s)
            g = g.detach().cpu().numpy()
            axs01 = fig.add_subplot(gs[0,1])
            axs01.plot(g, label='raw')
            axs01.title.set_text('Raw')

            s = self.scale[5]
            f = self.frequency[5]
            t = self.t
            g = torch.exp(-t**2 / (2 * s**2)) * torch.cos(2*np.pi*f * t/ s)
            g = g.detach().cpu().numpy()
            axs11 = fig.add_subplot(gs[1,1])
            axs11.plot(g, label='raw')
            axs11.title.set_text('Raw')

            s = self.scale[6]
            f = self.frequency[6]
            t = self.t
            g = torch.exp(-t**2 / (2 * s**2)) * torch.cos(2*np.pi*f * t/ s)
            g = g.detach().cpu().numpy()
            axs21 = fig.add_subplot(gs[2,1])
            axs21.plot(g, label='raw')
            axs21.title.set_text('Raw')
            
            s = self.scale[7]
            f = self.frequency[7]
            t = self.t
            g = torch.exp(-t**2 / (2 * s**2)) * torch.cos(2*np.pi*f * t/ s)
            g = g.detach().cpu().numpy()
            axs31 = fig.add_subplot(gs[3,1])
            axs31.plot(g, label='raw')
            axs31.title.set_text('Raw')
            fig.savefig(f'wavelets.png')

            plt.close()  

        return coeffs     

class WavConvNDLayer(nn.Module):
    def __init__(self, conv_class, conv_class_plus1, norm_class, input_dim, output_dim, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1, wav_version: str = 'base',
                 ndim: int = 2, dropout=0.0, wavelet_type='mexican_hat', **norm_kwargs):
        super(WavConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.norm_kwargs = norm_kwargs
        assert wavelet_type in ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon'], \
            ValueError(f"Unsupported wavelet type: {wavelet_type}")
        self.wavelet_type = wavelet_type

        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)

        self.base_conv = nn.ModuleList([conv_class(input_dim // groups,
                                                   output_dim // groups,
                                                   kernel_size,
                                                   stride,
                                                   padding,
                                                   dilation,
                                                   groups=1,
                                                   bias=False) for _ in range(groups)])
        if wav_version == 'base':
            self.wavelet_conv = WavConvND(
                        conv_class,
                        input_dim // groups,
                        output_dim // groups,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        ndim=ndim, wavelet_type=wavelet_type)

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **norm_kwargs) for _ in range(groups)])

        self.base_activation = nn.SiLU()

    # def forward_wavkan(self, x, group_ind):
        # You may like test the cases like Spl-KAN
        # base_output = self.base_conv[group_ind](self.base_activation(x))

        # if self.dropout is not None:
        #     x = self.dropout(x)

        # wavelet_output = self.wavelet_conv[group_ind](x)

        # combined_output = wavelet_output + base_output

        # Apply batch normalization
        # return self.layer_norm[group_ind](wavelet_output)
        # return wavelet_output

    # def forward(self, x):
    #     split_x = torch.split(x, self.inputdim // self.groups, dim=1)
    #     output = []
    #     for group_ind, _x in enumerate(split_x):
    #         y = self.forward_wavkan(_x, group_ind)
    #         output.append(y.clone())
    #     y = torch.cat(output, dim=1)
    #     return y

    def forward(self, x):
        wavelet_output = self.wavelet_conv(x)

        return wavelet_output




class WavConv1DLayer(WavConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 dropout=0.0, wavelet_type='mexican_hat', norm_layer=nn.BatchNorm1d,
                 wav_version: str = 'fast', **norm_kwargs):
        super(WavConv1DLayer, self).__init__(nn.Conv1d, nn.Conv2d, norm_layer, input_dim, output_dim, kernel_size,
                                                groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                ndim=1, dropout=dropout, wavelet_type=wavelet_type,
                                                wav_version=wav_version, **norm_kwargs)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if out_channels != in_channels or stride > 1:
            self.x_transform = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.x_transform = nn.Identity()

        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        x = F.relu(self.body(x) + self.x_transform(x))
        return x


class WavResNet54(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlock(2, 8),
            ResBlock(8, 8),
            ResBlock(8, 8),
            ResBlock(8, 8),
            ResBlock(8, 16, stride=2),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 32, stride=2),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(16, 32, 64), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 2, 1), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)

class WavResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if out_channels != in_channels or stride > 1:
            self.x_transform = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.x_transform = nn.Identity()

        
        self.conv1 = nn.Conv1d(in_channels, out_channels/2, kernel_size=3, stride=1, padding='same'),
        self.wav = WavConv1DLayer(in_channels, out_channels/2, kernel_size=3, stride=1, padding='same')

        self.bn1 = nn.BatchNorm1d(out_channels),
        self.relu = nn.ReLU(),
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = torch.concat(self.wav(x) + self.conv1(x))
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        x = F.relu(self.body(out) + self.x_transform(x))
        return x
    

class WavResNet54Double(nn.Module):
    def __init__(self, basis, base):
        super().__init__()
        self.wavelet = WavConv1DLayer(2, 16, kernel_size=3, wavelet_type=base, wav_version=basis)
        self.feature_extractor = nn.Sequential( 
            #base = ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon']
            #basis = ['fast_plus_one', 'fast', 'base']
            # WavConv1DLayer(2, 16, kernel_size=3, wavelet_type=base, wav_version=basis),
            # ResBlock(2, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            # WavKANConv1DLayer(16, 32, kernel_size=3, stride=2, wavelet_type=base, wav_version=basis),
            ResBlock(16, 32, stride=2),
            ResBlock(32, 32),
            ResBlock(32, 32),
            # WavKANConv1DLayer(16, 32, kernel_size=3, stride=2, wavelet_type=base, wav_version=basis),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            # WavKANConv1DLayer(16, 32, kernel_size=3, stride=2, wavelet_type=base, wav_version=basis),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128),
            # WavKANConv1DLayer(16, 32, kernel_size=3, stride=2, wavelet_type=base, wav_version=basis),
            ResBlock(128, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128),
            # WavKANConv1DLayer(16, 32, kernel_size=3, stride=2, wavelet_type=base, wav_version=basis),
            ResBlock(128, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(32, 64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 2, 1), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.wavelet(x)
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)
    

class WavResNet54Doublev2(nn.Module):
    def __init__(self, basis, base):
        super().__init__()
        self.feature_extractor = nn.Sequential( 
            #base = ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon']
            #basis = ['fast_plus_one', 'fast', 'base']
            WavResBlock(2, 16, kernel_size=3, wavelet_type=base, wav_version=basis),
            # ResBlock(2, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            # WavKANConv1DLayer(16, 32, kernel_size=3, stride=2, wavelet_type=base, wav_version=basis),
            ResBlock(16, 32, stride=2),
            ResBlock(32, 32),
            ResBlock(32, 32),
            # WavKANConv1DLayer(16, 32, kernel_size=3, stride=2, wavelet_type=base, wav_version=basis),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            # WavKANConv1DLayer(16, 32, kernel_size=3, stride=2, wavelet_type=base, wav_version=basis),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128),
            # WavKANConv1DLayer(16, 32, kernel_size=3, stride=2, wavelet_type=base, wav_version=basis),
            ResBlock(128, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128),
            # WavKANConv1DLayer(16, 32, kernel_size=3, stride=2, wavelet_type=base, wav_version=basis),
            ResBlock(128, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(32, 64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 2, 1), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)
    