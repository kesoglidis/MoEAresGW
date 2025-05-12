import torch
from torch import nn
from torch.nn import functional as F

from modules.kan_convs import WavKANConv1DLayer

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

class GaborAutoencoder(nn.Module):
    def __init__(self, signal_len=2048, n_channels=2, n_wavelets=32):
        super().__init__()
        self.signal_len = signal_len
        self.n_channels = n_channels
        self.n_wavelets = n_wavelets
        self.n_total = n_wavelets # * n_channels 
        self.counter = 100

        # Encoder: MLP to extract 5 parameters per wavelet
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_wavelets * 5)
        )

        # self.decoder = nn.Sequential(
        #     # nn.Linear(n_wavelets * 5, 256),
        #     # nn.ReLU(),
        #     nn.Linear(256, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 4096)
        # )

    # def forward(self, x, training_labels = None, training_clean = None):
    #     batch_size, C, T = x.shape 
    #     embed = self.encoder(x)  # (batch, n_total * 5)
    #     out = self.decoder(embed)
    #     out = out.view(batch_size, C, T)

    #     for b in range(batch_size):
    #                 if training_labels[b][0] == 0.0:
    #                     training_clean[b][:][:] = torch.zeros(2,2048)

    #     if training_labels is not None:
    #         self.counter = self.counter + 1

    #         if self.counter > 100:
    #             self.counter = 0

    #             labels = training_labels.detach().cpu().numpy()
    #             txt = ''
    #             if labels[0][0] == 1.0:
    #                 txt = 'with Injection'
    #             else:
    #                 txt =  'without Injection'
    #             # print(self.cutoff_predictor[-2].bias)

    #             fig = plt.figure(figsize=(30,20))
    #             fig.suptitle('Reconstuction of a Signal using Wavelets')

    #             gs = gridspec.GridSpec(4,2, figure=fig)

    #             channel_data = x[0][0].detach().cpu().numpy()
    #             minc = min(channel_data)
    #             maxc = max(channel_data)
                
    #             channel_data.clip(-10,10)
    #             axs00 = fig.add_subplot(gs[0,0])
    #             axs00.plot(channel_data)
    #             axs00.title.set_text('Raw Channel 1 '+ txt)
    #             axs00.set_ylim(minc, maxc)

    #             reconstucted = out[0][0].detach().cpu().numpy()
                        
    #             axs10 = fig.add_subplot(gs[2,0])
    #             axs10.plot(reconstucted, linewidth = 0.5)
    #             axs10.title.set_text('Reconstucted ')
    #             axs10.set_ylim(minc, maxc)

    #             clean = training_clean[0][0].detach().cpu().numpy()
    #             axs20 = fig.add_subplot(gs[1,0])
    #             axs20.plot(clean)
    #             axs20.title.set_text('Target signal')
    #             axs20.set_ylim(minc, maxc)
            
    #             # axs21 = fig.add_subplot(gs[2,1])
    #             # for wave in wavelets[0][0:self.n_wavelets]:
    #             #     # print(wave[0].detach().cpu().numpy())
    #             #     axs21.plot(wave[0].detach().cpu().numpy(), linewidth = 0.5)
    #             # axs21.title.set_text('Wavelets used in Reconstuction')
    #             # axs21.set_ylim(minc, maxc)

    #             axs30 = fig.add_subplot(gs[3,0])
    #             # for wave in wavelets[0][self.n_wavelets:2*self.n_wavelets]:
    #                 # print(wave[0].detach().cpu().numpy())
    #             axs30.plot(np.power(reconstucted - clean,2), linewidth = 0.5)
    #             axs30.title.set_text('Square Error Difference between reconstucted and target signal')
    #             # axs30.set_ylim(minc, maxc)

    #             channel_data = x[0][1].detach().cpu().numpy()

    #             axs01 = fig.add_subplot(gs[0,1])
    #             axs01.plot(channel_data)
    #             axs01.title.set_text('Raw Channel 2 '+ txt)
    #             axs01.set_ylim(minc, maxc)

    #             reconstucted = out[0][1].detach().cpu().numpy()
                        
    #             # axs10 = fig.add_subplot(gs[2,1])
    #             # axs10.plot(reconstucted, linewidth = 0.5)
    #             # axs10.title.set_text('Reconstucted')
    #             # axs10.set_ylim(minc, maxc)

    #             clean = training_clean[0][1].detach().cpu().numpy()
    #             axs11 = fig.add_subplot(gs[1,1])
    #             axs11.plot(clean)
    #             axs11.title.set_text('Target signal')
    #             axs11.set_ylim(minc, maxc)

    #             # axs31 = fig.add_subplot(gs[3,1])
    #             # for wave in wavelets[0][self.n_wavelets:2*self.n_wavelets]:
    #             #     # print(wave[0].detach().cpu().numpy())
    #             #     axs31.plot(wave[0].detach().cpu().numpy(), linewidth = 0.5)
    #             # axs31.title.set_text('Wavelets used in Reconstuction')
    #             # axs31.set_ylim(minc, maxc)

                

    #             axs31 = fig.add_subplot(gs[3,1])
    #             axs31.plot(np.power(reconstucted - clean,2), linewidth = 0.5)

    #             axs31.title.set_text('Square Error Difference between reconstucted and target signal')
    #             # axs31.set_ylim(minc, maxc)
    #             # axs01 = fig.add_subplot(gs[0:2,1])
    #             # axs01.bar(bins_ones[:-1], self.hist_ones, width=3)

    #             # axs01.title.set_text('Signal')
    #             # axs01.set_xlim(0,1024)

    #             # axs11 = fig.add_subplot(gs[2:4,1])
    #             # axs11.bar(bins_ones[:-1], self.hist_ones, width=3)
    #             # axs11.bar(bins_zeros[:-1], self.hist_zeros, width=3)
    #             # axs11.title.set_text('Both Signal and No Signal')
    #             # axs11.set_xlim(0,1024)


    #             fig.savefig(f'AutoWavelet.png')
    #             plt.close()
    #     return out, training_clean

    def forward(self, x, training_labels = None, training_clean = None):
        """
        x: (batch, 2, 2048)
        """
        batch_size, C, T = x.shape 
        params = self.encoder(x)  # (batch, n_total * 5)
        params = params.view(batch_size, self.n_total, 5)
       
        A = params[:, :, 0]
        t0 = torch.sigmoid(params[:, :, 1]) * self.signal_len  # map to [0, 2048]
        f = torch.sigmoid(params[:, :, 2]) * 0.5  # [0, 0.5] cycles per sample
        sigma = torch.sigmoid(params[:, :, 3]) * 200 + 2  # avoid near-zero widths
        phi = params[:, :, 4]  # no constraints
    
        signals = []
        wavelets = []

        # start = time.time()

        for i in range(batch_size):
            signal, wavelet = self.synthesize_wavelet_signal(A[i], t0[i], f[i], sigma[i], phi[i])
        
            signals.append(signal)
            wavelets.append(wavelet)

        recon = torch.stack(signals)
        # print(recon)

        if training_labels is not None:
            self.counter = self.counter + 1

            # for b in range(batch_size):
            #     if training_labels[b][0] == 0.0:
            #         training_clean[b][:][:] = torch.zeros(2,2048)

            if self.counter > 10:
                self.counter = 0

                labels = training_labels.detach().cpu().numpy()
                txt = ''
                if labels[0][0] == 1.0:
                    txt = 'with Injection'
                else:
                    txt =  'without Injection'
                # print(self.cutoff_predictor[-2].bias)

                fig = plt.figure(figsize=(30,20))
                fig.suptitle('Reconstuction of a Signal using Wavelets')

                gs = gridspec.GridSpec(4,2, figure=fig)

                channel_data = x[0][0].detach().cpu().numpy()
                minc = min(channel_data)
                maxc = max(channel_data)
                
                if minc == 0:
                    minc = 1
                    maxc = -1

                channel_data.clip(-10,10)
                axs00 = fig.add_subplot(gs[0,0])
                axs00.plot(channel_data)
                axs00.title.set_text('Raw Channel 1 '+ txt)
                axs00.set_ylim(minc, maxc)

                reconstucted = recon[0][0].detach().cpu().numpy()
                        
                axs10 = fig.add_subplot(gs[2,0])
                axs10.plot(reconstucted, linewidth = 0.5)
                axs10.title.set_text('Reconstucted ')
                axs10.set_ylim(minc, maxc)

                clean = training_clean[0][0].detach().cpu().numpy()
                axs20 = fig.add_subplot(gs[1,0])
                axs20.plot(clean)
                axs20.title.set_text('Target signal')
                axs20.set_ylim(minc, maxc)
            
                axs21 = fig.add_subplot(gs[2,1])
                for wave in wavelets[0][0:self.n_wavelets]:
                    axs21.plot(wave[0].detach().cpu().numpy(), linewidth = 0.5)
                axs21.title.set_text('Wavelets used in Reconstuction')
                axs21.set_ylim(minc, maxc)

                axs30 = fig.add_subplot(gs[3,0])
                axs30.plot(np.power(reconstucted - clean,2), linewidth = 0.5)
                axs30.title.set_text('Square Error Difference between reconstucted and target signal')
                # axs30.set_ylim(minc, maxc)

                channel_data = x[0][1].detach().cpu().numpy()

                axs01 = fig.add_subplot(gs[0,1])
                axs01.plot(channel_data)
                axs01.title.set_text('Raw Channel 2 '+ txt)
                axs01.set_ylim(minc, maxc)

                reconstucted = recon[0][1].detach().cpu().numpy()
                        
                clean = training_clean[0][1].detach().cpu().numpy()
                axs11 = fig.add_subplot(gs[1,1])
                axs11.plot(clean)
                axs11.title.set_text('Target signal')
                axs11.set_ylim(minc, maxc)

                axs31 = fig.add_subplot(gs[3,1])
                axs31.plot(np.power(reconstucted - clean,2), linewidth = 0.5)

                axs31.title.set_text('Square Error Difference between reconstucted and target signal')
                # axs31.set_ylim(minc, maxc)

                fig.savefig(f'AutoWavelet.png')
                plt.close()

        return recon, training_clean  # shape: (batch, 2, 2048)

    def synthesize_wavelet_signal(self, A, t0, f, sigma, phi):
        """
        Reconstruct 2x2048 signal using sum of N Morlet-Gabor wavelets per channel.
        """
        device = A.device
        t = torch.arange(self.signal_len, device=device).float().unsqueeze(0)  # [1, T]
        signal = torch.zeros((self.n_channels, self.signal_len), device=device)

        wavelets = []
        # for ch in range(self.n_channels):
        for i in range(self.n_wavelets):
            idx =  i # self.n_wavelets
            g = A[idx] * torch.exp(-((t - t0[idx]) ** 2) / (2 * sigma[idx] ** 2)) \
                * torch.cos(2 * np.pi * f[idx] * (t - t0[idx]) + phi[idx])
            wavelets.append(g)
            signal[0] += g.squeeze()
            signal[1] += g.squeeze()
        return signal, wavelets
    

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
    