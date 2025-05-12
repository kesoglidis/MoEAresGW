import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import scipy
from scipy.signal import welch, get_window
from modules.whiten import *

fs = 2048

class AdaptiveLowPassLayer(nn.Module):
    def __init__(self, kernel_size=101, fc_range=(300, 550)):
        super().__init__()
        self.fs = fs
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.fc_min, self.fc_max = fc_range
        self.hist_ones = np.zeros(100)
        self.hist_zeros = np.zeros(100)
        self.sum_ones = 0
        self.sum_zeros = 0

        self.counter = 100

        # Predictor network: maps signal -> cutoff frequency (normalized)
        # self.cutoff_predictor = nn.Sequential(
        #     nn.Linear(fs*2, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1),
        #     nn.Sigmoid()  # Output in [0, 1]
        # )
        self.cutoff_predictor = nn.Sequential(
            ResBlock(2 , 8 ),
            ResBlock(8 , 8 ),
            ResBlock(8 , 8 ),
            ResBlock(8 , 16, stride=4),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 32, stride=4),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 64, stride=4),
            ResBlock(64, 64),
            ResBlock(64, 64), #64*32
            # nn.AvgPool1d(kernel_size=3),
            nn.Flatten(),
            nn.Linear(64*32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, x, training_labels = None, training_clean = None):
        """
        x: [B, C, T] where B = batch size, C = number of channels (e.g., 2 for stereo), T = time steps
        """

        batch_size, C, T = x.shape  # Extract batch size, channels, and time steps
        # fc_norm = self.cutoff_predictor(x.view(batch_size, -1))  # [B, 1] (flattened input)
        fc_norm = self.cutoff_predictor(x)  # [B, 1] (flattened input)
        # fc_norm/1024.0
        fc_hz = self.fc_min + fc_norm * (self.fc_max - self.fc_min)  # [B, 1]

        # fc_hz = self.fc_min + fc_norm
        # print(fc_hz)
        # Create low-pass filter for each sample in the batch
        filters = []
        t = torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1, dtype=torch.float32, device=x.device)

        # Generate a filter for each sample in the batch
        for i in range(batch_size):
            fc = fc_hz[i] / self.fs  # Normalize cutoff frequency by sampling rate
            kernel = 2 * fc * torch.sinc(2 * fc * t)  # Sinc function for low-pass filter
            window = torch.hamming_window(self.kernel_size, device=x.device)  # Apply Hamming window
            kernel *= window
            kernel /= kernel.sum()  # Normalize the kernel to unit sum
            filters.append(kernel)

       # Stack the filters: [B, 1, K] where K = kernel_size
        filters = torch.stack(filters).unsqueeze(1)  # Shape: [B, 1, K]

        # Now we will filter each batch separately
        filtered_batches = []

        for b in range(batch_size):
            filtered_channels = []

            # For each channel in the batch, apply the filter independently
            for c in range(C):  # Iterate over channels (C)
                channel_data = x[b, c, :]  # Extract data for the current channel (Shape: [T])
                
                # Add a singleton dimension to fit conv1d input: [1, 1, T]
                channel_data = channel_data.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, T]

                # Apply the low-pass filter (same filter for this batch)
                filtered_channel = F.conv1d(channel_data, filters[b].unsqueeze(0), padding=self.padding)  # Shape: [1, 1, T]
                filtered_channels.append(filtered_channel)

            # if training_labels[b][1].item() == 1.0:
            #     # print("changed")
            #     training_clean[b][:][:] = x[b][:][:]
            # Stack filtered channels for the current batch, resulting in shape [C, T]
            filtered_batch = torch.cat(filtered_channels, dim=0)  # Shape: [C, T]
            filtered_batches.append(filtered_batch)

        # Stack all filtered batches together, resulting in shape [B, C, T]
        filtered = torch.stack(filtered_batches, dim=0).squeeze(2)  # Shape: [B, C, T]
        self.grad = filtered
        # print(filtered.shape)

        if training_labels is not None:
            self.counter = self.counter + 1 
            # print(fc_hz)
            # print(fc_hz.shape)
            # print(training_labels)
            # print(training_labels.shape)

            if self.counter > 10:
                
                fc = fc_hz.detach().cpu().numpy()
                labels = training_labels.detach().cpu().numpy()
                mask_ones = labels[:,0] == 1.0
                mask_zeros = labels[:,0] == 0.0

                # print(fc[mask_ones].shape)
                # print(fc[mask_zeros].shape)

                hist_ones, bins_ones = np.histogram(fc[mask_ones], bins=100, range = (250, 650)) #750
                hist_ones, bins_ones = np.histogram(fc[mask_ones], bins=100, range = (20, 1024)) #750

                self.sum_ones = self.sum_ones+ hist_ones.sum()
                self.hist_ones = self.hist_ones + hist_ones

                hist_zeros, bins_zeros = np.histogram(fc[mask_zeros], bins=100, range = (250, 650)) #750
                hist_zeros, bins_zeros = np.histogram(fc[mask_zeros], bins=100, range = (20, 1024)) #750

                self.sum_zeros = self.sum_zeros + hist_zeros.sum()
                self.hist_zeros = self.hist_zeros + hist_zeros

                self.counter = 0
                # print(self.cutoff_predictor[-2].bias)

                fig = plt.figure(figsize=(30,20))
                fig.suptitle('Histogram of frequency cutoffs')

                gs = gridspec.GridSpec(4,2, figure=fig)

                axs00 = fig.add_subplot(gs[0,0])
                channel_data = x[0][0].detach().cpu().numpy()
                axs00.plot(channel_data, label='raw')
                axs00.title.set_text('Raw Signal:'+ str (labels[0][0]) +' Noise:'+str(labels[0][1]))
                axs00.set_ylim(min(channel_data), max(channel_data))

                filtered_channel = filtered[0][0].detach().cpu().numpy()
                        
                axs10 = fig.add_subplot(gs[1,0])
                axs10.plot(filtered_channel, label='filtered')
                axs10.title.set_text('Filtered ' + str(round(float(fc_hz[-1][0]),2)) +'Hz')
                axs10.set_ylim(min(channel_data), max(channel_data))

                axs20 = fig.add_subplot(gs[2,0])
                axs20.psd(channel_data, Fs=fs, label='PSD raw')
                axs20.title.set_text('PSD raw')
                axs20.set_xlim(0,1024)
                axs20.set_ylim(-120,-20)
                
                axs30 = fig.add_subplot(gs[3,0])
                axs30.psd(filtered_channel, Fs=fs, label='PSD filtered')
                axs30.title.set_text('PSD filtered')
                axs30.set_xlim(0,1024)
                axs30.set_ylim(-120,-20)

                # clean = training_clean[0][0].detach().cpu().numpy()
                # axs20 = fig.add_subplot(gs[2,0])
                # axs20.plot(clean, label='Clean signal')
                # axs20.title.set_text('Clean signal')
                # axs20.set_ylim(min(channel_data), max(channel_data))

                # win = get_window('hann', fs)
                # f, pxx = welch(clean, fs=fs//2, window=win, nperseg=fs)
                # pxx_db = 10 * np.log10(pxx+1e-12)  # Convert to dB like plt.psd
              
            
                # axs30 = fig.add_subplot(gs[3,0])
                # axs30.psd(clean, Fs=fs, label='PSD filtered')
                # # axs30.plot(pxx_db)
                # axs30.title.set_text('PSD filtered')
                # axs30.set_xlim(0,1024)
                # axs30.set_ylim(-120,-20)

                
                
                axs01 = fig.add_subplot(gs[0:2,1])
                axs01.bar(bins_ones[:-1], self.hist_ones, width=3)

                axs01.title.set_text('Signal')
                axs01.set_xlim(0,1024)

                axs11 = fig.add_subplot(gs[2:4,1])
                axs11.bar(bins_ones[:-1], self.hist_ones, width=3)
                axs11.bar(bins_zeros[:-1], self.hist_zeros, width=3)
                axs11.title.set_text('Both Signal and No Signal')
                axs11.set_xlim(0,1024)


                fig.savefig(f'filter.png')
                plt.close()

        return filtered#, training_clean

    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck=False, stride=1):
        super().__init__()
        if out_channels != in_channels or stride > 1:
            self.x_transform = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.x_transform = nn.Identity()

        if bottleneck:
            width = int(out_channels/4.0)
            self.body = nn.Sequential(
                nn.Conv1d(in_channels, width, kernel_size=1),
                nn.BatchNorm1d(width),
                nn.Conv1d(width, width, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm1d(width),
                nn.Conv1d(width, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
        else:
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

class FIRResNet54(nn.Module):
    def __init__(self, bottleneck):
        super().__init__()
        self.low_pass = AdaptiveLowPassLayer()
        self.feature_extractor = nn.Sequential(
            ResBlock(2 , 8 , bottleneck),
            ResBlock(8 , 8 , bottleneck),
            ResBlock(8 , 8 , bottleneck),
            ResBlock(8 , 8 , bottleneck),
            ResBlock(8 , 16, bottleneck, stride=2),
            ResBlock(16, 16, bottleneck),
            ResBlock(16, 16, bottleneck),
            ResBlock(16, 32, bottleneck, stride=2),
            ResBlock(32, 32, bottleneck),
            ResBlock(32, 32, bottleneck),
            ResBlock(32, 64, bottleneck, stride=2),
            ResBlock(64, 64, bottleneck),
            ResBlock(64, 64, bottleneck),
            ResBlock(64, 64, bottleneck, stride=2),
            ResBlock(64, 64, bottleneck),
            ResBlock(64, 64, bottleneck),
            ResBlock(64, 64, bottleneck, stride=2),
            ResBlock(64, 64, bottleneck),
            ResBlock(64, 64, bottleneck),
            ResBlock(64, 32, bottleneck),
            ResBlock(32, 32, bottleneck),
            ResBlock(32, 32, bottleneck),
            ResBlock(32, 32, bottleneck),
            ResBlock(32, 32, bottleneck),
            ResBlock(32, 16, bottleneck),
            ResBlock(16, 16, bottleneck),
            ResBlock(16, 16, bottleneck),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(16, 32, 64), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 2, 1), nn.Softmax(dim=1)
        )

    def forward(self, x, training_labels):
        x = self.low_pass(x, training_labels)
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)


class FIRResNet54Double(nn.Module):
    def __init__(self, bottleneck=False):
        super().__init__()
        self.low_pass = AdaptiveLowPassLayer()
        self.feature_extractor = nn.Sequential(
            ResBlock(2  , 16 , bottleneck),
            ResBlock(16 , 16 , bottleneck),
            ResBlock(16 , 16 , bottleneck),
            ResBlock(16 , 16 , bottleneck),
            ResBlock(16 , 32 , bottleneck, stride=2),
            ResBlock(32 , 32 , bottleneck),
            ResBlock(32 , 32 , bottleneck),
            ResBlock(32 , 64 , bottleneck, stride=2),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 128, bottleneck, stride=2),
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck, stride=2),
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck, stride=2),
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 32 , bottleneck),
            ResBlock(32 , 32 , bottleneck),
            ResBlock(32 , 32 , bottleneck),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(32, 64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 2, 1), nn.Softmax(dim=1)
        )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.cls_head.parameters():
            param.requires_grad = False


    def forward(self, x, training_labels=None):
        x = self.low_pass(x, training_labels)
        
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)
    




class DeepGaussianFilter(nn.Module):
    def __init__(self, filter_size: int = 11, sigma: float = 1., order: int = 2) -> None:
        """
        The Gaussian filter is designed to smooth a signal.
        The parameter sigma which controls the amount of smoothing is a learnable parameter.
        The high-order Gaussian filters can also be applied to capture the higher-order derivatives of the signal.
        Args:
            filter_size: size of the filter
            sigma: the initialization of the standard deviation of the Gaussian filter
            order: the order up to which the signal will be returned
        """
        super().__init__()

        self.filter_size = filter_size
        self.order = order

        self.sigma = nn.Parameter(torch.tensor([sigma]))

        if order > 2:
            raise ValueError("The order of the filter cannot be greater than 2.")
        if order < 0:
            raise ValueError("The order of the filter cannot be negative.")

        if filter_size <= 1:
            raise ValueError("The filter size must be greater than 1.")
        if filter_size % 2 == 0:
            raise ValueError("The filter size should be odd.")

        self.counter = 990 
        
    def get_filters(self) -> torch.Tensor:
        """
        Get the Gaussian filters up to the specified order.
        Returns:
            filters: the filters to be applied
        """
        filters = []

        x = torch.linspace(-3, 3, self.filter_size, device=self.device)

        # The Gaussian filter
        if self.order >= 0:
            gaussian_filter = torch.exp(-x ** 2 / (2 * self.sigma ** 2.))
            gaussian_filter = gaussian_filter / torch.sum(gaussian_filter)
            filters.append(gaussian_filter)

        # The 1st order Gaussian filter
        if self.order >= 1:
            gaussian_filter_1 = filters[0] * x
            gaussian_filter_1 = gaussian_filter_1 / torch.std(gaussian_filter_1)
            filters.append(gaussian_filter_1)

        # The 2nd order Gaussian filter
        if self.order >= 2:
            gaussian_filter_2 = filters[0] * (x ** 2 - self.sigma ** 2)
            gaussian_filter_2 = gaussian_filter_2 / torch.std(gaussian_filter_2)
            filters.append(gaussian_filter_2)

        return torch.stack(filters).unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the Deep Gaussian Filter.
        Args:
            x: the input tensor. The shape is expected to be (batch_size, time_steps)
        Returns:
            y: the filtered signal along its high-order derivatives. Returned shape is (batch_size, order+1, time_steps)
        """
        out = x
        x = x.unsqueeze(1)
        x = self.pad(x, self.filter_size // 2)
        filters = self.get_filters()
        y = F.conv1d(x, filters)

        self.counter = self.counter+1
        if self.counter > 1000:
            self.counter = 0
            # print(self.cutoff_predictor[-2].bias)

            fig = plt.figure(figsize=(30,20))
            fig.suptitle('Histogram of frequency cutoffs')

            gs = gridspec.GridSpec(4,1, figure=fig)

            axs00 = fig.add_subplot(gs[0,0])
            channel_data = out[0].detach().cpu().numpy()
            filtered_channel = y[0].detach().cpu().numpy()

            # print(out.shape)
            # print(channel_data.shape)
            axs00.plot(channel_data, label='raw')
            axs00.set_ylim(min(channel_data), max(channel_data))

            # print(y.shape)
            # print(filtered_channel.shape)
            axs10 = fig.add_subplot(gs[1,0])
            axs10.plot(filtered_channel, label='filtered')
            # axs10.plot(filtered_channel[1], linewidth=0.5 ,label='filtered')
            # axs10.plot(filtered_channel[2], linewidth=0.5 ,label='filtered')

            axs10.set_ylim(min(channel_data), max(channel_data))

            axs20 = fig.add_subplot(gs[2,0])
            axs20.psd(channel_data, Fs=fs, label='PSD raw')
            axs20.title.set_text('PSD raw')
            axs20.set_xlim(0,1024)
            axs20.set_ylim(-120,-20)
            
            axs30 = fig.add_subplot(gs[3,0])
            axs30.psd(filtered_channel, Fs=fs, label='PSD filtered')
            # axs30.psd(filtered_channel[1], Fs=fs, label='PSD filtered')
            # axs30.psd(filtered_channel[2], Fs=fs, label='PSD filtered')

            axs30.title.set_text('PSD filtered')
            axs30.set_xlim(0,1024)
            axs30.set_ylim(-120,-20)

            fig.savefig(f'filtered.png')
            plt.close()
        return y

    @staticmethod
    def pad(x: torch.Tensor, n: int) -> torch.Tensor:
        """
        Pad the tensor with the first and last values
        Args:
            x: the tensor to pad
            n: the number of values to pad (left and right)
        Returns:
            x: the padded tensor
        """
        left_pad = x[:, :, 0].unsqueeze(1).repeat(1, 1, n)
        right_pad = x[:, :, x.shape[2] - 1].unsqueeze(1).repeat(1, 1, n)
        x = torch.cat((left_pad, x), dim=2)
        x = torch.cat((x, right_pad), dim=2)
        return x

    @property
    def device(self):
        return self.sigma.device

class FilterResNet54Double(nn.Module):
    def __init__(self, bottleneck=False):
        super().__init__()
        self.low_pass = DeepGaussianFilter(order=0)
        # self.channel_head = ResBlock(6, 2, False)
        self.feature_extractor = nn.Sequential(
            ResBlock(2  , 16 , bottleneck),
            ResBlock(16 , 16 , bottleneck),
            ResBlock(16 , 16 , bottleneck),
            ResBlock(16 , 16 , bottleneck),
            ResBlock(16 , 32 , bottleneck, stride=2),
            ResBlock(32 , 32 , bottleneck),
            ResBlock(32 , 32 , bottleneck),
            ResBlock(32 , 64 , bottleneck, stride=2),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 128, bottleneck, stride=2),
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck, stride=2),
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck, stride=2),
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 32 , bottleneck),
            ResBlock(32 , 32 , bottleneck),
            ResBlock(32 , 32 , bottleneck),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(32, 64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 2, 1), nn.Softmax(dim=1)
        )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.cls_head.parameters():
            param.requires_grad = False


    def forward(self, x):
        # print(x.shape)
        # print(x[:, 0, :].squeeze().shape)
        x1 = self.low_pass(x[:,0,:])

        # print(x1.shape)
        x2 = self.low_pass(x[:,1,:])
        x = torch.concat((x1,x2), dim=1)
        # x = self.channel_head(x)
        # print(x.shape)
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)
    