import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time 

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck=False, stride=1):
        super().__init__()
        if out_channels != in_channels or stride > 1:
            self.x_transform = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.x_transform = nn.Identity()

        self.out_channels = out_channels

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

    def forward(self, x, train=True):
        out = F.relu(self.body(x) + self.x_transform(x))

        # print(self.out_channels)
        # if self.out_channels == 16:
            
        #     fig = plt.figure(figsize=(24,15))
        #     fig.suptitle('Histogram of frequency cutoffs')

        #     gs = gridspec.GridSpec(1,1, figure=fig)

        #     axs00 = fig.add_subplot(gs[0,0])
        #     axs00.plot(x.detach().cpu().numpy()[0][0], label='raw')
        #     axs00.plot(out.detach().cpu().numpy()[0][0], label='output')

        #     axs00.title.set_text('Raw')
        #     # axs00.set_ylim(min(channel_data), max(channel_data))

        #     fig.savefig(f'layer.png')
        #     time.sleep(4)
        #     plt.close()
        return out

# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, bottleneck=False, stride=1):
#         super().__init__()
#         if out_channels != in_channels or stride > 1:
#             self.x_transform = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
#         else:
#             self.x_transform = nn.Identity()

#         self.body = nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(),
#             nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
#             nn.BatchNorm1d(out_channels)
#         )

#     def forward(self, x):
#         x = F.relu(self.body(x) + self.x_transform(x))
#         return x

class ResNet54(nn.Module):
    def __init__(self, bottleneck):
        super().__init__()
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

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)


class ResNet54Double(nn.Module):
    def __init__(self, bottleneck=False):
        super().__init__()
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

    def forward(self, x):
        
        # for i in range(x.shape[0]):
        #     fig = plt.figure(figsize=(30,20))

        #     plt.psd(x[i][0].detach().cpu().numpy(), Fs=2048)#*2/3*10e21)
        #     fig.savefig(f'sample2.png')
        #     plt.close
        #     import time
        #     time.sleep(2)


        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)
    