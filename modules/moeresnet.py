import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torch.distributions.normal import Normal

from modules.kan_convs.moe_utils import SparseDispatcher

from modules.resnet import ResBlock

# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: Yeonwoo Sung
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py

class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """
    def __init__(self, block, input_size, output_size, bottleneck, kernel_size=3, padding=1, stride=1, num_experts=4, k=2, temperature=1, noisy_gating=True):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = k
        self.temperature = temperature
        self.bottleneck = bottleneck

        self.usage = torch.zeros(num_experts)

        # instantiate experts
        # print(input_size)
        if block == ResBlock:
            self.experts = nn.ModuleList([block(self.input_size, self.output_size, self.bottleneck, stride=stride) for i in range(self.num_experts)])
            
        elif block == ClsHead:
            self.experts = nn.ModuleList([block(self.input_size, self.output_size) for i in range(self.num_experts)])

        else:
            self.experts = nn.ModuleList([block(self.input_size, self.output_size, kernel_size=kernel_size, padding=padding, stride=stride) for i in range(self.num_experts)])

        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self.avgpool = nn.AdaptiveAvgPool1d((1,))
        self.conv_dims = 1

        for i in range(1, num_experts):
            self.experts[i].load_state_dict(self.experts[0].state_dict())
        
        assert(self.k <= self.num_experts)


    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)


    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)


    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat,0 , threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob


    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        # print(x.shape)
        # print(self.w_gate.shape)
        clean_logits = x @ self.w_gate
        if self.noisy_gating:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits/self.temperature)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        with torch.no_grad():
            for idx in top_k_indices.reshape(-1):
                self.usage[idx] += 1

        if self.noisy_gating and self.k < self.num_experts:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load



    def forward(self, x, train=True, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """

        gate_x = torch.flatten(self.avgpool(x), 1).to('cuda:0')
        gates, load = self.noisy_top_k_gating(gate_x, train)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs, self.conv_dims)
        
        return y, loss

class MoEResBlock(nn.Module):
    temperature = 1
    def __init__(self, in_channels, out_channels, bottleneck, num_experts, top_k,  stride=1):
        super().__init__()
        if out_channels != in_channels or stride > 1:
            self.x_transform = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.x_transform = nn.Identity()

        self.bottleneck = bottleneck

        if bottleneck:
            width = int(out_channels/4.0)
            self.body = nn.Sequential(
                nn.Conv1d(in_channels, width, kernel_size=1),
                nn.BatchNorm1d(width),
                MoE(nn.Conv1d, width, width, kernel_size=3, stride=stride, padding=1, 
                    num_experts=num_experts, k=top_k, temperature=self.temperature),
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
                MoE(nn.Conv1d, out_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                     num_experts=num_experts, 
                    k=top_k, temperature=self.temperature),
                # nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x, train=True):
        moe_loss = 0
        if self.bottleneck:
            out = self.body[0](x)
            out = self.body[1](out)

            out, moe_loss = self.body[2](out)
            out = self.body[3](out)
            out = self.body[4](out)
            out = self.body[5](out)
        else:
            out = self.body[0](x)
            out = self.body[1](out)
            out = self.body[2](out)

            out, moe_loss = self.body[3](out)
            out = self.body[4](out)

        out = F.relu(out + self.x_transform(x))
        return out, moe_loss

class MoEResBlockv2(nn.Module):
    temperature = 1
    def __init__(self, in_channels, out_channels, bottleneck, num_experts, top_k, stride=1):
        super().__init__()
        if out_channels != in_channels or stride > 1:
            self.x_transform = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.x_transform = nn.Identity()

        self.body = nn.Sequential(MoE(ResBlock, in_channels, out_channels, bottleneck, stride=stride, num_experts=num_experts, k=top_k, temperature=self.temperature))
        self.usage = torch.zeros(num_experts)

    def forward(self, x, train=True):
        moe_loss = 0

        out, moe_loss = self.body[0](x)

        self.usage = self.body[0].usage
        out = F.relu(out + self.x_transform(x))
        return out, moe_loss

class MoEResNet54Double(nn.Module):
    def __init__(self, bottleneck, num_experts, top_k):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            MoEResBlock(2  , 16 , bottleneck, num_experts, top_k),
            MoEResBlock(16 , 16 , bottleneck, num_experts, top_k),
            MoEResBlock(16 , 16 , bottleneck, num_experts, top_k),
            MoEResBlock(16 , 16 , bottleneck, num_experts, top_k),
            MoEResBlock(16 , 32 , bottleneck, num_experts, top_k, stride=2),
            MoEResBlock(32 , 32 , bottleneck, num_experts, top_k),
            MoEResBlock(32 , 32 , bottleneck, num_experts, top_k),
            MoEResBlock(32 , 64 , bottleneck, num_experts, top_k, stride=2),
            MoEResBlock(64 , 64 , bottleneck, num_experts, top_k),
            MoEResBlock(64 , 64 , bottleneck, num_experts, top_k),
            MoEResBlock(64 , 128, bottleneck, num_experts, top_k, stride=2),
            MoEResBlock(128, 128, bottleneck, num_experts, top_k),
            MoEResBlock(128, 128, bottleneck, num_experts, top_k),
            MoEResBlock(128, 128, bottleneck, num_experts, top_k, stride=2),
            MoEResBlock(128, 128, bottleneck, num_experts, top_k),
            MoEResBlock(128, 128, bottleneck, num_experts, top_k),
            MoEResBlock(128, 128, bottleneck, num_experts, top_k, stride=2),
            MoEResBlock(128, 128, bottleneck, num_experts, top_k),
            MoEResBlock(128, 128, bottleneck, num_experts, top_k),
            MoEResBlock(128, 64 , bottleneck, num_experts, top_k),
            MoEResBlock(64 , 64 , bottleneck, num_experts, top_k),
            MoEResBlock(64 , 64 , bottleneck, num_experts, top_k),
            MoEResBlock(64 , 64 , bottleneck, num_experts, top_k),
            MoEResBlock(64 , 64 , bottleneck, num_experts, top_k),
            MoEResBlock(64 , 32 , bottleneck, num_experts, top_k),
            MoEResBlock(32 , 32 , bottleneck, num_experts, top_k),
            MoEResBlock(32 , 32 , bottleneck, num_experts, top_k) #32x64
        )
        
        self.cls_head = nn.Sequential(
            nn.Conv1d(32, 64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 2, 1), nn.Softmax(dim=1)
        )


    def forward(self, x, train: bool = True) -> Tensor:
        moe_loss = 0
        for block in self.feature_extractor:
            if block == MoEResBlock:
                x, _moe_loss = block(x, train)
                moe_loss += _moe_loss
            else:
                x = block(x, train)
        # print('Final', moe_loss)
        # x, moe_loss = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2), moe_loss
    
    # def forward(self, x):
    #     x, = self.feature_extractor(x)
    #     return self.cls_head(x).squeeze(2)
    
class MoEResNet54Doublev2(nn.Module):
    def __init__(self, bottleneck, num_experts, top_k):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlock(2  , 16 , bottleneck), # 16x2048
            ResBlock(16 , 16 , bottleneck),
            ResBlock(16 , 16 , bottleneck),
            ResBlock(16 , 16 , bottleneck),
            MoEResBlockv2(16 , 32 , bottleneck, num_experts, top_k, stride=2), # 32x1024
            ResBlock(32 , 32 , bottleneck),
            ResBlock(32 , 32 , bottleneck),
            MoEResBlockv2(32 , 64 , bottleneck, num_experts, top_k, stride=2), # 64x512
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            MoEResBlockv2(64 , 128, bottleneck, num_experts, top_k, stride=2), # 128x256
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck),
            MoEResBlockv2(128, 128, bottleneck, num_experts, top_k, stride=2), # 128x128
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck),
            MoEResBlockv2(128, 128, bottleneck, num_experts, top_k, stride=2), # 128x64
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 64 , bottleneck), #64x64
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 32 , bottleneck), #32x64
            ResBlock(32 , 32 , bottleneck),
            ResBlock(32 , 32 , bottleneck)  
        )
        
        self.cls_head = nn.Sequential(
            nn.Conv1d(32, 64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 2, 1), nn.Softmax(dim=1)
        )


    def forward(self, x, train: bool = True) -> Tensor:
        moe_loss = 0
        for block in self.feature_extractor:
            if type(block) == MoEResBlockv2:
                x, _moe_loss = block(x, train)
                moe_loss += _moe_loss
            else:
                x = block(x, train)
        return self.cls_head(x).squeeze(2), moe_loss

class MoEResNet54Doublev3(nn.Module):
    def __init__(self, bottleneck, num_experts, top_k):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlock(2  , 16 , bottleneck), # 16x2048
            ResBlock(16 , 16 , bottleneck),
            ResBlock(16 , 16 , bottleneck),
            ResBlock(16 , 16 , bottleneck),
            ResBlock(16 , 32 , bottleneck, stride=2),
            # MoEResBlockv2(16 , 32 , bottleneck, num_experts, top_k, stride=2), # 32x1024
            ResBlock(32 , 32 , bottleneck),
            ResBlock(32 , 32 , bottleneck),
            ResBlock(32 , 64 , bottleneck, stride=2),
            # MoEResBlockv2(32 , 64 , bottleneck, num_experts, top_k, stride=2), # 64x512
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 128 , bottleneck, stride=2),
            # MoEResBlockv2(64 , 128, bottleneck, num_experts, top_k, stride=2), # 128x256
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck, stride=2),
            # MoEResBlockv2(128, 128, bottleneck, num_experts, top_k, stride=2), # 128x128
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck, stride=2),
            # MoEResBlockv2(128, 128, bottleneck, num_experts, top_k, stride=2), # 128x64
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 64 , bottleneck), #64x64
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 32 , bottleneck), #32x64
            ResBlock(32 , 32 , bottleneck),
            ResBlock(32 , 32 , bottleneck)  
        )
        
        self.cls_head = nn.Sequential(
            nn.Conv1d(32, 64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 2, 1), nn.Softmax(dim=1)
        )


    def forward(self, x, train: bool = True) -> Tensor:
        moe_loss = 0
        for block in self.feature_extractor:
            if type(block) == MoEResBlockv2:
                x, _moe_loss = block(x, train)
                moe_loss += _moe_loss
            else:
                x = block(x, train)
            # print(x.shape)
        # print('Final', moe_loss)
        # x, moe_loss = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2), moe_loss

class MoEResNet54Doublev4(nn.Module):
    def __init__(self, bottleneck, num_experts, top_k):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            ResBlock(2  , 16 , bottleneck), # 16x2048
            ResBlock(16 , 16 , bottleneck),
            ResBlock(16 , 16 , bottleneck),
            ResBlock(16 , 16 , bottleneck),
            MoEResBlockv2(16 , 32 , bottleneck, num_experts, top_k, stride=2), # 32x1024
            ResBlock(32 , 32 , bottleneck),
            ResBlock(32 , 32 , bottleneck),
            MoEResBlockv2(32 , 64 , bottleneck, num_experts, top_k, stride=2), # 64x512
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            MoEResBlockv2(64 , 128, bottleneck, num_experts, top_k, stride=2), # 128x256
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck),
            MoEResBlockv2(128, 128, bottleneck, num_experts, top_k, stride=2), # 128x128
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck),
            MoEResBlockv2(128, 128, bottleneck, num_experts, top_k, stride=2), # 128x64
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 64 , bottleneck), #64x64
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 32 , bottleneck), #32x64
            ResBlock(32 , 32 , bottleneck),
            MoEResBlockv2(32, 32, bottleneck, num_experts, top_k) 
        )
        
        self.cls_head = nn.Sequential(
            nn.Conv1d(32, 64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 2, 1), nn.Softmax(dim=1)
        )
        # self.cls_head = MoE(ClsHead, 32, 64)

    def forward(self, x, train: bool = True) -> Tensor:
        moe_loss = 0
        for block in self.feature_extractor:
            if type(block) == MoEResBlockv2:
                x, _moe_loss = block(x, train)
                moe_loss += _moe_loss
            else:
                x = block(x, train)
        x = self.cls_head(x)
            # print(x.shape)
        # print('Final', moe_loss)
        # x, moe_loss = self.feature_extractor(x)
        return x.squeeze(2), moe_loss 

class ClsHead(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(input, output, 64), nn.ReLU(),
            nn.Conv1d(output, 2, 1), nn.Softmax(dim=1))
    
    def forward(self, x):
        return self.body(x)

# self.feature_extractor = nn.Sequential(
        #     ResBlock(2  , 16 , bottleneck),
        #     ResBlock(16 , 16 , bottleneck),
        #     ResBlock(16 , 16 , bottleneck),
        #     ResBlock(16 , 16 , bottleneck),
        #     MoEResBlockv2(16 , 32 , bottleneck, num_experts, top_k, stride=2), #strided
        #     # MoEResBlockv2(32 , 32 , bottleneck, num_experts, top_k), #expand
        #     ResBlock(32 , 32 , bottleneck),
        #     ResBlock(32 , 32 , bottleneck),
        #     MoEResBlockv2(32 , 64 , bottleneck, num_experts, top_k, stride=2),
        #     # MoEResBlockv2(64 , 64 , bottleneck, num_experts, top_k),
        #     ResBlock(64 , 64 , bottleneck),
        #     ResBlock(64 , 64 , bottleneck),
        #     MoEResBlockv2(64 , 128, bottleneck, num_experts, top_k, stride=2),
        #     # MoEResBlockv2(128, 128, bottleneck, num_experts, top_k),
        #     ResBlock(128, 128, bottleneck),
        #     ResBlock(128, 128, bottleneck),
        #     MoEResBlockv2(128, 128, bottleneck, num_experts, top_k, stride=2),
        #     # MoEResBlockv2(128, 128, bottleneck, num_experts, top_k),
        #     ResBlock(128, 128, bottleneck),
        #    ResBlock(128, 128, bottleneck),
        #     MoEResBlockv2(128, 128, bottleneck, num_experts, top_k, stride=2),
        #     #  MoEResBlockv2(128, 128, bottleneck, num_experts, top_k),
        #     ResBlock(128, 128, bottleneck),
        #     ResBlock(128, 128, bottleneck),
        #     ResBlock(128, 64 , bottleneck),
        #     ResBlock(64 , 64 , bottleneck),
        #     ResBlock(64 , 64 , bottleneck),
        #     ResBlock(64 , 64 , bottleneck),
        #     ResBlock(64 , 64 , bottleneck),
        #     ResBlock(64 , 32 , bottleneck),
        #     ResBlock(32 , 32 , bottleneck),
        #     ResBlock(32 , 32 , bottleneck) #32x64
        # )
        
class MoEResNet54Double(nn.Module):
    def __init__(self, bottleneck, num_experts, top_k):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlock(2  , 16 , bottleneck),
            ResBlock(16 , 16 , bottleneck),
            ResBlock(16 , 16 , bottleneck),
            ResBlock(16 , 16 , bottleneck),
            MoEResBlockv2(16 , 32 , bottleneck, num_experts, top_k, stride=2), #strided
            ResBlock(32 , 32 , bottleneck),
            ResBlock(32 , 32 , bottleneck),
            MoEResBlockv2(32 , 64 , bottleneck, num_experts, top_k, stride=2),
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            MoEResBlockv2(64 , 128, bottleneck, num_experts, top_k, stride=2),
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
            ResBlock(32 , 32 , bottleneck) #32x64
        )



        