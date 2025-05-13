import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import torch.distributed as dist
from torch.distributions.normal import Normal

from modules.kan_convs.moe_utils import SparseDispatcher

from modules.resnet import ResBlock

from typing import Callable, List, Union

def _maybe_repeat(x, n):
    if isinstance(x, list):
        assert len(x) == n
        return x
    else:
        return [x] * n

def transpose_list_of_lists(lol):
    return list(map(list, zip(*lol)))

class Parallelism():
    def __init__(self, devices: List[Union[str, torch.device]]):
        self.devices = devices
        self.n = len(devices)

    def __call__(self, fn: Union[Callable, List[Callable]], *args, **kwargs):
        fns = _maybe_repeat(fn, self.n)
        args_per_device = transpose_list_of_lists([_maybe_repeat(arg, self.n) for arg in args]) if args else [[] for _ in range(self.n)]
        kwargs_per_device = [{} for _ in range(self.n)]

        for k, v in kwargs.items():
            vals = _maybe_repeat(v, self.n)
            for i in range(self.n):
                kwargs_per_device[i][k] = vals[i]

        outputs = []
        for i in range(self.n):
            with torch.cuda.device(self.devices[i]):
                outputs.append(fns[i](*args_per_device[i], **kwargs_per_device[i]))

        if isinstance(outputs[0], tuple):
            return tuple([list(t) for t in zip(*outputs)])
        else:
            return outputs
        
class Expert(nn.Module):
    def __init__(self, input_size, output_size, bottleneck, stride):
        super().__init__()
        self.net = ResBlock(input_size, output_size, bottleneck, stride)
    def forward(self, x):
        return self.net(x)


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
    def __init__(self, input_size, output_size, bottleneck, stride=1, num_experts=4, k=2, temperature=1, noisy_gating=True):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = k
        self.temperature = temperature

        self.usage = torch.zeros(num_experts)

        # instantiate experts
        # print(input_size)
        
     
        self.gating_device = 'cuda:0'  # or specify
        self.expert_devices = [f"cuda:{i}" for i in range(num_experts)]
        self.parallel = Parallelism(self.expert_devices)

        self.experts = nn.ModuleList([
            Expert(input_size, output_size, bottleneck, stride).to(self.expert_devices[i])
            for i in range(num_experts)
        ])

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

        # Dispatch inputs per expert
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)

        # Forward expert inputs in parallel
        expert_outputs = [None] * self.num_experts
        futures = []

        for i, expert_input in enumerate(expert_inputs):
            if expert_input is None or expert_input.numel() == 0:
                continue

            expert_input = expert_input.to(self.expert_devices[i])
            expert_module = self.experts[i]

            futures.append((
                i,
                expert_module(expert_input).to(x.device)  # bring back to the input device
            ))

        for i, result in futures:
            expert_outputs[i] = result

        # Combine outputs
        y = dispatcher.combine(expert_outputs, self.conv_dims)

        return y, loss

class MoEResBlock(nn.Module):
    temperature = 1
    def __init__(self, in_channels, out_channels, bottleneck, num_experts, top_k, stride=1):
        super().__init__()
        if out_channels != in_channels or stride > 1:
            self.x_transform = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.x_transform = nn.Identity()

        self.body = nn.Sequential(MoE(ResBlock, in_channels, out_channels, stride=stride, num_experts=num_experts, k=top_k, temperature=self.temperature))
        self.usage = torch.zeros(num_experts)

    def forward(self, x, train=True):
        moe_loss = 0

        out, moe_loss = self.body[0](x)

        self.usage = self.body[0].usage
        out = F.relu(out + self.x_transform(x))
        return out, moe_loss
    
    def forward(self, x, train=True):
        # x: [batch_size, channels, seq_len]
        batch_size, channels, seq_len = x.size()
        x_flat = x.mean(dim=2)  # Global average pooling over seq_len
        logits = self.gate(x_flat)  # [batch_size, num_experts]
        topk_vals, topk_indices = torch.topk(logits, self.top_k, dim=1)

        # Create a mask for selected experts
        mask = torch.zeros_like(logits)
        mask.scatter_(1, topk_indices, 1)
        mask = mask.unsqueeze(2).unsqueeze(3)  # [batch_size, num_experts, 1, 1]

       
        # Apply experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Select inputs assigned to expert i
            expert_mask = (topk_indices == i).float().unsqueeze(2).unsqueeze(3)
            expert_input = x_all * expert_mask
            expert_output = expert(expert_input)
            expert_outputs.append(expert_output)

        moe_loss = 0

        out, moe_loss = self.body[0](x)

        self.usage = self.body[0].usage
        out = F.relu(out + self.x_transform(x))
        return out, moe_loss

    
    
class MoDEResNet54Double(nn.Module):
    def __init__(self, bottleneck, num_experts, top_k):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlock(2  , 16 , bottleneck), # 16x2048
            ResBlock(16 , 16 , bottleneck),
            ResBlock(16 , 16 , bottleneck),
            ResBlock(16 , 16 , bottleneck),
            MoEResBlock(16 , 32 , bottleneck, num_experts, top_k, stride=2), # 32x1024
            ResBlock(32 , 32 , bottleneck),
            ResBlock(32 , 32 , bottleneck),
            MoEResBlock(32 , 64 , bottleneck, num_experts, top_k, stride=2), # 64x512
            ResBlock(64 , 64 , bottleneck),
            ResBlock(64 , 64 , bottleneck),
            MoEResBlock(64 , 128, bottleneck, num_experts, top_k, stride=2), # 128x256
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck),
            MoEResBlock(128, 128, bottleneck, num_experts, top_k, stride=2), # 128x128
            ResBlock(128, 128, bottleneck),
            ResBlock(128, 128, bottleneck),
            MoEResBlock(128, 128, bottleneck, num_experts, top_k, stride=2), # 128x64
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

        # Gather inputs from all devices
        world_size = dist.get_world_size()
        print("World size: ", world_size)
        

        for block in self.feature_extractor:
            if type(block) == MoEResBlock:
                x_list = [torch.zeros_like(x) for _ in range(world_size)]
        
                dist.all_gather(x_list, x)
                print(x_list)
                print(x_list.shape)
                x_all = torch.cat(x_list, dim=0)  # [batch_size * world_size, channels, seq_len]

                x, _moe_loss = block(x, train)
                moe_loss += _moe_loss
            else:
                x = block(x, train)
        return self.cls_head(x).squeeze(2), moe_loss
