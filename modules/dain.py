import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torch.distributions.normal import Normal

from modules.kan_convs.moe_utils import SparseDispatcher

# class MoEDAIN_Layer(nn.Module):

#     """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
#     Args:
#     input_size: integer - size of the input
#     output_size: integer - size of the input
#     num_experts: an integer - number of experts
#     hidden_size: an integer - hidden size of the experts
#     noisy_gating: a boolean
#     k: an integer - how many experts to use for each batch element
#     """
#     def __init__(self, mode='full', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001, input_dim=144, num_experts=4, k=2, temperature=1, noisy_gating=True):
#         super(MoEDAIN_Layer, self).__init__()
#         self.noisy_gating = noisy_gating
#         self.num_experts = num_experts
#         self.input_size = input_dim
#         input_size = input_dim
#         self.k = k
#         self.temperature = temperature

#         self.usage = torch.zeros(num_experts)

#         # instantiate experts
#         # print(input_size)
#         self.experts = nn.ModuleList([DAIN_Layer(input_dim=self.input_size) for i in range(self.num_experts)])
#         self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
#         self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)


#         self.softplus = nn.Softplus()
#         self.softmax = nn.Softmax(1)
#         self.register_buffer("mean", torch.tensor([0.0]))
#         self.register_buffer("std", torch.tensor([1.0]))

#         self.avgpool = nn.AdaptiveAvgPool1d((1,))
#         self.conv_dims = 1

#         for i in range(1, num_experts):
#             self.experts[i].load_state_dict(self.experts[0].state_dict())
        
#         assert(self.k <= self.num_experts)

#     def cv_squared(self, x):
#         """The squared coefficient of variation of a sample.
#         Useful as a loss to encourage a positive distribution to be more uniform.
#         Epsilons added for numerical stability.
#         Returns 0 for an empty Tensor.
#         Args:
#         x: a `Tensor`.
#         Returns:
#         a `Scalar`.
#         """
#         eps = 1e-10
#         # if only num_experts = 1

#         if x.shape[0] == 1:
#             return torch.tensor([0], device=x.device, dtype=x.dtype)
#         return x.float().var() / (x.float().mean() ** 2 + eps)



#     def _gates_to_load(self, gates):
#         """Compute the true load per expert, given the gates.
#         The load is the number of examples for which the corresponding gate is >0.
#         Args:
#         gates: a `Tensor` of shape [batch_size, n]
#         Returns:
#         a float32 `Tensor` of shape [n]
#         """
#         return (gates > 0).sum(0)




#     def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
#         """Helper function to NoisyTopKGating.
#         Computes the probability that value is in top k, given different random noise.
#         This gives us a way of backpropagating from a loss that balances the number
#         of times each expert is in the top k experts per example.
#         In the case of no noise, pass in None for noise_stddev, and the result will
#         not be differentiable.
#         Args:
#         clean_values: a `Tensor` of shape [batch, n].
#         noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
#           normally distributed noise with standard deviation noise_stddev.
#         noise_stddev: a `Tensor` of shape [batch, n], or None
#         noisy_top_values: a `Tensor` of shape [batch, m].
#            "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
#         Returns:
#         a `Tensor` of shape [batch, n].
#         """

#         batch = clean_values.size(0)
#         m = noisy_top_values.size(1)
#         top_values_flat = noisy_top_values.flatten()
        
#         threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
#         threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
#         is_in = torch.gt(noisy_values, threshold_if_in)
#         threshold_positions_if_out = threshold_positions_if_in - 1
#         threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat,0 , threshold_positions_if_out), 1)
#         # is each value currently in the top k.
#         normal = Normal(self.mean, self.std)
#         prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
#         prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
#         prob = torch.where(is_in, prob_if_in, prob_if_out)
#         return prob


#     def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
#         """Noisy top-k gating.
#           See paper: https://arxiv.org/abs/1701.06538.
#           Args:
#             x: input Tensor with shape [batch_size, input_size]
#             train: a boolean - we only add noise at training time.
#             noise_epsilon: a float
#           Returns:
#             gates: a Tensor with shape [batch_size, num_experts]
#             load: a Tensor with shape [num_experts]
#         """
#         # print(x.shape)
#         # print(self.w_gate.shape)
#         clean_logits = x @ self.w_gate
#         if self.noisy_gating:
#             raw_noise_stddev = x @ self.w_noise
#             noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
#             noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
#             logits = noisy_logits
#         else:
#             logits = clean_logits

#         # calculate topk + 1 that will be needed for the noisy gates
#         logits = self.softmax(logits/self.temperature)
#         top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
#         top_k_logits = top_logits[:, :self.k]
#         top_k_indices = top_indices[:, :self.k]
#         top_k_gates = self.softmax(top_k_logits)

        
#         zeros = torch.zeros_like(logits, requires_grad=True)
#         gates = zeros.scatter(1, top_k_indices, top_k_gates)

        
#         with torch.no_grad():
#             for idx in top_k_indices.reshape(-1):
#                 self.usage[idx] += 1


#         if self.noisy_gating and self.k < self.num_experts:
#             load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
#         else:
#             load = self._gates_to_load(gates)
#         return gates, load



#     def forward(self, x, train=True, loss_coef=1e-2):
#         """Args:
#         x: tensor shape [batch_size, input_size]
#         train: a boolean scalar.
#         loss_coef: a scalar - multiplier on load-balancing losses
#         Returns:
#         y: a tensor with shape [batch_size, output_size].
#         extra_training_loss: a scalar.  This should be added into the overall
#         training loss of the model.  The backpropagation of this loss
#         encourages all experts to be approximately equally used across a batch.
#         """

#         gate_x = torch.flatten(self.avgpool(x), 1).to('cuda:0')
#         gates, load = self.noisy_top_k_gating(gate_x, train)
#         # calculate importance loss
#         importance = gates.sum(0)
#         #
#         loss = self.cv_squared(importance) + self.cv_squared(load)
#         loss *= loss_coef
#         self.loss = loss

#         dispatcher = SparseDispatcher(self.num_experts, gates)
#         expert_inputs = dispatcher.dispatch(x)
#         gates = dispatcher.expert_to_gates()
#         expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
#         y = dispatcher.combine(expert_outputs, self.conv_dims)
        
#         return y


class MoEDAIN_Layer(nn.Module):
    def __init__(self, mode='full', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001, input_dim=144):
        super(MoEDAIN_Layer, self).__init__()
        print("Mode = ", mode)

        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive scaling
        self.gating_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim*4),
            nn.Linear(input_dim*4, input_dim)
        )
        # self.gating_layer = nn.Linear(input_dim, input_dim)
        self.eps = 1e-8

    def forward(self, x):
        # Expecting  (n_samples, dim,  n_feature_vectors)
        # Step 1:
        avg = torch.mean(x, 2)
        adaptive_avg = self.mean_layer(avg)
        adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
        x = x - adaptive_avg

        # # Step 2:
        std = torch.mean(x ** 2, 2)
        std = torch.sqrt(std + self.eps)
        adaptive_std = self.scaling_layer(std)
        adaptive_std[adaptive_std <= self.eps] = 1

        adaptive_std = adaptive_std.resize(adaptive_std.size(0), adaptive_std.size(1), 1)
        x = x / adaptive_std

        # Step 3:
        avg = torch.mean(x, 2)
        gate = torch.sigmoid(self.gating_layer(avg))
        gate = gate.resize(gate.size(0), gate.size(1), 1)
        x = x * gate
        return x


# code from https://github.com/passalis/dain
class DAIN_Layer(nn.Module):
    def __init__(self, mode='full', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001, input_dim=144):
        super(DAIN_Layer, self).__init__()
        print("Mode = ", mode)

        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive scaling
        # self.gating_layer = nn.Sequential(
        #     nn.Linear(input_dim, input_dim*4),
        #     nn.Linear(input_dim*4, input_dim)
        # )
        self.gating_layer = nn.Linear(input_dim, input_dim)
        self.eps = 1e-8

    def forward(self, x):
        # Expecting  (n_samples, dim,  n_feature_vectors)
        # Step 1:
        avg = torch.mean(x, 2)
        adaptive_avg = self.mean_layer(avg)
        adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
        x = x - adaptive_avg

        # # Step 2:
        std = torch.mean(x ** 2, 2)
        std = torch.sqrt(std + self.eps)
        adaptive_std = self.scaling_layer(std)
        adaptive_std[adaptive_std <= self.eps] = 1

        adaptive_std = adaptive_std.resize(adaptive_std.size(0), adaptive_std.size(1), 1)
        x = x / adaptive_std

        # Step 3:
        avg = torch.mean(x, 2)
        gate = torch.sigmoid(self.gating_layer(avg))
        gate = gate.resize(gate.size(0), gate.size(1), 1)
        x = x * gate
        return x
