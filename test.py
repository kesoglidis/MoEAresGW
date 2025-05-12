#! /usr/bin/env python3
import argparse
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils.eval_utils import get_triggers, get_clusters
from modules.dain import DAIN_Layer
from modules.whiten import CropWhitenNet
from modules.resnet import ResNet54Double
from modules.find_model import find_model
from modules.dain import MoEDAIN_Layer

# weights filename
path = 'trained_models/improved_d4_model/weights.pt'

# Set data type to be used
dtype = torch.float32

sample_rate = 2048
delta_t = 1. / sample_rate
delta_f = 1 / 1.25


def main(args):
    train_device = 'cuda:0' if torch.cuda.device_count() > 0 else 'cpu'

    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = path

    checkpoint = torch.load(weights_path, weights_only=False)
    # print(checkpoint.keys)
    if ('args' in checkpoint.keys()):
        model, double, basis, base, bottleneck, grid_size, num_experts, top_k = checkpoint['args']
        # model, double, basis, base  = checkpoint['args']
        print(model)
        # base_model = find_model(model, double, basis, base, False, 0, 4, 2, train_device)

        base_model, MoE, labels = find_model(model, double, basis, base,  bottleneck, grid_size, num_experts, top_k, train_device)
    else:
        base_model, MoE, labels = find_model(args.model, args.double, args.basis, args.base, args.bottleneck, args.grid_size, args.num_experts, args.top_k, train_device)

    print(type(base_model))
    # print("Moe: ",MoE)

    # MoE = False
    # base_model = ResNet54Double()
    if model == 'MoEv3':
        print("MoE DAIN")
        norm = MoEDAIN_Layer(input_dim=2).to(train_device)

        # print(norm.experts[0])
        # for i in range(num_experts):
        #     print(i)
        #     print('Mean: ', norm.experts[i].mean_layer.weight.data)
        #     print('Scaling: ', norm.experts[i].scaling_layer.weight.data)
        #     print('Gating: ', norm.experts[i].gating_layer.weight.data)

        # fig = plt.figure(figsize=(30,20))
        # # fig.suptitle('Histogram of frequency cutoffs')

        # gs = gridspec.GridSpec(4,2, figure=fig)

        # axs00 = fig.add_subplot(gs[0,0])
        # mean = norm.experts[0].mean_layer.weight.data.detach().cpu().numpy()
        # print(mean)
        # axs00.plot(mean, label='mean')
        # # axs00.title.set_text('Raw Signal:'+ str (labels) +' Noise:'+str(labels[0][1]))

        # axs10 = fig.add_subplot(gs[0,0])
        # scaling = norm.experts[0].scaling_layer.weight.data.detach().cpu().numpy()
        # print(scaling)

        # axs10.plot(scaling, label='scaling')

        # axs20 = fig.add_subplot(gs[0,0])
        # gating = norm.experts[0].gating_layer.weight.data.detach().cpu().numpy()
        # print(gating)

        # axs20.plot(gating, label='gating')

        # fig.legend()
        # fig.savefig(f'mean.png')
        # plt.close()

    else:
        norm = DAIN_Layer(input_dim=2).to(train_device)
        
       

    net = CropWhitenNet(base_model, norm).to(train_device)
    net.deploy = True
    
    # checkpoint = torch.load(weights_path, weights_only=False)
    net.load_state_dict(checkpoint['model'])
    
    # if model == 'MoEv3':
    #     for i in range(num_experts):
    #         print(i)
    #         print('Usage: ', norm.usage)
    #         print('Mean: ', norm.experts[i].mean_layer.weight.data)
    #         print('Scaling: ', norm.experts[i].scaling_layer.weight.data)
    #         print('Gating: ', norm.experts[i].gating_layer.weight.data)
    # else:
    print('Mean: ', norm.mean_layer.weight.data)
    print('Scaling: ', norm.scaling_layer.weight.data)
    print('Gating: ', norm.gating_layer.weight.data)
    # net.load_state_dict(torch.load(weights_path, map_location=train_device))
 
    net.eval()

    # run on foreground
    inputfile = args.inputfile
    outputfile = args.outputfile
    step_size = 5.1
    slice_dur = 6.25
    trigger_threshold = 0.5
    cluster_threshold = 0.35
    var = 0.5

    test_batch_size = 1
    with torch.no_grad():
        triggers = get_triggers(net,
                                inputfile,
                                step_size=step_size,
                                trigger_threshold=trigger_threshold,
                                device=train_device,
                                verbose=True,
                                batch_size=test_batch_size,
                                whiten=False,
                                slice_length=int(slice_dur * 2048),
                                MoE=MoE)

    time, stat, var = get_clusters(triggers, cluster_threshold, var=var)

    with h5py.File(outputfile, 'w') as outfile:
        print("Saving clustered triggers into %s." % outputfile)

        outfile.create_dataset('time', data=time)
        outfile.create_dataset('stat', data=stat)
        outfile.create_dataset('var', data=var)

        print("Triggers saved, closing file.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('inputfile', type=str, help="The path to the input data file.")
    parser.add_argument('outputfile', type=str, help="The path where to store the triggers.")

    # parser.add_argument('--model', type=str, help='Model architecture.', default='double')
    # parser.add_argument('--double', type=str, default='no', help='If we want double neurons. Default is no.')
    # parser.add_argument('--base', type=str, default='SiLU', help='Base activation function. Options: SiLU, SELU, ELU, GELU, CELU, LeakyReLU, RReLU, Tanh, Tahnshrink and LogSigmoid')
    # parser.add_argument('--basis', type=str, default='FastKAN', help='Basis spline functions. Options: FastKAN, KABN, KAJN, KALN, KAN, KAGN, KAGNBN, ReLUKAN and ReLUKANBN')

    parser.add_argument('--weights', type=str, help='Custom weights path.', default=None)

    args = parser.parse_args()
    print(args.weights)
    main(args)
