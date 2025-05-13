import torch.nn as nn 

from modules.resnet import ResNet54Double, ResNet54
from modules.filter import FIRResNet54Double, FIRResNet54, FilterResNet54Double
from modules.moeresnet import MoEResNet54Double, MoEResNet54Doublev2, \
MoEResNet54Doublev3, MoEResNet54Doublev4
from modules.moeparallel import MoDEResNet54Double
from modules.myresnetkan import ResNet54DoubleKAN, ResNet54KAN, ResNet54KAXN, ResNet54DoubleKAXN
from modules.myresnetkanV2 import ResNet54DoubleKANv2, ResNet54KANv2, ResNet54KAXNv2, ResNet54DoubleKAXNv2
from modules.wavelet import WavResNet54Double
from modules.mymoe import MoEResNet54KAN, MoEResNet54DoubleKAN
from modules.kan_convs.fast_kan_conv import FastKANConv1DLayer
from modules.kan_convs import KABNConv1DLayer, KAJNConv1DLayer, KALNConv1DLayer, KANConv1DLayer, \
    KAGNConv1DLayerV2, KAGNConv1DLayer, BottleNeckKAGNConv1DLayer, ReLUKANConv1DLayer, BottleNeckReLUKANConv1DLayer


def find_model(model_type, double, basis_type, base_type, bottleneck, grid_size, num_experts, top_k, train_device):
    MoE = False
    labels = False

    print(train_device)
    
    if model_type == 'ResNet54':
        if double == 'no':
            model = ResNet54(bottleneck).to(train_device)
            print("Architecture is: ResNet54")
        else:
            model = ResNet54Double(bottleneck).to(train_device)
            print("Architecture is: ResNet54Double")

    elif model_type == 'FIR':
        if double == 'no':
            model = FIRResNet54(bottleneck).to(train_device)
            print("Architecture is: ResNet54")
        else:
            model = FIRResNet54Double(bottleneck).to(train_device)
            print("Architecture is: ResNet54Double")
        labels = True

    elif model_type == 'Filter':
        if double == 'no':
            # model = FIRResNet54(bottleneck).to(train_device)
            print("Architecture is: ResNet54")
        else:
            model = FilterResNet54Double().to(train_device)
            print("Architecture is: ResNet54Double")
        labels = True

    elif model_type == 'WAV' or model_type == 'Wav':
        print(basis_type)
        print(base_type)
        model = WavResNet54Double(basis_type, base_type).to(train_device)
        print("Architecture is: ResNet54DoubleWavKAN")

    elif 'KAN' in model_type:

        if basis_type == 'FastKAN': 
            basis = FastKANConv1DLayer
        elif basis_type == 'KAGN':
            basis = KAGNConv1DLayer
        elif basis_type == 'KAGNv2':
            basis = KAGNConv1DLayerV2
        elif basis_type == 'KAGNBN':
            basis = BottleNeckKAGNConv1DLayer
        elif basis_type == 'ReLUKAN':
            basis = ReLUKANConv1DLayer
        elif basis_type == 'ReLUKANBN':
            basis = BottleNeckReLUKANConv1DLayer
        elif basis_type == 'KABN':
            basis = KABNConv1DLayer
        elif basis_type == 'KAJN':
            basis = KAJNConv1DLayer
        elif basis_type == 'KALN':
            basis = KALNConv1DLayer
        elif basis_type == 'KAN':
            basis = KANConv1DLayer
        else:
            raise TypeError(f"Basis {basis_type} is not supported")
        print(basis_type)
        print("Basis is " + str(basis))

        if basis_type == 'FASTKAN' or basis_type == 'KAN':

            if base_type == 'SiLU':
                base = nn.SiLU
            elif base_type == 'SELU':
                base = nn.SELU
            elif base_type == 'ELU':
                base = nn.ELU
            elif base_type == 'LeakyReLU':
                base = nn.LeakyReLU
            elif base_type == 'GELU':
                base = nn.GELU
            elif base_type == 'CELU':
                base = nn.CELU
            elif base_type == 'RRELU':
                base = nn.RReLU
            elif base_type == 'Tanh':
                base = nn.Tanh
            elif base_type == 'Tanhshrink':
                base = nn.Tanhshrink
            elif base_type == 'LogSigmoid':
                base = nn.LogSigmoid
            else:
                raise TypeError(f"Base {base_type} is not supported")
            print(base_type)
            print("Base is " + str(base))
            
            if model_type == 'KANv2':
                if double == 'no':
                    model = ResNet54KANv2(basis, base, bottleneck).to(train_device)
                    print("Architecture is: ResNet54KANv2")
                else:
                    model = ResNet54DoubleKANv2(basis, base, bottleneck).to(train_device)
                    print("Architecture is: ResNet54DoubleKANv2")
            else:
                if double == 'no':
                    model = ResNet54KAN(basis, base).to(train_device)
                    print("Architecture is: ResNet54KAN")
                else:
                    model = ResNet54DoubleKAN(basis, base).to(train_device)
                    print("Architecture is: ResNet54DoubleKAN")

        else:
            if model_type == 'KANv2':
                if double == 'no':
                    model = ResNet54KAXNv2(basis, bottleneck).to(train_device)
                    print("Architecture is: ResNet54KAXNv2")
                else:
                    model = ResNet54DoubleKAXNv2(basis, bottleneck).to(train_device)
                    print("Architecture is: ResNet54DoubleKAXNv2")
            else:
                if double == 'no':
                    model = ResNet54KAXN(basis).to(train_device)
                    print("Architecture is: ResNet54KAXN")
                else:
                    model = ResNet54DoubleKAXN(basis).to(train_device)
                    print("Architecture is: ResNet54DoubleKAXN")
    elif model_type == 'MoEKAN':
        MoE = True
        if double == 'no':
            model = MoEResNet54KAN( num_experts, top_k).to(train_device)
            print("Architecture is: MoEResNet54KAN")
        else:
            model = MoEResNet54DoubleKAN(bottleneck, num_experts, top_k).to(train_device)
            print("Architecture is: MoEResNet54DoubleKAN")

    elif model_type == 'MoE':
        MoE = True
        if double == 'no':
            # model = MoEResNet54(num_experts, top_k).to(train_device)
            print("Architecture is: MoEResNet54")
        else:
            model = MoEResNet54Double(bottleneck, num_experts, top_k).to(train_device)
            print("Architecture is: MoEResNet54Double")
    elif model_type == 'MoEv2':
        MoE = True
        if double == 'no':
            # model = MoEResNet54(num_experts, top_k).to(train_device)
            print("Architecture is: MoEResNet54")
        else:
            model = MoEResNet54Doublev2(bottleneck, num_experts, top_k).to(train_device)
            print("Architecture is: MoEResNet54Doublev2")
    elif model_type == 'MoEv3':
        MoE = True
        if double == 'no':
            # model = MoEResNet54(num_experts, top_k).to(train_device)
            print("Architecture is: MoEResNet54")
        else:
            model = MoEResNet54Doublev3(bottleneck, num_experts, top_k).to(train_device)
            print("Architecture is: MoEResNet54Doublev3")
    elif model_type == 'MoEv4':
        MoE = True
        if double == 'no':
            # model = MoEResNet54(num_experts, top_k).to(train_device)
            print("Architecture is: MoEResNet54")
        else:
            model = MoEResNet54Doublev4(bottleneck, num_experts, top_k).to(train_device)
            print("Architecture is: MoEResNet54Doublev4")
    elif model_type == 'MoDE':
        MoE = True
        if double == 'no':
            # model = MoEResNet54(num_experts, top_k).to(train_device)
            print("Architecture is: MoDEResNet54")
        else:
            model = MoDEResNet54Double(bottleneck, num_experts, top_k).to(train_device)
            print("Architecture is: MoDEResNet54Doublev4")
    else:
        raise TypeError(f"Model {model_type} is not supported")
    
    return model, MoE, labels