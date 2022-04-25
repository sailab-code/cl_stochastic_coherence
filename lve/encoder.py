from torch import nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms, datasets, models
import os
from torchvision.transforms import Compose
from dpt_intel.dpt.models import DPTSegmentationModel
from dpt_intel.dpt.transforms import Resize, NormalizeImage, PrepareForNet
import torchvision

from torchsummary import summary


class EncoderFactory:
    @staticmethod
    def createEncoder(options):
        if options["architecture"] == "standard":
            return Encoder(options)
        elif options["architecture"] == "standard_sigmoidal":
            return SigmoidalEncoder(options)
        elif options["architecture"] == "larger_standard":
            return LargerStandardEncoder(options)
        elif options["architecture"] == "resnetu":
            # az = ResNetUnetEncoder(options)
            # summary(az.to("cuda:0"), input_size=(3, options["w"], options["h"]))
            return ResNetUnetEncoder(options)
        elif options["architecture"] == "resnetunobn":
            return ResNetUnetNoBNEncoder(options)
        elif options["architecture"] == "resnetunoskip":
            return ResNetUnetNoSkipEncoder(options)
        elif options["architecture"] == "resnetunolastskip":
            return ResNetUnetNoLastSkipEncoder(options)
        elif options["architecture"] == "deeplab_resnet101_backbone":
            return DeepLabv3_Resnet101BackBone(options)
        elif options["architecture"] == "deeplab_resnet101_classifier":
            return DeepLabv3_Resnet101Classifier(options)
        elif options["architecture"] == "dpt_backbone":
            return DPTHybridEncoderBackbone(options)
        elif options["architecture"] == "dpt_classifier":
            return DPTHybridEncoderClassifier(options)
        elif options["architecture"] == "identity":
            return IdentityEncoder(options)
        else:
            raise AttributeError(f"Architecture {options['architecture']} unknown.")


class Encoder(nn.Module):

    def __init__(self, options):
        super(Encoder, self).__init__()
        self.options = options
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=options['c'], out_channels=8, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=options['num_what'], kernel_size=7, padding=3,
                      bias=True),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        # list containing all the neural activations after the application of the non-linearity
        activations = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if "activation" in str(type(self.layers[i])):  # activation modules have this string...watch out
                activations.append(x)
        return x, activations


class IdentityEncoder(nn.Module):

    def __init__(self, options):
        super(IdentityEncoder, self).__init__()
        self.options = options

    def forward(self, x):
        # list containing all the neural activations after the application of the non-linearity
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        return x, None


class LargerStandardEncoder(nn.Module):

    def __init__(self, options):
        super(LargerStandardEncoder, self).__init__()
        self.options = options
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=options['c'], out_channels=16, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=options['num_what'], kernel_size=5, padding=2,
                      bias=True),
        )

    def forward(self, x):
        # list containing all the neural activations after the application of the non-linearity

        return self.layers(x), None


### Sigmoidal Nets

class SigmoidalEncoder(Encoder):
    def __init__(self, options):
        super(Encoder, self).__init__()
        self.options = options
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=options['c'], out_channels=8, kernel_size=5, padding=2, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, padding=3, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, padding=3, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=16, out_channels=options['num_what'], kernel_size=7, padding=3,
                      bias=True),
            nn.Sigmoid(),
        ])


#####
base_model = models.resnet18(pretrained=False)


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


def recursive_avoid_bn(base_model):
    for id, (name, child_model) in enumerate(base_model.named_children()):
        if isinstance(child_model, nn.BatchNorm2d):
            setattr(base_model, name, nn.Identity())

    for name, immediate_child_module in base_model.named_children():
        recursive_avoid_bn(immediate_child_module)


# from https://github.com/usuyama/pytorch-unet/blob/master/pytorch_resnet18_unet.ipynb
class ResNetUNet(nn.Module):

    def __init__(self, n_class, batch_norm):
        super().__init__()

        if not batch_norm:
            recursive_avoid_bn(base_model)
        self.base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        # repeat grey channel for rgb
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)

        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


class ResNetUnetEncoder(Encoder):
    def __init__(self, options):
        super(Encoder, self).__init__()
        self.options = options
        self.layers = ResNetUNet(options['num_what'], batch_norm=True)

    def forward(self, x):
        return self.layers(x), None


class ResNetUnetNoBNEncoder(Encoder):
    def __init__(self, options):
        super(Encoder, self).__init__()
        self.options = options
        self.layers = ResNetUNet(options['num_what'], batch_norm=False)

    def forward(self, x):
        return self.layers(x), None


#### No SKip


class ResNetUNetNoSkip(nn.Module):

    def __init__(self, n_class, batch_norm):
        super().__init__()

        if not batch_norm:
            recursive_avoid_bn(base_model)
        self.base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(512, 512, 3, 1)
        self.conv_up2 = convrelu(512, 256, 3, 1)
        self.conv_up1 = convrelu(256, 256, 3, 1)
        self.conv_up0 = convrelu(256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        # repeat grey channel for rgb
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        x = self.conv_up3(x)

        x = self.upsample(x)
        x = self.conv_up2(x)

        x = self.upsample(x)
        x = self.conv_up1(x)

        x = self.upsample(x)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


class ResNetUnetNoSkipEncoder(Encoder):
    def __init__(self, options):
        super(Encoder, self).__init__()
        self.options = options
        self.layers = ResNetUNetNoSkip(options['num_what'], batch_norm=True)

    def forward(self, x):
        return self.layers(x), None


### NO last skip


class ResNetUNetNoLastSkip(nn.Module):

    def __init__(self, n_class, batch_norm):
        super().__init__()

        if not batch_norm:
            recursive_avoid_bn(base_model)
        self.base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        # repeat grey channel for rgb
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


class ResNetUnetNoLastSkipEncoder(Encoder):
    def __init__(self, options):
        super(Encoder, self).__init__()
        self.options = options
        self.layers = ResNetUNetNoLastSkip(options['num_what'], batch_norm=True)

    def forward(self, x):
        return self.layers(x), None


#
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 64, 256, 256]           1,792
#               ReLU-2         [-1, 64, 256, 256]               0
#             Conv2d-3         [-1, 64, 256, 256]          36,928
#               ReLU-4         [-1, 64, 256, 256]               0
#             Conv2d-5         [-1, 64, 128, 128]           9,408
#        BatchNorm2d-6         [-1, 64, 128, 128]             128
#               ReLU-7         [-1, 64, 128, 128]               0
#          MaxPool2d-8           [-1, 64, 64, 64]               0
#             Conv2d-9           [-1, 64, 64, 64]          36,864
#       BatchNorm2d-10           [-1, 64, 64, 64]             128
#              ReLU-11           [-1, 64, 64, 64]               0
#            Conv2d-12           [-1, 64, 64, 64]          36,864
#       BatchNorm2d-13           [-1, 64, 64, 64]             128
#              ReLU-14           [-1, 64, 64, 64]               0
#        BasicBlock-15           [-1, 64, 64, 64]               0
#            Conv2d-16           [-1, 64, 64, 64]          36,864
#       BatchNorm2d-17           [-1, 64, 64, 64]             128
#              ReLU-18           [-1, 64, 64, 64]               0
#            Conv2d-19           [-1, 64, 64, 64]          36,864
#       BatchNorm2d-20           [-1, 64, 64, 64]             128
#              ReLU-21           [-1, 64, 64, 64]               0
#        BasicBlock-22           [-1, 64, 64, 64]               0
#            Conv2d-23          [-1, 128, 32, 32]          73,728
#       BatchNorm2d-24          [-1, 128, 32, 32]             256
#              ReLU-25          [-1, 128, 32, 32]               0
#            Conv2d-26          [-1, 128, 32, 32]         147,456
#       BatchNorm2d-27          [-1, 128, 32, 32]             256
#            Conv2d-28          [-1, 128, 32, 32]           8,192
#       BatchNorm2d-29          [-1, 128, 32, 32]             256
#              ReLU-30          [-1, 128, 32, 32]               0
#        BasicBlock-31          [-1, 128, 32, 32]               0
#            Conv2d-32          [-1, 128, 32, 32]         147,456
#       BatchNorm2d-33          [-1, 128, 32, 32]             256
#              ReLU-34          [-1, 128, 32, 32]               0
#            Conv2d-35          [-1, 128, 32, 32]         147,456
#       BatchNorm2d-36          [-1, 128, 32, 32]             256
#              ReLU-37          [-1, 128, 32, 32]               0
#        BasicBlock-38          [-1, 128, 32, 32]               0
#            Conv2d-39          [-1, 256, 16, 16]         294,912
#       BatchNorm2d-40          [-1, 256, 16, 16]             512
#              ReLU-41          [-1, 256, 16, 16]               0
#            Conv2d-42          [-1, 256, 16, 16]         589,824
#       BatchNorm2d-43          [-1, 256, 16, 16]             512
#            Conv2d-44          [-1, 256, 16, 16]          32,768
#       BatchNorm2d-45          [-1, 256, 16, 16]             512
#              ReLU-46          [-1, 256, 16, 16]               0
#        BasicBlock-47          [-1, 256, 16, 16]               0
#            Conv2d-48          [-1, 256, 16, 16]         589,824
#       BatchNorm2d-49          [-1, 256, 16, 16]             512
#              ReLU-50          [-1, 256, 16, 16]               0
#            Conv2d-51          [-1, 256, 16, 16]         589,824
#       BatchNorm2d-52          [-1, 256, 16, 16]             512
#              ReLU-53          [-1, 256, 16, 16]               0
#        BasicBlock-54          [-1, 256, 16, 16]               0
#            Conv2d-55            [-1, 512, 8, 8]       1,179,648
#       BatchNorm2d-56            [-1, 512, 8, 8]           1,024
#              ReLU-57            [-1, 512, 8, 8]               0
#            Conv2d-58            [-1, 512, 8, 8]       2,359,296
#       BatchNorm2d-59            [-1, 512, 8, 8]           1,024
#            Conv2d-60            [-1, 512, 8, 8]         131,072
#       BatchNorm2d-61            [-1, 512, 8, 8]           1,024
#              ReLU-62            [-1, 512, 8, 8]               0
#        BasicBlock-63            [-1, 512, 8, 8]               0
#            Conv2d-64            [-1, 512, 8, 8]       2,359,296
#       BatchNorm2d-65            [-1, 512, 8, 8]           1,024
#              ReLU-66            [-1, 512, 8, 8]               0
#            Conv2d-67            [-1, 512, 8, 8]       2,359,296
#       BatchNorm2d-68            [-1, 512, 8, 8]           1,024
#              ReLU-69            [-1, 512, 8, 8]               0
#        BasicBlock-70            [-1, 512, 8, 8]               0
#            Conv2d-71            [-1, 512, 8, 8]         262,656
#              ReLU-72            [-1, 512, 8, 8]               0
#          Upsample-73          [-1, 512, 16, 16]               0
#            Conv2d-74          [-1, 256, 16, 16]          65,792
#              ReLU-75          [-1, 256, 16, 16]               0
#            Conv2d-76          [-1, 512, 16, 16]       3,539,456
#              ReLU-77          [-1, 512, 16, 16]               0
#          Upsample-78          [-1, 512, 32, 32]               0
#            Conv2d-79          [-1, 128, 32, 32]          16,512
#              ReLU-80          [-1, 128, 32, 32]               0
#            Conv2d-81          [-1, 256, 32, 32]       1,474,816
#              ReLU-82          [-1, 256, 32, 32]               0
#          Upsample-83          [-1, 256, 64, 64]               0
#            Conv2d-84           [-1, 64, 64, 64]           4,160
#              ReLU-85           [-1, 64, 64, 64]               0
#            Conv2d-86          [-1, 256, 64, 64]         737,536
#              ReLU-87          [-1, 256, 64, 64]               0
#          Upsample-88        [-1, 256, 128, 128]               0
#            Conv2d-89         [-1, 64, 128, 128]           4,160
#              ReLU-90         [-1, 64, 128, 128]               0
#            Conv2d-91        [-1, 128, 128, 128]         368,768
#              ReLU-92        [-1, 128, 128, 128]               0
#          Upsample-93        [-1, 128, 256, 256]               0
#            Conv2d-94         [-1, 64, 256, 256]         110,656
#              ReLU-95         [-1, 64, 256, 256]               0
#            Conv2d-96         [-1, 32, 256, 256]           2,080
#        ResNetUNet-97         [-1, 32, 256, 256]               0
# ================================================================
# Total params: 17,801,824
# Trainable params: 17,801,824
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.75
# Forward/backward pass size (MB): 492.50
# Params size (MB): 67.91
# Estimated Total Size (MB): 561.16
# ----------------------------------------------------------------


#####

modelzoo_transform = Compose([
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class SSModelZooEncoder(nn.Module):
    def __init__(self, options):
        super(SSModelZooEncoder, self).__init__()
        self.options = options
        self.model = self.get_model()

        self.layers = None

    def get_backbone(self, model):

        encoder = None
        for name, child_model in model.named_children():
            if name == 'backbone':
                encoder = child_model
        return encoder

    def get_classifier(self, model):
        for name, child_model in model.named_children():
            if name == 'classifier':
                child_model[-1] = torch.nn.Identity()
        return model

    def get_model(self, ):
        return

    def predict(self, out):
        return

    def forward(self, x):

        # unnormalization
        # normalization (customizable)
        # frame = (frame - 0.5) / 0.25
        x = (x * 0.25) + 0.5

        # hack force_gray
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # modelzoo normalization
        x = modelzoo_transform(x)

        temp_out = self.layers(x)
        net_output = self.predict(temp_out)
        return net_output, None


####### Deeplab Resnet50


class DeepLabv3_Resnet50BackBone(SSModelZooEncoder):

    # 2048 feature maps

    def __init__(self, options):
        super(DeepLabv3_Resnet50BackBone, self).__init__(options)

        self.layers = self.get_backbone(self.model)

    def get_model(self, ):
        return models.segmentation.deeplabv3_resnet50(pretrained=True).eval()

    def predict(self, out):
        return F.interpolate(out['out'], size=(self.options["w"], self.options["h"]),
                             mode='bilinear', align_corners=False)


class DeepLabv3_Resnet50Classifier(SSModelZooEncoder):
    # 512 feature maps

    def __init__(self, options):
        super(DeepLabv3_Resnet50Classifier, self).__init__(options)

        self.layers = self.get_classifier(self.model)

    def get_model(self, ):
        return models.segmentation.deeplabv3_resnet50(pretrained=True).eval()

    def predict(self, out):
        return out["out"]


####### Deeplab Resnet101


class DeepLabv3_Resnet101BackBone(SSModelZooEncoder):

    def __init__(self, options):
        super(DeepLabv3_Resnet101BackBone, self).__init__(options)

        self.layers = self.get_backbone(self.model)

    def get_model(self, ):
        return models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

    def predict(self, out):
        return F.interpolate(out['out'], size=(self.options["w"], self.options["h"]),
                             mode='bilinear', align_corners=False)


class DeepLabv3_Resnet101Classifier(SSModelZooEncoder):
    # 256 feature maps

    def __init__(self, options):
        super(DeepLabv3_Resnet101Classifier, self).__init__(options)

        self.layers = self.get_classifier(self.model)

    def get_model(self, ):
        return models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

    def predict(self, out):
        return out["out"]


####### Deeplab Mobilenet


class DeepLabv3_MobilenetBackBone(SSModelZooEncoder):

    # 2048 feature maps

    def __init__(self, options):
        super(DeepLabv3_MobilenetBackBone, self).__init__(options)

        self.layers = self.get_backbone(self.model)

    def get_model(self, ):
        return models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True).eval()  # pytorch >= 1.8

    def predict(self, out):
        return F.interpolate(out['out'], size=(self.options["w"], self.options["h"]),
                             mode='bilinear', align_corners=False)


class DeepLabv3_MobilenetClassifier(SSModelZooEncoder):
    # 512 feature maps

    def __init__(self, options):
        super(DeepLabv3_MobilenetClassifier, self).__init__(options)

        self.layers = self.get_classifier(self.model)

    def get_model(self, ):
        return models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True).eval()

    def predict(self, out):
        return out["out"]


#### DPT

default_models = {
    "dpt_large": "dpt_large-ade20k-b12dca68.pt",
    "dpt_hybrid": "dpt_hybrid-ade20k-53898607.pt",
}

model_folder = os.path.join("dpt_intel", "weights")

net_w = net_h = 480
dpt_transform = Compose(
    [
        torchvision.transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ]
)


class DPTHybridEncoderBackbone(nn.Module):
    def __init__(self, options):
        super(DPTHybridEncoderBackbone, self).__init__()
        self.options = options
        self.model = DPTSegmentationModel(
            150,
            path=os.path.join(model_folder, default_models["dpt_hybrid"]),
            backbone="vitb_rn50_384",
        )
        self.model.eval()
        self.model.scratch.output_conv = nn.Identity()

    def forward(self, x):
        # unnormalization
        # normalization (customizable)
        # frame = (frame - 0.5) / 0.25
        x = (x * 0.25) + 0.5

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = dpt_transform(x)
        # x = F.interpolate(x, size=(net_w, net_h), mode='bilinear', align_corners=False)

        out = self.model(x)

        return F.interpolate(out, size=(self.options["w"], self.options["h"]),
                             mode='bilinear', align_corners=False), None


class DPTHybridEncoderClassifier(nn.Module):
    def __init__(self, options):
        super(DPTHybridEncoderClassifier, self).__init__()
        self.options = options
        self.model = DPTSegmentationModel(
            150,
            path=os.path.join(model_folder, default_models["dpt_hybrid"]),
            backbone="vitb_rn50_384",
        )
        self.model.eval()
        self.model.scratch.output_conv[3], self.model.scratch.output_conv[4], self.model.scratch.output_conv[5] = \
            nn.Identity(), nn.Identity(), nn.Identity()

    def forward(self, x):
        # unnormalization
        # normalization (customizable)
        # frame = (frame - 0.5) / 0.25
        x = (x * 0.25) + 0.5

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = dpt_transform(x)

        out = self.model(x)

        return F.interpolate(out, size=(self.options["w"], self.options["h"]),
                             mode='bilinear', align_corners=False), None
