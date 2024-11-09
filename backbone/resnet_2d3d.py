## modified from https://github.com/kenshohara/3D-ResNets-PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from scipy.signal import butter, filtfilt
from torch.nn import init

__all__ = [
    'resnet9_2d3d_full', 'ResNet2d3d_full', 'resnet18_2d3d_full', 'resnet34_2d3d_full', 'resnet50_2d3d_full', 'resnet101_2d3d_full',
    'resnet152_2d3d_full', 'resnet200_2d3d_full', 'resnet18_2d3d_two_stream'
]

def conv3x3x3(in_planes, out_planes, stride=1, bias=False):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias)

def conv1x3x3(in_planes, out_planes, stride=1, bias=False):
    # 1x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,3,3),
        stride=(1,stride,stride),
        padding=(0,1,1),
        bias=bias)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, track_running_stats=True, use_final_relu=True):
        super(BasicBlock3d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = conv3x3x3(inplanes, planes, stride, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, bias=bias)
        self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu: out = self.relu(out)

        return out


class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, track_running_stats=True, use_final_relu=True):
        super(BasicBlock2d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = conv1x3x3(inplanes, planes, stride, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes, bias=bias)
        self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu: out = self.relu(out)

        return out


class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, track_running_stats=True, use_final_relu=True):
        super(Bottleneck3d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm3d(planes * 4, track_running_stats=track_running_stats)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu: out = self.relu(out)

        return out


class Bottleneck2d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, track_running_stats=True, use_final_relu=True):
        super(Bottleneck2d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1,3,3), stride=(1,stride,stride), padding=(0,1,1), bias=bias)
        self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm3d(planes * 4, track_running_stats=track_running_stats)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu: out = self.relu(out)

        return out


class ResNet2d3d_full(nn.Module):
    def __init__(self, block, layers, track_running_stats=True):
        super(ResNet2d3d_full, self).__init__()
        self.inplanes = 64
        self.track_running_stats = track_running_stats
        bias = False
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,3,3), stride=(1, 2, 2), padding=(0, 3, 3), bias=bias)
        self.bn1 = nn.BatchNorm3d(64, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        if not isinstance(block, list):
            block = [block] * 4

        self.layer1 = self._make_layer(block[0], 64, layers[0])
        self.layer2 = self._make_layer(block[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block[3], 256, layers[3], stride=2, is_final=True)
        # modify layer4 from exp=512 to exp=256
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, is_final=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # customized_stride to deal with 2d or 3d residual blocks
            if (block == Bottleneck2d) or (block == BasicBlock2d):
                customized_stride = (1, stride, stride)
            else:
                customized_stride = stride

            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=customized_stride, bias=False), 
                nn.BatchNorm3d(planes * block.expansion, track_running_stats=self.track_running_stats)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, track_running_stats=self.track_running_stats))
        self.inplanes = planes * block.expansion
        if is_final: # if is final block, no ReLU in the final output
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes, track_running_stats=self.track_running_stats))
            layers.append(block(self.inplanes, planes, track_running_stats=self.track_running_stats, use_final_relu=False))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, track_running_stats=self.track_running_stats))
                
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x)

        print("x shape:", x.shape)
        # Pool away index 2 (temporal dimension)
        x = F.adaptive_avg_pool3d(x, (1, x.size(3), x.size(4)))
        #x = x.squeeze(2)  # Remove the temporal dimension
        #print("x shape after pooling:", x.shape)
        return x

class ResNet2d3d_half(nn.Module):
    def __init__(self, block, layers, track_running_stats=True):
        super(ResNet2d3d_half, self).__init__()
        self.inplanes = 32  # Half of the original 64
        self.track_running_stats = track_running_stats
        bias = False
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(1,3,3), stride=(1, 2, 2), padding=(0, 3, 3), bias=bias)
        self.bn1 = nn.BatchNorm3d(32, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        if not isinstance(block, list):
            block = [block] * 4

        self.layer1 = self._make_layer(block[0], 32, layers[0])
        self.layer2 = self._make_layer(block[1], 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block[2], 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block[3], 128, layers[3], stride=2, is_final=True)
        # modify layer4 from exp=256 to exp=128
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, is_final=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # customized_stride to deal with 2d or 3d residual blocks
            if (block == Bottleneck2d) or (block == BasicBlock2d):
                customized_stride = (1, stride, stride)
            else:
                customized_stride = stride

            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=customized_stride, bias=False), 
                nn.BatchNorm3d(planes * block.expansion, track_running_stats=self.track_running_stats)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, track_running_stats=self.track_running_stats))
        self.inplanes = planes * block.expansion
        if is_final: # if is final block, no ReLU in the final output
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes, track_running_stats=self.track_running_stats))
            layers.append(block(self.inplanes, planes, track_running_stats=self.track_running_stats, use_final_relu=False))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, track_running_stats=self.track_running_stats))
                
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x)

        return x

class ResNet2d3d_two_stream(nn.Module):
    def __init__(self, block, layers, track_running_stats=True, num_heads=8):
        super(ResNet2d3d_two_stream, self).__init__()
        self.stream1 = ResNet2d3d_half(block, layers, track_running_stats)
        self.stream2 = ResNet2d3d_half(block, layers, track_running_stats)

        self.num_heads = num_heads
        self.head_dim = 128 // num_heads
        
        self.query_layer = nn.Linear(128, 128)
        self.key_layer = nn.Linear(128, 128)
        self.value_layer = nn.Linear(128, 128)

        self.norm = nn.LayerNorm(128)
        
        self.output_layer = nn.Linear(128, 128)

        init.xavier_uniform_(self.query_layer.weight)
        if self.query_layer.bias is not None:
            init.constant_(self.query_layer.bias, 0)

        init.xavier_uniform_(self.key_layer.weight)
        if self.key_layer.bias is not None:
            init.constant_(self.key_layer.bias, 0)

        init.xavier_uniform_(self.value_layer.weight)
        if self.value_layer.bias is not None:
            init.constant_(self.value_layer.bias, 0)

        self.norm = nn.LayerNorm(128)
        
        self.output_layer = nn.Linear(128, 128)
        init.xavier_uniform_(self.output_layer.weight)  # Added Xavier initialization
        if self.output_layer.bias is not None:
            init.constant_(self.output_layer.bias, 0)

    def forward(self, x1):

        #SL = x1.size(2)
        #x1_static = x1[:, :, 0:1, :, :].repeat(1, 1, SL, 1, 1)

        out1 = self.stream1(x1)
        out2 = self.stream2(x1)

        #out1 = F.adaptive_avg_pool3d(out1, (1, 3, 3))
        #out2 = F.adaptive_avg_pool3d(out2, (1, 3, 3))

        #print("original output", out1.shape)

        # Pool only the temporal dimension for out1
        out1 = F.adaptive_avg_pool3d(out1, (1, 3, 3))
        out2 = F.adaptive_avg_pool3d(out2, (1, 3, 3))
        #print("after temporal pooling", out1.shape)
    
        # Reshape out1 to (batch_size, channels, 9)
        out1 = out1.view(out1.size(0), out1.size(1), -1)
        out2 = out2.view(out2.size(0), out2.size(1), -1)
        #print("after first reshape", out1.shape)

        # Transpose to (batch_size, 9, channels)
        out1 = out1.transpose(1, 2)

        out2 = out2.transpose(1, 2)

        # Multi-head attention
        query = self.query_layer(out1).view(out1.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_layer(out2).view(out2.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_layer(out2).view(out2.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        # Concatenate heads and pass through output layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(out1.size(0), -1, 128)
        attn_output = self.output_layer(attn_output)

        # Add attention output to out1 (residual connection)
        out1_with_attn = out1 + attn_output

        # Apply layer normalization after residual connection
        out1_with_attn = self.norm(out1_with_attn)

        # Reshape out1_with_attn back to (batch_size, channels, 1, 4, 4)
        out1_with_attn = out1_with_attn.transpose(1, 2).view(out1_with_attn.size(0), -1, 1, 3, 3)
        out2 = out2.transpose(1,2).view(out1_with_attn.size(0), -1, 1, 3, 3)
        #print("out_with_attn:", out1_with_attn.shape)
        #print("out2:", out2.shape)

        # Ensure out2 has the same shape as out1_with_attn
        #out2 = out2.view(out2.size(0), out2.size(1), 1, 3, 3)

        # Concatenate out1_with_attn and out2 along the channel dimension
        out = torch.cat((out1_with_attn, out2), dim=1)
        #out = torch.cat((out1, out2), dim=1)
    
        return out

## full resnet
def resnet18_2d3d_full(**kwargs):
    '''Constructs a ResNet-18 model. '''
    model = ResNet2d3d_full([BasicBlock2d, BasicBlock2d, BasicBlock3d, BasicBlock3d], 
                   [2, 2, 2, 2], **kwargs)
    return model

def resnet9_2d3d_full(**kwargs):
    '''Constructs a ResNet-9 model.'''
    model = ResNet2d3d_full([BasicBlock2d, BasicBlock2d, BasicBlock3d, BasicBlock3d], 
                   [1, 1, 1, 1], **kwargs)
    return model

def resnet18_2d3d_two_stream(**kwargs):
    '''Constructs a two-stream ResNet-18 model.'''
    model = ResNet2d3d_two_stream([BasicBlock2d, BasicBlock2d, BasicBlock3d, BasicBlock3d], 
                                  [1, 1, 1, 1], **kwargs)
    return model

def resnet34_2d3d_full(**kwargs):
    '''Constructs a ResNet-34 model. '''
    model = ResNet2d3d_full([BasicBlock2d, BasicBlock2d, BasicBlock3d, BasicBlock3d], 
                   [3, 4, 6, 3], **kwargs)
    return model

def resnet50_2d3d_full(**kwargs):
    '''Constructs a ResNet-50 model. '''
    model = ResNet2d3d_full([Bottleneck2d, Bottleneck2d, Bottleneck3d, Bottleneck3d], 
                   [3, 4, 6, 3], **kwargs)
    return model

def resnet101_2d3d_full(**kwargs):
    '''Constructs a ResNet-101 model. '''
    model = ResNet2d3d_full([Bottleneck2d, Bottleneck2d, Bottleneck3d, Bottleneck3d], 
                   [3, 4, 23, 3], **kwargs)
    return model

def resnet152_2d3d_full(**kwargs):
    '''Constructs a ResNet-101 model. '''
    model = ResNet2d3d_full([Bottleneck2d, Bottleneck2d, Bottleneck3d, Bottleneck3d], 
                   [3, 8, 36, 3], **kwargs)
    return model

def resnet200_2d3d_full(**kwargs):
    '''Constructs a ResNet-101 model. '''
    model = ResNet2d3d_full([Bottleneck2d, Bottleneck2d, Bottleneck3d, Bottleneck3d], 
                   [3, 24, 36, 3], **kwargs)
    return model

def neq_load_customized(model, pretrained_dict):
    ''' load pre-trained model in a not-equal way,
    when new model has been partially modified '''
    model_dict = model.state_dict()
    tmp = {}
    print('\n=======Check Weights Loading======')
    print('Weights not used from pretrained file:')
    for k, v in pretrained_dict.items():
        if k in model_dict:
            tmp[k] = v
        else:
            print(k)
    print('---------------------------')
    print('Weights not loaded into new model:')
    for k, v in model_dict.items():
        if k not in pretrained_dict:
            print(k)
    print('===================================\n')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':
    mymodel = resnet18_2d3d_full()
    mydata = torch.FloatTensor(4, 3, 16, 128, 128)
    nn.init.normal_(mydata)
    import ipdb; ipdb.set_trace()
    mymodel(mydata)
