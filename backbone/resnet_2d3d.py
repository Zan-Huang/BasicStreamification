## modified from https://github.com/kenshohara/3D-ResNets-PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from torch.autograd import Variable
import math
import numpy as np
from scipy.signal import butter, filtfilt
from torch.nn import init
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

__all__ = [
    'resnet9_2d3d_full', 'ResNet2d3d_full', 'resnet18_2d3d_full', 'resnet34_2d3d_full', 'resnet50_2d3d_full', 'resnet101_2d3d_full',
    'resnet152_2d3d_full', 'resnet200_2d3d_full', 'resnet18_2d3d_two_stream'
]

"""def visualize_features_with_torch(out1, out2, filename='features_visualization.png'):
    # Assuming out1 and out2 are of shape (batch_size, channels, 1, height, width)
    batch_size, channels, _, height, width = out1.shape

    # Select the first sample from the batch for visualization
    out1_sample = out1[0].squeeze(0)  # Remove the singleton dimension
    out2_sample = out2[0].squeeze(0)

    # Create a grid of images for out1 and out2
    grid_out1 = make_grid(out1_sample, nrow=8, normalize=True, scale_each=True)  # Adjust nrow for larger images
    grid_out2 = make_grid(out2_sample, nrow=8, normalize=True, scale_each=True)

    # Convert the grids to numpy arrays
    grid_out1_np = grid_out1.cpu().numpy().transpose((1, 2, 0))
    grid_out2_np = grid_out2.cpu().numpy().transpose((1, 2, 0))

    # Create a figure to display the grids side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))  # Increase figsize for larger display
    axes[0].imshow(grid_out1_np)
    axes[0].set_title('out1 Features')
    axes[0].axis('off')

    axes[1].imshow(grid_out2_np)
    axes[1].set_title('out2 Features')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(filename)  # Save the figure as an image file
    plt.close(fig)"""

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
    def __init__(self, block, layers, track_running_stats=True, num_heads=4):
        super(ResNet2d3d_two_stream, self).__init__()
        self.stream1 = ResNet2d3d_half(block, layers, track_running_stats)
        self.stream2 = ResNet2d3d_half(block, layers, track_running_stats)

        self.num_heads = num_heads
        self.head_dim = 256 // num_heads

        # Add layer norms
        self.pre_attn_norm1 = nn.LayerNorm(256)
        self.pre_attn_norm2 = nn.LayerNorm(256)
        
        self.query_layer = nn.Linear(256, 256)
        self.key_layer = nn.Linear(256, 256)
        self.value_layer = nn.Linear(256, 256)

        self.reverse_query_layer = nn.Linear(256, 256)
        self.reverse_key_layer = nn.Linear(256, 256)
        self.reverse_value_layer = nn.Linear(256, 256)

        self.ffn1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.Dropout(0.1)
        )

        self.ffn2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.Dropout(0.1)
        )

        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)

        self.ffn_norm1 = nn.LayerNorm(256)
        self.ffn_norm2 = nn.LayerNorm(256)

        self.attn_dropout = nn.Dropout(0.1)
        
        self.output_layer1 = nn.Linear(256, 256)
        self.output_layer2 = nn.Linear(256, 256)
        for ffn in [self.ffn1, self.ffn2]:
            for layer in ffn:
                if isinstance(layer, nn.Linear):
                    init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        init.constant_(layer.bias, 0)

        init.xavier_uniform_(self.query_layer.weight)
        if self.query_layer.bias is not None:
            init.constant_(self.query_layer.bias, 0)

        init.xavier_uniform_(self.key_layer.weight)
        if self.key_layer.bias is not None:
            init.constant_(self.key_layer.bias, 0)

        init.xavier_uniform_(self.value_layer.weight)
        if self.value_layer.bias is not None:
            init.constant_(self.value_layer.bias, 0)

        init.xavier_uniform_(self.reverse_query_layer.weight)
        if self.reverse_query_layer.bias is not None:
            init.constant_(self.reverse_query_layer.bias, 0)

        init.xavier_uniform_(self.reverse_key_layer.weight)
        if self.reverse_key_layer.bias is not None:
            init.constant_(self.reverse_key_layer.bias, 0)

        init.xavier_uniform_(self.reverse_value_layer.weight)
        if self.reverse_value_layer.bias is not None:
            init.constant_(self.reverse_value_layer.bias, 0)

        self.norm = nn.LayerNorm(256)
        
        init.xavier_uniform_(self.output_layer1.weight)
        if self.output_layer1.bias is not None:
            init.constant_(self.output_layer1.bias, 0)

        init.xavier_uniform_(self.output_layer2.weight)
        if self.output_layer2.bias is not None:
            init.constant_(self.output_layer2.bias, 0)

        self.gaussian_blur = GaussianBlur(kernel_size=11, sigma=21.0)

    def forward(self, x1):
        #(batch, channels, temporal_dimension, height, width) )
        x1_gray = x1.mean(dim=1, keepdim=True)
        x1_gray = x1_gray.expand(-1, 3, -1, -1, -1) 
        zero_frame = torch.zeros_like(x1_gray[:, :, 0:1, :, :])

        x_temporal_derivative = x1_gray[:, :, 1:, :, :] - x1_gray[:, :, :-1, :, :]
        x_temporal_derivative = torch.cat((x_temporal_derivative, zero_frame), dim=2)

        blurred_temporal_derivative = torch.stack([
            self.gaussian_blur(frame) for frame in x_temporal_derivative.unbind(dim=2)
        ], dim=2)

        b, c, t, h, w = blurred_temporal_derivative.size()
        reshaped_input = blurred_temporal_derivative.view(b * t, c, h, w)

        downsampled_reshaped = F.interpolate(
        reshaped_input, scale_factor=0.5, mode='bilinear', align_corners=False)
        downsampled_temporal_derivative = downsampled_reshaped.view(b, c, t, downsampled_reshaped.size(2), downsampled_reshaped.size(3))

        upsampled_frames = []
        for i in range(t):
            frame = downsampled_temporal_derivative[:, :, i, :, :]
            upsampled_frame = F.interpolate(
                frame, 
                size=(h, w),  # Specify the original height and width
                mode='bilinear',  # Use bilinear interpolation for upsampling
                align_corners=False
            )
            upsampled_frames.append(upsampled_frame)
        
        upsampled_temporal_derivative = torch.stack(upsampled_frames, dim=2)

        static_image = x1[:, :, 0:1, :, :].clone()
        static_image_repeated = static_image.expand(-1, -1, x1.size(2), -1, -1)

        out1 = self.stream1(static_image_repeated)
        out2 = self.stream2(upsampled_temporal_derivative)

        #out1 = F.adaptive_avg_pool3d(out1, (1, 3, 3))
        #out2 = F.adaptive_avg_pool3d(out2, (1, 3, 3))

        #print("original output", out1.shape)

        # Pool only the temporal dimension for out1
        out1 = F.adaptive_avg_pool3d(out1, (1, 3, 3))
        out2 = F.adaptive_avg_pool3d(out2, (1, 3, 3))
        #print("after temporal pooling", out1.shape)

        #visualize_features_with_torch(out1, out2, filename='features_visualization.png')
    
        # Reshape out1 to (batch_size, channels, 9)
        out1 = out1.view(out1.size(0), out1.size(1), -1)
        out2 = out2.view(out2.size(0), out2.size(1), -1)
        #print("after first reshape", out1.shape)

        # Transpose to (batch_size, 9, channels)
        out1 = out1.transpose(1, 2)
        out2 = out2.transpose(1, 2)

        out1 = self.pre_attn_norm1(out1)
        out2 = self.pre_attn_norm2(out2)

        # Multi-head attention
        query = self.query_layer(out1).view(out1.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_layer(out2).view(out2.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_layer(out2).view(out2.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(F.softmax(attn_weights, dim=-1))
        attn_output = torch.matmul(attn_weights, value)

        # Concatenate heads and pass through output layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(out1.size(0), -1, 256)
        attn_output = self.output_layer1(attn_output)

        # Add attention output to out1 (residual connection)
        out1_with_attn = out1 + attn_output


        out1_with_attn = self.norm1(out1_with_attn)  # First normalization
        out1_ffn = self.ffn1(out1_with_attn)
        out1_with_attn = out1_with_attn + out1_ffn  # Add FFN output
        out1_with_attn = self.ffn_norm1(out1_with_attn)

        # Reverse attention
        reverse_query = self.reverse_query_layer(out2).view(out2.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        reverse_key = self.reverse_key_layer(out1).view(out1.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        reverse_value = self.reverse_value_layer(out1).view(out1.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)

        reverse_scores = torch.matmul(reverse_query, reverse_key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        reverse_attn_weights = F.softmax(reverse_scores, dim=-1)
        reverse_attn_weights = self.attn_dropout(F.softmax(reverse_attn_weights, dim=-1))
        reverse_attn_output = torch.matmul(reverse_attn_weights, reverse_value)

        reverse_attn_output = reverse_attn_output.transpose(1, 2).contiguous().view(out1.size(0), -1, 256)
        reverse_attn_output = self.output_layer2(reverse_attn_output)

        out2_with_reverse_attn = out2 + reverse_attn_output


        #out2 = out2.transpose(1,2).view(out1_with_attn.size(0), -1, 1, 3, 3)
        #print("out_with_attn:", out1_with_attn.shape)
        #print("out2:", out2.shape)
        out2_with_reverse_attn = self.norm2(out2_with_reverse_attn)  # First normalization
        out2_ffn = self.ffn2(out2_with_reverse_attn)
        out2_with_reverse_attn = out2_with_reverse_attn + out2_ffn  # Add FFN output
        out2_with_reverse_attn = self.ffn_norm2(out2_with_reverse_attn) 

        # Ensure out2 has the same shape as out1_with_attn
        #out2 = out2.view(out2.size(0), out2.size(1), 1, 3, 3)
        out1_with_attn = out1_with_attn.transpose(1, 2).view(out1_with_attn.size(0), -1, 1, 3, 3)
        out2_with_reverse_attn = out2_with_reverse_attn.transpose(1, 2).view(out2_with_reverse_attn.size(0), -1, 1, 3, 3)

        # Concatenate out1_with_attn and out2 along the channel dimension
        out = torch.cat((out1_with_attn, out2_with_reverse_attn), dim=1)
        #out = torch.cat((out1, out2), dim=1)
    
        return out
    
    """def visualize_conv1_filters(self):
        # Extract conv1 filters from both streams
        conv1_filters_stream1 = self.stream1.conv1.weight.data.clone()
        conv1_filters_stream2 = self.stream2.conv1.weight.data.clone()

        # Normalize the filters for better visualization
        conv1_filters_stream1 = (conv1_filters_stream1 - conv1_filters_stream1.min()) / (conv1_filters_stream1.max() - conv1_filters_stream1.min())
        conv1_filters_stream2 = (conv1_filters_stream2 - conv1_filters_stream2.min()) / (conv1_filters_stream2.max() - conv1_filters_stream2.min())

        # Reshape the filters to [B, C, H, W] by selecting the middle slice of the temporal dimension
        conv1_filters_stream1 = conv1_filters_stream1[:, :, 0, :, :]
        conv1_filters_stream2 = conv1_filters_stream2[:, :, 0, :, :]

        # Convert to uint8
        conv1_filters_stream1 = (conv1_filters_stream1 * 255).byte()
        conv1_filters_stream2 = (conv1_filters_stream2 * 255).byte()

        # Ensure the tensor has four dimensions (B, C, H, W)
        if conv1_filters_stream1.dim() == 3:
            conv1_filters_stream1 = conv1_filters_stream1.unsqueeze(1)  # Add channel dimension
        if conv1_filters_stream2.dim() == 3:
            conv1_filters_stream2 = conv1_filters_stream2.unsqueeze(1)  # Add channel dimension

        # Create a grid of images for each stream's filters
        grid_stream1 = make_grid(conv1_filters_stream1, nrow=8, normalize=False)
        grid_stream2 = make_grid(conv1_filters_stream2, nrow=8, normalize=False)

        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir='log_dir/conv1_filters')

        # Log the grids to TensorBoard
        writer.add_image('Stream1/conv1_filters', grid_stream1, 0)
        writer.add_image('Stream2/conv1_filters', grid_stream2, 0)

        # Close the writer
        writer.close()"""

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
