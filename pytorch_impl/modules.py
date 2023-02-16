from turtle import forward
import torch
from torch import nn

from torch import Tensor
from typing import Optional, List, Tuple, Union, Any


class DiceLoss(nn.Module):
    def __init__(self, eps: float=1e-6, *args: Any, **kwargs: Any) -> None:
        super(DiceLoss, self).__init__(*args, **kwargs)
        self.eps = eps
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        inputs, targets = inputs.view(inputs.size(0), -1), targets.view(targets.size(0), -1)
        intersection = torch.sum(inputs * targets, -1)
        denom = torch.sum(inputs + targets, -1)
        dice_score = 2. * intersection / (denom + self.eps)
        return 1. - dice_score

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, pooling: Optional[int]=1 , *args: Any, **kwargs: Any) -> None:
        super(ResidualBlock, self).__init__(*args, **kwargs)
        self.conv_1d_0 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn_0 = nn.BatchNorm1d(channels)
        self.conv_1d_1 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn_1 = nn.BatchNorm1d(channels)
        self.conv_1d_pool = nn.Conv1d(channels, channels, pooling, stride=pooling)
    
    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.conv_1d_0(input)
        x = self.bn_0(x)
        x = nn.functional.relu(x)
        x = self.conv_1d_1(x)
        x = self.bn_1(x)
        x = input + x
        x_residual = nn.functional.relu(x)
        x = self.conv_1d_pool(x_residual)
        return x, x_residual


class ResidualBlockNoAux(nn.Module):
    def __init__(self, channels: int, kernel_size: int, pooling: Optional[int]=1 , *args: Any, **kwargs: Any) -> None:
        super(ResidualBlockNoAux, self).__init__(*args, **kwargs)
        self.conv_1d_0 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn_0 = nn.BatchNorm1d(channels)
        self.conv_1d_1 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn_1 = nn.BatchNorm1d(channels)
        self.conv_1d_pool = nn.Conv1d(channels, channels, pooling, stride=pooling)
    
    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.conv_1d_0(input)
        x = self.bn_0(x)
        x = nn.functional.relu(x)
        x = self.conv_1d_1(x)
        x = self.bn_1(x)
        x = input + x
        x_residual = nn.functional.relu(x)
        x = self.conv_1d_pool(x_residual)
        return x


class ResidualBlockUpsampling(nn.Module):
    def __init__(self, channels: int, kernel_size: int, pooling: Optional[int]=1 , *args, **kwargs) -> None:
        super(ResidualBlockUpsampling, self).__init__(*args, **kwargs)
        self.conv_1d_0 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn_0 = nn.BatchNorm1d(channels)
        self.conv_1d_1 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn_1 = nn.BatchNorm1d(channels)
        self.upsampling = nn.Upsample(scale_factor=pooling)
    
    def forward(self, inputs: Tensor, residual: Tensor) -> Tensor:
        x = self.conv_1d_0(inputs)
        x = self.bn_0(x)
        x = nn.functional.relu(x)
        x = self.conv_1d_1(x)
        x = self.bn_1(x)
        x = inputs + x + residual
        x = nn.functional.relu(x)
        x = self.upsampling(x)
        return x


class Generator(nn.Module):
    def __init__(self, 
                convs: Optional[List[int]]=[4,4], 
                channels: Optional[List[int]]=[32,64], 
                pooling: Optional[List[int]]=[2,2], 
                kernel_size: Optional[List[int]]=[5,5], 
                middle_blocks_channels: Optional[List[int]]=[64,128,256],
                middle_blocks_kernel_size: Optional[List[int]]=[5,5,5],
                dropout: Optional[float]=0.25,
                *args: Any, 
                **kwargs: Any) -> None:
        super(Generator, self).__init__(*args, **kwargs)
        self.dropout = nn.Dropout1d(dropout)
        self.output_conv_1d = nn.Conv1d(channels[0], 1, 1, padding='same')
        self.make_encoder_layers(convs, channels, pooling, kernel_size)
        self.make_middle_layers(middle_blocks_channels, middle_blocks_kernel_size)
        self.make_decoder_layers(convs, channels, pooling, kernel_size)
        

    def make_encoder_layers(self,
                            convs: Optional[List[int]]=[4,4], 
                            channels: Optional[List[int]]=[32,64], 
                            pooling: Optional[List[int]]=[2,2], 
                            kernel_size: Optional[List[int]]=[5,5]
                            ) -> None:
        self.downsampling_blocks = nn.ModuleList()
        ch_in = 1
        for nc, ch, ks, pl in zip(convs, channels, kernel_size, pooling):
            block = nn.ModuleList()
            block.append(nn.Conv1d(ch_in, ch, ks, padding='same'))
            for _ in range(nc):
                block.append(ResidualBlock(ch, ks, pl))
            self.downsampling_blocks.append(block)
            ch_in = ch
    
        self.final_enc_conv_1d = nn.Conv1d(ch_in, ch_in, kernel_size[-1], padding='same')
        self.final_enc_bn = nn.BatchNorm1d(ch_in)
        self.ch_out_final = ch_in

    def make_middle_layers(self,
                            middle_blocks_channels: Optional[List[int]]=[64,128,256],
                            middle_blocks_kernel_size: Optional[List[int]]=[5,5,5],
                            ) -> None:
        ch_in = self.ch_out_final

        self.middle_blocks_A = nn.ModuleList()
        for ch, ks in zip(middle_blocks_channels, middle_blocks_kernel_size):
            self.middle_blocks_A.append(nn.ModuleList([
                nn.Conv1d(ch_in, ch, ks, padding='same'),
                nn.BatchNorm1d(ch)
            ]))
            ch_in = ch

        self.middle_blocks_B = nn.ModuleList()
        rev_middle_blocks_channels = middle_blocks_channels[::-1]
        rev_middle_blocks_kernel_size = middle_blocks_kernel_size[::-1]
        for i, (ch, ks) in enumerate(zip(rev_middle_blocks_channels[:-1], rev_middle_blocks_kernel_size[:-1])):
            self.middle_blocks_B.append(nn.ModuleList([
                nn.Conv1d(ch_in, ch, ks, padding='same'),
                nn.BatchNorm1d(ch)
            ]))
            self.middle_blocks_B.append(nn.ModuleList([
                    nn.Conv1d(ch, rev_middle_blocks_channels[i+1], rev_middle_blocks_kernel_size[i+1], padding='same'),
                    nn.BatchNorm1d(rev_middle_blocks_channels[i+1])
                ]))
            ch_in = rev_middle_blocks_channels[i+1]
        self.middle_blocks_B.append(nn.ModuleList([
                nn.Conv1d(ch_in, rev_middle_blocks_channels[-1], rev_middle_blocks_kernel_size[-1], padding='same'),
                nn.BatchNorm1d(rev_middle_blocks_channels[-1])
            ]))
        
    def make_decoder_layers(self,
                            convs: Optional[List[int]]=[4,4], 
                            channels: Optional[List[int]]=[32,64], 
                            pooling: Optional[List[int]]=[2,2], 
                            kernel_size: Optional[List[int]]=[5,5],
                            ) -> None:
        convs, channels, kernel_size, pooling = convs[::-1], channels[::-1], kernel_size[::-1], pooling[::-1]
        self.upsampling_blocks = nn.ModuleList()
        self.upsampling_blocks.append(nn.ModuleList([ResidualBlockUpsampling(channels[0], kernel_size[0], pooling[0])]))
        
        for i, (nc, ch, ks, pl) in enumerate(zip(convs, channels, kernel_size, pooling)):
            block = nn.ModuleList()
            for j in range(nc-1):
                block.append(ResidualBlockUpsampling(ch, ks, pl))
            if i < len(pooling)-1:
                block.append(ResidualBlockUpsampling(ch, ks, pooling[i+1]))
                self.upsampling_blocks.append(block)
                self.upsampling_blocks.append(nn.Conv1d(ch, channels[i+1], kernel_size[i+1], padding='same'))
            else:
                block.append(ResidualBlockUpsampling(ch, ks, 1))
                self.upsampling_blocks.append(block)

        self.final_block = ResidualBlock(channels[-1], kernel_size[-1], 1)
    
    def forward(self, inputs: Tensor, *args: Any, **kwargs: Any) -> Tuple[Tensor, List[Tensor]]:
        residual_stack = []
        classifier_residual_list = []
        x = inputs
        for block in self.downsampling_blocks: # blocks are a list of residual blocks with the same ch_out
            x = block[0](x)
            for layer in block[1:]: # layers are residual blocks
                x, residual = layer(x)
                residual_stack.append(residual)
                classifier_residual_list.append(residual)
        residual_stack.append(x)

        x = self.final_enc_conv_1d(x)
        x = self.final_enc_bn(x)
        x = nn.functional.relu(x)

        for block in self.middle_blocks_A[:-1]: # each block has a single conv and bn
            for layer in block:
                x = layer(x)
            x = nn.functional.relu(x) # conv-bn-relu
            residual_stack.append(x)

        for layer in self.middle_blocks_A[-1]:
            x = layer(x)
        x = nn.functional.relu(x) # conv-bn-relu
        x = self.dropout(x)

        for i, block in enumerate(self.middle_blocks_B[:-1]):
            for layer in block:
                x = layer(x)
            x = nn.functional.relu(x) # conv-bn-relu
            if i % 2 == 1:
                x = x + residual_stack.pop()

        for layer in self.middle_blocks_B[-1]:
            x = layer(x)
        x = nn.functional.relu(x) # conv-bn-relu

        for block in self.upsampling_blocks:
            if isinstance(block, nn.ModuleList):
                for layer in block:
                    residual = residual_stack.pop()
                    x = layer(x, residual)
            else:
                x = block(x)
        _, x = self.final_block(x)
        
        x = self.output_conv_1d(x)
        output = torch.sigmoid(x)
        return output, classifier_residual_list


class Discriminator(nn.Module):
    def __init__(self, 
                conv_filters: Optional[List[int]]=[64,64,64,64,64,128,128,128,128,256],
                kernel_sizes: Optional[List[int]]=[5,5,5,5,5,5,5,5,5,5],
                strides: Optional[List[int]]=[2,2,2,2,2,2,2,2,2,2],
                linear_units: Optional[List[int]]=[128,128],
                lrelu_slope: Optional[float]=0.2,
                dropout: Optional[float]=0.25,
                *args: Any, 
                **kwargs: Any) -> None:
        super(Discriminator, self).__init__(*args, **kwargs)
        layers = self.create_model(conv_filters, kernel_sizes, strides, linear_units, lrelu_slope, dropout)
        self.model = nn.Sequential(*layers)

    def forward(self, inputs: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        x = self.model(inputs)
        return x

    def create_model(self,
                conv_filters: Optional[List[int]]=[64,64,64,64,64,128,128,128,128,256],
                kernel_sizes: Optional[List[int]]=[5,5,5,5,5,5,5,5,5,5],
                strides: Optional[List[int]]=[2,2,2,2,2,2,2,2,2,2],
                linear_units: Optional[List[int]]=[128,128],
                lrelu_slope: Optional[float]=0.2,
                dropout: Optional[float]=0.25) -> List[nn.Module]:
        layers = []
        ch_in = 1
        for ch, ks, st in zip(conv_filters[:-1], kernel_sizes[:-1], strides[:-1]):
            layers.append(nn.Conv1d(ch_in, ch, kernel_size=ks, stride=st, padding=ks//2))
            layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.LeakyReLU(lrelu_slope))
            layers.append(nn.Dropout1d(dropout))
            ch_in = ch

        layers.append(nn.Conv1d(ch_in, conv_filters[-1], kernel_size=kernel_sizes[-1], stride=strides[-1], padding=kernel_sizes[-1]//2))
        layers.append(nn.BatchNorm1d(conv_filters[-1]))
        layers.append(nn.LeakyReLU(lrelu_slope))
        ch_in = conv_filters[-1]
        
        layers.append(nn.AdaptiveMaxPool1d(1))
        layers.append(nn.Dropout1d(dropout))
        layers.append(nn.Flatten())

        for ch in linear_units:
            layers.append(nn.Linear(ch_in, ch))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout1d(dropout))
            ch_in = ch
        layers.append(nn.Linear(ch_in, 1))
        return layers

class Classifier(nn.Module):
    def __init__(self, 
                convs: Optional[List[int]]=[3,4], 
                channels: Optional[List[int]]=[32,64], 
                pooling: Optional[List[int]]=[2,2], 
                kernel_size: Optional[List[int]]=[5,5], 
                extra_convs: Optional[List[int]]=[3], 
                extra_channels: Optional[List[int]]=[64], 
                extra_pooling: Optional[List[int]]=[2], 
                extra_kernel_size: Optional[List[int]]=[5], 
                middle_blocks_channels: Optional[List[int]]=[128, 256],
                middle_blocks_kernel_size: Optional[List[int]]=[5,5],
                lrelu_slope: Optional[float]=0.2,
                dropout: Optional[float]=0.25,
                linear_units: Optional[List[int]]=[256,256],
                linear_dropout: Optional[float]=0.4,
                *args: Any, 
                **kwargs: Any) -> None:
        super(Classifier, self).__init__(*args, **kwargs)
        self.make_encoder_layers(convs, channels, pooling, kernel_size)
        self.make_middle_layers(extra_convs, extra_channels, extra_pooling, extra_kernel_size, middle_blocks_channels, middle_blocks_kernel_size, lrelu_slope, dropout)
        self.make_head(middle_blocks_channels[-1], linear_units, linear_dropout)
    
    def make_head(self,
                ch_in: int,
                linear_units: Optional[List[int]]=[256,256],
                linear_dropout: Optional[float]=0.4) -> None:
        layers = []
        layers.append(nn.AdaptiveMaxPool1d(1))
        layers.append(nn.Flatten())
        for ch in linear_units:
            layers.append(nn.Linear(ch_in, ch))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(linear_dropout))
            ch_in = ch
        layers.append(nn.Linear(ch_in, 1))
        self.head = nn.Sequential(*layers)

    def make_middle_layers(self,
                        extra_convs: Optional[List[int]]=[3], 
                        extra_channels: Optional[List[int]]=[64], 
                        extra_pooling: Optional[List[int]]=[2], 
                        extra_kernel_size: Optional[List[int]]=[5], 
                        middle_blocks_channels: Optional[List[int]]=[128, 256],
                        middle_blocks_kernel_size: Optional[List[int]]=[5,5],
                        lrelu_slope: Optional[float]=0.2,
                        dropout: Optional[float]=0.25) -> None:
        
        middle_layers = []
        ch_in = 1
        for n, ch, pl, ks in zip(extra_convs, extra_channels, extra_pooling, extra_kernel_size):
            for _ in range(n):
                middle_layers.append(ResidualBlockNoAux(ch, ks, pl))
                middle_layers.append(nn.Dropout1d(dropout))
            ch_in = ch

        for ch, ks in zip(middle_blocks_channels, middle_blocks_kernel_size):
                middle_layers.append(nn.Conv1d(ch_in, ch, ks, padding='same'))
                middle_layers.append(nn.BatchNorm1d(ch))
                middle_layers.append(nn.LeakyReLU(lrelu_slope))
                middle_layers.append(nn.Dropout1d(dropout))
                ch_in = ch
        self.middle_layers = nn.Sequential(*middle_layers)

    def make_encoder_layers(self,
                            convs: Optional[List[int]]=[3,4], 
                            channels: Optional[List[int]]=[32,64], 
                            pooling: Optional[List[int]]=[2,2], 
                            kernel_size: Optional[List[int]]=[5,5],
                            dropout: Optional[float]=0.25) -> None:
        self.initial_conv = nn.Conv1d(1, channels[0], kernel_size[0], padding='same')
        self.initial_dropout = nn.Dropout1d(dropout)
        self.encoder_layers = nn.ModuleList()
        ch_in = channels[0]
        for i, (n, ch, pl, ks) in enumerate(zip(convs, channels, pooling, kernel_size)):

            if i > 0: block = nn.Conv1d(ch_in, ch, ks, padding='same')
            else: block = nn.Identity()

            block = nn.ModuleList([block, nn.ModuleList(), nn.ModuleList()])
            for _ in range(n):
                block[1].append(ResidualBlockNoAux(ch, ks, pl))
                block[2].append(nn.Dropout1d(dropout))
            ch_in = ch
            self.encoder_layers.append(block)

    def forward(self, inputs: Tensor, classifier_residual_list: List[Tensor], *args: Any, **kwargs: Any) -> Tensor:
        x = self.initial_conv(inputs)
        x = x + classifier_residual_list.pop(0)
        x = self.initial_dropout(x)
        for block in self.encoder_layers:
            x = block[0](x)
            for layer, dropout in zip(block[1], block[2]):
                x = layer(x)
                x = x + classifier_residual_list.pop(0)
                x = dropout(x)
        x = self.middle_layers(x)
        x = self.head(x)
        return x
