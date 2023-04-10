import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from complex_modules import *

__all__ = ['UNet']

USE_BIAS = False   

def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, groups=1, ndims=2):
    if ndims == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, groups=groups)
    elif ndims == 3:
        return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, groups=groups)
    else:
        raise ValueError(f'Convolution  must be 2D or 3D passed {ndims}')
        
class ComplexNorm(nn.Module):
    '''
    Performs batch norm independently on real and imaginary part.
    '''
    def __init__(self, num_features, normType, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ComplexNorm, self).__init__()
        self.bn_r = normType(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = normType(num_features, eps, momentum, affine, track_running_stats)

    def forward(self,input):
        return self.bn_r(input.real) +1J*self.bn_i(input.imag)


class VarNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.05, affine=True, track_running_stats=True):
        super(VarNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size()
            with torch.no_grad():
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            var = self.running_var

        input = input / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None]

        return input
    
class VarNorm3d(nn.BatchNorm3d):
    def __init__(self, num_features, eps=1e-5, momentum=0.05, affine=True, track_running_stats=True):
        super(VarNorm3d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            # use biased var in train
            var = input.var([0, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size()
            with torch.no_grad():
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            var = self.running_var

        input = input / (torch.sqrt(var[None, :, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None]

        return input
    
class ConvBlock(nn.Sequential):
    """
    Create a convolution block with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of convolution channel
        order (list): order of modules 
        padding (int): add zero-padding to the input
        ndims (int): 2 or 3 to indicate 2D or 3D convolutions
        complex_input (bool): True -> the input is complex, False -> the input is real values
        complex_kernel (bool): If True the convolution will use a complex kernel
    Return:
        A nn.Module object containing all the specified layers
    """
    def __init__(self, in_channels, out_channels, kernel_size, order, padding=1, ndims=2, complex_kernel=False):
        super(ConvBlock, self).__init__()
        for i, char in enumerate(order):
            if char == 'relu':
                self.add_module(f'ComplexReLU{i}', ComplexReLU())
            elif char == 'mod relu':
                self.add_module(f'modReLU{i}', modReLU(in_channels=in_channels, ndims=ndims))
            elif char == 'convolution':
                # add learnable bias only in the absence of gatchnorm/groupnorm
                # bias = not ('g' in order or 'b' in order)
                # bias = not ('g' in order)
                self.add_module(f'complex_conv{i}',
                                ComplexConv(in_channels, out_channels, kernel_size,
                                            bias=False, padding=padding, complex_kernel=complex_kernel, ndims=ndims))
                in_channels = out_channels
            elif char == 'separable convolution':
                self.add_module(f'conv{i}',
                            ComplexDepthwiseSeparableConv(in_channels, out_channels, kernel_size,
                                                          bias=False, padding=padding, ndims=ndims))
                in_channels = out_channels
            elif char == 'instance norm':
                if ndims == 2:
                    layer = ComplexNorm(out_channels, normType=nn.InstanceNorm2d)
                else: 
                    layer = ComplexNorm(out_channels, normType=nn.InstanceNorm3d)
                self.add_module(f'instancenorm{i}', layer)
            elif char == 'batch norm':
                if ndims == 2:
                    layer = ComplexNorm(out_channels, normType=nn.BatchNorm2d)
                else: 
                    layer = ComplexNorm(out_channels, normType=nn.BatchNorm3d)
                self.add_module(f'batchnorm{i}', layer)
            elif char == 'variation norm':
                if ndims == 2:
                    layer = ComplexNorm(out_channels, normType=VarNorm2d)
                else: 
                    layer = ComplexNorm(out_channels, normType=VarNorm3d)
                self.add_module(f'varnorm{i}', layer)
            else:
                raise ValueError(f"Unsupported layer type '{char}'")

class DoubleConv(nn.Module):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm2d+ReLU+Conv2d).
    We use (Conv2d+ReLU+GroupNorm2d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv2d+BatchNorm2d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int): size of the convolving kernel
        order (list): determines the order of layers
    """

    def __init__(self, in_channels, out_channels, encoder, t_dim=32, kernel_size=3, order='crg', padding=1, ndims=2, complex_kernel=False):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.ConvBlock1 = ConvBlock(conv1_in_channels, conv1_out_channels, kernel_size, order, padding,
                                   ndims, complex_kernel)
        # conv2
        self.ConvBlock2 = ConvBlock(conv2_in_channels, conv2_out_channels, kernel_size, order, padding,
                                   ndims, complex_kernel)
        # time encoding
        self.time_mlp = nn.Sequential(
            ComplexSinusoidalPositionEmbeddings(t_dim),
            ComplexReLU(),
            ComplexLinear(t_dim, conv1_out_channels, complex_kernel=complex_kernel),
            ComplexReLU()
        )
    def forward(self, x, t):
        time_encoding = self.time_mlp(t)
        x = self.ConvBlock1(x)
        x = x+time_encoding.unsqueeze(-1).unsqueeze(-1)
        x = self.ConvBlock2(x)
        return x


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int): size of the convolving kernel
        downsample (bool): if True use downsampling convolution
        scale_factor (int): downsampling scale
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3,
                 basic_module=DoubleConv, downsample=True, conv_layer_order='crb',
                 scale_factor=2, ndims=2, complex_kernel=False,
                 padding=1):
        super(Encoder, self).__init__()

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         ndims=ndims,
                                         complex_kernel=complex_kernel,
                                         padding=padding)

        if downsample:
            self.downsampler = ComplexConv(in_channels,
                                           in_channels,
                                           kernel_size=2,
                                           stride=scale_factor,
                                           padding=0,
                                           bias=False,
                                           ndims=ndims,
                                           complex_kernel=complex_kernel,
                                           groups=in_channels)
        else:
            self.downsampler = nn.Identity()

    def forward(self, x, t):
        x = self.downsampler(x)
        x = self.basic_module(x, t)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose2d or interpolation) followed by a DoubleConv
    module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose2d, must reverse the MaxPool2d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
    """

    def __init__(self, in_channels, out_channels, add_features, kernel_size=3,
                 scale_factor=2, basic_module=DoubleConv, conv_layer_order='crg', ndims=2,
                 complex_kernel=False, padding=1):
        super(Decoder, self).__init__()

        self.upsample = ComplexConvTranspose(in_channels,
                                             in_channels,
                                             kernel_size=2,
                                             stride=scale_factor,
                                             padding=0,
                                             output_padding=0,
                                             bias=False,
                                             ndims=ndims,
                                             complex_kernel=complex_kernel,
                                             groups=in_channels)

        self.basic_module = basic_module(in_channels + add_features, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         ndims=ndims,
                                         complex_kernel=complex_kernel,
                                         padding=padding)

    def forward(self, encoder_features, x, t):
        # use ConvTranspose2d and summation joining
        x = self.upsample(x)
        x = torch.cat([encoder_features, x], dim=1)
        x = self.basic_module(x, t)
        return x


def create_feature_maps(init_channel_number, number_of_fmaps, growth_rate=2.0):
    channel_number = init_channel_number
    fmaps = []
    fmaps.append(init_channel_number)
    for k in range(number_of_fmaps - 1):
        channel_number = int(channel_number * growth_rate)
        fmaps.append(channel_number)

    return fmaps


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]), requires_grad=True)

    def forward(self, x):
       return x * self.scale


class UNet(nn.Module):
    r"""
    Args:
        in_channels (int): Channels of input
        out_channels (int): Channels of output
        f_maps (int or tuple): either the starting number of feature maps or a list of feature maps
                               at each level.
        layer_order (str):
        depth (int): The depth of the UNet. Only used if f_maps is an int
        layer_growth (float): The rate of increase in feature maps per layer.
        residual (bool): Add a shortcut og the input to the output before returning.
        ndims (int): 2 or 3 to switch between 2D and 3D UNets
        complex_input (bool): If True the input is expected to be complex
        complex_kernel (bool): Use complex convolution kernels. Only used if input is complex.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 f_maps=64,
                 layer_order=['convolution', 'mod relu'],
                 depth=4,
                 layer_growth=2.0,
                 residual=True,
                 scaled_residual=True,
                 ndims=2,
                 complex_kernel=False,
                 padding=1):

        super(UNet, self).__init__()

        # Create feature maps as list if specified by integer
        if isinstance(f_maps, int):
            f_maps = create_feature_maps(f_maps, number_of_fmaps=depth, growth_rate=layer_growth)

        self.padding = padding

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        crop_amount = []
        current_crop = 0
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, downsample=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, ndims=ndims, complex_kernel=complex_kernel,
                                  padding=padding)

                # Last layer layer applies 4 convolutions with no scaling
                current_crop = (current_crop + 4)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, ndims=ndims, complex_kernel=complex_kernel,
                                  padding=padding)

                # Each layer applies 4 convolutions and scales by 2
                current_crop = (current_crop + 4)*2

            crop_amount.append( tuple([-current_crop for _ in range(2*ndims)]))
            encoders.append(encoder)

        self.crop_amount = crop_amount[:-1]
        self.encoders = nn.ModuleList(encoders)
        print(f'Crop amount {self.crop_amount}')

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i]
            add_feature_num = reversed_f_maps[i + 1]  # features from past layer
            out_feature_num = reversed_f_maps[i + 1]  # features from past layer
            decoder = Decoder(in_feature_num, out_feature_num, add_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, ndims=ndims, complex_kernel=complex_kernel,
                              padding=padding)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = ComplexConv(f_maps[0], out_channels,
                                      kernel_size=1, complex_kernel=complex_kernel, bias=False, ndims=ndims)

        # Store boolean to specify if input is added
        self.residual = residual
        self.scaled_residual = scaled_residual
        if self.residual and self.scaled_residual:
            # Use a 1x1 convolution if the channels out does not equal channels in
            if out_channels != in_channels:
                self.residual_conv = ComplexConv(in_channels, out_channels,
                                                 kernel_size=1,
                                                 complex_kernel=complex_kernel,
                                                 bias=False,
                                                 ndims=ndims)
            else:
                self.residual_conv = ScaleLayer()
        elif self.residual:
            self.residual_conv = nn.Identity()

        if self.padding == 0:
            self.output_pad = tuple([-4 + p for p in self.crop_amount[-1]])
        else:
            self.output_pad = tuple([0 for _ in range(ndims * 2)])

    def check_spatial_dimensions(self, input_size):

        input_size = np.array(input_size)
        #print(f'Input size = {input_size}')
        for d in range(len(self.encoders)-1):
            # Convolutions
            input_size -= 4

            # Downsampling
            if np.any(input_size % 2 != 0):
                #print(f'   Down layer size not divisible {input_size}')
                return False
            input_size = input_size // 2


            #print(f'   Down Layer size {input_size}')

        # Flat layer
        input_size -= 4
        #print(f'Flat Layer size {input_size}')

        # Now up layers
        for u in range(len(self.decoders)):
            # Upsample
            input_size *= 2

            # Convolution
            input_size -= 4

            #print(f'Up Layer size {input_size}')

        return True

    def forward(self, x, t):

        # Keep x
        input = x

        # encoder part
        encoders_features = []
        for encoder in self.encoders[:-1]:
            x = encoder(x, t)

            # reverse the encoder outputs to be aligned with the decoder
            #print(x.shape)
            encoders_features.insert(0, x)

        # Last encoder is flat
        x = self.encoders[-1](x, t)
        #print(x.shape)

        # decoder part
        for decoder, encoder_output, crop_amount in zip(self.decoders, encoders_features, self.crop_amount):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            #print(f'Decoder {x.shape} {encoder_output.shape} {crop_amount}')
            if self.padding != 1:
                encoder_output = F.pad(encoder_output, crop_amount)
            #print(f'Decoder {x.shape} {encoder_output.shape}')
            x = decoder(encoder_output, x, t)

        x = self.final_conv(x)

        # Keep skip to end
        if self.residual:
            if self.padding != 1:
                #print(f'Residual {x.shape} {input.shape} {self.output_pad}')
                x += F.pad(self.residual_conv(input), self.output_pad)
            else:
                x += self.residual_conv(input)

        return x