import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(CausalConv3d, self).__init__(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=0, dilation=dilation,
                                           groups=groups, bias=bias)

        # Calculate the causal padding for the time dimension
        time_pad = (self.kernel_size[0] - 1) * self.dilation[0] - self.stride[0] + 1
        
        # Ensure the time dimension (first dimension) is causal
        self.padding = (time_pad, padding if isinstance(padding, int) else padding[1], 
                        padding if isinstance(padding, int) else padding[2])

    def forward(self, x):
        # Apply causal padding only to the time dimension
        x = F.pad(x, (self.padding[2], self.padding[2],  # Depth (non-causal)
                      self.padding[1], self.padding[1],  # Height (non-causal)
                      self.padding[0], 0))  # Time (causal)
        
        return super(CausalConv3d, self).forward(x)


class CausalConvTranspose3d(nn.ConvTranspose3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super(CausalConvTranspose3d, self).__init__(in_channels, out_channels, kernel_size,
                                                    stride=stride, padding=padding, output_padding=output_padding,
                                                    groups=groups, bias=bias, dilation=dilation)

    def forward(self, x):
        # Apply the transposed convolution
        output = super(CausalConvTranspose3d, self).forward(x)
        
        # Only return :time * stride[0] of the output feature map
        return output[..., :x.shape[2] * self.stride[0], ...]
