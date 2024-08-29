import torch
import torch.nn as nn
import torch.nn.functional as F
def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, if_IN=False, IN_affine=False, if_BN=False):
    padding = ((kernel_size - 1) * dilation) // 2
    if kernel_size % 2 == 0:  # Extra adjustment for even kernel sizes
            padding=1
    if isReLU:
        if if_IN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=padding, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.InstanceNorm2d(out_planes, affine=IN_affine)
            )
        elif if_BN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=padding, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(out_planes, affine=IN_affine)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=padding, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
            )
    else:
        if if_IN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=padding, bias=True),
                nn.InstanceNorm2d(out_planes, affine=IN_affine)
            )
        elif if_BN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=padding, bias=True),
                nn.BatchNorm2d(out_planes, affine=IN_affine)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=padding, bias=True)
            )

input_tensor = torch.randn(12, 115, 5, 10)  # Example batch size, channels, height, width

# Initialize the convolution with kernel size 2
conv_layer = conv(in_planes=115, out_planes=128, kernel_size=2)

# Pass the input through the convolution
output_tensor = conv_layer(input_tensor)

# Print the shapes to verify
print("Input shape:", input_tensor.shape)
print("Output shape:", output_tensor.shape)

class Conv2dCustomPadding(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=1, dilation=1, isReLU=True):
        super(Conv2dCustomPadding, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=0, bias=True)
        self.isReLU = isReLU
        if isReLU:
            self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # Custom padding: manually add padding before applying the convolution
        x_padded = F.pad(x, (1, 0, 1, 0))  # (left, right, top, bottom) asymmetric padding
        x = self.conv(x_padded)
        if self.isReLU:
            x = self.relu(x)
        return x

# Example usage
input_tensor = torch.randn(12, 115, 5, 10)  # Example input tensor
conv_layer = Conv2dCustomPadding(in_planes=115, out_planes=128)
output_tensor = conv_layer(input_tensor)

print("Input shape:", input_tensor.shape)
print("Output shape:", output_tensor.shape)