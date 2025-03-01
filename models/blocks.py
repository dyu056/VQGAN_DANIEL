import torch
from torch import nn

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, device=None):
        super(Down, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, device=None):
        super(Up, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, 
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, device=None):
        super(Residual, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class Non_Local(nn.Module):
    def __init__(self, in_channels, device=None):
        super(Non_Local, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2

        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.W = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        batch_size, C, H, W = x.size()
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        theta_x = phi_x.permute(0, 2, 1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = f / f.size(-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, H, W)
        W_y = self.W(y)
        z = W_y + x

        return z

class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, device=None):
        super(GroupNorm, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.group_norm = nn.GroupNorm(num_groups, num_channels)
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.group_norm(x)

class Swish(nn.Module):
    def __init__(self, device=None):
        super(Swish, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return x * torch.sigmoid(x) 