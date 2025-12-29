import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=True)

    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(self.avg_pool(x)), inplace=True))
        max_out = self.fc2(F.relu(self.fc1(self.max_pool(x)), inplace=True))
        out = avg_out + max_out
        return x * torch.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv1(concat))
        return x * attn

class CBAM(nn.Module):
    def __init__(self, channels, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# Attention Gating Block
class AttentionGatingBlock(nn.Module):
    def __init__(self, in_channels_x, in_channels_g, inter_channels):
        super(AttentionGatingBlock, self).__init__()
        self.W_g = nn.Conv2d(in_channels_g, inter_channels, kernel_size=1)
        self.W_x = nn.Conv2d(in_channels_x, inter_channels, kernel_size=2, stride=2)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)
        self.final_conv = nn.Conv2d(in_channels_x, in_channels_x, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels_x)

    def forward(self, x, g):
        theta_x = self.W_x(x)
        phi_g = self.W_g(g)
        upsample_g = F.interpolate(phi_g, size=theta_x.shape[2:], mode='bilinear', align_corners=True)
        concat_xg = F.relu(theta_x + upsample_g)
        psi = torch.sigmoid(self.psi(concat_xg))

        upsample_psi = F.interpolate(psi, size=x.shape[2:], mode='bilinear', align_corners=True)
        upsample_psi = upsample_psi.repeat(1, x.shape[1], 1, 1)

        y = x * upsample_psi
        out = self.final_conv(y)
        return self.bn(out)


# 2 lớp Conv2D + BatchNorm + ReLU (giống block trong U-Net)
class UnetConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm=True):
        super(UnetConv2D, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if is_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if is_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# Gating signal (1x1 conv + BatchNorm + ReLU)
class UnetGatingSignal(nn.Module):
    def __init__(self, in_channels, is_batchnorm=True):
        super(UnetGatingSignal, self).__init__()
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels) if is_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        self.gate = nn.Sequential(*layers)

    def forward(self, x):
        return self.gate(x)

class PyramidCbamGateUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(PyramidCbamGateUNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)

        # Input scaling
        self.scale2_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding="same")
        self.scale3_conv = nn.Conv2d(in_channels, 32, kernel_size=3, padding="same")
        self.scale4_conv = nn.Conv2d(in_channels, 16, kernel_size=3, padding="same")

        # Encoder
        self.conv1 = UnetConv2D(in_channels, 32)
        self.conv2 = UnetConv2D(64 + 32, 64)
        self.conv3 = UnetConv2D(32 + 64, 128)
        self.conv4 = UnetConv2D(16 + 128, 64)
        self.center = UnetConv2D(64, 512)

        # Attention
        # self.cbam1 = attach_attention_module(self.conv1, 'cbam_block')
        # self.cbam2 = attach_attention_module(self.conv2,)
        # self.cbam3 = attach_attention_module(self.conv3)
        # self.cbam4 = attach_attention_module(self.conv4)
        self.cbam1 = CBAM(32)
        self.cbam2 = CBAM(64)
        self.cbam3 = CBAM(128)
        self.cbam4 = CBAM(64)
        self.gating1 = UnetGatingSignal(512)
        self.gating2 = UnetGatingSignal(32+64)
        self.gating3 = UnetGatingSignal(128+32)

        self.attgatingblock1 =AttentionGatingBlock(64,512,128)
        self.attgatingblock2 = AttentionGatingBlock(128,96,64)
        self.attgatingblock3 = AttentionGatingBlock(64,160,32)

        # self.attgatingblock =AttentionGatingBlock(64,512,64)
        # Decoder
        self.up1_transpose = nn.Sequential(
                        nn.ConvTranspose2d(512, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.ReLU(inplace=True)
)
        self.up2_transpose = nn.Sequential(
                        nn.ConvTranspose2d(32 + 64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.ReLU(inplace=True)
) 

        self.up3_transpose = nn.ConvTranspose2d(32 + 128, 32, 3, stride=2, padding=1, output_padding=1)
        self.up4_transpose = nn.ConvTranspose2d(32 + 64, 32, 3, stride=2, padding=1, output_padding=1)
        
        self.conv9 = UnetConv2D(32 + 32, 32)
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Scale inputs
        scale2 = self.avgpool(x)
        scale3 = self.avgpool(scale2)
        scale4 = self.avgpool(scale3)

        # Encoder path
        conv1 = self.conv1(x)
        pool1 = self.pool(conv1)

        scale2 = self.scale2_conv(scale2)
        conv2_in = torch.cat([scale2, pool1], dim=1)
        conv2 = self.conv2(conv2_in)
        pool2 = self.pool(conv2)

        scale3 = self.scale3_conv(scale3)
        conv3_in = torch.cat([scale3, pool2], dim=1)
        conv3 = self.conv3(conv3_in)
        pool3 = self.pool(conv3)

        scale4 = self.scale4_conv(scale4)
        conv4_in = torch.cat([scale4, pool3], dim=1)
        conv4 = self.conv4(conv4_in)
        pool4 = self.pool(conv4)

        center = self.center(pool4)

        # # Attention
        conv1 = self.cbam1(conv1)
        conv2 = self.cbam2(conv2)
        conv3 = self.cbam3(conv3)
        conv4 = self.cbam4(conv4)

        # # Decoder with Attention Gate
        # gating = UnetGatingSignal(center.shape[1])
        # g1 = UnetGatingSignal(gating)
        g1 = self.gating1(center)
        attn1 = self.attgatingblock1(conv4, g1)

        up1 = torch.cat([self.up1_transpose(center), attn1], dim=1)

        g2 = self.gating2(up1)
        attn2 = self.attgatingblock2(conv3, g2)
        up2 = torch.cat([self.up2_transpose(up1), attn2], dim=1)

        g3 = self.gating3(up2)

        attn3 = self.attgatingblock3(conv2, g3)

        up3 = torch.cat([self.up3_transpose(up2), attn3], dim=1)

        up4 = torch.cat([self.up4_transpose(up3), conv1], dim=1)

        conv9 = self.conv9(up4)
        out = self.final(conv9)
        return out
