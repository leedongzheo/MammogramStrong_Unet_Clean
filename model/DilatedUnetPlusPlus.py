import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[1, 2, 4, 8]):
        super(ASPP, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv3x3_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[3], dilation=rates[3], bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Dropout(0.5))

    def forward(self, x):
        size = x.shape[-2:]
        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3_1(x)
        feat3 = self.conv3x3_2(x)
        feat4 = self.conv3x3_3(x)
        feat5 = self.global_avg_pool(x)
        feat5 = F.interpolate(feat5, size=size, mode='bilinear', align_corners=False)
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        return self.project(out)

# --- Class DilatedUnetPlusPlus (ĐÃ FIX FORWARD) ---
class DilatedUnetPlusPlus(smp.UnetPlusPlus):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, **kwargs):
        super().__init__(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes, **kwargs)
        last_channel = self.encoder.out_channels[-1]
        self.aspp = ASPP(in_channels=last_channel, out_channels=last_channel)
        
    def forward(self, x):
        features = self.encoder(x)
        
        # Chèn ASPP vào feature cuối
        features = list(features)
        last_feature = features[-1]
        dilated_feature = self.aspp(last_feature)
        features[-1] = dilated_feature
        
        # FIX: Bỏ dấu * (unpacking) vì Decoder version này muốn nhận 1 list
        decoder_output = self.decoder(features)
        
        masks = self.segmentation_head(decoder_output)
        return masks
