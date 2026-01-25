import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        
        # 1. 1x1 Convolution
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. 3x3 Convolutions với các dilation rate khác nhau
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3x3_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[3], dilation=rates[3], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3. Global Average Pooling (Image Level Features)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 4. Gom tất cả lại và project về số kênh mong muốn
        # Input của lớp này là 5 nhánh gộp lại -> 5 * out_channels
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

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
class DilatedUnetPlusPlus(smp.UnetPlusPlus):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, **kwargs):
        # Khởi tạo mô hình cha (U-Net++ gốc)
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            **kwargs
        )
        
        # Lấy số kênh tại điểm đáy (Bottleneck) của Encoder
        # Với ResNeSt50d, thường là 2048 kênh
        last_channel = self.encoder.out_channels[-1]
        
        # Khởi tạo khối ASPP
        # Quan trọng: out_channels của ASPP PHẢI BẰNG last_channel 
        # để Decoder của U-Net++ hoạt động bình thường mà không báo lỗi lệch kênh.
        self.aspp = ASPP(in_channels=last_channel, out_channels=last_channel)
        
        print(f"[INFO] Initialized Dilated U-Net++ with ASPP at bottleneck (Channels: {last_channel})")

    def forward(self, x):
        # 1. Chạy Encoder -> Ra list các features map [f0, f1, f2, f3, f4, f5]
        features = self.encoder(x)
        
        # 2. Lấy feature map cuối cùng (Bottleneck)
        last_feature = features[-1]
        
        # 3. Đưa qua khối ASPP
        dilated_feature = self.aspp(last_feature)
        
        # 4. Thay thế feature cũ bằng feature đã qua ASPP
        # Cần chuyển tuple thành list để sửa đổi
        features = list(features)
        features[-1] = dilated_feature
        
        # 5. Đưa vào Decoder như bình thường
        decoder_output = self.decoder(*features)
        
        # 6. Segmentation Head
        masks = self.segmentation_head(decoder_output)
        
        return masks
