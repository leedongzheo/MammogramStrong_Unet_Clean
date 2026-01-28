import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class HybridDualEncoderUnetPlusPlus(nn.Module):
    def __init__(self, classes=1, encoder_weights="imagenet"):
        super().__init__()
        
        # --- 1. BACKBONE 1: CNN (ResNeSt50d) ---
        # Chuyên trị chi tiết cục bộ (Local features)
        self.cnn_encoder = smp.encoders.get_encoder(
            name="tu-resnest50d", 
            weights=encoder_weights,
            depth=5,
            output_stride=32
        )
        
        # --- 2. BACKBONE 2: Transformer (MiT-B3) ---
        # Chuyên trị ngữ cảnh toàn cục (Global context)
        self.trans_encoder = smp.encoders.get_encoder(
            name="mit_b3", 
            weights=encoder_weights,
            depth=5,
            output_stride=32
        )
        
        # --- 3. TÍNH TOÁN KÊNH SAU KHI GỘP (FUSION) ---
        # Lấy danh sách số kênh của từng encoder
        # Ví dụ ResNeSt: (3, 64, 256, 512, 1024, 2048)
        # Ví dụ MiT-B3:  (3, 64, 128, 320, 512, 512)
        cnn_channels = self.cnn_encoder.out_channels
        trans_channels = self.trans_encoder.out_channels
        
        # Cộng gộp số kênh vì mình sẽ Concatenate (nối đuôi) 2 features lại
        # self.fusion_channels sẽ là đầu vào cho Decoder
        self.fusion_channels = [c + t for c, t in zip(cnn_channels, trans_channels)]
        
        # --- 4. DECODER: Att-UNet++ ---
        # Chúng ta dùng Decoder có sẵn của SMP nhưng khai báo kênh đầu vào tùy chỉnh
        self.decoder = smp.decoders.UnetPlusPlusDecoder(
            encoder_channels=self.fusion_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type="scse"  # <--- Giữ Attention scSE mạnh mẽ
        )
        
        # --- 5. HEAD DỰ ĐOÁN ---
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=16,
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x):
        # 1. Chạy CNN Encoder
        # features_cnn là một list các feature maps từ to đến nhỏ
        features_cnn = self.cnn_encoder(x)
        
        # 2. Chạy Transformer Encoder
        features_trans = self.trans_encoder(x)
        
        # 3. Gộp Features (Feature Fusion)
        fused_features = []
        for f_c, f_t in zip(features_cnn, features_trans):
            # Transformer features có thể bị lệch kích thước 1-2 pixel do cách chia patch
            # Nên cần resize f_t cho khớp với f_c trước khi nối
            if f_c.shape[2:] != f_t.shape[2:]:
                f_t = F.interpolate(f_t, size=f_c.shape[2:], mode='bilinear', align_corners=False)
            
            # Nối theo chiều kênh (Dim 1)
            f_cat = torch.cat([f_c, f_t], dim=1)
            fused_features.append(f_cat)
            
        # 4. Đưa vào Decoder UNet++
        decoder_output = self.decoder(*fused_features)
        
        # 5. Dự đoán mask
        masks = self.segmentation_head(decoder_output)
        
        return masks
