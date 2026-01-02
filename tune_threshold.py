import torch
import numpy as np
import os
from tqdm import tqdm
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import get_dataloaders
from config import DEVICE, BASE_OUTPUT

# 1. Load lại SWA Model
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    decoder_attention_type="scse",
    deep_supervision=True,
    encoder_params={"dropout_rate": 0.5} 
)

# Load checkpoint fixed
checkpoint_path = os.path.join(BASE_OUTPUT, "best_model_swa_fixed.pth")
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# 2. Load Validation Set
_, valid_loader, _ = get_dataloaders(aug_mode='none')

print("[INFO] Tuning Threshold on Validation Set...")
true_masks = []
pred_probs = []

with torch.no_grad():
    for images, masks, _ in tqdm(valid_loader):
        images = images.to(DEVICE)
        
        # Predict
        logits = model(images)
        if isinstance(logits, (list, tuple)): logits = logits[0]
        probs = torch.sigmoid(logits)
        
        # Chỉ lấy mask khối u (bỏ qua background nếu cần, ở đây bài toán 1 class nên ok)
        pred_probs.append(probs.cpu().numpy())
        true_masks.append(masks.cpu().numpy())

# Gộp lại thành mảng lớn
pred_probs = np.concatenate(pred_probs)
true_masks = np.concatenate(true_masks)

# 3. Quét ngưỡng để tìm Best Dice
thresholds = np.arange(0.1, 0.95, 0.05)
best_dice = 0.0
best_thresh = 0.5

print("\nResult:")
print(f"{'Threshold':<10} | {'Dice Score':<10}")
print("-" * 25)

for thresh in thresholds:
    # Tính Dice cứng
    preds = (pred_probs > thresh).astype(np.float32)
    
    # Chỉ tính trên vùng có khối u (Mass only) để so sánh công bằng
    # (Nếu tính cả ảnh đen thì Dice sẽ rất cao ảo)
    intersection = (preds * true_masks).sum()
    union = preds.sum() + true_masks.sum()
    dice = (2. * intersection) / (union + 1e-7)
    
    print(f"{thresh:.2f}       | {dice:.4f}")
    
    if dice > best_dice:
        best_dice = dice
        best_thresh = thresh

print("-" * 25)
print(f"✅ BEST THRESHOLD: {best_thresh}")
print(f"🚀 BEST DICE:      {best_dice:.4f}")
