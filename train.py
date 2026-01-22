import argparse
from dataset import*

def get_args():
    # Tham số bắt buộc nhập
    parser = argparse.ArgumentParser(description="Train, Pretrain hoặc Evaluate một model AI")
    parser.add_argument("--epoch", type=int, help="Số epoch để train")
    # parser.add_argument("--model", type=str, required=True, help="Đường dẫn đến model")
    parser.add_argument("--mode", type=str, choices=["train", "pretrain", "evaluate"], required=True, help="Chế độ: train hoặc pretrain hoặc evaluate")
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn đến dataset đã giải nén")
    # Tham số trường hợp
    parser.add_argument("--checkpoint", type=str, help="Đường dẫn đến file checkpoint (chỉ dùng cho chế độ pretrain)")
    parser.add_argument("--augment", action='store_true', help="Bật Augmentation cho dữ liệu đầu vào")
    # Tham số mặc định(default)
    parser.add_argument("--saveas", type=str, help="Thư mục lưu checkpoint")
    parser.add_argument("--lr0", type=float, help="learning rate, default = 0.0001")
    parser.add_argument("--batchsize", type=int, help="Batch size, default = 8")

    parser.add_argument("--weight_decay", type=float,  help="weight_decay, default = 1e-6")
    parser.add_argument("--img_size", type=int, nargs=2,  help="Height and width of the image, default = [256, 256]")
    parser.add_argument("--numclass", type=int, help="shape of class, default = 1")
    parser.add_argument("--warmup", type=int, default=10, help="Số epoch để warm-up (augment nhẹ)")
    """
    # Với img_size, cách chạy: python script.py --img_size 256 256
    Nếu muốn nhập list dài hơn 3 phần tử, gõ 
    parser.add_argument("--img_size", type=int, nargs='+', default=[256, 256], help="Image dimensions")
    Chạy:
    python script.py --img_size 128 128 3
    """
    parser.add_argument("--loss", type=str, choices=["Dice_loss", "Hybric_loss", "BCEDice_loss", "BCEwDice_loss", "BCEw_loss", "SoftDice_loss", "Combo_loss", "Tversky_loss", "FocalTversky_loss" ], default="Combo_loss", help="Hàm loss sử dụng, default = Combo_loss")
    parser.add_argument("--optimizer", type=str, choices=["Adam", "SGD", "AdamW"], default="AdamW", help="Optimizer sử dụng, default = AdamW")
    args = parser.parse_args()
    
    # Kiểm tra logic tham số
    if args.mode in ["pretrain", "evaluate"] and not args.checkpoint:
        parser.error(f"--checkpoint là bắt buộc khi mode là '{args.mode}'")
        
    return args
def set_seed():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# --- [THÊM HÀM NÀY] HÀM HỖ TRỢ ĐÓNG/MỞ BĂNG ---
def set_grad_status(model, freeze=True):
    """
    Hàm đóng băng hoặc mở băng Backbone/Encoder.
    Hỗ trợ cả Model Custom (self.backbone) và Model SMP (self.encoder).
    """
    target_module = None
    
    # 1. Kiểm tra nếu là Model Custom (PyramidCbamGateResNetUNet)
    if hasattr(model, 'backbone'):
        target_module = model.backbone
        name = "Backbone (ResNet)"
    # 2. Kiểm tra nếu là Model SMP (DeepLabV3+, Unet++, ...)
    elif hasattr(model, 'encoder'):
        target_module = model.encoder
        name = "Encoder (SMP)"
    
    if target_module:
        for param in target_module.parameters():
            param.requires_grad = not freeze # Freeze = True -> requires_grad = False
        
        status = "FROZEN ❄️" if freeze else "UNFROZEN 🔥"
        print(f"[INFO] {name} is now {status}")
    else:
        print("[WARNING] Could not find 'backbone' or 'encoder' to freeze!")
def main(args):  
    print(f"\n[DEBUG TRAIN] args.loss bạn nhập từ bàn phím = {args.loss}")
    print("-" * 50)
    import numpy as np    
    from trainer import Trainer
    from model import Unet, unet_pyramid_cbam_gate, Swin_unet
    # from model import Swin_unet
    import optimizer as optimizer_module
    from dataset import get_dataloaders
    from result import export, export_evaluate
    global trainer
    from utils import get_loss_instance, _focal_tversky_global
    from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
    import shutil
    # from utils import loss_func
    from torch.optim.lr_scheduler import _LRScheduler
    print("-" * 50)
    print(f"[INFO] Mode: {args.mode.upper()}")
    print("-" * 50)

    set_seed()
    
    # 1. Khởi tạo Model
    print(f"[INFO] Initializing Model...")
    # model = unet_pyramid_cbam_gate.PyramidCbamGateResNet50UNet(
    #     in_channels=3, 
    #     out_channels=1, 
    #     deep_supervision=True,
    #     dropout_prob=0.5)
    # model = smp.DeepLabV3Plus(
    #         # encoder_name="tu-resnest50d", # ResNeSt rất mạnh cho y tế, hoặc dùng efficientnet-b3
    #         # encoder_name = "efficientnet-b4"
    #         encoder_name="tu-resnest50d",
    #         encoder_weights="imagenet",
    #         in_channels=3,
    #         classes=1,
    #         drop_path_rate=0.2
    # )
    # UNET++ (used)
#     model = smp.UnetPlusPlus(
#         encoder_name="tu-resnest50d", 
#         encoder_weights="imagenet",
#         in_channels=3,
#         classes=1,
#         drop_path_rate=0.5
# )
    # SwinUnet => Size ảnh ko khớp => Bỏ
    
#     model = smp.Unet(
#         # --- CẤU HÌNH QUAN TRỌNG NHẤT ---
#         # Thay backbone CNN (ResNest) bằng Swin Transformer
#         # Các lựa chọn: 
#         # - 'swin_tiny_patch4_window7_224' (Nhẹ nhất, ~ResNet50)
#         # - 'swin_small_patch4_window7_224' (~ResNet101)
#         # - 'swin_base_patch4_window7_224' (Mạnh, rất nặng VRAM)
#         encoder_name="tu-swin_base_patch4_window7_224", 
        
#         encoder_weights="imagenet",
#         in_channels=3,
#         classes=1,
        
#         # Swin Transformer không nhận drop_path_rate ở ngoài
#         # Phải đưa vào encoder_params
#         encoder_params={
#             "drop_path_rate": 0.2, # Swin khá nhạy cảm, nên để thấp (0.2-0.3)
#         },
        
#         # Tùy chọn: Thêm Attention cho Decoder để "full option"
#         decoder_attention_type="scse" 
# )
    #  Thay thế SwinUnet bằng ConvNeXt
# -------------BO DROP-----------------------
#     model = smp.Unet(
#         # ConvNeXt Tiny: Mạnh ~ ResNet50 / Swin-Tiny
#         # ConvNeXt Base: Mạnh ~ ResNet101 / Swin-Base
#         # Thêm tiền tố "tu-" vì nó lấy từ timm
#         encoder_name="tu-convnext_tiny", 
        
#         encoder_weights="imagenet",
#         in_channels=3,
#         classes=1,
        
#         # ConvNeXt dùng Drop Path giống Swin
#         drop_path_rate=0.2,
#         # Vẫn nên thêm Attention cho Decoder
#         decoder_attention_type="scse"
# )
# -------------BO DROP-----------------------
    #  Thay thế SwinUnet bằng ConvNeX
    def set_drop_path_rate(model, drop_rate=0.2):
        count = 0
        # Duyệt qua tất cả các module trong encoder
        for module in model.encoder.modules():
            # Kiểm tra xem module có phải là DropPath của timm không
            # (Thường tên class sẽ chứa chữ 'DropPath')
            if "DropPath" in module.__class__.__name__:
                module.drop_prob = drop_rate
                count += 1
        print(f"[INFO] Đã cập nhật DropPath rate = {drop_rate} cho {count} blocks trong Encoder.")

    # Gọi hàm để set rate là 0.2
    set_drop_path_rate(model, drop_rate=0.2)
    # tranUnet (using)
    # Thay vì TransUNet (chưa có trong SMP), ta dùng Unet với Encoder là Transformer
#     model = smp.Unet(
#         # mit_b3 là backbone của SegFormer, mạnh tương đương ResNet50/101
#         # nhưng dùng cơ chế Self-Attention.
#         encoder_name="mit_b3",        
#         encoder_weights="imagenet",
#         in_channels=3,
#         classes=1,
#         # Các backbone Transformer trong SMP thường không nhận tham số drop_path_rate 
#         # trực tiếp ở đây, nên ta bỏ dòng đó đi để tránh lỗi.
#         decoder_use_batchnorm=True,
# )
    # UNET++ attention (Used)
#     model = smp.UnetPlusPlus(
#         encoder_name="tu-resnest50d", 
#         encoder_weights="imagenet",
#         in_channels=3,
#         classes=1,
        
#         # --- QUAN TRỌNG: THÊM DÒNG NÀY ĐỂ CÓ ATTENTION ---
#         # scse giúp mô hình vừa lọc không gian (Spatial) vừa lọc kênh (Channel)
#         decoder_attention_type="scse",
        
#         # --- SỬA LỖI DROP_PATH_RATE ---
#         # Đưa vào encoder_params mới đúng cú pháp
#         drop_path_rate=0.5
# )
    # UNet thường (Used)
#     model = smp.Unet(
#         encoder_name="tu-resnest50d", 
#         encoder_weights="imagenet",
#         in_channels=3,
#         classes=1,
#         # drop_path_rate nên được đưa vào encoder_params để truyền xuống backbone timm
#         drop_path_rate=0.5
# )
    # attentionUnet (Using)
#     model = smp.Unet(
#         encoder_name="tu-resnest50d", 
#         encoder_weights="imagenet",
#         in_channels=3,
#         classes=1,
        
#         # --- THÊM DÒNG NÀY ĐỂ THÀNH ATTENTION UNET ---
#         # scse: Spatial and Channel Squeeze & Excitation Attention
#         # Nó sẽ chèn các block attention vào sau mỗi tầng Decoder
#         decoder_attention_type="scse",
        
#         # --- SỬA LỖI DROP_PATH_RATE ---
#         # Phải đưa vào encoder_params mới đúng, để ở ngoài sẽ không có tác dụng hoặc báo lỗi
#         drop_path_rate=0.5
# )
    # Segformer (Using)
#     model = smp.Segformer(
#         # Encoder chuẩn của SegFormer là dòng MiT (Mix Transformer)
#         # mit_b0 (nhẹ nhất) -> mit_b5 (nặng nhất)
#         # mit_b3 là lựa chọn cân bằng, mạnh tương đương ResNet50/ResNest50d
#         encoder_name="mit_b3",        
        
#         encoder_weights="imagenet",
#         in_channels=3,
#         classes=1,
        
#         # Encoder params vẫn dùng để truyền drop_path_rate
#         # Lưu ý: Với Transformer, drop_path_rate thường để thấp (0.1) thay vì 0.5
#         encoder_params={"drop_path_rate": 0.1} 
# )
    # 2. Khởi tạo Optimizer
    opt = optimizer_module.optimizer(model=model) 
    # --- [CHÍNH XÁC: KHỞI TẠO SEQUENTIAL LR TẠI ĐÂY] ---
    warmup_epochs = args.warmup if args.warmup > 0 else 10
    scheduler_warmup = LinearLR(
        opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    # B. Main Cosine
    scheduler_cosine = CosineAnnealingWarmRestarts(
        opt, T_0=10, T_mult=2, eta_min=1e-6
    )
    # C. Hợp thể (Dùng cho Giai đoạn 1 & 2)
    scheduler_initial = SequentialLR(
        opt, 
        schedulers=[scheduler_warmup, scheduler_cosine], 
        milestones=[warmup_epochs] 
    )
    # 3. KHỞI TẠO LOSS (Thay thế hàm get_loss_function)
    # Logic: Nếu chọn FocalTversky thì lấy biến toàn cục, còn lại thì khởi tạo class
    criterion_init = get_loss_instance(args.loss)
    # 4. Khởi tạo Trainer
    # Lưu ý: Trainer lưu reference tới criterion_init. 
    # Nếu criterion_init là _focal_tversky_loss, mọi thay đổi trên _focal_tversky_loss sẽ tự động cập nhật trong Trainer.
    trainer = Trainer(model=model, optimizer=opt, criterion=criterion_init, scheduler=scheduler_initial, patience=10, device=DEVICE)

    if args.mode == "train":
        if not os.path.exists(BASE_OUTPUT):
            os.makedirs(BASE_OUTPUT)
        # resume_checkpoint = None
        if args.augment:
            mode_stage1 = 'weak'
            mode_stage23 = 'strong'
            print(f"[INFO] Augmentation: ON (Stage 1: {mode_stage1} -> Stage 2/3: {mode_stage23})")
        else:
            mode_stage1 = 'none'
            mode_stage23 = 'none'
            print(f"[INFO] Augmentation: OFF (All Stages: none)")
        # if args.augment and args.warmup > 0:
        if args.warmup > 0:

            # =========================================================
            # GIAI ĐOẠN 1: WARM-UP
            # =========================================================
            print("\n" + "="*40)
            print(" GIAI ĐOẠN 1: WARM-UP (Freeze Backbone) (10 Epochs)")
            print(f" Config: Light Augment | Loss: {args.loss}")
            print("="*40)

            trainLoader_weak, validLoader, _ = get_dataloaders(aug_mode=mode_stage1)
            
            # Đảm bảo params đúng cho GD1 (nếu dùng Focal)
            if args.loss == "FocalTversky_loss":
                _focal_tversky_global.update_params(alpha=0.7, beta=0.3, gamma=1.33)
            # --- [THÊM] ĐÓNG BĂNG BACKBONE ---
            set_grad_status(model, freeze=True)
            trainer.num_epochs = args.warmup
            trainer.patience = 999      
            trainer.train(trainLoader_weak, validLoader, resume_path=None)
            # --- [THÊM] MỞ BĂNG BACKBONE (Để chuẩn bị cho GD2) ---
            set_grad_status(model, freeze=False)
            resume_checkpoint = "best_dice_mass_model.pth" 
        else:
            print("\n[INFO] Skipping Stage 1 (Warm-up). Starting directly with Main Training.")
            # Đảm bảo chắc chắn là đã Unfreeze nếu không chạy Stage 1
            set_grad_status(model, freeze=False)
            resume_checkpoint = None
        # =========================================================
        # GIAI ĐOẠN 2: INTERMEDIATE TUNING (Chỉ FocalTversky)
        # =========================================================
        if args.loss == "FocalTversky_loss":
            print("\n" + "="*40)
            print(" GIAI ĐOẠN 2: INTERMEDIATE TUNING (Full Finetune)")
            print(" Config: Heavy Augment | Alpha=0.7")
            print("="*40)
            # Đảm bảo backbone đã được mở khóa (double check)
            set_grad_status(model, freeze=False)
            trainLoader_strong, validLoader, _ = get_dataloaders(aug_mode=mode_stage23)
            
            # Update Params (Thực tế GD2 vẫn dùng 0.7, nhưng gọi lại cho chắc chắn hoặc nếu bạn muốn chỉnh khác)
            # KHÔNG CẦN gán trainer.criterion = ... vì nó đã trỏ cùng 1 vùng nhớ
            _focal_tversky_global.update_params(alpha=0.7, beta=0.3, gamma=1.33)
            
            trainer.num_epochs = 150 
            trainer.patience = 10   
            trainer.early_stop_counter = 0
            
            trainer.train(trainLoader_strong, validLoader, resume_path=resume_checkpoint)
            resume_checkpoint = "best_dice_mass_model.pth"
            print(f"[TRANSITION] Stage 2 Finished. Best model '{resume_checkpoint}' will be loaded for Stage 3.")
            if os.path.exists(resume_checkpoint):
                backup_name = "stage2_final_best.pth" # Tên file backup
                shutil.copy(resume_checkpoint, backup_name)
                print(f"[BACKUP] Safe copy created: {resume_checkpoint} -> {backup_name}")
        # ----------------------------------------
        else:
            print("\n[INFO] Skipping Stage 2 (Only for FocalTversky).")

        # =========================================================
        # GIAI ĐOẠN 3: FINAL TRAINING
        # =========================================================
        # print("\n[INFO] MANUAL RESUME: Skipping Stage 2.")
        # resume_checkpoint = "best_dice_mass_model.pth"
        # if os.path.exists(resume_checkpoint):
        #      print(f"[INFO] Found Stage 2 Checkpoint: {resume_checkpoint}. Proceeding to Stage 3.")
        # else:
        #      print(f"[ERROR] Checkpoint {resume_checkpoint} not found! Check file name.")
        #      return # Dừng chương trình nếu không thấy file
        print("\n" + "="*40)
        print(" GIAI ĐOẠN 3: FINAL TRAINING")
        # [BƯỚC 1: QUAN TRỌNG] Load Checkpoint thủ công TRƯỚC KHI chỉnh sửa bất cứ thứ gì
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"[INFO] Manually loading checkpoint for Stage 3 setup: {resume_checkpoint}")
            trainer.load_checkpoint(resume_checkpoint)
        else:
            print("[WARNING] No checkpoint found for Stage 3! Training from scratch?")
            
        set_grad_status(model, freeze=False)
        if args.loss == "FocalTversky_loss":
            # >> CHIẾN LƯỢC 3 GIAI ĐOẠN (Focal) <<
            print(" Config: Heavy Augment | Alpha=0.4 (Reduce FP) | LR REDUCED Strategy: Start Low (1e-5) -> Restart High (1e-4)")
            
            # 1. Update params "nóng"
            _focal_tversky_global.update_params(alpha=0.4, beta=0.6, gamma=1.33)
            
            # 2. Reset Best Loss (Vì scale loss thay đổi)
            trainer.best_val_loss = float('inf')
            # 3. Giảm LR
            # current_lr = trainer.optimizer.param_groups[0]['lr']
            # new_lr = current_lr * 0.1
            new_lr = 1e-5
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"[SWITCH] Switching logic from SequentialLR -> ReduceLROnPlateau for Final Stage with LR forced to {new_lr}")
            
            # # # Cập nhật "trần" cho Scheduler để các chu kỳ sau không vượt quá 1e-5
            # if hasattr(trainer.scheduler, 'base_lrs'):
            #      trainer.scheduler.base_lrs = [new_lr] * len(trainer.optimizer.param_groups)

            # print(f"[CONFIG] Scheduler continued! New Peak LR set to: {new_lr}")
            
            # 4. KHỞI TẠO LẠI SCHEDULER (Hack thời gian)
            CYCLE_START = 10
            CYCLE_ADD = 10
            # fake_last_epoch = 7  # Mẹo: Giả vờ là đã chạy được 7 epoch -> Đang ở gần đáy chu kỳ

            trainer.scheduler = ReduceLROnPlateau(
                trainer.optimizer, 
                mode='max',      # Theo dõi Dice Mass (càng cao càng tốt)
                factor=0.5,      # Giảm 1 nửa khi bão hòa
                patience=5,     # Chờ 10 epoch
                # verbose=True,
                min_lr=1e-6      # Đáy để kích hoạt reset
            )
            print(f"[CONFIG] Scheduler Reset! Mode: Arithmetic (10 -> 20 -> 30...)")            
        else:
            # >> CHIẾN LƯỢC 2 GIAI ĐOẠN (Loss khác) <<
            print(f" Config: Heavy Augment | Loss: {args.loss} | KEEP LR")
            # Không giảm LR, Không đổi params
            # Trainer vẫn giữ nguyên criterion khởi tạo từ đầu

        print("="*40)

        # Load Data Strong (Dùng chung cho cả 2 nhánh)
        trainLoader_strong, validLoader, _ = get_dataloaders(aug_mode=mode_stage23)
        
        trainer.num_epochs = NUM_EPOCHS # Max epoch
        trainer.patience = 25  # Patient 20 cho GD3         
        trainer.early_stop_counter = 0
        # Chạy GD3 đến khi Early Stop kích hoạt        
        trainer.train(trainLoader_strong, validLoader, resume_path=None) 
        print("\n[INFO] Exporting Main Training Results (Stage 1-3)...")
        export(trainer)
        # =========================================================
        # GIAI ĐOẠN 4: SWA (STOCHASTIC WEIGHT AVERAGING)
        # =========================================================
        # Chỉ chạy SWA nếu đang dùng FocalTversky (chiến lược của bạn)
        # if args.loss == "FocalTversky_loss":
        #     print("\n" + "="*40)
        #     print(" GIAI ĐOẠN 4: SWA FINETUNING (The Secret Weapon)")
        #     print(" Strategy: Constant LR | No Early Stop | 20 Epochs")
        #     print("="*40)

        #     # 1. QUAN TRỌNG: Load lại BEST MODEL của GD3 (Không dùng model cuối cùng)
        #     # best_model_path = "best_dice_mass_model.pth"
        #     best_ep = trainer.best_epoch_dice
        #     best_d = trainer.best_dice_mass
        #     folder_name = f"output_epoch{best_ep}_diceMass{best_d:.4f}"
        #     exported_best_model_path = os.path.join(BASE_OUTPUT, folder_name, "best_dice_mass_model.pth")
        #     if os.path.exists(exported_best_model_path):
        #         print(f"[INFO] Loading BEST model from Stage 3 for SWA: {exported_best_model_path}")
        #         trainer.load_checkpoint(exported_best_model_path)
        #     else:
        #         print("[WARNING] Could not find exported best model. Trying local 'best_dice_mass_model.pth'...")
        #         if os.path.exists("best_dice_mass_model.pth"):
        #             trainer.load_checkpoint("best_dice_mass_model.pth")

        #     # 2. Khởi tạo SWA
        #     swa_model = AveragedModel(trainer.model)
        #     # LR cho SWA: Cao hơn GD3 một chút để thoát hố (5e-5 là an toàn với AdamW)
        #     swa_lr = 5e-5 
        #     swa_scheduler = SWALR(trainer.optimizer, swa_lr=swa_lr, anneal_epochs=3)
            
        #     print(f"[CONFIG] SWA Scheduler set. LR: {swa_lr}")

        #     # 3. Cấu hình vòng lặp SWA
        #     SWA_EPOCHS = 5 # Chạy cố định
        #     trainer.patience = 999 # Tắt Early Stop
        #     trainer.early_stop_counter = 0
            
        #     # Chúng ta sẽ dùng lại hàm train() của Trainer nhưng chạy từng epoch một
        #     # để chèn logic update_parameters vào giữa.
            
        #     print("[INFO] Starting SWA Loop...")
        #     for epoch in range(SWA_EPOCHS):
        #         # Hack: Set epoch = 1 để Trainer chạy 1 vòng rồi thoát ra
        #         trainer.num_epochs = 1 
        #         trainer.start_epoch = 0 
        #         # Gán scheduler SWA vào trainer
        #         trainer.scheduler = swa_scheduler
                
        #         # Train 1 epoch (Không load checkpoint, chạy tiếp từ bộ nhớ)
        #         # Lưu ý: Trainer sẽ in ra log validation, cứ kệ nó.
        #         print(f"\n[SWA] Epoch {epoch+1}/{SWA_EPOCHS}")
        #         trainer.train(trainLoader_strong, validLoader, resume_path=None)
                
        #         # Cập nhật trọng số trung bình
        #         swa_model.update_parameters(trainer.model)
                
        #         # Step Scheduler
        #         swa_scheduler.step()
                
        #     # 4. Cập nhật Batch Norm (Bước bắt buộc)
        #     print("\n[INFO] Updating Batch Normalization statistics for SWA Model...")
        #     update_bn(trainLoader_strong, swa_model, device=DEVICE)

        #     # 5. Lưu và Đánh giá SWA Model
        #     swa_save_path = os.path.join(BASE_OUTPUT, "best_model_swa.pth")
        #     print(f"[INFO] Saving SWA Model to {swa_save_path}")
        #     swa_checkpoint = {
        #         'epoch': SWA_EPOCHS,
        #         'model_state_dict': swa_model.state_dict(),         # <--- Đã sửa để khớp tên layer
        #         'optimizer_state_dict': trainer.optimizer.state_dict(), # Để không lỗi optimizer
                
        #         # Các chỉ số thống kê (Lấy từ trainer hiện tại để lưu làm kỷ niệm)
        #         'best_val_loss': trainer.best_val_loss, 
        #         'best_dice_mass': trainer.best_dice_mass,
        #         'best_iou_mass': trainer.best_iou_mass,
        #         # 'history': trainer.history,
                
        #         # QUAN TRỌNG: KHÔNG ĐƯỢC THÊM 'scheduler_state_dict' VÀO ĐÂY
        #         # Nếu thêm 'scheduler_state_dict': None -> Sẽ bị lỗi NoneType crash ngay.
        #     }
        #     torch.save(swa_checkpoint, swa_save_path)
        #     # export(trainer)
        #     # Đánh giá Model SWA
        #     print("\n[INFO] Evaluating SWA Model...")
        #     # Gán model SWA vào trainer để evaluate
        #     trainer.model = swa_model
            
        #     visual_folder = os.path.join(BASE_OUTPUT, "prediction_images_swa")
        #     os.makedirs(visual_folder, exist_ok=True)
            
        #     trainer.evaluate(
        #         test_loader=validLoader, 
        #         checkpoint_path=swa_save_path,
        #         save_visuals=True,          
        #         output_dir=visual_folder    
        #     )
        #     export_evaluate(trainer, split_name="valid_swa")
            
    # (Giữ nguyên phần pretrain/evaluate)
    elif args.mode == "pretrain":
        aug_type = 'strong' if args.augment else 'none'
        trainLoader, validLoader, _ = get_dataloaders(aug_mode=aug_type)
        trainer.patience = 20
        trainer.train(trainLoader, validLoader, resume_path=args.checkpoint)
        export(trainer)
    elif args.mode == "evaluate":
        print(f"[INFO] Mode: EVALUATING FULL DATASET")
        
        trainLoader, validLoader, testLoader = get_dataloaders(aug_mode='none', state='evaluate')
        
        eval_tasks = [
            # (trainLoader, "train"),
            (validLoader, "valid"),
            (testLoader, "test")
        ]
        
        for loader, split_name in eval_tasks:
            print(f"\n" + "="*40)
            print(f" [EVALUATING] Processing: {split_name.upper()} SET")
            print("="*40)
            
            visual_folder = os.path.join(BASE_OUTPUT, f"prediction_images_{split_name}")
            if not os.path.exists(visual_folder):
                os.makedirs(visual_folder)
            
            trainer.evaluate(
                test_loader=loader, 
                checkpoint_path=args.checkpoint,
                save_visuals=True,          
                output_dir=visual_folder    
            )
            
            # --- GỌI HÀM VỚI THAM SỐ MỚI ---
            print(f"[INFO] Exporting metrics for {split_name}...")
            export_evaluate(trainer, split_name=split_name)

if __name__ == "__main__":
    args = get_args()
    main(args)
