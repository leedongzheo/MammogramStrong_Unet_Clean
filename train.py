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
def main(args):  
    print(f"\n[DEBUG TRAIN] args.loss bạn nhập từ bàn phím = {args.loss}")
    print("-" * 50)
    import numpy as np    
    from trainer import Trainer
    # from model import Unet, unet_pyramid_cbam_gate, Swin_unet
    from model import Swin_unet
    import optimizer as optimizer_module
    from dataset import get_dataloaders
    from result import export, export_evaluate
    global trainer
    from utils import get_loss_instance, _focal_tversky_global


    # from utils import loss_func

    print("-" * 50)
    print(f"[INFO] Mode: {args.mode.upper()}")
    print("-" * 50)

    set_seed()
    
    # 1. Khởi tạo Model
    print(f"[INFO] Initializing Model...")
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b2", 
        encoder_weights="imagenet",     
        in_channels=3, 
        classes=1, 
        decoder_attention_type="scse"   
    )
    
    # 2. Khởi tạo Optimizer
    opt = optimizer_module.optimizer(model=model) 

    # 3. KHỞI TẠO LOSS (Thay thế hàm get_loss_function)
    # Logic: Nếu chọn FocalTversky thì lấy biến toàn cục, còn lại thì khởi tạo class
    criterion_init = get_loss_instance(args.loss)

    # 4. Khởi tạo Trainer
    # Lưu ý: Trainer lưu reference tới criterion_init. 
    # Nếu criterion_init là _focal_tversky_loss, mọi thay đổi trên _focal_tversky_loss sẽ tự động cập nhật trong Trainer.
    trainer = Trainer(model=model, optimizer=opt, criterion=criterion_init, patience=10, device=DEVICE)

    if args.mode == "train":
        if not os.path.exists(BASE_OUTPUT):
            os.makedirs(BASE_OUTPUT)
        if args.augment:
            mode_stage1 = 'weak'
            mode_stage23 = 'strong'
            print(f"[INFO] Augmentation: ON (Stage 1: {mode_stage1} -> Stage 2/3: {mode_stage23})")
        else:
            mode_stage1 = 'none'
            mode_stage23 = 'none'
            print(f"[INFO] Augmentation: OFF (All Stages: none)")
        if args.augment and args.warmup > 0:

            # =========================================================
            # GIAI ĐOẠN 1: WARM-UP
            # =========================================================
            print("\n" + "="*40)
            print(" GIAI ĐOẠN 1: WARM-UP (10 Epochs)")
            print(f" Config: Light Augment | Loss: {args.loss}")
            print("="*40)

            trainLoader_weak, validLoader, _ = get_dataloaders(aug_mode=mode_stage1)
            
            # Đảm bảo params đúng cho GD1 (nếu dùng Focal)
            if args.loss == "FocalTversky_loss":
                _focal_tversky_global.update_params(alpha=0.7, beta=0.3, gamma=1.33)

            trainer.num_epochs = args.warmup
            trainer.patience = 999      
            trainer.train(trainLoader_weak, validLoader, resume_path=None)
            
            resume_checkpoint = "last_model.pth" 
        else:
            print("\n[INFO] Skipping Stage 1 (Warm-up). Starting directly with Main Training.")
            resume_checkpoint = None
        # =========================================================
        # GIAI ĐOẠN 2: INTERMEDIATE TUNING (Chỉ FocalTversky)
        # =========================================================
        if args.loss == "FocalTversky_loss":
            print("\n" + "="*40)
            print(" GIAI ĐOẠN 2: INTERMEDIATE TUNING")
            print(" Config: Heavy Augment | Alpha=0.7")
            print("="*40)

            trainLoader_strong, validLoader, _ = get_dataloaders(aug_mode=mode_stage23)
            
            # Update Params (Thực tế GD2 vẫn dùng 0.7, nhưng gọi lại cho chắc chắn hoặc nếu bạn muốn chỉnh khác)
            # KHÔNG CẦN gán trainer.criterion = ... vì nó đã trỏ cùng 1 vùng nhớ
            _focal_tversky_global.update_params(alpha=0.7, beta=0.3, gamma=1.33)
            
            trainer.num_epochs = 60 
            trainer.patience = 10   
            trainer.early_stop_counter = 0

            trainer.train(trainLoader_strong, validLoader, resume_path=resume_checkpoint)
            resume_checkpoint = "last_model.pth"
        else:
            print("\n[INFO] Skipping Stage 2 (Only for FocalTversky).")

        # =========================================================
        # GIAI ĐOẠN 3: FINAL TRAINING
        # =========================================================
        print("\n" + "="*40)
        print(" GIAI ĐOẠN 3: FINAL TRAINING")
        # [BƯỚC 1: QUAN TRỌNG] Load Checkpoint thủ công TRƯỚC KHI chỉnh sửa bất cứ thứ gì
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"[INFO] Manually loading checkpoint for Stage 3 setup: {resume_checkpoint}")
            trainer.load_checkpoint(resume_checkpoint)
        else:
            print("[WARNING] No checkpoint found for Stage 3! Training from scratch?")
            
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
            print(f"[CONFIG] Optimizer LR forced to: {new_lr}")
            
            # # # Cập nhật "trần" cho Scheduler để các chu kỳ sau không vượt quá 1e-5
            # if hasattr(trainer.scheduler, 'base_lrs'):
            #      trainer.scheduler.base_lrs = [new_lr] * len(trainer.optimizer.param_groups)

            # print(f"[CONFIG] Scheduler continued! New Peak LR set to: {new_lr}")
            
            # 4. KHỞI TẠO LẠI SCHEDULER (Hack thời gian)
            CYCLE_LEN = 10
            fake_last_epoch = 7  # Mẹo: Giả vờ là đã chạy được 7 epoch -> Đang ở gần đáy chu kỳ

            trainer.scheduler = CosineAnnealingWarmRestarts(
                trainer.optimizer, 
                T_0=CYCLE_LEN, 
                T_mult=1, 
                eta_min=1e-6, 
                last_epoch=-1  # <--- Không cần hack nữa vì Epoch 37 tự khớp rồi
            )
            target_high_lr = 1e-4
            if hasattr(trainer.scheduler, 'base_lrs'):
                 trainer.scheduler.base_lrs = [target_high_lr] * len(trainer.optimizer.param_groups)

            [QUAN TRỌNG] Reset scheduler về step 0 để bắt đầu chu kỳ mới mượt mà
            trainer.scheduler.last_epoch = -1
            print(f"[CONFIG] Scheduler Reset! Current LR ~{new_lr}. Next Restart Peak: {target_high_lr}")
            
        else:
            # >> CHIẾN LƯỢC 2 GIAI ĐOẠN (Loss khác) <<
            print(f" Config: Heavy Augment | Loss: {args.loss} | KEEP LR")
            # Không giảm LR, Không đổi params
            # Trainer vẫn giữ nguyên criterion khởi tạo từ đầu

        print("="*40)

        # Load Data Strong (Dùng chung cho cả 2 nhánh)
        trainLoader_strong, validLoader, _ = get_dataloaders(aug_mode=mode_stage23)
        
        trainer.num_epochs = NUM_EPOCHS # Max epoch
        trainer.patience = 20           
        trainer.early_stop_counter = 0

        trainer.train(trainLoader_strong, validLoader, resume_path=None) 
        export(trainer)

    # (Giữ nguyên phần pretrain/evaluate)
    elif args.mode == "pretrain":
        aug_type = 'strong' if args.augment else 'none'
        trainLoader, validLoader, _ = get_dataloaders(aug_mode=aug_type)
        trainer.patience = 20
        trainer.train(trainLoader, validLoader, resume_path=args.checkpoint)
        export(trainer)
    elif args.mode == "evaluate":
        print(f"[INFO] Mode: EVALUATING")
        _, validLoader, _ = get_dataloaders(aug_mode='none')
        visual_folder = os.path.join(BASE_OUTPUT, "prediction_images")
        trainer.evaluate(
            test_loader=validLoader, 
            checkpoint_path=args.checkpoint,
            save_visuals=True,          # <--- Bật chế độ lưu ảnh
            output_dir=visual_folder    # <--- Truyền đường dẫn lưu
        )
        export_evaluate(trainer)

if __name__ == "__main__":
    args = get_args()
    main(args)
