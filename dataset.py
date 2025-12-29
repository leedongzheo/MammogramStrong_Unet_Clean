from config import*

class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)
	def __getitem__(self, idx):
		imagePath = self.imagePaths[idx]
		maskPath = self.maskPaths[idx]
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE) # với cv2.IMREAD_GRAYSCALE = 0
		if self.transforms:
			# grab the image path from the current index
			augmented = self.transforms(image=image, mask=mask)
			image = augmented["image"]
			mask = augmented["mask"]
			# print("shape_mask1: ", mask.shape)
			mask = (mask > 127).float() 
			# mask = (mask > 127).astype("float32")        # chuyển về float32: giá trị 0.0 hoặc 1.0
			# mask = torch.from_numpy(mask)  
			mask = mask.unsqueeze(0)                     # shape (1, H, W)
		return image, mask, imagePath	
def seed_worker(worker_id):
		np.random.seed(SEED + worker_id)
		random.seed(SEED + worker_id)			
def get_dataloaders(aug_mode='none'):    
	# 1. Định nghĩa Transform dựa trên mode
	if aug_mode == 'strong':
		print("[INFO] Loading STRONG Augmentation (Elastic, Distortions...)")
		train_transform = A.Compose([
            A.Resize(height=INPUT_IMAGE_WIDTH, width=INPUT_IMAGE_WIDTH, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            # Các biến đổi mạnh
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.ElasticTransform(alpha=0.2*512, sigma=0.08*512, alpha_affine=0.04*512, border_mode=cv2.BORDER_REFLECT_101, p=0.3),
            A.GridDistortion(border_mode=cv2.BORDER_REFLECT_101, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
	elif aug_mode == 'weak':
		print("[INFO] Loading WEAK Augmentation (Flip only - for Warmup)")
		train_transform = A.Compose([
            A.Resize(height=INPUT_IMAGE_WIDTH, width=INPUT_IMAGE_WIDTH, interpolation=cv2.INTER_LINEAR),
            # Chỉ lật nhẹ nhàng
            A.HorizontalFlip(p=0.5), 
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
	else:
		print("[INFO] Not using AUGMENTATION")
		train_transform = A.Compose([
            A.Resize(
	        height=INPUT_IMAGE_WIDTH,
	        width=INPUT_IMAGE_WIDTH,
	        interpolation=cv2.INTER_LINEAR,          # cho ảnh
	        # mask_interpolation=cv2.INTER_NEAREST     # cho mask
    ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

	valid_transform = A.Compose([
	A.Resize(
	        height=INPUT_IMAGE_WIDTH,
	        width=INPUT_IMAGE_WIDTH,
	        interpolation=cv2.INTER_LINEAR,          # cho ảnh
	        # mask_interpolation=cv2.INTER_NEAREST     # cho mask
    ),
        # A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
	    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    
	# g = torch.Generator()
	# g.manual_seed(SEED)

	trainImagesPaths = sorted(list(paths.list_images(IMAGE_TRAIN_PATH)))
	trainMasksPaths = sorted(list(paths.list_images(MASK_TRAIN_PATH)))

	validImagesPaths = sorted(list(paths.list_images(IMAGE_VALID_PATH)))
	validMasksPaths = sorted(list(paths.list_images(MASK_VALID_PATH)))

	testImagesPaths = sorted(list(paths.list_images(IMAGE_TEST_PATH)))
	testMasksPaths = sorted(list(paths.list_images(MASK_TEST_PATH)))

	trainDS = SegmentationDataset(trainImagesPaths, trainMasksPaths, transforms=train_transform)
	validDS = SegmentationDataset(validImagesPaths, validMasksPaths, transforms=valid_transform)
	testDS = SegmentationDataset(testImagesPaths, testMasksPaths, transforms=valid_transform)
	print(f"[INFO] found {len(trainDS)} examples in the training set...")
	print(f"[INFO] found {len(validDS)} examples in the valid set...")
	print(f"[INFO] found {len(testDS)} examples in the test set...")
	# --- 3. XỬ LÝ BALANCED SAMPLING (QUAN TRỌNG) ---
	print("[INFO] Scanning training masks to compute class weights (This may take a while)...")
	train_targets = []
    
    # Duyệt qua tất cả mask train để xem cái nào là Mass (1), cái nào là Normal (0)
    # Dùng tqdm để hiển thị thanh loading vì bước này đọc ổ cứng hơi lâu
	for maskPath in tqdm(trainMasksPaths, desc="Scanning Masks"):
        # Đọc chế độ grayscale (0) cho nhanh
		mask = cv2.imread(maskPath, 0) 
        # Nếu có pixel nào > 0 thì là Mass (Class 1), ngược lại là Normal (Class 0)
		if cv2.countNonZero(mask) > 0: 
			train_targets.append(1) # Mass
		else:
			train_targets.append(0) # Normal
    
	train_targets = torch.tensor(train_targets)
    
    # Đếm số lượng mỗi loại
	class_counts = torch.bincount(train_targets)
	num_normal = class_counts[0].item()
	num_mass = class_counts[1].item() if len(class_counts) > 1 else 0
    
	print(f"[INFO] Class distribution - Normal: {num_normal}, Mass: {num_mass}")
    
    # Tính trọng số cho từng Class (Class ít thì trọng số cao)
    # Tránh lỗi chia cho 0
	weight_normal = 1. / num_normal if num_normal > 0 else 0
	weight_mass = 1. / num_mass if num_mass > 0 else 0
	class_weights = torch.tensor([weight_normal, weight_mass])
    
    # Gán trọng số cho từng mẫu (Sample Weights)
	samples_weights = class_weights[train_targets]
    
    # Tạo Sampler
    # replacement=True: Cho phép bốc lặp lại mẫu (quan trọng để cân bằng)
	sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    
    # --- 4. TẠO DATALOADERS ---
	g = torch.Generator()
	g.manual_seed(SEED)

    # Train Loader: Dùng Sampler thì shuffle PHẢI LÀ FALSE
	trainLoader = DataLoader(
        trainDS, 
        batch_size=batch_size, 
        sampler=sampler,        # <--- Dùng sampler ở đây
        shuffle=False,          # <--- Bắt buộc tắt shuffle
        pin_memory=PIN_MEMORY,
        num_workers=4, 
        worker_init_fn=seed_worker, 
        generator=g
    )

    # Valid và Test Loader: Không cần Sampler, giữ nguyên
	validLoader = DataLoader(
        validDS, 
        shuffle=False, 
        batch_size=batch_size*2, 
        pin_memory=PIN_MEMORY,
        num_workers=4, 
        worker_init_fn=seed_worker, 
        generator=g
    )
    
	testLoader = DataLoader(
        testDS, 
        shuffle=False,
        batch_size=batch_size*2, 
        pin_memory=PIN_MEMORY,
        num_workers=4, 
        worker_init_fn=seed_worker, 
        generator=g
    )	
	return trainLoader, validLoader, testLoader
