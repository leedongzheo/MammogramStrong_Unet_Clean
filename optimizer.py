from config import*
# from train import get_args
# import model
# def optimizer():
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     return optimizer

def optimizer(model):
    # args = get_args()
    if optim == "Adam":
        # optimizer = Adam(model.parameters(), lr=lr0, weight_decay=weight_decay)
        optimizer = Adam(
            model.parameters(),
            lr = lr0,
            betas = BETA,
            weight_decay = weight_decay1,
            amsgrad = AMSGRAD  # nếu bạn muốn bật AMSGrad giống như bài báo có đề cập
        )
        return optimizer
    if optim == "AdamW":
        # optimizer = Adam(model.parameters(), lr=lr0, weight_decay=weight_decay)
        optimizer = AdamW(
            model.parameters(),
            lr = lr0,
            betas = BETA,
            weight_decay = weight_decay2,
            amsgrad = AMSGRAD  # nếu bạn muốn bật AMSGrad giống như bài báo có đề cập
        )
        return optimizer
    elif optim == "SGD":
        optimizer = SGD(model.parameters(), lr=lr0, weight_decay=weight_decay1) 
        return optimizer
