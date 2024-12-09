import os
import random
import argparse
import warnings
import numpy as np
from functools import partial

import torch
from torch.backends import cudnn
import torch.nn as nn

from dataset import GetDataList, get_loader
from loss import DiceCELoss
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR, PolynomialLRDecay
from utils import Dice
from train import Train
from network_architecture.Ablation.TransFusion import MultiScaleCascadeTrans
from monai.inferers import sliding_window_inference
from monai.metrics import HausdorffDistanceMetric


parser = argparse.ArgumentParser(description='LAA segmentation')

parser.add_argument('--data_path', default='/public2/wumusheng/segmentation/')
parser.add_argument('--data_list', default='./DataList.txt')
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--amp', default=True, type=bool, help='whether to use amp for training')
parser.add_argument('--optimizer', default='Adam', type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_schedule', default='cosine_anneal', type=str, help='type of learning rate scheduler')
parser.add_argument('--warmup_epochs', default=30, type=int, help='number of warmup epochs')
parser.add_argument('--momentum', default=0.99, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--log_dir', default=r'./logs', type=str)
parser.add_argument('--log_dir_name', default='TransHRNet_v2', type=str, help='logs saving dir name')
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--roi", default=(128, 128, 128), type=tuple, help="roi size")
parser.add_argument('--model_infer', default=True, type=bool, help='whether to use sliding window inference in Validation')
parser.add_argument('--sw_batch_size', default=2, type=int, help='number of sliding window batch size')
parser.add_argument('--infer_window_size', default=(128, 128, 128), type=tuple, help='the size of infer window size')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--save_period', default=50, type=int, help='save checkpoint every period')
parser.add_argument('--resume', default=False, help="resume training from saved checkpoint")
parser.add_argument('--pretrained_path', default=r'./pretrained_model/', type=str, help="pretrained checkpoint directory")
parser.add_argument('--pretrained_model_name', default=r'best_acc_model.pt', type=str, help="pretrained model name")
parser.add_argument("--test-mode", default=False, type=bool)
parser.add_argument("--seed", default=12345, type=int)
parser.add_argument("--workers", default=4, type=int)


def init_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    init_random(12345)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using {} device.'.format(device))

    GetDataList(args.data_path)

    Train_Loader, Val_Loader = get_loader(args)
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    torch.multiprocessing.set_sharing_strategy('file_system')
    model = MultiScaleCascadeTrans(in_channels=1,
                                   num_classes=1,
                                   patch_size=(4, 4, 4),
                                   window_size=(7, 7, 7),
                                   depths=[1, 1, 1, 1]).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay,
                                     amsgrad=True)
    else:
        raise KeyError('The optimizer type is wrong!!!')

    DCE = DiceCELoss().to(device)

    if args.model_infer:
        model_infer = partial(sliding_window_inference,
                              roi_size=args.infer_window_size,
                              sw_batch_size=args.sw_batch_size,
                              predictor=model,
                              overlap=args.infer_overlap,
                              mode='gaussian')
    else:
        model_infer = None

    # Scheduler
    if args.lr_schedule == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.warmup_epochs,
                                                  max_epochs=args.epochs)
    elif args.lr_schedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)
    elif args.lr_schedule == 'poly':
        scheduler = PolynomialLRDecay(optimizer, max_decay_steps=args.epochs, power=0.9)
    else:
        scheduler = None

    hd = HausdorffDistanceMetric(include_background=False,
                                 distance_metric='euclidean',
                                 percentile=95.,
                                 reduction='mean')

    start_epoch = 0
    best_acc = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.pretrained_path, args.pretrained_model_name), map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("Resume training from pretrained weights !!!\n")
        print("=> loaded checkpoint '{}' (epoch {}) (best acc {})\n".format(args.checkpoint, start_epoch, best_acc))
        print("start train from epoch = {}".format(start_epoch))

    acc_func = [
        Dice,
        hd
    ]

    accuracy = Train(args=args,
                     model=model,
                     device=device,
                     train_loader=Train_Loader,
                     val_loader=Val_Loader,
                     loss_func=DCE,
                     acc_func=acc_func,
                     optimizer=optimizer,
                     scheduler=scheduler,
                     model_inferer=model_infer,
                     start_epoch=start_epoch)

    return accuracy


if __name__ == '__main__':
    main()
