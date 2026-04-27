import argparse
import sys
import time
import numpy as np
from data.metrics import psnr,ssim
import warnings
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils
from torch import optim, nn
from torch.utils.data import DataLoader
from data import utils
from data.dataloader import TrainDataloader
from networks import RJNet
from losses import VGG19PerceptualLoss
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from torchvision.utils import save_image

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--hazy_dir', type=str, default='./datasets/reals/',help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
parser.add_argument('--category', type=str, default='real',help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
parser.add_argument('--save_model_dir', default='./checkpoints/T_models/',help='Directory to save the model')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--decay_epoch', type=int, default=50)
parser.add_argument('--start_epoch', type=float, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--l1_weight', type=float, default=1.0)
parser.add_argument('--perceptual_weight', type=float, default=0.1)
args = parser.parse_args('')

if __name__ == "__main__":

    transforms_train = [
        transforms.ToTensor()  # range [0, 255] -> [0.0,1.0]
    ]

    train_dataset = TrainDataloader(args.hazy_dir,args.hazy_dir, is_rotate=True, transform=transforms_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    dataset_length = len(train_loader)

    # logger_train=utils.Logger(args.max_epoch,dataset_length)

    T = RJNet.JNet().to(device)

    print('The models are initialized successfully!')

    T.train()

    total_params = sum(p.numel() for p in T.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(total_params))

    opt_T = optim.Adam(T.parameters(), lr=args.lr, betas=(0.9, 0.999))

    lr_scheduler_T = torch.optim.lr_scheduler.LambdaLR(opt_T, lr_lambda=utils.LambdaLR(args.max_epoch, args.start_epoch, args.decay_epoch).step)

    loss_l1 = nn.L1Loss().to(device)   # L1 loss
    loss_per = VGG19PerceptualLoss.PerceptualLoss().to(device)   # perceptual loss

    max_ssim = 0
    max_psnr = 0
    all_ssims = []
    all_psnrs = []

    for epoch in range(args.start_epoch, args.max_epoch + 1):

        ssims = []  # 每轮清空
        psnrs = []  # 每轮清空

        start_time = time.time()

        for i, batch in enumerate(train_loader):

            x = batch[0].to(device)   # clear images

            output, _ = T(x)

            loss_L1 = loss_l1(output, x) * args.l1_weight
            loss_Per = loss_per(output, x) * args.perceptual_weight

            loss = loss_L1 + loss_Per

            opt_T.zero_grad()
            loss.backward()
            opt_T.step()


            psnr1 = psnr(output, x)
            ssim1 = ssim(output, x).item()

            psnrs.append(psnr1)
            ssims.append(ssim1)

            # logger_train.log_train({},images={'input': x, 'Output': output})

            sys.stdout.write(
                '\rEpoch %03d/%03d [%04d/%04d] -- Loss %.6f --Max_PSNR：%.6f --Max_SSIM：%.6f' % (
                epoch, args.max_epoch, i + 1, dataset_length, loss.item(), max_psnr, max_ssim))

        one_epoch_time = time.time() - start_time
        psnr_eval = np.mean(psnrs)
        ssim_eval = np.mean(ssims)

        if psnr_eval > max_psnr:

            max_psnr = max(max_psnr, psnr_eval)
            max_ssim = ssim_eval

            torch.save(T.state_dict(), args.save_model_dir + args.category + "_T.pth")

        lr_scheduler_T.step()

        utils.print_log(epoch,args.max_epoch,one_epoch_time=one_epoch_time,val_psnr=psnr_eval,val_ssim=ssim_eval)









