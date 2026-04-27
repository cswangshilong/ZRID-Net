import argparse
import sys
import torchvision
import itertools
import warnings
import torch
import torch.backends.cudnn as cudnn
from torch import optim, nn
from torch.utils.data import DataLoader
from data import utils
from data.dataloader import TrainDataloader
from networks import DCP,RJNet,RTNet,PF,GF
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from losses import cap_loss,VGG19CR

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--hazy_dir', type=str, default='./datasets/reals/haze/',help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
parser.add_argument('--CLAHE_dir', type=str, default='./datasets/reals/CLAHE/',help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
parser.add_argument('--category', type=str, default='real',help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
parser.add_argument('--save_model_dir', default='./checkpoints/',help='Directory to save the model')
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--decay_epoch', type=int, default=50)
parser.add_argument('--start_epoch', type=float, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lambda_cap', type=float, default=1.0)
parser.add_argument('--lambda_rec', type=float, default=10.0)
parser.add_argument('--lambda_cr', type=float, default=0.1)
parser.add_argument('--lambda_rec2', type=float, default=1.0)
parser.add_argument('--lambda_rkt', type=float, default=0.1)
args = parser.parse_args('')

if __name__ == "__main__":

    transforms_train = [
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    train_dataset = TrainDataloader(args.hazy_dir, args.CLAHE_dir, is_rotate=False, transform=transforms_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    dataset_length_train = len(train_loader)

    logger_train = utils.Logger(args.max_epoch,dataset_length_train)

    DCP = DCP.DCPDehazeGenerator().to(device) # No learnable parameters
    Refined_T = GF.Refined_T().to(device) # No learnable parameters

    JNet = RJNet.JNet().to(device)
    TNet = RTNet.TNet().to(device)

    print('The networks are instantiated successfully！')

    total_params = sum(p.numel() for p in JNet.parameters() if p.requires_grad) + sum(p.numel() for p in TNet.parameters() if p.requires_grad)

    print("Total_params: ==> {}".format(total_params))

    opt_JT = optim.Adam(itertools.chain(JNet.parameters(), TNet.parameters()), lr=args.lr, betas=(0.9, 0.999))

    lr_scheduler_JT = torch.optim.lr_scheduler.LambdaLR(opt_JT, lr_lambda=utils.LambdaLR(args.max_epoch, args.start_epoch, args.decay_epoch).step)

    loss_mse = nn.MSELoss().to(device)
    loss_cr = VGG19CR.ContrastLoss().to(device)
    loss_rkt = RJNet.RKTLoss(t_path="./checkpoints/T_models/{}_T.pth".format(args.category)).to(device)

    test_time = 0

    JNet.train()
    TNet.train()

    for epoch in range(args.start_epoch, args.max_epoch + 1):

        for i, batch in enumerate(train_loader):

            I = batch[0].to(device)   # hazy images
            CLAHE = batch[1].to(device)  # CLAHE images
            image_name = batch[2][0]

            J_DCP, T_DCP, A_DCP = DCP(I)

            T_Coarse = TNet(torch.cat([T_DCP, I], dim=1))

            T_Refined = Refined_T(I, T_Coarse)

            J_Refined, res_g3w = JNet(J_DCP)

            J_ASM = utils.reverse_fog_asm(I, T_Refined, A_DCP)

            I_Rec = utils.synthesize_fog(J_Refined, T_Refined, A_DCP)

            J_Fusion = PF.Perceptual_Fusion(I, J_ASM, J_Refined).to(device)

            ###################### Loss of JNet and TNet ###############
            loss_Rec = loss_mse(I_Rec, I) * args.lambda_rec  # Reconstruction loss

            ###################### Loss of JNet #############################
            loss_CAP = cap_loss.caploss(J_Refined) * args.lambda_cap  # CAP loss

            loss_CR = loss_cr(J_Refined, CLAHE, I) * args.lambda_cr  # Unsupervised comparative loss

            loss_Rec2 = loss_mse(J_Refined, CLAHE) * args.lambda_rec2  # Reconstruction loss

            loss_RKT = loss_rkt(res_g3w, I) * args.lambda_rkt

            ###################### Total Loss #############################
            loss = loss_Rec + loss_CAP + loss_CR + loss_Rec2 + loss_RKT

            opt_JT.zero_grad()
            loss.backward()
            opt_JT.step()

            logger_train.log_train({},images={'I': I, 'Rec': I_Rec,'Refined_T': T_Refined,'ASM': J_ASM, 'J_Refined': J_Refined, 'J_Fusion': J_Fusion})

            sys.stdout.write(
                '\rEpoch %03d/%03d [%04d/%04d] -- Loss %.6f' % (
                epoch, args.max_epoch, i + 1, dataset_length_train, loss.item()))

        torch.save(TNet.state_dict(), args.save_model_dir + args.category + "_TNet.pth")
        torch.save(JNet.state_dict(), args.save_model_dir + args.category + "_JNet.pth")

        lr_scheduler_JT.step()
















