import os
import pandas as pd
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'          ## setting the environment
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0,1'                           ## using GPU core 0 and 1


def warn(*args, **kwargs):                                          ## Hiding the warnings
    pass
import warnings
warnings.warn = warn

import torch
import numpy as np
from scipy import linalg
from utils import load_checkpoint, save_val_examples
import torch.nn as nn
import torch.optim as optim
from dataset import DeblurData
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
import torchvision.models as models
from tqdm import tqdm
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_activation_statistics(images, model):
    model.eval()
    with torch.no_grad():
        activations = model(images).view(images.shape[0], -1)
        mu = torch.mean(activations, dim=0)
        # sigma = torch.inverse(torch.diag(torch.var(activations, dim=0, unbiased=False).sqrt()))
        sigma = torch.inverse(torch.diag(torch.var(activations, dim=0, unbiased=False).sqrt() + 1e-6)) #adding a small term so that the matrix is not singular

    return mu, sigma

def calculate_fid_score(images1, images2, model):
    model.eval()
    images1 = F.resize(images1, [299, 299])
    images2 = F.resize(images2, [299, 299])
    images1 = images1.to(DEVICE)
    images2 = images2.to(DEVICE)
    images1 = (images1 + 1) * 127.5
    images2 = (images2 + 1) * 127.5
    mu1, sigma1 = calculate_activation_statistics(images1, model)
    mu2, sigma2 = calculate_activation_statistics(images2, model)
    ssdiff = torch.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.cpu() @ sigma2.cpu())
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_score = ssdiff + torch.trace(sigma1.cpu() + sigma2.cpu() - 2.0 * covmean)
    return fid_score.item()

def main():
    disc = Discriminator().to(DEVICE)
    disc = nn.DataParallel(disc)
    gen = Generator().to(DEVICE)
    gen = nn.DataParallel(gen)

    opt_disc = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))

    load_checkpoint("checkpoints/gen/gen.pth_10.tar", gen, opt_gen, 2e-4)
    load_checkpoint("checkpoints/disc/disc_10.pth.tar", disc, opt_disc, 2e-4)

    inception = models.inception_v3(pretrained=True, aux_logits=True)
    inception.fc = nn.Identity()
    inception.to(DEVICE)

    val_dataset = DeblurData(path="test_gopro")          
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    val_loop = tqdm(val_loader, leave=True)
    psnr_list = []
    ssim_list = []
    ms_ssim_list = []
    f_sim_list = []
    vif_list = []
    fid_list = []

    for idx, (Image_blur, Image_sharp) in enumerate(val_loop):
        psnr, ssim, ms_ssim, f_sim, vif =  save_val_examples(Image_blur,Image_sharp,gen,idx,folder="results/val")
        fid_score = calculate_fid_score(Image_blur, Image_sharp, inception)
        fid_list.append(fid_score)
        psnr_list.append(psnr), ssim_list.append(ssim),ms_ssim_list.append(ms_ssim), f_sim_list.append(f_sim), vif_list.append(vif)
    df = pd.DataFrame({'PSNR': psnr_list,'SSIM': ssim_list,'MS-SSIM': ms_ssim_list,'F_SIM': f_sim_list,'VIF': vif_list, 'FID': fid_list})
    df.to_csv('image_metrics.csv')


if __name__ == "__main__":
    main()