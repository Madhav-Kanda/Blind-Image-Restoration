# Reference: https://www.youtube.com/watch?v=SuddDSqGRzg&list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va&index=7

import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'          ## setting the environment
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0,1'                           ## using GPU core 0 and 1


def warn(*args, **kwargs):                                          ## Hiding the warnings
    pass
import warnings
warnings.warn = warn

import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples, lossplot, save_val_examples
import torch.nn as nn
import torch.optim as optim
import lpips
from dataset import DeblurData
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from diff_augment import DiffAugment
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## Config variables
LOAD_MODEL = False
SAVE_MODEL = True
NUM_EPOCHS = 512

diff_augment_policies = "color,translation"


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, lpips_loss , bce, g_scaler, d_scaler,epoch, epochs):

    for idx, (Image_blur, Image_sharp) in enumerate(loader):            ## Iterating over all the images in the train loader
        Image_blur = Image_blur.to(DEVICE)       
        Image_sharp = Image_sharp.to(DEVICE)       

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(Image_blur)
            
            y_fake_aug      = DiffAugment(y_fake, policy=diff_augment_policies)
            Image_sharp_aug = DiffAugment(Image_sharp, policy=diff_augment_policies)

            D_real = disc(Image_sharp_aug).view(-1)

            D_real_loss = bce(D_real, torch.ones_like(D_real).to(DEVICE))

            D_fake = disc(y_fake_aug.detach()).view(-1)
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake).to(DEVICE))
            D_loss = D_real_loss + D_fake_loss

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
        	# y_fake_aug = DiffAugment(y_fake, policy=diff_augment_policies)
            D_fake = disc(y_fake_aug.detach()).view(-1)
            Adversarial_loss = torch.mean(-1*torch.log(D_fake))

            L1_LPIPS = torch.mean(l1_loss(y_fake, Image_sharp)*170 + lpips_loss(y_fake,Image_sharp)*145)
            G_loss = Adversarial_loss + L1_LPIPS

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 50 == 0:                           ## After feeding every 50 images printing the losses
            print('[%d/%d][%d/%d]\tD_loss: %.4f\tG_loss: %.4f' % (epoch, epochs, idx, len(loader), 
            D_loss.item(), G_loss.item()))
    return D_loss, G_loss


def main():
    disc = Discriminator().to(DEVICE)
    disc = nn.DataParallel(disc)
    gen = Generator().to(DEVICE)
    gen = nn.DataParallel(gen)

    opt_disc = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Different losses used in the implementation
    BCE = nn.BCEWithLogitsLoss()        
    L1_LOSS = nn.L1Loss()
    LPIPS = lpips.LPIPS(net='vgg').to(DEVICE)     

    if LOAD_MODEL:
        load_checkpoint("gen.pth.tar", gen, opt_gen, 2e-4)
        load_checkpoint("disc.pth.tar", disc, opt_disc, 2e-4)

    g_scaler = torch.cuda.amp.GradScaler()      ## Prevents underfitting or overfitting
    d_scaler = torch.cuda.amp.GradScaler()

    ## Dataset Loaders
    train_dataset = DeblurData(path='train_gopro', data_type='train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,num_workers=16, pin_memory=True)

    val_dataset = DeblurData(path="test_gopro")          
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    ## Lists for appending Discriminator and Generator loss
    D_LossList = []
    G_LossList = []

    for epoch in range(NUM_EPOCHS):         ## Runing for the specified number of epochs in config variables
        d_loss,g_loss = train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, LPIPS, BCE, g_scaler,d_scaler,epoch,NUM_EPOCHS)
        
        D_LossList.append(d_loss.item())        ## Appending the loss after each epoch
        G_LossList.append(g_loss.item())

        if SAVE_MODEL and epoch % 10 == 0:      ## Saving the model checkpoint after every 10 iterations
            save_checkpoint(gen, opt_gen, filename=f"checkpoints/gen/gen.pth_{epoch}.tar")
            save_checkpoint(disc, opt_disc, filename=f"checkpoints/disc/disc_{epoch}.pth.tar")

        save_some_examples(gen, train_loader, epoch, folder="results/train")     ## Saving training examples
    
    # Loss plot
    lossplot(G_LossList,D_LossList,"lossplot")                            ## Creating loss plot for Gen and Disc

    # Runing the Generator on Validation images
    val_loop = tqdm(val_loader, leave=True)
    for idx, (Image_blur, Image_sharp) in enumerate(val_loop):
        val_loader = save_val_examples(Image_blur,Image_sharp,gen,idx,folder="results/val")


if __name__ == "__main__":
    main()