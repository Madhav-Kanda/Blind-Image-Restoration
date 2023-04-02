import os
import cv2
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0,1'

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import torch
import torch.nn as nn
import torch.optim as optim
from generator import netG
from discriminator import netD
from loss import generator_loss, adversarial_loss
from dataset import DeblurData
from torch.utils.data import DataLoader
import lpips
from torchvision.utils import save_image
from utils import save_checkpoint



torch.manual_seed(45)


class netD_on_G(nn.Module):
  def __init__(self,modelG,modelD,channel_rate=32,image_size=256,dropout=0.0):
    super(netD_on_G,self).__init__()
    self.channel_rate = channel_rate
    self.patch_size = self.channel_rate
    self.image_size = image_size
    self.dropout = dropout
    self.modelG = modelG
    self.modelD = modelD

  def forward(self,x):
    x = self.modelG(x)
    x = self.modelD(x)
    return x


def fit(modelG,modelD,modelD_on_G,train_loader,loss_fn_vgg,epochs,device):
  optimizerG = optim.Adam(modelG.parameters(),lr=0.0002, betas=(0.9, 0.999))
  optimizerD = optim.Adam(modelD.parameters(),lr=0.0002, betas=(0.9, 0.999))
  optimizerD_on_G = optim.SGD(modelD_on_G.parameters(),lr=0.0002,momentum=0.9,nesterov=True, weight_decay=1e-6)
  criteriaD = nn.BCELoss()
  G_losses,D_losses,D_on_G_losses=[],[],[]
  #criteriaG : generator_loss
  #cireriaG_on_D: adversarial_loss
  for epoch in range(epochs):
    for i,data in enumerate(train_loader):
      image_blur, image_full = data[0].to(device),data[1].to(device)
      ############################
      # (1) Update D network
      ###########################
      ## Train with all-real batch
      optimizerD.zero_grad()
      pred = modelD(image_full).view(-1)
      b_size = image_blur.size(0)
      y = torch.full((b_size,), 1.0, device=device)  
      errD_real = criteriaD(pred,y)
      errD_real.backward()
      D_x = pred.mean().item()

      ## Train with all-fake batch
      fake_clear = modelG(image_blur)
      y.fill_(0)
      pred = modelD(fake_clear.detach()).view(-1)
      errD_fake = criteriaD(pred,y)
      errD_fake.backward()
      D_G_z1 = pred.mean().item()
      errD = errD_real + errD_fake

      optimizerD.step()

      ############################
      # (2) Update D_on_G network
      ###########################
      optimizerD_on_G.zero_grad()
      pred = modelD(modelG(image_blur).detach()).view(-1)
      adv_loss = torch.mean(adversarial_loss(pred))
      adv_loss.backward()
      D_G_z2 = pred.mean().item()
      optimizerD_on_G.step()

      ############################
      # (3) Update G network
      ###########################
      optimizerG.zero_grad()
      pred = modelG(image_blur)
      # Calculate G's loss based on this output
      errG = torch.mean(generator_loss(pred,image_full,loss_fn_vgg))
      # Calculate gradients for G
      errG.backward()
      G_z = pred.mean().item()
      # Updating G
      optimizerG.step()

      if i % 50 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss D_on_G: %.4f' % (epoch, epochs, i, len(train_loader), 
          errD.item(), errG.item(), adv_loss.item()))
        
    

      
      # Saving Losses of each iter
      G_losses.append(errG.item())
      D_losses.append(errD.item())
      D_on_G_losses.append(adv_loss.item())

    with torch.no_grad():
            # img_blur, img_sharp = traindata_obj[0][0].to(device),traindata_obj[0][1].to(device)
            # img_blur, img_sharp = img_blur.to(device), img_sharp.to(device)
            
            fake_clear = modelG(image_blur)
            # save_image(image_blur, f"blurcheck_{epoch+1}.png", normalize=True)
            # save_image(fake_clear,f"fakeclear_{epoch+1}.png",normal=True)
            image_blur_norm = (image_blur - image_blur.min()) / (image_blur.max() - image_blur.min())
            image_full_norm = (image_full - image_full.min()) / (image_full.max() - image_full.min())
            fake_clear_norm = (fake_clear - fake_clear.min()) / (fake_clear.max() - fake_clear.min())
            # Concatenate normalized tensors with fake_clear
            output = torch.cat((image_blur_norm, image_full_norm, fake_clear_norm), dim=3)
            
            #output= cv2.hconcat([image_full,image_blur,fake_clear])
            save_image(output, f"generated_{epoch+1}.png", normalize=False)
    if epoch % 10 == 0:
          save_checkpoint(modelG, optimizerG, filename=f"gen_{epoch}.pth.tar")
          save_checkpoint(modelD, optimizerD, filename=f"disc_{epoch}.pth.tar")        
  return G_losses,D_losses,D_on_G_losses


epochs=100
use_cuda=True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
modelG = netG().to(device)
modelD = netD().to(device)
modelD_on_G = netD_on_G(modelG,modelD).to(device)
modelG = nn.DataParallel(modelG)
modelD = nn.DataParallel(modelD)
modelD_on_G = nn.DataParallel(modelD_on_G)
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

traindata_obj = DeblurData(path='datasets/train', data_type='train')
# print(len(traindata_obj[0][0]))
# save_image(traindata_obj[0][0], 'train_immage.png')
train_loader   = DataLoader(traindata_obj, batch_size=2, shuffle=True, num_workers=16, pin_memory=True) 

G_losses,D_losses,D_on_G_losses = fit(modelG,modelD,modelD_on_G,train_loader,loss_fn_vgg,epochs,device)

torch.save(modelG.state_dict(),'./modelG.pt')
torch.save(modelD.state_dict(),'./modelD.pt')
torch.save(modelD_on_G.state_dict(),'./modelD_on_G.pt')