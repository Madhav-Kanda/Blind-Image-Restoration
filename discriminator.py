import torch
import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self,channel_rate=32,patch_size=64,image_size=256,device='cuda'):
    super(Discriminator,self).__init__()
    self.channel_rate = channel_rate
    self.patch_size = patch_size
    self.image_size = image_size
    self.device = device

    self.model = nn.Sequential(
       nn.Conv2d(3,self.channel_rate,3,2),        ## (in_channels, out_channels, kernel_size, stride)
       nn.InstanceNorm2d(self.channel_rate),
       nn.LeakyReLU(0.2),                         ## Leaky relu slope = 0.2   

       nn.Conv2d(self.channel_rate,2*self.channel_rate,3,2),
       nn.InstanceNorm2d(2*self.channel_rate),
       nn.LeakyReLU(0.2),

       nn.Conv2d(2*self.channel_rate,3*self.channel_rate,3,2),
       nn.InstanceNorm2d(3*self.channel_rate),
       nn.LeakyReLU(0.2),

       nn.Flatten(1),
       nn.Linear(3*self.channel_rate*3*3,1),
       nn.Sigmoid()
    )

  def forward(self, x):
    # create a list of row and column indices for each patch
    row_idx_list = [(i * self.channel_rate, (i + 1) * self.channel_rate) for i in range(int(self.image_size / self.patch_size))]
    col_idx_list = [(i * self.channel_rate, (i + 1) * self.channel_rate) for i in range(int(self.image_size / self.patch_size))]

    # create a list of patches from the input image
    list_patch = [x[:, :, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]] for row_idx in row_idx_list for col_idx in col_idx_list]

    # pass each patch through the model and compute the average output
    list_output = [self.model(patch) for patch in list_patch]
    output = sum(list_output) / len(list_patch)

    return output


def test_discriminator():
    x = torch.randn(1,3,256,256)
    model = Discriminator()
    pred = model(x)
    print(pred.shape)   ## Should be torch.Size([1, 1])


if __name__ == '__main__':
    test_discriminator()