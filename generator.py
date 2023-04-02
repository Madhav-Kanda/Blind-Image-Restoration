import torch
import torch.nn as nn


class Block(nn.Module):                                                                     ## Class for Dense field blocks
    def __init__(self,nChannels=128,channel_rate=32,drop_rate=0.0, lnum = 1):               ## lnum = block number
        super(Block,self).__init__()
        layers = []
        layers.append(nn.Sequential(
                    nn.InstanceNorm2d(nChannels + (lnum-1)*channel_rate),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(in_channels = nChannels + (lnum-1)*channel_rate,out_channels = 4*channel_rate,kernel_size = 1,bias=None),
                    nn.InstanceNorm2d(4*channel_rate),
                ))
        
        if(lnum == 3 or lnum ==7):                          ## Dilation and Padding corresponding to the block number
            layers.append(nn.Conv2d(4*channel_rate,channel_rate,3,dilation=2,padding=2,bias=None))
        elif(lnum == 5):
            layers.append(nn.Conv2d(4*channel_rate,channel_rate,3,dilation=3,padding=3,bias=None))
        else:
            layers.append(nn.Conv2d(4*channel_rate,channel_rate,3,dilation=1,padding=1,bias=None))

        layers.append(nn.Sequential(
                nn.InstanceNorm2d(channel_rate),
                nn.Dropout2d(drop_rate)
            ))
        self.conv = nn.ModuleList(layers)

    def forward(self,x):
        for layer in self.conv:
            x = layer(x)
        return x


class Generator(nn.Module):
    def __init__(self,nChannels=128,channel_rate=32,drop_rate=0.0,dilation=1):
        super(Generator,self).__init__()
        self.nChannels = nChannels
        self.channel_rate = channel_rate
        self.drop_rate = drop_rate
        self.dilation = dilation

        # Head
        self.head = nn.Conv2d(3,4*self.channel_rate,3,dilation=1,padding=1)

        # Dense Block
        self.block1 = Block(self.nChannels,self.channel_rate,self.drop_rate,lnum = 1)                   ## Block 1
        self.block2 = Block(self.nChannels,self.channel_rate,self.drop_rate,lnum = 2)                   ## Block 2
        self.block3 = Block(self.nChannels,self.channel_rate,self.drop_rate,lnum = 3)                   ## Block 3
        self.block4 = Block(self.nChannels,self.channel_rate,self.drop_rate,lnum = 4)                   ## Block 4
        self.block5 = Block(self.nChannels,self.channel_rate,self.drop_rate,lnum = 5)                   ## Block 5
        self.block6 = Block(self.nChannels,self.channel_rate,self.drop_rate,lnum = 6)                   ## Block 6
        self.block7 = Block(self.nChannels,self.channel_rate,self.drop_rate,lnum = 7)                   ## Block 7
        self.block8 = Block(self.nChannels,self.channel_rate,self.drop_rate,lnum = 8)                   ## Block 8
        self.block9 = Block(self.nChannels,self.channel_rate,self.drop_rate,lnum = 9)                   ## Block 9
        self.block10 = Block(self.nChannels,self.channel_rate,self.drop_rate, lnum = 10)                ## Block 10

        # Tail  
        self.tail = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.channel_rate,self.channel_rate,1,bias= None),
            nn.InstanceNorm2d(self.channel_rate),
            nn.Dropout2d(self.drop_rate)
        )

        # Last Layer
        self.last_layer = nn.Sequential(
            nn.Conv2d(5*self.channel_rate,self.channel_rate,3,dilation=1,padding=1,bias=None),
            nn.PReLU(self.channel_rate),
            nn.Conv2d(self.channel_rate,3,3,dilation=1,padding=1,bias=None),
            nn.Tanh()
        )

    def forward(self,x):
        x = self.head(x)                                                ## Head
        x1 = x
        for i in range(1,10):                                           ## Dense blocks 1-9
            d = self.__getattr__('block'+str(i))(x)
            x = torch.cat([x,d],1)  

        d = self.block10(x)                                             ## 10th Dense block
        x = self.tail(d)                                                ## Tail
        x = torch.cat([x,x1],1)                                         ## Global Skip connect
        x = self.last_layer(x)                                          ## Last layer
        return x

def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator()
    preds = model(x)
    print(preds.shape)                                                   ## Output shape should be (1, 3, 256, 256)


if __name__ == "__main__":
    test()