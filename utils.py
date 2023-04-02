import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def save_some_examples(gen, loader, epoch, folder):
    x, y = next(iter(loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        fake_clear = (y_fake - y_fake.min())/ (y_fake.max() - y_fake.min())
        blur = (x - x.min())/ (x.max() - x.min())
        clear = (y - y.min())/ (y.max() - y.min())

        output = torch.cat((blur, clear, fake_clear), dim=3)
        save_image(output, folder + f"/img_{epoch}.png")
    gen.train()

def save_val_examples(Image_blur,Image_sharp,gen,num,folder):
    x,y = Image_blur,Image_sharp
    x,y = x.to(DEVICE),y.to(DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        fake_clear = (y_fake - y_fake.min())/ (y_fake.max() - y_fake.min())
        blur = (x - x.min())/ (x.max() - x.min())
        clear = (y - y.min())/ (y.max() - y.min())

        output = torch.cat((blur, clear, fake_clear), dim=3)
        save_image(output, folder + f"/img_{num}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def lossplot(GenData, DiscData, filename):
    plt.plot(GenData, label='Gen')
    plt.plot(DiscData, label='Disc')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss plots of Generator and Discriminator')
    plt.legend()
    plt.savefig(filename)
    plt.clf()

