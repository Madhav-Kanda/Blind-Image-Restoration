# Reference:
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7
# https://stackoverflow.com/questions/65447992/pytorch-how-to-apply-the-same-random-transformation-to-multiple-image

from torch.utils.data import Dataset, DataLoader
from glob import glob
from natsort import natsorted
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch


class DeblurData(Dataset):
    def __init__(self, path, data_type="train", shape=256):
        self.paths = natsorted(glob(path + "/*"))
        self.data_type = data_type
        self.shape = shape
        # print(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        ## Opening Image as PIL object --> Numpy object --> Pytorch Tensor object
        image = torch.from_numpy(np.array(Image.open(self.paths[idx])))

        ## Use one half of image --> Normalize --> Permute its dimensions
        ## Pytorch CNN expects the dimensions as (batch_size, channel, height, width) where as usually it is (height, width, channel)
        sharp_image = torch.permute(image[:, :256, :] / 255.0, (2, 0, 1))
        blur_image = torch.permute(image[:, 256:, :] / 255.0, (2, 0, 1))

        ## If you want to check the images are correctly taken then run the below code:
        # cv2.imwrite('Image Sharp.png', np.uint8(sharp_image.numpy()))
        # cv2.imwrite('Image Blur.png', np.uint8(blur_image.numpy()))
        # pdb.set_trace()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return blur_image, sharp_image


if __name__ == "__main__":
    traindata_obj = DeblurData(path="datasets/train", data_type="train")
    train_batch = DataLoader(
        traindata_obj, batch_size=1, shuffle=True, num_workers=16, pin_memory=True
    )

    testdata_obj = DeblurData(path="datasets/test", data_type="test")
    test_batch = DataLoader(
        testdata_obj, batch_size=1, shuffle=True, num_workers=16, pin_memory=True
    )

    tq = tqdm(train_batch)
    for x, y in tq:
        print(x.shape, y.shape)
        pass
