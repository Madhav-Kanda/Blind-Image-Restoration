# DeblurGAN

Pytorch-based implementation of Deep Generative Filter for Motion Deblurring. Incorporated the latest techniques to enhance the results.

This repository contains a PyTorch-based implementation of Deep Generative Filter for Motion Deblurring. We have incorporated the latest techniques to enhance the results.

## Features

1. In contrast to the VGG loss commonly used in generator networks, we employ the Learned Perceptual Image Patch Similarity (LPIPS) loss function in our approach, as it provides more accurate coverage of color contrast.

2. Implemented differential augmentation technique to ensure a consistent flow of gradients from the discriminator during training, preventing them from becoming zero and promoting effective learning and generalization of our model.

3. Incorporated WGAN loss as it introduces a gradient penalty to enforce the Lipschitz constraint on the discriminator. This helps to improve stability and convergence of training process.

Feel free to explore the code and experiment with different settings to achieve optimal results.

<img width="1007" alt="image" src="https://github.com/Madhav-Kanda/DeblurGAN/assets/76394914/c3de020b-dedd-40b7-b443-42d1c5604a1e">
