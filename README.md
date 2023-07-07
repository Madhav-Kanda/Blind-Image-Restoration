# DeblurGAN
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)

<div align = center>
<img width="800" alt="image" src="https://github.com/Madhav-Kanda/DeblurGAN/assets/76394914/ab017047-007b-4b74-89d6-ede7a410a48e">
</div>

--------------------------------------------------------------------------------
This repository presents a PyTorch-based implementation of [Deep Generative Filter for Motion Deblurring](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w43/Ramakrishnan_Deep_Generative_Filter_ICCV_2017_paper.pdf) with the integration of advanced techniques to significantly enhance the model's performance.
This repository contains a PyTorch-based implementation of Deep Generative Filter for Motion Deblurring. We have incorporated the latest techniques to enhance the results.

## Features

1. In contrast to the VGG loss commonly used in generator networks, we employ the Learned Perceptual Image Patch Similarity (LPIPS) loss function in our approach, as it provides more accurate coverage of color contrast.

2. Implemented differential augmentation technique to ensure a consistent flow of gradients from the discriminator during training, preventing them from becoming zero and promoting effective learning and generalization of our model.

3. Incorporated WGAN loss as it introduces a gradient penalty to enforce the Lipschitz constraint on the discriminator. This helps to improve stability and convergence of training process.

Feel free to explore the code and experiment with different settings to achieve optimal results.

## How to Run the Code?

1. To train the model, execute the following command:
   ```shell
   python train.py
   ```
   This will initiate training and optimize the model parameters based on the provided dataset. Adjust the training settings and hyperparameters as needed.

2. To evaluate the performance of the trained model and generate various metrics, use the following command:
    ```shell
    python evaluate.py
    ```
    This will assess the model's performance on the test dataset and provide valuable metrics for analysis and comparison.

3. The additional files in the repository serve specific purposes:

    - **generator.py:** contains the code for the generator network responsible for motion deblurring.
    
    - **discriminator.py:** includes the code for the discriminator network used for adversarial training.
    
    - **utils.py:** provides utility functions and helper code used throughout the project.

