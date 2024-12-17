# README

## EFREI Lab 1: Variational Autoencoders and Generative Adversarial Networks

### Course: Generative AI for Computer Vision 
**Level:** M2 (Master’s 2)  
**Duration:** 4 hours  

### Objective
In this lab, you will:
1. Implement a Variational Autoencoder (VAE) to learn how to reconstruct images by sampling from a latent space.
2. Understand how the components of a VAE can be adapted to conceptualize and implement a Generative Adversarial Network (GAN).
3. Compare the roles of VAEs and GANs in generative modeling.
4. Reflect on the strengths and weaknesses of each approach.

---

## Part 1: Implementing a Variational Autoencoder (VAE)

### Instructions
1. **Download the MNIST dataset** of handwritten digits.
2. **Implement the following components in TensorFlow/Keras:**
   - **Encoder:** Compresses images into a latent space of dimension `latent_dim`.
   - **Reparameterization Trick:** Samples latent variables from a Gaussian distribution.
   - **Decoder:** Reconstructs images from the latent space.
3. **Define the VAE loss function:**
   - **Reconstruction Loss** (binary cross-entropy).
   - **KL Divergence Loss** (regularizes the latent space).
4. **Train the VAE and visualize the reconstructed images.**

### Code
The implementation consists of a modular structure with separate files for different components, including the encoder and decoder.

### Questions
1. **Why do we use the reparameterization trick in VAEs?**  
   The reparameterization trick allows for the computation of gradients during backpropagation by enabling the sampling of the latent variables to be differentiable.
   
2. **How does the KL divergence loss affect the latent space?**  
   The KL divergence loss ensures that the latent space maintains a normal distribution, which facilitates smooth interpolation and regularizes the network.

3. **How does changing the latent space dimension (`latent_dim`) impact the reconstruction quality?**  
   Increasing `latent_dim` generally improves reconstruction quality by allowing more capacity to encode information, but may risk overfitting. Decreasing it may fail to capture necessary details.

---

## Part 2: From VAE to GAN

### Conceptual Discussion
- **Explain how the VAE decoder can be used as a GAN generator:**  
  The decoder of a VAE, which generates images from latent variables, can be repurposed as the generator in a GAN architecture to create realistic images without the need for direct reconstruction.

- **Discuss the differences between the VAE encoder and the GAN discriminator:**  
  While a VAE encoder maps input data to a latent space, a GAN discriminator differentiates between real and synthetic data, learning to classify inputs.

### Implementation
- A GAN was implemented using the VAE decoder as the generator, emphasizing the adaptability of the VAE components.

---

## Results

### Hyperparameters
<img width="181" alt="result3" src="https://github.com/user-attachments/assets/a9176ba3-82ed-4f4e-96cd-ce0cb7f7ec2d" />
---
### Training Loss Over Time
![result2](https://github.com/user-attachments/assets/f1565d0d-1c77-4de0-ad63-b5c3f438ee38)
---
### VAE Image Generation
<img width="427" alt="result1" src="https://github.com/user-attachments/assets/bb2df0ae-6dc1-4f2d-be01-0ee282758de9" />
---
### Reconstructed Images and Latent Space Visualization
<img width="869" alt="result4" src="https://github.com/user-attachments/assets/f14691b1-7339-4fac-ac41-776636a7155b" />
---
