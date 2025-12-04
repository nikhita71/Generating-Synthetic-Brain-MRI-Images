# Generating Brain MRI Images Using DC-GAN

## Overview
This project implements a Deep Convolutional Generative Adversarial Network (DC-GAN) to generate synthetic brain MRI images. The model learns from a dataset of brain tumor MRI scans and generates realistic-looking brain MRI images from random noise.

## Project Description
Generative Adversarial Networks (GANs) consist of two neural networks competing against each other:
- **Generator**: Creates synthetic images from random noise
- **Discriminator**: Distinguishes between real and generated images

Through adversarial training, the generator learns to produce increasingly realistic brain MRI images that can fool the discriminator.

## Dataset
- **Source**: Brain tumor MRI dataset
- **Path**: `./Downloads/brain dataset/brain_tumor_dataset/yes`
- **Images Used**: 20 selected images from 155 total images
- **Image Specifications**: 128x128 pixels, grayscale

## Model Architecture

### Generator
The generator transforms random noise (latent vectors) into 128x128 grayscale images:
- **Input**: 100-dimensional noise vector
- **Architecture**:
  - Dense layer (32×32×256)
  - LeakyReLU activation
  - Reshape to (32, 32, 256)
  - Conv2DTranspose (128 filters, 4×4 kernel, stride 2)
  - Conv2DTranspose (128 filters, 4×4 kernel, stride 2)
  - Conv2D output layer (1 channel, tanh activation)
- **Total Parameters**: 27,265,281 (104.01 MB)

### Discriminator
The discriminator classifies images as real or fake:
- **Input**: 128×128×1 image
- **Architecture**:
  - Conv2D (64 filters, 3×3 kernel)
  - Conv2D (128 filters, 3×3 kernel, stride 2)
  - Conv2D (128 filters, 3×3 kernel, stride 2)
  - Conv2D (256 filters, 3×3 kernel, stride 2)
  - Flatten and Dropout (0.4)
  - Dense output layer (sigmoid activation)
- **Total Parameters**: 582,785 (2.22 MB)

## Hyperparameters
```python
NOISE_DIM = 100          # Dimensionality of random noise input
BATCH_SIZE = 4           # Number of images per training batch
STEPS_PER_EPOCH = 3750   # Training steps per epoch
EPOCHS = 10              # Total training epochs
IMAGE_SIZE = 128×128     # Output image dimensions
CHANNELS = 1             # Grayscale images
OPTIMIZER = Adam(lr=0.0002, beta_1=0.5)
```

## Requirements
```
numpy
matplotlib
tqdm
opencv-python (cv2)
tensorflow
keras
seaborn
```

## Installation
```bash
pip install numpy matplotlib tqdm opencv-python tensorflow keras seaborn
```

## Usage

### 1. Prepare Dataset
Ensure your brain MRI dataset is organized in the following structure:
```
Downloads/
  brain dataset/
    brain_tumor_dataset/
      yes/
        *.jpg
```

### 2. Run the Notebook
Execute the cells in sequence:
1. Import libraries and set hyperparameters
2. Load and preprocess brain MRI images
3. Build generator and discriminator models
4. Train the GAN model
5. Generate synthetic brain MRI images

### 3. Generate Images
After training, use the `sample_images()` function to generate new brain MRI images:
```python
noise = np.random.normal(0, 1, size=(10, NOISE_DIM))
sample_images(noise, (2, 5))
```

## Training Process
The GAN is trained using the following procedure:
1. Generate fake images using the generator with random noise
2. Sample real images from the training dataset
3. Train discriminator on both real and fake images
4. Train generator to fool the discriminator
5. Repeat for specified epochs and steps

During training, the model displays:
- Generator loss
- Discriminator loss
- Sample generated images after each epoch

## Data Preprocessing
Images are preprocessed with the following steps:
1. Convert to grayscale using OpenCV
2. Resize to 128×128 pixels
3. Normalize pixel values to [-1, 1] range
4. Reshape to (batch_size, 128, 128, 1)

## Model Files
After training, you can save the models:
- `generator_final.h5`: Trained generator model
- `discriminator_final.h5`: Trained discriminator model

## Key Features
- Uses LeakyReLU activation for better gradient flow
- Implements batch normalization for training stability
- Adam optimizer with learning rate 0.0002
- Binary cross-entropy loss function
- Dropout regularization in discriminator

## Results
The trained generator can produce synthetic brain MRI images that resemble real brain tumor scans. The quality of generated images improves progressively through training epochs as the generator learns to capture the underlying patterns and features of brain MRI scans.

## Applications
- **Data Augmentation**: Increase training dataset size for medical imaging models
- **Privacy Preservation**: Generate synthetic medical data without patient privacy concerns
- **Research**: Study brain tumor characteristics and variations
- **Education**: Create training materials for medical students

## Limitations
- Limited to 128×128 resolution
- Trained on small sample size (20 images)
- Generates only brain tumor MRI images (from "yes" class)
- May require more training epochs for higher quality results

## Future Improvements
- Increase training dataset size for better generalization
- Implement higher resolution image generation
- Add conditional GAN (cGAN) for controlled generation
- Include multiple tumor classes
- Implement evaluation metrics (FID, IS scores)
- Add batch normalization to generator layers

## References
- Goodfellow, I., et al. (2014). "Generative Adversarial Networks"
- Radford, A., et al. (2015). "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"

## License
This project is for educational and research purposes.

## Author
Nikhita Gowda

---

**Note**: This implementation uses a subset of the brain tumor dataset for demonstration purposes. For production use, consider training on larger datasets with more epochs for better quality results.
