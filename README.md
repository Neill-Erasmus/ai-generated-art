# AI Generated Art (Generative Adversarial Networks)

<p align="center">
  <img src="https://github.com/Neill-Erasmus/ai-generated-art/assets/141222943/9ae4e0e0-aa48-4496-930a-41663c6e5fd5" alt="fake_samples_epoch_075">
</p>

A generative adversarial network for generating fake portraits.

##  Generative Adversarial Networks (GANs)

A class of machine learning models developed in 2014. GANs consist of two neural networks, a generator, and a discriminator, which are trained simultaneously through adversarial training.

### Generator (G)

The generator takes random noise as input and tries to generate data (e.g., images).
Initially, its output is likely to be random and not resemble the real data.

### Discriminator (D)

The discriminator evaluates inputs and tries to distinguish between real data and the data generated by the generator.
It is trained with real data and data generated by the generator.

### Training Process

During training, the generator and discriminator are in a constant feedback loop.
The generator aims to produce data that is indistinguishable from real data, while the discriminator aims to improve its ability to differentiate between real and generated data.

### Adversarial Loss

The generator and discriminator are trained using an adversarial loss function.
The generator seeks to minimize this loss by generating realistic data, while the discriminator aims to maximize the loss by correctly classifying real and fake data.

### Equilibrium

Ideally, the GAN reaches an equilibrium where the generator produces data that is so realistic that the discriminator cannot distinguish it from real data.
At this point, the generator has learned the distribution of the real data.

### Generator Output

Once trained, the generator can be used to generate new, synthetic data that resembles the training data.

### Applications

GANs are widely used for image generation, style transfer, data augmentation, and more.
They can be adapted for various domains, such as art, fashion, and even text generation.

## Dataset Overview

A dataset for Generative Adversarial Networks (GANs) typically consists of a collection of images that the GAN will use to learn and generate new, similar images. Here are key components and considerations for such a dataset:

1. Image Content

The dataset may focus on a specific type of content, such as faces, animals, landscapes, or objects, depending on the application. For instance, there are datasets like CelebA for celebrity faces, CIFAR-10 for object recognition, or LSUN for scenes.

2. Image Size and Resolution

Images in the dataset are often standardized to a specific size and resolution. Common resolutions include 64x64, 128x128, or 256x256 pixels.

3. Diversity

A diverse dataset with a wide range of variations helps the GAN learn to generate diverse and realistic samples. This diversity can include variations in lighting, angles, poses, and backgrounds.

4. Annotations

Some datasets come with annotations or labels. For instance, in a dataset of faces, each image might be labeled with attributes such as age, gender, or emotion. Annotations can be useful for specific tasks or conditional generation.

5. Number of Images

The size of the dataset can vary widely. Larger datasets often lead to better generalization, but they also require more computational resources for training.
6. Quality and Authenticity

High-quality images contribute to better training outcomes. It's important to ensure that the images are authentic and not artificially generated, as the GAN's objective is to generate realistic samples based on the training data.

7. Preprocessing

Images are often preprocessed to normalize colors, resize them to a standard resolution, and enhance features. Preprocessing steps may include normalization, cropping, or augmentation.

8. Privacy and Ethical Considerations

If the dataset involves human subjects, privacy and ethical considerations are crucial. Proper consent and anonymization of sensitive information should be taken into account.

Our dataset contains over 6000 images of portraits which will be used by the neural networks to generate new portraits.

## Architecture of the Generator Neural Network

### Class Definition

The Generator class inherits from nn.Module, which is the base class for all neural network modules in PyTorch.

### Initialization

In the __init__ method, the generator is defined using the nn.Sequential container to arrange a series of layers sequentially.

### Sequential Layers

The generator consists of several transposed convolutional layers (nn.ConvTranspose2d) followed by batch normalization (nn.BatchNorm2d) and rectified linear unit (ReLU) activation functions (nn.ReLU).

### Transposed Convolutional Layers

Transposed convolutional layers are used for upsampling or generating higher-resolution feature maps. They effectively learn to "deconvolve" the input.
The kernel size is set to 4x4 for each transposed convolution.
The stride parameter determines the step size of the convolution operation.
The padding parameter is used to control the spatial size of the output.

### Batch Normalization

Batch normalization is applied after each transposed convolutional layer to normalize the input, which can help stabilize and speed up the training process.

### ReLU Activation

Rectified Linear Unit (ReLU) activation functions are used after each batch normalization to introduce non-linearity to the model.

### Tanh Activation

The final layer uses a transposed convolutional layer followed by the hyperbolic tangent (nn.Tanh) activation function. Tanh squashes the output values to the range [-1, 1], which is common in GANs for image generation.

### Forward Method

The forward method defines the forward pass of the network. It reshapes the input tensor using view and applies the sequential layers defined in the __init__ method.

### Input Size

The generator expects an input tensor of shape (batch_size, 100, 1, 1). The 100 channels are likely meant for random noise that the generator will use to produce synthetic images.

## Architecture of the Discriminator Neural Network

### Class Definition

The Discriminator class inherits from nn.Module, the base class for all PyTorch neural network modules.

### Initialization

The __init__ method initializes the discriminator architecture using the nn.Sequential container for sequential layers.

### Sequential Layers

The discriminator consists of several convolutional layers (nn.Conv2d) followed by batch normalization (nn.BatchNorm2d) and leaky rectified linear unit (LeakyReLU) activation functions (nn.LeakyReLU).

### Convolutional Layers

Convolutional layers are used for extracting features from the input image. The kernel size is set to 4x4 for each convolution.
The stride parameter determines the step size of the convolution operation.
The padding parameter is used to control the spatial size of the output.

### Leaky ReLU Activation

Leaky Rectified Linear Unit (LeakyReLU) activation functions introduce non-linearity. They allow a small gradient when the input is negative, preventing the "dying ReLU" problem.

### Batch Normalization

Batch normalization is applied after certain convolutional layers to normalize the input, which can help stabilize and accelerate the training process.

### Sigmoid Activation

The final layer uses a convolutional layer followed by the sigmoid activation function (nn.Sigmoid). The sigmoid activation outputs a probability score, indicating the likelihood that the input image is real.

### Forward Method

The forward method defines the forward pass of the network. It applies the sequential layers to the input tensor and uses view(-1) to flatten the output into a one-dimensional tensor.

### Input Size

The discriminator expects an input tensor of shape (batch_size, 3, height, width), representing RGB images with three channels.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
