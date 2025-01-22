## CSMUnet

## Abstract

Hyperspectral image classification is a critical component of remote sensing studies. The emergence of deep learning techniques has driven the extensive application of neural network-based methods in this domain, leading to significant improvements in classification performance. In traditional image semantic segmentation, particularly in medical image analysis, U-Net has become a benchmark model due to its high precision and robust segmentation performance. However, hyperspectral remote sensing data, characterized by high acquisition costs, hundreds of spectral bands, and large image sizes, makes it challenging to directly use entire images as training or testing sets, as is common in traditional image segmentation tasks. This poses difficulties for directly applying the traditional U-Net architecture. This paper proposes a hybrid model, CSMUnet, which combines the strengths of Mamba and U-Net to effectively capture local spectral-spatial features while leveraging U-Net's semantic segmentation capabilities, thereby improving classification accuracy. To address the compatibility issue between hyperspectral image dataset partitioning and U-Net application, a Center-Sampling Method is introduced. This method calculates loss and performs gradient descent using only the center pixel of local windows, significantly enhancing classification performance. Experiments demonstrate that this method effectively operates on the entire region during convolutional operations, further validating the efficacy of the optimization strategy. Experiments on widely-used hyperspectral datasets, including Indian Pines, Pavia University, and Houston 2013, demonstrate that the CSMUnet model exceeds other state-of-the-art models. It achieves impressive classification accuracies of 98.12%, 99.69%, and 99.30% on these datasets, respectively. This study demonstrates the effectiveness of the Mamba architecture, U-Net, and the Center-Sampling Method in hyperspectral image classification tasks, providing valuable references and technical pathways for future research and promoting further advancements in this field.

## Requirements:

- Python 3.7
- PyTorch >= 1.12.1

## Usage:

python main.py

