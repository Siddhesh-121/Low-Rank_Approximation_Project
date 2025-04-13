# Low-Rank_Approximation_Project

# Neural Network Compression via Low-Rank Approximation

This repository contains code implementation for neural network compression using Singular Value Decomposition (SVD) applied to MNIST and EMNIST datasets.

## Project Overview

This project implements low-rank approximation techniques to compress neural network models while maintaining or even improving performance. The implementation focuses on:

1. Using SVD to decompose weight matrices into lower-rank approximations
2. Evaluating different energy retention thresholds (50%, 75%, and 90%)
3. Measuring the impact on model size, computational efficiency, and accuracy
4. Applying fine-tuning to recover accuracy after compression

## Datasets

### MNIST Dataset
- Handwritten digit recognition dataset (0-9)
- 42,000 samples of 28×28 pixel grayscale images
- Dataset split: 80% training (33,600 images), 20% validation (8,400 images)
- **Download link**: [MNIST Dataset on Kaggle](https://www.kaggle.com/competitions/digit-recognizer/data)

### EMNIST Dataset 
- Extended MNIST with handwritten characters and digits
- 112,800 samples of 28×28 pixel grayscale images
- Dataset split: 80% training (90,240 images), 20% validation (22,560 images)
- **Download links**: 
  - [EMNIST Dataset on Kaggle (Main page)](https://www.kaggle.com/datasets/crawford/emnist)
  - [EMNIST Balanced Train CSV](https://www.kaggle.com/datasets/crawford/emnist?select=emnist-balanced-train.csv)
  - [EMNIST Balanced Test CSV](https://www.kaggle.com/datasets/crawford/emnist?select=emnist-balanced-test.csv)

**Note**: Due to GitHub's 25MB file size limit, we cannot include these datasets directly in the repository. Please download them from the links provided above before running the notebooks.

## Implementation Details

The implementation performs the following key steps:

1. **Training the original model**: A mini-ResNet architecture is trained on the MNIST/EMNIST dataset to establish baseline performance.
2. **SVD-based layer decomposition**: For each Dense and Conv2D layer, SVD is applied to decompose weight matrices into lower-rank approximations.
3. **Energy threshold determination**: Three different energy retention thresholds (50%, 75%, and 90%) control the compression level by determining how many singular values to retain.
4. **Layer replacement with low-rank equivalents**: Original layers are replaced with sequential blocks containing two smaller layers that together approximate the original computation.
5. **Fine-tuning**: The compressed model is retrained to recover accuracy lost during compression.

## Key Files

- `mnist_svd_compression.ipynb`: Google Colab notebook for MNIST compression implementation
- `emnist_svd_compression.ipynb`: Google Colab notebook for EMNIST compression implementation
- `requirements.txt`: Required libraries and dependencies

## Usage

1. Upload the notebooks to Google Colab or run locally
2. Install dependencies: `pip install -r requirements.txt`
3. Run each cell sequentially in the notebook
4. **Important**: Each notebook contains separate code cells for different energy retention thresholds (50%, 75%, and 90%). Run these sections individually to compare results across different compression rates.

**Note:** The implementation for 50%, 75%, and 90% energy retention thresholds is contained within different cells of the same notebook. Execute each threshold's section separately to avoid conflicts between implementations.

## Results Summary

| Metric | MNIST (75% ER) | EMNIST (75% ER) |
|--------|---------------|-----------------|
| Parameter Reduction | ~92-93% | 49.5% |
| Model Size Reduction | Significant | 46.7% |
| Accuracy (Original) | 97.10% | 87.85% |
| Accuracy (Compressed) | 97.85% | 88.16% |
| Actual Speedup | Improved significantly | 0.92x (slight slowdown) |

## Key Findings

1. **Compression-Accuracy Trade-off**: The 75% energy retention threshold offers the optimal balance between compression rate and minimal accuracy loss.
2. **Layer-Specific Compression**: Different layers have varying sensitivity to compression. Some layers can be compressed more aggressively than others without significant accuracy impact.
3. **Fine-tuning Impact**: Post-compression fine-tuning is essential and can sometimes lead to improved accuracy compared to the original model.
4. **Theoretical vs. Actual Performance**: While theoretical speedup calculated from FLOPs reduction suggests significant acceleration, the actual speedup may be lower due to memory access patterns and hardware limitations.

## Recommendations

1. **Optimal Energy Threshold Selection**: For production deployment, a 75% energy threshold offers a good balance between compression rate and minimal accuracy loss.
2. **Layer-Wise Adaptive Compression**: Apply different thresholds to different layers based on their sensitivity to compression.
3. **Post-Compression Fine-Tuning**: Always perform fine-tuning after compression to recover accuracy, especially for aggressive compression rates.
4. **Hardware-Aware Compression**: Consider the target hardware architecture when selecting compression parameters, as memory access patterns significantly impact actual speedup.

## Future Work

1. Combine low-rank approximation with quantization for even greater compression
2. Develop methods to automatically determine the optimal energy threshold for each layer
3. Further investigate how to optimize the compressed model structure for specific hardware architectures
4. Explore techniques to dynamically adjust ranks during training rather than using fixed energy thresholds
