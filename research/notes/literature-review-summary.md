## Introduction
This is the summary of the document literature review from [[Deepfake_Detection_Literature_Review.pdf|here]]

**Deep fake:** manipulated High quality realistic videos.

Deepfake detection methods
- Deep Learning techniques
- Classical ML techniques
- Statistical techniques
- Blockchain techniques

Deep learning techniques are more effective in conclusion

### Generation of Deep Fakes?
Deep fakes are generated using two Neural Networks
1. Generative Network
2. Discrimination Network

A generative network creates fake images using an encoder and a decoder. The discriminating network defines the authenticity of the newly generated
images. The combination of these two networks is called
**Generative Adversarial Networks (GANs)**

---
## What does this cover?

All together 112 articles from 10 selected popular repositories by considering their relevance and availability as listed below:
- Web of Science
- IEEE Xplore Digital Library
- ACM Digital Library
- ScienceDirect (ELSEVIER)
- SpringerLink
- Google Scholar
- Semantic Scholar
- Cornell University
- Computing Research Repository
- Database Systems and Logic Programming (DBLP)

The repositories include journals, conferences, and archives. Search duration from January 2018 to December 2020.

---
## Research Questions
Here are the targeted research questions addressed by the literature review.
### RQ-1: What are the popular deepfake detection techniques?

All the techniques are classified into the below four categories
#### 1. Machine Learning Methods
Here are the approaches
- Traditional machine learning (ML) algorithms
- Tree-based ML approaches (Decision Tree, Random Forest, Extremely Randomized Trees)
- GAN-based methods
- Feature-based methods (focusing on specific facial regions)
- Biological sign consistency measurement (Lip syncing, Heartbeat consistency etc.)
- 3D head pose approximation
- MLP (Multilayer Perceptron) for visual artifact detection

machine learning-based Deepfake detection methods can achieve up to 98% accuracy. However, this performance heavily depends on:

- The type of dataset used
- The selected features
- The alignment between training and test sets

When using similar datasets split into training and testing sets, higher accuracy is achieved. However, performance drops significantly (close to 50%) when using unrelated datasets.

#### 2. Deep Learning Methods
1. GAN Simulator
    - Replicates GAN-image artifacts
    - Uses these artifacts to train a classifier
2. Physiological Measurement
    - Detects deepfakes based on physiological signs like heartbeat
3. Inception Modules
    - Uses Meso-4 and MesoInception-4 networks
    - Focuses on mean squared error for training
4. Convolutional Neural Networks (CNNs)
    - Both deep and shallow CNNs are used
    - Deep CNNs generally outperform shallow ones
5. Feature Extraction Methods
    - Extracts handcrafted features
    - Analyzes spatiotemporal features
    - Examines common textures
    - Uses facial landmarks (e.g., eyes, teeth, lip movements)
6. Data Augmentation and Super-resolution
    - Enhances input data for better detection
7. Attention Mechanism
    - Focuses on important parts of the image/video
8. Capsule Networks (CN)
    - Requires fewer parameters than very deep networks
9. Ensemble Learning
    - Combines multiple models for improved accuracy
10. Recurrent Neural Networks (RNNs)
    - Extracts features at micro and macroscopic levels
    - Useful for analyzing video sequences
11. Optical Flow Techniques
    - Analyzes motion between video frames
12. Autoencoder-based Architectures
    - Helps in reducing overfitting
13. Pixel-wise Masking
    - Focuses on affected areas of the face
14. Adversarial Training
    - Improves model robustness
15. Frequency Domain Analysis
    - Examines latent patterns in images
16. Multimodal Approaches
    - Combines audio and visual analysis
17. Patch-based Methods
    - Focuses on local image patches rather than global structure

These approaches use various deep learning techniques to detect deepfakes by analyzing different aspects of images and videos, from facial features and movements to physiological signs and image artifacts.
#### 3. Statistical Methods
- Photo Response Non-Uniformity (PRNU)
    - Analyzes unique noise patterns in digital images
    - Uses normalized cross-correlation scores
    - Applies t-tests for statistical significance
- Expectation-Maximization (EM) Algorithm
    - Extracts regional features
    - Validates against various GAN architectures
- Statistical Framework with Hypothesis Testing
    - Measures distance between original and GAN-created image distributions
    - Longer distances indicate easier detection
#### 4. Block Chain Methods
- Decentralized Verification
    - Uses public blockchain to verify content authenticity
    - Tracks video/image origin to trusted sources
- IPFS-based Decentralized Storage
    - Integrates with Ethereum Name Service
- LSTM Networks with Blockchain
    - Uses LSTM CNNs to hash and encode image/video content
    - Stores information in permission-based blockchain

#### Method Distribution
- Deep learning-based: 77%
- Machine learning-based: 18%
- Statistical methods: 3%
- Blockchain-based: 2%

---