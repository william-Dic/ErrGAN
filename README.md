# ErrGAN: Error Recognition and Intention Guided GAN

![ErrGAN Logo](https://github.com/william-Dic/ErrGAN/blob/main/assets/errgan_logo.png)  

## Overview

**ErrGAN** is a novel approach for error recognition in sequential decision-making tasks. By leveraging Generative Adversarial Networks (GANs), the framework not only recognizes errors but also backtracks to identify the root cause and correct it using an intention-guided mechanism.

The framework comprises two key components:

1. **V_GAN**: Discriminator that evaluates the value of the current trajectory using state, image sequences, and language embeddings.
2. **Q_GAN**: Generator that predicts the optimal action for the current trajectory using state, image sequences, and language embeddings.

### Key Concepts

#### Error Recognition
- **Backtracking**: The model identifies errors and backtracks to the moment when the error intention was formed. This is followed by intention-guided corrections.
- **Error + Backtracking + Intention Guided**: The combination of error detection, backtracking, and guided correction to refine the trajectory.

#### Intention Guided
- The model learns to guide the trajectory correction after recognizing an error, ensuring that the new trajectory aligns with the optimal action sequence.

### Mathematical Formulation

- **Q Function**: The Q value is calculated as the cosine similarity between the sampled K actions and the optimal action, providing a measure of how closely the generated action aligns with the optimal one.

### Workflow

1. **Error Detection**: The discriminator (V_GAN) identifies errors in the trajectory.
2. **Backtracking**: The system backtracks 20 steps before the detected error to understand the error intention and its root cause.
3. **Intention-Guided Correction**: The generator (Q_GAN) proposes corrective actions to guide the trajectory back on track.

### Future Work

- **Backtracking Efficiency**: Current backtracking considers the previous 20 steps. Future work involves finding a more efficient backtracking mechanism.

## Repository Structure

```plaintext
ErrGAN/
├── assets/                 # Visual assets such as logos and diagrams
├── data/                   # Data files for training and evaluation
├── models/                 # Implementation of V_GAN and Q_GAN models
├── scripts/                # Scripts for data processing and model training
├── experiments/            # Experimental setups and results
├── notebooks/              # Jupyter notebooks for detailed analysis
├── tests/                  # Unit tests for model components
└── README.md               # This README file
