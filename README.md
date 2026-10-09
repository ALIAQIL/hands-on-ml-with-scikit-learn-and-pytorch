# Hands-on Machine Learning — Road to Glory

## From the first idea to modern AI systems

Welcome to my machine learning journey.

This repository documents the road from understanding the foundations of machine learning to building neural networks, computer vision systems, sequence models, transformers, and modern language-model experiments.

It is organized as a sequence of chapters. Each folder represents a stage of the work: the concepts I studied, the algorithms I implemented, the experiments I ran, and the problems I solved while progressing from classical machine learning to deep learning.

The purpose of this repository is not only to collect code. It is to show the complete learning process:

```text
understand the problem
        ↓
prepare and explore the data
        ↓
train a simple model
        ↓
measure errors and improve it
        ↓
build deeper and more capable models
        ↓
study modern AI systems
```

---

## What this project covers

- Machine learning fundamentals and the ML problem landscape
- End-to-end data preparation and model development
- Classification and regression
- Linear models, logistic regression, and regularization
- Support vector machines
- Dimensionality reduction and manifold learning
- Clustering, Gaussian mixtures, and anomaly detection
- Artificial neural networks from first principles
- PyTorch model building and automatic differentiation
- Convolutional neural networks for computer vision
- Semantic segmentation and medical image analysis
- Time-series forecasting and recurrent neural networks
- Tokenization, sentiment analysis, and sequence-to-sequence models
- Attention, transformers, GPT-style language models, and NMT
- Knowledge distillation and transfer to smaller models
- Optimizer experiments, including Adam-family methods and Muon

---

## The learning road

### Chapter 0 — Preface

Folder: [`0-preface/`](0-preface/)

This is the starting point of the journey. It introduces why machine learning became so important, what kinds of problems it can solve, and the major ideas that appear throughout the repository.

The chapter establishes the main roadmap: data, models, cost functions, optimization, feature engineering, evaluation, neural-network architectures, large-scale training, and reinforcement learning.

**Work represented here:**

- Understanding what machine learning is designed to solve
- Learning the difference between a model, a task, training experience, and evaluation
- Identifying the major families of machine learning systems
- Building a mental map for the chapters that follow

---

### Chapter 1 — The Machine Learning Landscape

Folder: [`1-the-ml-landscape/`](1-the-ml-landscape/)

This chapter develops the vocabulary and intuition needed to reason about ML systems. It studies why machine learning is useful when manually written rules become difficult to maintain, and how models discover patterns from examples.

**Topics and work:**

- Supervised, unsupervised, semi-supervised, self-supervised, and reinforcement learning
- Batch learning, online learning, and out-of-core learning
- Instance-based and model-based learning
- Training, inference, validation, and generalization
- Data quality, sampling bias, irrelevant features, underfitting, and overfitting
- Holdout validation, cross-validation, and the danger of adapting to the test set
- Practical applications such as spam filtering, fraud detection, recommendation, segmentation, and forecasting
- The No Free Lunch theorem and why model choice depends on the data

This chapter turns machine learning from a collection of algorithms into a way of thinking about problems.

---

### Chapter 2 — The End-to-End ML Project

Folder: [`2-end-to-end-ml-project/`](2-end-to-end-ml-project/)

This is the first complete machine learning workflow. The work follows the full path from understanding the big picture to preparing data, selecting a model, tuning it, and presenting a solution.

**Workflow developed here:**

1. Look at the big picture and define the objective.
2. Get and inspect the data.
3. Explore the training set and visualize patterns.
4. Split data correctly and avoid leakage.
5. Prepare numerical and categorical features.
6. Handle missing values and create useful transformations.
7. Select a model and establish a baseline.
8. Fine-tune the model and evaluate it honestly.
9. Present the solution.
10. Think about launch, monitoring, and maintenance.

The chapter uses practical preprocessing ideas such as stratified sampling, imputation, feature scaling, pipelines, and feature combinations. It is the foundation for treating an ML model as part of a larger system rather than as an isolated training call.

---

### Chapter 3 — Classification

Folder: [`3-Classification/`](3-Classification/)

This chapter focuses on making decisions between classes. The main work uses handwritten-digit classification as a concrete problem and explores how to evaluate classifiers beyond a single accuracy number.

**Topics and work:**

- Binary and multiclass classification
- MNIST data and handwritten-digit recognition
- Stochastic gradient descent classification
- Confusion matrices
- Precision, recall, and the precision-recall trade-off
- Decision thresholds and the ROC curve
- Cross-validation and validation predictions
- Error analysis and visual inspection of misclassified digits
- Multilabel and multioutput classification
- Practical dataset loading with the Titanic example

The central lesson is that a classifier is only as useful as the evaluation questions we ask about its mistakes.

---

### Chapter 4 — Training Models

Folder: [`4-Training-Models/`](4-Training-Models/)

This chapter goes deeper into how models learn. It studies the mathematics and behavior behind optimization instead of treating training as a black box.

**Topics and work:**

- Linear regression and the normal equation
- The pseudoinverse and singular value decomposition
- Batch gradient descent, stochastic gradient descent, and minibatch training
- Learning curves and diagnosing underfitting versus overfitting
- Polynomial regression
- Ridge, Lasso, and Elastic Net regularization
- Logistic regression and probability-based classification
- Decision boundaries
- Softmax regression for multiclass problems
- Cross-entropy as a learning objective
- The bias-variance trade-off and irreducible error

This folder represents the work of understanding not only which model wins, but why a model behaves the way it does as data, noise, capacity, and regularization change.

---

### Chapter 5 — Support Vector Machines

Folder: [`5-support-vector-machine/`](5-support-vector-machine/)

This chapter studies support vector machines as powerful models for both classification and regression.

**Topics and work:**

- Maximum-margin classification
- Support vectors and decision boundaries
- Linear SVMs
- Kernel methods and nonlinear decision boundaries
- Polynomial and Gaussian-style feature transformations
- SVM regression
- The importance of scaling features
- The trade-off between margin, errors, and model complexity

SVMs provide a different perspective on learning: instead of only fitting a function, the model searches for a boundary with a strong geometric margin.

---

### Chapter 8 — Dimensionality Reduction

Folder: [`8-Dimesionality-reduction/`](8-Dimesionality-reduction/)

High-dimensional data can be difficult to visualize, expensive to train on, and vulnerable to the curse of dimensionality. This chapter explores methods for finding simpler representations while preserving useful structure.

**Topics and work:**

- The curse of dimensionality
- Projection and manifold learning
- The Swiss roll problem
- Principal Component Analysis
- Principal components, explained variance, and compression
- Randomized and incremental PCA
- Random projection
- Locally Linear Embedding
- Multidimensional Scaling
- A from-scratch t-SNE implementation
- Visualization of high-dimensional datasets such as MNIST
- Comparing original spaces with reduced representations

The work in this chapter connects mathematical representation learning with practical communication: reducing dimensions can help both a model and the person trying to understand the data.

---

### Chapter 9 — Unsupervised Learning

Folder: [`9-Unsupervised-learning/`](9-Unsupervised-learning/)

This chapter studies learning without target labels. The objective is to discover groups, density, structure, and unusual examples hidden inside the data.

**Topics and work:**

- Clustering as a tool for data analysis
- K-Means and K-Means++ initialization
- Hard clustering and soft clustering
- Choosing the number of clusters
- Elbow methods and silhouette scores
- Cluster visualization and interpretation
- Gaussian mixture models
- Density estimation
- Anomaly detection
- Image segmentation using clustering
- Semi-supervised learning and active learning

The chapter emphasizes that there is no universal definition of a cluster. A useful grouping depends on the structure of the data and the decision we want to support.

---

### Chapter 10 — Introduction to Artificial Neural Networks

Folder: [`10-Intro-to-ANN/`](10-Intro-to-ANN/)

This chapter marks the transition from classical ML to deep learning. It explores how networks composed of layers and nonlinear functions can learn representations from data.

**Topics and work:**

- The history and intuition behind artificial neural networks
- Perceptrons and activation functions
- Hidden layers and multilayer perceptrons
- Forward propagation and prediction
- Backpropagation and reverse-mode automatic differentiation
- Gradient-based optimization
- Loss functions and training behavior
- Classification with neural networks
- A CIFAR-10 neural-network project

The goal is to understand what is happening inside a neural network before relying on high-level abstractions.

---

### Chapter 11 — Building Neural Networks with PyTorch

Folder: [`building-nn-with-pytorch/`](building-nn-with-pytorch/)

This chapter turns neural-network concepts into practical PyTorch programs. It focuses on the mechanics that make modern deep-learning experimentation possible.

**Topics and work:**

- Tensors and tensor operations
- Dynamic computation graphs
- Automatic differentiation
- Detaching tensors and understanding shared memory
- Building custom models with `torch.nn`
- Loss functions, optimizers, and training loops
- Image classification experiments
- CIFAR-10 work
- Saving and reusing model states
- Inspecting gradients and model behavior

This is the engineering bridge between knowing how neural networks work and being able to build, debug, and train them in practice.

---

### Chapter 12 — Computer Vision with CNNs

Folder: [`11-computervision-usingCNN/`](11-computervision-usingCNN/)

Convolutional networks are designed to learn visual structure. This chapter studies how local patterns, spatial hierarchies, pooling, residual ideas, and segmentation architectures allow neural networks to understand images.

**Topics and work:**

- Convolutional layers and receptive fields
- Feature maps and spatial hierarchies
- Pooling and spatial downsampling
- Image classification
- Deeper networks and the degradation problem
- Skip connections and residual learning
- Semantic segmentation
- Instance segmentation concepts
- Fully Convolutional Networks implemented in PyTorch
- A toy segmentation dataset with geometric shapes
- Medical image analysis with a pneumonia classification task
- Class imbalance and the motivation for techniques such as SMOTE

The computer-vision work combines numerical training with visual inspection: predictions should be examined as images, not only as numbers.

---

### Chapter 13 — Recurrent Networks, NLP, and Transformers

Folder: [`13-rnn/`](13-rnn/)

This chapter explores data where order matters: time series, sentences, and token sequences. It follows the evolution from recurrent networks to attention and transformer-based models.

**Topics and work:**

- Recurrent neural networks and hidden state
- Sequence-to-sequence and sequence-to-vector learning
- Time-series forecasting
- Naive forecasting and seasonality
- Autocorrelation and ARIMA ideas
- Deep recurrent forecasting
- Sentiment analysis
- Tokenization and subword representations
- Packed sequences and variable-length batches
- GRUs and reusable pretrained embeddings
- Attention mechanisms
- Transformer-based natural language processing
- Neural machine translation
- Semantic textual similarity and sentence embeddings
- GPT-2-style language-model experiments

This chapter represents the move from models that learn individual examples to models that learn context across sequences.

---

### Chapter 14 — Knowledge Distillation

Folder: [`distillation-project/`](distillation-project/)

Knowledge distillation studies how a smaller student model can learn from a larger teacher model. The project explores the relationship between model quality, model size, and practical deployment constraints.

**Work represented here:**

- Teacher and student model thinking
- Soft targets and transferred information
- The trade-off between accuracy and efficiency
- Compressing learned behavior into a more accessible model
- Connecting research ideas to deployable systems

This folder extends the journey from “can the model learn?” to “can the model be useful under real constraints?”

---

### Chapter 15 — After the LLM

Folder: [`after-llm/`](after-llm/)

This is the latest experimental area of the repository. It represents the move beyond introductory transformer usage toward questions about what comes after simply using a large language model.

**Work represented here:**

- Continuing to explore language-model systems
- Connecting previous NLP knowledge with newer model ideas
- Investigating how models, training methods, and representations can evolve
- Keeping the roadmap open for future experiments

This chapter is intentionally a work in progress: it is where the next part of the road is being built.

---

## Additional experiments and research work

The root of the repository also contains focused experiments that support the main chapters:

- [`aa_moun.py`](aa_moun.py) — a GPT-style training benchmark comparing AdamW with a Muon-based hybrid optimizer on streamed text.
- [`nlp_optimizer.py`](nlp_optimizer.py) — optimizer comparisons on a small transformer-style NLP model, including loss, gradient norms, parameter norms, and step time.
- [`train_muon.py`](train_muon.py) — optimizer experiments on a multilayer perceptron regression task.
- [`optimizer_loss_plot.png`](optimizer_loss_plot.png) — a generated visualization from optimizer work.
- [`13-rnn/attention-from-scratch.py`](13-rnn/attention-from-scratch.py) — an implementation-focused attention experiment.
- [`mini-projet.ipynb`](mini-projet.ipynb) and [`se.ipynb`](se.ipynb) — additional exploratory workspaces.
- [`VoxCeleb1/`](VoxCeleb1/) and [`frame.webp`](frame.webp) — supporting media and audio-related experimentation.

These experiments are part of the same journey: they are smaller investigations into optimization, language modeling, audio, and representation learning.

---

## How the chapters connect

The folders are not isolated tutorials. They form a progression:

| Stage | Main question | Chapters |
| --- | --- | --- |
| Foundations | What is machine learning and when should we use it? | 0–1 |
| Reliable workflows | How do we turn raw data into a trustworthy baseline? | 2–5 |
| Structure | What hidden patterns and representations exist in the data? | 8–9 |
| Neural foundations | How do networks learn representations? | 10 and PyTorch |
| Vision | How can a model understand images and pixels? | 11 |
| Sequences | How can a model use order and context? | 13 |
| Modern systems | How can models become more capable, efficient, and practical? | distillation, after-LLM, optimizer work |

The project grows in complexity, but the core loop stays the same:

1. Understand the problem.
2. Prepare and inspect the data.
3. Build a baseline.
4. Choose a model that matches the structure of the problem.
5. Train and evaluate it.
6. Study the errors.
7. Improve the system deliberately.
8. Document what was learned.

---

## How to use this repository

If you are following the road from the beginning, read the folders in numerical order. If you are looking for a particular capability, use the map below:

- Start with `0-preface/` and `1-the-ml-landscape/` for the big picture.
- Use `2-end-to-end-ml-project/` for the complete classical workflow.
- Study `3-Classification/`, `4-Training-Models/`, and `5-support-vector-machine/` for supervised learning.
- Study `8-Dimesionality-reduction/` and `9-Unsupervised-learning/` for discovering structure.
- Continue with `10-Intro-to-ANN/` and `building-nn-with-pytorch/` for deep-learning foundations.
- Move to `11-computervision-usingCNN/` for visual models.
- Move to `13-rnn/` for time series, NLP, attention, and transformers.
- Finish with `distillation-project/`, `after-llm/`, and the optimizer experiments for modern AI systems.

Most work is written as Jupyter notebooks so that the reasoning, equations, visualizations, and results stay together. The Python scripts contain focused implementations and experiments that can be read independently.

## Environment

The work uses Python and a mixture of the following tools:

- NumPy and pandas for numerical work and data preparation
- Matplotlib and related visualization tools
- Scikit-learn for classical machine learning
- PyTorch and TorchVision for neural networks and computer vision
- Transformers, tokenizers, and datasets for NLP and language-model experiments

The exact dependencies may differ by chapter. Some experiments require downloaded datasets, pretrained models, or GPU resources; read the notebook or script before running a full training experiment.

---

## Final goal

The final goal of this repository is to become a complete record of the ability to:

- reason about a machine learning problem;
- prepare and understand data;
- select an appropriate baseline;
- implement algorithms from first principles;
- use modern deep-learning frameworks;
- evaluate models honestly;
- understand failure cases;
- and connect research ideas to practical AI systems.

This is the **Road to Glory**: one chapter at a time, from the first classification model to systems that can see, read, remember context, and learn from increasingly rich representations.
