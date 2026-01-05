# Context-Guided Mixture-of-Experts with Multimodal LLMs for Lesion Detection in Buccal Mucosa Images

## Table of Contents
- Abstract
- Dataset
- Repository Structure
- Environment and Dependencies

---

## Abstract

This repository accompanies the research paper titled **“Context-Guided Mixture-of-Experts with Multimodal LLMs for Lesion Detection in Buccal Mucosa Images”**. It contains the complete experimental codebase used to investigate oral lesion detection from smartphone-acquired buccal mucosa images using a combination of convolutional neural networks (CNNs) and context-aware multimodal large language models (MLLMs).

The central contribution of this work is a **context-guided Mixture-of-Experts (MoE)** framework that integrates prompting-based reasoning, retrieval-augmented context, and specialist MLLM decision-making to improve diagnostic sensitivity and robustness. The repository is organized to facilitate reproducibility, controlled comparisons with CNN baselines, and systematic analysis of prompting and expert aggregation strategies.

---

## Dataset

### Dataset Availability
The dataset used in this study is currently hosted in **private mode** on the Dryad Digital Repository. It will be made **publicly available upon acceptance of the manuscript**.

### Dataset Description
The dataset consists of smartphone-acquired buccal mucosa images annotated by clinical experts. It is designed to support research in oral lesion detection, medical image classification, few-shot learning, and multimodal diagnostic reasoning.

---

## Repository Structure

The repository is organized to mirror the experimental design presented in the paper, with **prompting-based multimodal reasoning methods evaluated first**, followed by **CNN-based supervised baselines**. 

```plaintext
.
├── COT.ipynb                                         # Chain-of-Thought prompting experiments
├── Few shot learning.ipynb                           # Few-shot prompting–based lesion detection
├── RAG.ipynb                                         # Retrieval-Augmented Generation experiments
├── Majority Voting.ipynb                             # Multi-expert majority voting baseline
├── Generate_COT_Dict.ipynb                           # Code to generate chain of thought based on expert annotations
├── RAG+COT (experts) and Mixture of Experts.ipynb    # Proposed RAG + CoT with Mixture-of-Experts framework
├── ResNet18.ipynb                                    # CNN baseline using ResNet-18
├── ResNet34.ipynb                                    # CNN baseline using ResNet-34
├── ResNet50.ipynb                                    # CNN baseline using ResNet-50
├── EfficientNet B4.ipynb                             # CNN baseline using EfficientNet-B4
├── InceptionV4.ipynb                                 # CNN baseline using InceptionV4
└── README.md                                         # Documentation file (this file)
```


### Prompting-Based and Contextual Reasoning Methods

These components implement the prompting methods explored using multimodal large language models (MLLMs):

- **Few-shot Prompting** – Context injection using a small number of representative examples  
- **Chain-of-Thought (CoT) Prompting** – Explicit reasoning traces to guide multimodal decision-making  
- **Retrieval-Augmented Generation (RAG)** – Dynamic retrieval of relevant visual context at inference time  
- **RAG + CoT** – Combined retrieval and reasoning for context-guided decision-making  
- **Mixture-of-Experts (MoE)** – Proposed framework that aggregates multiple prompting experts using a specialist MLLM  

These methods analyze how contextual relevance, reasoning structure, and expert aggregation influence sensitivity and robustness in lesion detection.

---

### CNN-Based Baseline Models

These components implement supervised convolutional neural network baselines used for comparison with the prompting-based and MoE frameworks:

- ResNet18  
- ResNet34  
- ResNet50  
- EfficientNet-B4  
- InceptionV4  

These baselines provide a controlled reference point for evaluating the benefits of contextual prompting and expert aggregation.

---

## Environment and Dependencies

All experiments in this repository are implemented in Python using Jupyter notebooks.
The codebase combines CNN-based image classification with prompting-based multimodal
reasoning and statistical analysis. The dependencies listed below are derived directly
from the imported libraries used across the notebooks.

### Core Environment
- Python ≥ 3.8
- Jupyter Notebook / JupyterLab

### Deep Learning and Vision
- torch
- torchvision
- Pillow (PIL)

### Data Handling and Utilities
- numpy
- pandas
- os
- glob
- shutil
- random
- json
- time
- re
- configparser
- textwrap
- base64
- io (BytesIO)

### Visualization
- matplotlib
- plotly

### Machine Learning and Evaluation
- scikit-learn  
  - Metrics: accuracy, precision, recall, F1-score  
  - Confusion matrix and classification reports  
  - Pairwise similarity (cosine similarity)

### Statistical Analysis and Distance Metrics
- scipy  
  - Wasserstein distance  
  - Energy distance  
  - Euclidean distance  
  - Mahalanobis distance  
  - Bray–Curtis distance  
  - Vector norms

### Prompting and Multimodal LLM Interface
- openai (Python SDK)

> **Note:**  
> Prompting-based experiments (Few-shot, CoT, RAG, and Mixture-of-Experts) rely on
> external multimodal large language models accessed via the OpenAI API.  
> API keys and credentials are **not included** in this repository and must be
> configured separately by the user.

### Progress Monitoring
- tqdm

## Citation
If you use this dataset in your research, please cite:

Kumar, P. D. M., Ranganathan, K. S., Rajeshwari, P. M. C., Lavanya, Nayak, A., Kestur, R., Behera, S., & Diddigi, R. B. (2025).  
*SMART-OM: A smartphone-based expert-annotated dataset of oral mucosa images*.  
Dryad Digital Repository. https://doi.org/10.5061/dryad.gtht76hz6 *(currently in private mode)*

If you use this codebase or refer to the proposed methodology, please cite the associated manuscript:

Nayak, A., Diddigi, R. B., Kestur, R., Kumar, P. D. M., Ranganathan, K., Lavanya, C., Rajeshwari, S., & Behera, S. (2025).  
*Context-guided mixture-of-experts with multimodal LLMs for lesion detection in buccal mucosa images*.  
Manuscript submitted for publication to *International Dental Journal*.
