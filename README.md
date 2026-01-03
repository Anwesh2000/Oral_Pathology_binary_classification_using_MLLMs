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

### Citation
If you use this dataset in your research, please cite:

> Kumar, P. D. Madan; Ranganathan, K. S.; Rajeshwari, Pavithra M. C.; Lavanya; Nayak, Anwesh; Kestur, Ramesh; Behera, Sushree; Diddigi, Raghuram Bharadwaj (2025).  
> **SMART-OM: A SMARTphone-based expert-annotated dataset of Oral Mucosa images.**  
> Dryad Digital Repository.  
> https://doi.org/10.5061/dryad.gtht76hz6 *(currently in private mode)*

---

## Repository Structure

The repository is organized to mirror the experimental design presented in the paper, with **prompting-based multimodal reasoning methods evaluated first**, followed by **CNN-based supervised baselines**. 

├── COT.ipynb # Chain-of-Thought prompting experiments
├── Few shot learning.ipynb # Few-shot prompting–based lesion detection
├── RAG.ipynb # Retrieval-Augmented Generation experiments
├── Majority Voting.ipynb # Multi-expert majority voting baseline
├── RAG+COT (experts) and Mixture of Experts # Proposed RAG + CoT with Mixture-of-Experts framework
├── ResNet18 # CNN baseline using ResNet-18
├── ResNet34 # CNN baseline using ResNet-34
├── ResNet50 # CNN baseline using ResNet-50
├── EfficientNet B4 # CNN baseline using EfficientNet-B4
├── InceptionV4 # CNN baseline using InceptionV4
└── README.md # Documentation file (this file)

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

The experiments are implemented in Python using Jupyter notebooks. Core dependencies include:

- Python ≥ 3.8  
- PyTorch  
- Torchvision  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Jupyter Notebook  

Exact dependency versions may vary slightly across experiments.
