# DLERKm
## Table of Contents

1. [Introduction](#introduction)
2. [Python Environment](#python-environment)
3. [Project Structure](#Project-Structure)
   1. [Dataset](#Dataset)
   2. [Model](#Model)
   3. [script](#script)
---


## 1. Introduction
In this study, we introduce a novel deep learning model, DLERKm, designed for predicting Km values. The model integrates substrates, products, and enzyme sequences as inputs and leverages deep learning techniques to extract features from enzymatic reactions for accurate Km value prediction.

## 2. Python Environment

Python 3.9 and packages version:

- torch==1.12.0
- torchvision==0.13.0
- transformers==4.22.2
- numpy==1.22.4
- pandas==1.4.4
- scikit-learn==1.1.1

- ## 3. Project Structure

### 3.1 **Dataset**
In this paper, we first retrieved 146 enzymatic reaction data entries containing Km values from the Sabio-RK database. Next, we selected enzymatic reaction samples from the two species with the highest data availability on the platform, Homo sapiens (H. sapiens) and Escherichia coli (E. coli), resulting in a total of 16,322 samples. We then filtered out samples that lacked Km values or corresponding UniProt IDs for enzyme sequences, leaving 8,672 and 4,146 enzymatic reaction samples, respectively. Using the SabioReactionID and UniProtKB AC identifiers from these samples, we queried the Sabio-RK and UniProt databases to obtain substrate SMILES, product SMILES, and enzyme amino acid sequences. Finally, samples that were unavailable in the databases or lacked mutation site information were excluded, yielding a total of 10,122 enzymatic reaction samples.

### 3.2 **Model**
The overall architectures of DLERkm is presented in the following figure, which consists of a enzyme sequence extraction module, a enzyme reaction extraction module, a molecular set feature extraction module, and downstream prediction module.
