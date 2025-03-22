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

Python 3.8 and packages version:
- cuda==12.1
- matplotlib==3.2.2
- seaborn==0.13.2
- torch==2.1.0
- numpy==1.24.3
- pandas==1.3.4
- scipy==1.4.1
- scikit-learn==0.23.1
- ## 3. Project Structure

### 3.1 **Dataset**
In this paper, we first retrieved 146 enzymatic reaction data entries containing Km values from the Sabio-RK database. Next, we selected enzymatic reaction samples from the two species with the highest data availability on the platform, Homo sapiens (H. sapiens) and Escherichia coli (E. coli), resulting in a total of 16,322 samples. We then filtered out samples that lacked Km values or corresponding UniProt IDs for enzyme sequences, leaving 8,672 and 4,146 enzymatic reaction samples, respectively. Using the SabioReactionID and UniProtKB AC identifiers from these samples, we queried the Sabio-RK and UniProt databases to obtain substrate SMILES, product SMILES, and enzyme amino acid sequences. Finally, samples that were unavailable in the databases or lacked mutation site information were excluded, yielding a total of 10,122 enzymatic reaction samples.

### 3.2 **Model**
   - The overall architectures of DLERkm is presented in the following figure, which consists of a enzyme sequence extraction module, a enzyme reaction extraction module, a molecular set feature extraction module, and downstream prediction module.
   ![Model Architecture](https://github.com/yulglee/DLERKm/blob/main/Dataset_file/Figure1_model_framework.jpg)
   
   - To load the RXNFP, we can use the following code:
   ```python
   import torch
   from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
   )
   model, tokenizer = get_default_model_and_tokenizer()
   rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
   reaction_vector = rxnfp_generator.convert(r_smiles)
   ```
   - To load the EMS-2, we can use the following code:
   ```python
   import esm 
   batch_labels, batch_strs, batch_tokens = self.batch_converter(pro_seq)
   results = self.model(batch_tokens.to(device=self.device), repr_layers=[33], return_contacts=True)
   token_representations = results["representations"][33]
    ```

### 3.3 **script**
   - `dataloader.py` randomly splits enzymatic reactions and feed them into the DLERKm in batchsizes for training and testing.
   - `CBAM.py` implements the channel attention mechanism to enhance the representation of local features by emphasizing important channel information.
   - `DLERKm.py` implements DLERKm for Km value prediction.
   -`test_program.py` is used to calculate the performance metrics of DLERKm.
   - `train.py` provides the training pipeline for DLERKm.

