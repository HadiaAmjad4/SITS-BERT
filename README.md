# 🌍 Self-Supervised Learning for Satellite Image Time Series (SITS-BERT)

This project extends and reimplements **SITS-BERT**, a Transformer-based self-supervised learning framework for **Satellite Image Time Series (SITS)** classification.  
The work involved **pre-training**, **fine-tuning**, and **Low-Rank Adaptation (LoRA)** integration for efficient Transformer adaptation.

By leveraging the vast amount of unlabeled Sentinel-2 imagery, this project demonstrates how self-supervised learning can overcome the limitations of traditional supervised methods that rely on scarce labeled data.  
Through extensive experiments on datasets such as **TimeSen2Crop** and **California Labeled**, the results show significant improvements in classification accuracy and model efficiency, highlighting the potential of **Transformer-based SSL methods** for large-scale Earth observation tasks.

---

## 🧠 Overview

SITS-BERT applies the **BERT architecture** to satellite image time series, learning temporal–spectral representations from unlabeled data.  
My work focused on reproducing and extending the original model through:

- ✅ **Pre-training** SITS-BERT on unlabeled Sentinel-2 data  
- ✅ **Fine-tuning** the model on labeled datasets (California and TimeSen2Crop)  
- ✅ **Implementing LoRA** (Low-Rank Adaptation) for parameter-efficient fine-tuning  
- ✅ **Benchmarking** against state-of-the-art methods for SITS classification  

---

## 📊 Experimental Results

| Dataset | Configuration | Overall Accuracy (OA) |
|----------|----------------|------------------------|
| **TimeSen2Crop** | SITS-BERT (no LoRA) | **95.59%** |
| **TimeSen2Crop** | SITS-BERT (with LoRA) | 93.94% |
| **California Labeled** | Pretrained on California (no LoRA) | **93.88%** |
| **California Labeled** | Pretrained on California (with LoRA) | 79.71% |
| **California Labeled** | Pretrained on TimeSen2Crop (no LoRA) | 90.79% |
| **California Labeled** | Pretrained on TimeSen2Crop (with LoRA) | 73.01% |

> LoRA significantly reduced trainable parameters (2.48M → 49K) but required further tuning to maintain accuracy.

---

## 🧩 Methodology

### 1. **Self-Supervised Pre-Training**
- Random contamination of input sequences (simulated noise)
- Model predicts original (clean) values → learns temporal–spectral dependencies  
- Objective: Mean Squared Error (MSE)

### 2. **Fine-Tuning**
- End-to-end adaptation for land cover or crop-type classification  
- Cross-entropy loss on labeled datasets (California Labeled, TimeSen2Crop)

### 3. **LoRA Fine-Tuning**
- Integrated **Low-Rank Adaptation (LoRA)** for efficient Transformer adaptation  
- Reduced parameters by >95% with minimal accuracy loss on some configurations

---

## 🛰️ Datasets

- **[TimeSen2Crop](https://ieeexplore.ieee.org/document/9408357)** – Sentinel-2 time series for crop-type classification  
- **California Labeled Dataset** – Land cover dataset used in the original SITS-BERT paper  

Preprocessing included:
- Conversion of Julian dates → Day of Year (DOY)
- Flattening and padding time series for Transformer input
- Cleaning and validation for uniform structure across samples

---

## ⚙️ Requirements

```bash
python==3.6.10
pytorch==1.6.0
numpy==1.19.1
tqdm==4.48.2
tensorboard==2.3.0

```
---

🚀 Running the Code

Pre-Training
python pretraining.py \
  --dataset_path '../data/Pre-Training-Data.csv' \
  --pretrain_path '../checkpoints/pretrain/' \
  --epochs 100 \
  --batch_size 256 \
  --hidden_size 256 \
  --layers 3 \
  --attn_heads 8 \
  --learning_rate 1e-4

Fine-Tuning
python finetuning.py \
  --file_path '../data/California-Labeled/' \
  --pretrain_path '../checkpoints/pretrain/' \
  --finetune_path '../checkpoints/finetune/' \
  --epochs 100 \
  --batch_size 128 \
  --learning_rate 2e-4

LoRA Fine-Tuning
python finetuning_lora.py \
  --rank 16 \
  --alpha 24 \
  --dropout 0.01 \
  --learning_rate 1e-5

---

🔍 Key Contributions

--Implemented SITS-BERT end-to-end using PyTorch
--Conducted pre-training on unlabeled satellite data
--Performed fine-tuning and cross-dataset transfer learning
--Integrated and evaluated LoRA for parameter-efficient fine-tuning
--Compared results with baseline methods (TempCNN, LSTM, Transformer, etc.)
--Documented findings in a full research report (see /docs/REPPORT.pdf).

