# PEFT — Parameter-Efficient Fine-Tuning

## 📌 Overview

**Parameter-Efficient Fine-Tuning (PEFT)** is a technique for adapting large pre-trained deep learning models to specific tasks by training only a small number of parameters instead of updating the entire model.

PEFT reduces:

- 🧠 GPU memory usage
- ⚡ Training time
- 💰 Computational cost
- 💾 Model storage requirements

## 🔑 Key Techniques

- **LoRA (Low-Rank Adaptation)**
- **QLoRA (Quantized LoRA)**
- **Adapter Tuning**
- **Prefix Tuning**
- **Prompt Tuning**
- **IA³**

## ⭐ LoRA

LoRA is one of the most popular PEFT methods. It keeps the original model weights frozen and learns small low-rank matrices to represent the required weight updates.

```text
Pre-trained Model
       ↓
Freeze Base Model
       ↓
Add LoRA Adapters
       ↓
Train Small Parameters
       ↓
Task-Specific Model
