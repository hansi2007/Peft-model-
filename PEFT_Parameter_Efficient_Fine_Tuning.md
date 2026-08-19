# PEFT (Parameter-Efficient Fine-Tuning)

## Introduction

**PEFT** stands for **Parameter-Efficient Fine-Tuning**. It is a set of techniques used to adapt large pre-trained deep learning models to specific tasks while updating only a small portion of the model's parameters.

Traditional fine-tuning updates almost all parameters of a pre-trained model. This becomes expensive when the model contains millions or billions of parameters.

PEFT solves this problem by keeping most of the original model frozen and training only a small number of additional or selected parameters.

---

## Why PEFT Is Needed

Modern models such as BERT, LLaMA, Mistral, T5, and other Large Language Models (LLMs) can contain hundreds of millions or billions of parameters.

Full fine-tuning can require:

- Large GPU memory
- High computational power
- Long training time
- Large storage requirements
- Separate copies of the model for different tasks

For example, imagine a model with **7 billion parameters**.

With full fine-tuning:

```text
Pre-trained Model
       ↓
Update all 7B parameters
       ↓
Fine-tuned Model
```

This can be computationally expensive.

With PEFT:

```text
              ┌────────────────────┐
              │  Pre-trained Model │
              │   7B parameters    │
              │      FROZEN        │
              └─────────┬──────────┘
                        │
                        ↓
                Small Trainable
                    Adapter
                        │
                        ↓
                 Fine-tuned Model
```

Only a small fraction of parameters are trained.

---

# Full Fine-Tuning vs PEFT

## Full Fine-Tuning

In full fine-tuning, the parameters of the pre-trained model are updated using the new dataset.

```text
Pre-trained Model
       ↓
Train all parameters
       ↓
Updated Model
```

### Advantages

- Maximum flexibility
- Can achieve excellent task-specific performance
- All model parameters can adapt to the new task

### Disadvantages

- High GPU memory requirement
- Expensive training
- Longer training time
- Large model checkpoints
- Difficult to maintain multiple task-specific versions

---

## PEFT

In PEFT, most parameters remain frozen.

```text
Pre-trained Model
       ↓
Freeze most parameters
       +
Train small PEFT parameters
       ↓
Task-specific Model
```

### Advantages

- Lower memory usage
- Faster training
- Smaller checkpoints
- Lower computational cost
- Easier to maintain multiple task-specific adapters
- Useful when computing resources are limited

---

# How PEFT Works

The general PEFT workflow is:

```text
                    Pre-trained Model
                           │
                           ↓
                    Freeze Base Model
                           │
                           ↓
                 Add PEFT Components
                           │
                           ↓
                    Train Small Set
                    of Parameters
                           │
                           ↓
                 Task-specific Adapter
```

The base model contains most of the knowledge learned during pre-training.

The PEFT component learns how to adapt that knowledge to a new task.

---

# Major PEFT Techniques

Some important PEFT methods are:

1. LoRA
2. QLoRA
3. Adapter Tuning
4. Prefix Tuning
5. Prompt Tuning
6. IA³

---

# 1. LoRA

## Low-Rank Adaptation

**LoRA** is one of the most popular PEFT techniques.

Instead of directly modifying the original weight matrix, LoRA learns a small low-rank update.

Suppose the original weight matrix is:

\[
W
\]

Traditional fine-tuning updates it as:

\[
W' = W + \Delta W
\]

LoRA represents the update as:

\[
\Delta W = BA
\]

Therefore:

\[
W' = W + BA
\]

where:

- \(W\) = original pre-trained weights
- \(A\) = trainable low-rank matrix
- \(B\) = trainable low-rank matrix
- \(BA\) = learned update

The original matrix \(W\) remains frozen.

---

## LoRA Example

Suppose:

\[
W = 1000 \times 1000
\]

The original matrix contains:

\[
1,000,000
\]

parameters.

Instead of training the entire matrix, LoRA can use a rank of 8:

\[
A = 8 \times 1000
\]

\[
B = 1000 \times 8
\]

Trainable parameters:

\[
8000 + 8000 = 16,000
\]

Therefore:

```text
Full Fine-Tuning
≈ 1,000,000 trainable parameters

LoRA
≈ 16,000 trainable parameters
```

This dramatically reduces the number of trainable parameters.

---

# Why Is It Called Low-Rank?

The matrices used by LoRA have a much smaller dimension called the **rank**.

For example:

```text
Original:

1000 × 1000

LoRA:

1000 × 8
8 × 1000
```

Here, the rank is 8.

A smaller rank means fewer trainable parameters.

The rank is usually represented by:

\[
r
\]

The number of trainable parameters becomes approximately:

\[
r(d_{in}+d_{out})
\]

instead of:

\[
d_{in}d_{out}
\]

for the original matrix.

---

# LoRA Scaling

LoRA commonly uses a scaling factor:

\[
\frac{\alpha}{r}
\]

The updated weight can therefore be represented as:

\[
W' = W + \frac{\alpha}{r}BA
\]

where:

- \(r\) = LoRA rank
- \(\alpha\) = LoRA scaling parameter

The scaling controls the influence of the LoRA update.

---

# 2. QLoRA

## Quantized LoRA

**QLoRA** combines:

- Quantization
- LoRA

The base model is stored in a lower-precision format while LoRA adapters remain trainable.

General idea:

```text
Large Pre-trained Model
          ↓
      Quantization
          ↓
   Frozen Quantized Model
          +
       LoRA Adapters
          ↓
       Fine-tuning
```

QLoRA significantly reduces memory requirements and is useful when fine-tuning large language models on limited GPU hardware.

---

# LoRA vs QLoRA

| Feature | LoRA | QLoRA |
|---|---|---|
| Base model | Frozen | Frozen |
| Quantization | Not required | Yes |
| Trainable adapters | Yes | Yes |
| Memory usage | Reduced | Usually much lower |
| Main idea | Low-rank adaptation | Quantization + LoRA |
| Large-model training | Useful | Especially useful |

---

# 3. Adapter Tuning

Adapter tuning inserts small trainable neural network modules into the original model.

```text
Transformer Layer
       ↓
Original Layer
       ↓
Small Adapter
       ↓
Next Layer
```

The original model remains mostly frozen.

Only the adapter parameters are trained.

This allows the same base model to support different tasks using different adapters.

---

# 4. Prefix Tuning

Prefix tuning adds trainable vectors to the input of Transformer layers.

Instead of modifying the entire model, the model learns a small set of task-specific vectors.

Conceptually:

```text
Task Input
   +
Trainable Prefix
   ↓
Transformer
   ↓
Output
```

The base Transformer remains frozen.

---

# 5. Prompt Tuning

Prompt tuning learns a small number of continuous prompt embeddings.

Traditional prompting uses human-written text:

```text
"Classify the following text as spam or ham:"
```

Prompt tuning instead learns numerical prompt representations.

```text
Trainable Prompt Embeddings
          +
Input
          ↓
Pre-trained Model
          ↓
Output
```

Only the prompt parameters are trained.

---

# 6. IA³

**IA³** stands for **Infused Adapter by Inhibiting and Amplifying Inner Activations**.

Instead of adding large trainable modules, IA³ learns scaling vectors that modify internal activations.

The number of trainable parameters can therefore be extremely small.

---

# PEFT Architecture

A simplified architecture looks like this:

```text
                 Input
                   │
                   ↓
            Tokenization
                   │
                   ↓
          Pre-trained Model
          ┌────────────────┐
          │ Frozen Weights │
          └───────┬────────┘
                  │
          ┌───────▼────────┐
          │ PEFT Component  │
          │ LoRA / Adapter  │
          └───────┬────────┘
                  │
                  ↓
              Output
```

---

# PEFT Training Process

## Step 1: Select a Pre-trained Model

Examples:

- BERT
- RoBERTa
- T5
- LLaMA
- Mistral

## Step 2: Load the Dataset

For example:

```text
Text                          Label
--------------------------------------
"Win a free prize!"            Spam
"Meeting at 5 PM"              Ham
```

## Step 3: Tokenize the Data

The tokenizer converts text into tokens and numerical IDs.

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
```

## Step 4: Freeze the Base Model

Most parameters are not updated during training.

## Step 5: Add PEFT Parameters

For example, with LoRA, small trainable matrices are added to selected layers.

## Step 6: Train

Only PEFT parameters are optimized.

## Step 7: Save the Adapter

Instead of saving a complete copy of the huge model, the smaller adapter can be saved.

---

# PEFT for NLP

PEFT is commonly used for:

- Text classification
- Sentiment analysis
- Spam detection
- Question answering
- Text generation
- Named Entity Recognition
- Summarization
- Translation
- Instruction tuning
- Chatbots

---

# Example: Spam Detection

Suppose we want to adapt BERT for SMS spam detection.

```text
SMS Dataset
     ↓
BERT Tokenizer
     ↓
Pre-trained BERT
     ↓
Freeze BERT
     ↓
Add LoRA
     ↓
Train LoRA Parameters
     ↓
Classification Layer
     ↓
Spam / Ham
```

The BERT model already contains general language knowledge.

LoRA helps adapt that knowledge to the spam classification task.

---

# PEFT with LLMs

PEFT is especially useful for Large Language Models.

Example:

```text
              Base LLM
                 │
        ┌────────┼────────┐
        ↓        ↓        ↓
      Coding   Medical  Finance
      LoRA      LoRA      LoRA
     Adapter   Adapter   Adapter
```

A single base model can be combined with different adapters.

This can be more storage-efficient than maintaining a completely separate full model for every task.

---

# PEFT and Transfer Learning

PEFT is closely related to transfer learning.

### Transfer Learning

A model trained on a large dataset is reused for a new task.

```text
Large Dataset
     ↓
Pre-training
     ↓
Pre-trained Model
     ↓
New Task
     ↓
Fine-tuning
```

### PEFT

PEFT makes this adaptation more efficient:

```text
Pre-trained Model
       ↓
Freeze Most Parameters
       ↓
Train Small PEFT Parameters
       ↓
New Task
```

---

# Advantages of PEFT

## 1. Lower Memory Usage

Since most model parameters are frozen, fewer parameters need gradients and optimizer states.

## 2. Faster Fine-Tuning

Only a small number of parameters are optimized.

## 3. Smaller Checkpoints

Only the PEFT adapter may need to be stored for a task.

## 4. Lower Computational Cost

PEFT can make large-model adaptation more practical with limited hardware.

## 5. Multiple Task Adapters

One base model can support multiple task-specific adapters.

## 6. Useful for Large Models

PEFT is particularly valuable for models with billions of parameters.

---

# Limitations of PEFT

PEFT is not automatically better than full fine-tuning in every situation.

Possible limitations include:

- Choosing PEFT hyperparameters can be difficult.
- Very large task/domain shifts may require more adaptation capacity.
- Different PEFT methods work differently for different models and tasks.
- The best rank or target layers for LoRA may require experimentation.
- PEFT still requires a suitable base model and training infrastructure.

---

# Important Hyperparameters in LoRA

Some commonly used LoRA parameters are:

### `r`

The rank of the low-rank matrices.

Higher `r` generally means more trainable parameters.

### `lora_alpha`

Controls the scaling of the LoRA update.

### `lora_dropout`

Applies dropout to the LoRA branch and can help regularization.

### `target_modules`

Specifies which model layers receive LoRA adapters.

For Transformer models, attention projection layers are common targets.

Example:

```python
target_modules=["q_proj", "v_proj"]
```

The exact target modules depend on the model architecture.

---

# Conceptual Python Example

Using the Hugging Face PEFT ecosystem, a LoRA configuration can look conceptually like:

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
```

The configuration tells the training system how the LoRA adapters should be created.

> Note: `target_modules` and `task_type` must match the model architecture and task. The example above is typical for some causal language models, not a universal configuration.

---

# PEFT Ecosystem

The Hugging Face ecosystem provides tools for implementing many PEFT methods.

A common workflow is:

```text
Transformers
     +
PEFT
     +
Datasets
     +
Trainer / Training Framework
     ↓
Efficient Fine-Tuning
```

A typical installation is:

```bash
pip install peft transformers datasets accelerate
```

For quantized workflows such as QLoRA, additional libraries may be required depending on the hardware and model.

---

# PEFT vs Full Fine-Tuning

| Feature | Full Fine-Tuning | PEFT |
|---|---|---|
| Base parameters | Updated | Mostly frozen |
| Trainable parameters | Very high | Low |
| GPU memory | High | Lower |
| Training cost | High | Lower |
| Checkpoint size | Large | Small |
| Multiple task versions | Expensive | Easier |
| Large LLM adaptation | Difficult | More practical |

---

# PEFT vs Prompt Engineering

These are different concepts.

### Prompt Engineering

You change the input prompt.

```text
Prompt
  ↓
Pre-trained Model
  ↓
Output
```

No model parameters are trained.

### PEFT

You train a small number of model-related parameters.

```text
Input
  +
Trainable PEFT Parameters
  ↓
Pre-trained Model
  ↓
Output
```

---

# PEFT vs RAG

PEFT and RAG solve different problems.

### PEFT

Changes how the model behaves by adapting its parameters.

### RAG

Provides external information to the model at inference time.

```text
PEFT:
Model → Adapt behavior

RAG:
Question → Retrieve information → Model → Answer
```

They can also be combined.

For example:

```text
RAG + PEFT + LLM
```

can be used to build specialized AI assistants.

---

# Real-World Applications

PEFT is useful in:

- Customer support chatbots
- Domain-specific LLMs
- Medical NLP research
- Financial text analysis
- Code generation
- Sentiment analysis
- Spam detection
- Enterprise AI assistants
- Instruction-following models
- Personalized language models

---

# Key Terms to Remember

| Term | Meaning |
|---|---|
| **PEFT** | Parameter-Efficient Fine-Tuning |
| **LoRA** | Low-Rank Adaptation |
| **QLoRA** | Quantized LoRA |
| **Adapter** | Small trainable module |
| **Rank (r)** | Dimension used in low-rank adaptation |
| **Fine-tuning** | Adapting a pre-trained model |
| **Quantization** | Representing model weights with lower precision |
| **Frozen parameters** | Parameters that are not updated |
| **Trainable parameters** | Parameters updated during training |

---

# Simple Intuition

Imagine a huge textbook containing everything a student already knows.

### Full Fine-Tuning

Rewrite the entire textbook to prepare for one new exam.

### PEFT

Keep the textbook unchanged and create a small set of additional notes specifically for that exam.

The notes are much smaller, easier to train, and can be replaced for another task.

---

# Summary

**PEFT (Parameter-Efficient Fine-Tuning)** is a collection of methods that efficiently adapts pre-trained deep learning models by training only a small subset of parameters or additional parameters.

The most important technique to learn is **LoRA**, which represents weight updates using low-rank matrices.

The main idea is:

```text
Huge Pre-trained Model
        ↓
Freeze Most Parameters
        ↓
Add Small Trainable Parameters
        ↓
Efficient Fine-Tuning
        ↓
Task-Specific Model
```

For large language models, PEFT can significantly reduce the memory, storage, and computational requirements associated with adapting a model to new tasks.

---

## Learning Path

If you are learning PEFT as a beginner, follow this order:

```text
Neural Networks
      ↓
Transformers
      ↓
Attention Mechanism
      ↓
BERT / LLMs
      ↓
Fine-Tuning
      ↓
PEFT
      ↓
LoRA
      ↓
QLoRA
      ↓
Practical LLM Fine-Tuning
```

## One-Line Definition

> **PEFT is a technique for adapting large pre-trained models to new tasks by training only a small number of parameters instead of updating the entire model.**
