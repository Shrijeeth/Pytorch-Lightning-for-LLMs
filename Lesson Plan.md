
# 📚 The Curriculum: PyTorch Lightning for LLMs

---

## Module 1: The Lightning Philosophy (Refactoring)

**Goal:** Understand why Lightning exists by refactoring a messy "Vanilla" PyTorch loop into a clean Lightning structure.

* **Math & Theory:**
  * The Computational Graph.
  * The 5 steps of the optimization loop: **Forward**, **Loss**, **Backward**, **Step**, **Zero_grad**.
  * How Lightning automates hardware placement.
* **Project:** Build a simple Transformer block classifier, first in raw PyTorch, then refactor it to Lightning.

---

## Module 2: The Data Pipeline (LightningDataModule)

**Goal:** Master efficient data loading for text. LLMs starve without fast data.

* **Math & Theory:**
  * Tokenization logic (BPE/WordPiece).
  * Sequence Packing.
  * Dynamic Padding (strategies to minimize compute waste).
* **Project:** Create a reusable `LLMDataModule` that ingests a raw text dataset (like Wikitext) and outputs tokenized, batched tensors ready for training.

---

## Module 3: The LLM Core (LightningModule)

**Goal:** Wrap a pre-trained Hugging Face model (e.g., TinyLlama or GPT-2) into a Lightning system.

* **Math & Theory:**
  * The Cross-Entropy Loss formula: $H(p, q) = -\sum p(x) \log q(x)$.
  * Perplexity calculations.
  * Weight Decay logic in AdamW.
* **Project:** **"BabyGPT Trainer"** – A complete training script that fine-tunes a small GPT model on a specific style of text (e.g., Shakespeare or Python code).

---

## Module 4: Advanced Training (Precision & Optimization)

**Goal:** Train faster and fit larger models on smaller GPUs.

* **Math & Theory:**
  * Mixed Precision (FP16/BF16 vs. FP32).
  * Gradient Accumulation math.
  * Learning Rate Schedulers (Cosine Decay with Warmup).
* **Project:** Benchmark the "BabyGPT" from Module 3 to see how 16-bit precision and accumulation change memory usage and speed.

---

## Module 5: Scaling Up (FSDP & Multi-GPU)

**Goal:** The "Secret Sauce" of modern LLMs. How to train models that don't fit on one GPU.

* **Math & Theory:**
  * Data Parallel (DDP) vs. Fully Sharded Data Parallel (FSDP).
  * Visualizing how gradients and optimizer states are sharded across devices.
* **Project:** **"The Scalable Trainer"** – A script configured to train across multiple GPUs (or simulated on Colab) using the FSDP strategy.

---

## Module 6: Efficient Fine-Tuning (PEFT & LoRA)

**Goal:** Fine-tune massive models using minimal parameters.

* **Math & Theory:**
  * Low-Rank Decomposition: $W = A \times B$.
  * Mathematical derivation of the parameter savings formula.
* **Project:** **"Local-LoRA"** – Fine-tune a 7B parameter model (like Llama 3 8B or Mistral) on a consumer GPU using PyTorch Lightning + PEFT + LoRA.
