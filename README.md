# PyTorch Lightning for LLMs

A comprehensive, math-first curriculum for training Large Language Models (LLMs) using PyTorch Lightning. This course bridges the gap between theory and practice by explaining the mathematical foundations behind every optimization decision while building production-ready training pipelines.

## 🎯 Course Philosophy

This curriculum treats neural networks not as "magic boxes" but as sophisticated versions of linear regression. Every module combines:

- **Mathematical Theory**: The equations and principles behind each concept
- **PyTorch Lightning Implementation**: Clean, scalable code patterns
- **Hands-on Projects**: Real-world LLM training scenarios

## 📚 Course Modules

### Module 1: The Lightning Philosophy (Refactoring)

**Goal**: Understand why Lightning exists by refactoring messy "Vanilla" PyTorch into clean Lightning structure.

**Key Concepts**:

- The 5-step optimization loop: Forward → Loss → Backward → Step → Zero_grad
- Computational graphs and automatic differentiation
- Hardware placement automation
- Neural networks as extensions of linear regression ($y = mx + b$)

**Project**: Build a Transformer block classifier, first in raw PyTorch, then refactor to Lightning.

**Notebooks**: `Module-1/notebooks/3_The Lightning Solution.ipynb`

---

### Module 2: The Data Pipeline (LightningDataModule)

**Goal**: Master efficient data loading to prevent GPU starvation.

**Key Concepts**:

- The "Starving GPU" problem and bottleneck analysis
- Tokenization (BPE/WordPiece) and vocabulary management
- Dynamic padding vs. static padding for efficiency
- Sequence packing strategies
- Reproducibility through deterministic data splits

**Project**: Create a reusable `LLMDataModule` for datasets like Wikitext with tokenization and batching.

**Notebooks**: `Module-2/notebooks/3_The LLM DataModule.ipynb`

---

### Module 3: The LLM Core (LightningModule)

**Goal**: Wrap pre-trained Hugging Face models into Lightning training systems.

**Key Concepts**:

- Cross-Entropy Loss: $H(P, Q) = -\sum_{x \in V} P(x) \log Q(x)$
- Perplexity as a human-interpretable metric: $\text{PPL} = e^{\text{Loss}}$
- AdamW optimizer and decoupled weight decay
- Mixed precision training (FP16/BF16)

**Project**: **"BabyGPT Trainer"** – Fine-tune a small GPT model on domain-specific text (Shakespeare, Python code, etc.).

**Notebooks**: `Module-3/notebooks/2_Baby GPT Trainer.ipynb`

---

### Module 4: Advanced Training (Precision & Optimization)

**Goal**: Train faster and fit larger models on consumer GPUs.

**Key Concepts**:

- Floating-point formats: FP32 vs. FP16 vs. BF16
- IEEE 754 bit structure (Sign, Exponent, Mantissa)
- Gradient accumulation mathematics
- Learning rate schedulers (Cosine Decay with Warmup)
- Tensor Core acceleration on modern GPUs

**Project**: Benchmark BabyGPT to measure how 16-bit precision and accumulation affect memory and speed.

**Notebooks**: `Module-4/notebooks/3_Benchmarking-Baby-GPT.ipynb`

---

### Module 5: Scaling Up (FSDP & Multi-GPU)

**Goal**: Train models that don't fit on a single GPU using sharding strategies.

**Key Concepts**:

- The Memory Wall: Why 70B models need 840GB+ VRAM
- DDP (Data Parallel) vs. FSDP (Fully Sharded Data Parallel)
- Gradient and optimizer state sharding across devices
- Memory breakdown: Weights + Gradients + Optimizer States

**Project**: **"The Scalable Trainer"** – Configure multi-GPU training with FSDP strategy.

**Notebooks**: `Module-5/notebooks/3_The Scalable Trainer FSDP.ipynb`

---

### Module 6: Efficient Fine-Tuning (PEFT & LoRA)

**Goal**: Fine-tune massive models using minimal parameters and memory.

**Key Concepts**:

- The "Full Fine-Tune" cost problem (7x memory overhead)
- Low-Rank Decomposition: $W = A \times B$
- Parameter-Efficient Fine-Tuning (PEFT) strategies
- LoRA (Low-Rank Adaptation) mathematics
- Freezing base model weights

**Project**: **"Local-LoRA"** – Fine-tune a 7B+ parameter model (Llama 3, Mistral) on consumer hardware.

**Notebooks**: `Module-6/notebooks/` (in development)

---

## 🛠️ Technical Stack

- **Framework**: PyTorch Lightning
- **Models**: Hugging Face Transformers (GPT-2, TinyLlama, Llama 2/3, Mistral)
- **Optimization**: AdamW with weight decay
- **Precision**: Mixed precision (BF16/FP16)
- **Scaling**: FSDP for multi-GPU training
- **Fine-tuning**: PEFT + LoRA for parameter efficiency

## 🚀 Getting Started

### Prerequisites

```bash
# Core dependencies
pip install torch pytorch-lightning transformers datasets
pip install peft accelerate bitsandbytes
```

### Project Structure

```text
Pytorch-Lightning-for-LLMs/
├── Module-1/          # Lightning basics and refactoring
├── Module-2/          # Data pipelines and tokenization
├── Module-3/          # LLM training core
├── Module-4/          # Precision and optimization
├── Module-5/          # Multi-GPU scaling with FSDP
├── Module-6/          # PEFT and LoRA fine-tuning
├── Official Lightning Notebooks/  # Reference implementations
└── Lesson Plan.md     # Detailed curriculum outline
```

## 📖 Learning Path

1. **Start with Module 1** to understand the optimization loop fundamentals
2. **Progress sequentially** through modules as each builds on previous concepts
3. **Complete the projects** in each module's notebooks
4. **Experiment** with different datasets and model architectures
5. **Scale up** gradually from single GPU to multi-GPU setups

## 🎓 Key Takeaways

By completing this curriculum, you will:

- Understand the mathematical foundations of LLM training
- Build production-ready training pipelines with PyTorch Lightning
- Optimize memory usage and training speed
- Scale training across multiple GPUs
- Fine-tune large models on consumer hardware
- Avoid common pitfalls (NaN losses, GPU starvation, memory crashes)

## 📚 References

Each module includes citations to:

- Academic papers (Attention Is All You Need, LoRA, FSDP, etc.)
- Official documentation (PyTorch, Lightning, Hugging Face)
- Industry best practices (NVIDIA, Google Brain)

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

This is a learning resource. Feel free to:

- Report issues or unclear explanations
- Suggest additional topics or modules
- Share your training results and experiments

---

**Note**: This curriculum emphasizes understanding *why* each technique works, not just *how* to implement it. Every optimization decision is grounded in mathematics and hardware constraints.
