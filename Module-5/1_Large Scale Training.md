# Large Scale Training

## 1. The Problem: The Memory Wall

In Module 4, we learned how to make a model run *faster* using Mixed Precision. But speed doesn't matter if the model **doesn't fit** on the chip.

This is the "Memory Wall." It is the single biggest hurdle in modern AI.

### The Math: Why is 70B so big?

You might think: "My hard drive has 1 Terabyte. Why can't I load a 70 Billion parameter model?"
The problem isn't storage (disk); it is **VRAM (Video RAM)**. The GPU needs fast access to the numbers to do math.

Let's audit the memory cost of a **70 Billion Parameter Model** (like [Llama-2-70B](https://huggingface.co/meta-llama/Llama-2-70b-hf)) in **FP16** (2 bytes per number).

| Component | The Math | Size in GB | What is it? |
| :--- | :--- | :--- | :--- |
| **Model Weights** | $70B \times 2 \text{ bytes}$ | **140 GB** | The actual brain (the static file). |
| **Gradients** | $70B \times 2 \text{ bytes}$ | **140 GB** | The temporary error calculation for every weight. |
| **Optimizer (Adam)** | $70B \times 8 \text{ bytes}$ | **560 GB** | **The Silent Killer.** Adam needs to store 2 extra states (Momentum & Variance) for every parameter in FP32 precision to be accurate. |
| **Total Required** | | **~840 GB** | *This doesn't even count the data batch!* |

### The Reality Check

* **The Best GPU:** NVIDIA H100 or A100 (each described in the [NVIDIA Data Center lineup](https://www.nvidia.com/en-us/data-center/)).
* **Max Capacity:** 80 GB.
* **The Result:** $840 \text{ GB} \gg 80 \text{ GB}$.

You cannot load this model. If you try `model.to("cuda")`, your Python script crashes instantly with an OOM (Out Of Memory) error.

---

## 2. The Old Way: DDP (Distributed Data Parallel)

"But wait," you say, "I'll just buy 10 GPUs! $80 \text{ GB} \times 10 = 800 \text{ GB}$!"

This is where the old method, **DDP (Distributed Data Parallel)**, fails us (see the [PyTorch DDP docs](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)).

### How DDP Works (The "Copy-Paste" Strategy)

DDP was designed when models were small (like ResNet-50). Its strategy for speed is simple (summarized in the [official tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)):

1. **Replicate:** It copies the **entire model** onto every single GPU.
2. **Split Data:** If you have a batch of 32 items and 4 GPUs, each GPU gets 8 items.
3. **Sync:** After they calculate their small batch, they average their answers together.

### The Classroom Analogy

Imagine you have a textbook that is **5,000 pages long** (The Model).

* **DDP Approach:** Every student (GPU) must have their **own copy** of the 5,000-page textbook on their desk.
* **The Problem:** The student's desk (VRAM) can only fit 500 pages.

It doesn't matter if you hire 1,000 students. If the book doesn't fit on *one* desk, nobody can open it. Adding more GPUs with DDP does **not** increase your max model size; it only increases your training speed.

**Visualizing the Failure:**

| GPU ID | VRAM Capacity | What DDP tries to load | Result |
| :--- | :--- | :--- | :--- |
| **GPU 0** | 80 GB | Full Model (840 GB) | 💥 **CRASH** |
| **GPU 1** | 80 GB | Full Model (840 GB) | 💥 **CRASH** |
| **GPU 2** | 80 GB | Full Model (840 GB) | 💥 **CRASH** |

### The Conclusion

DDP is useless for Large Language Models. We need a way to take that 840 GB monster and **break it into pieces**, putting a different piece on each GPU.

This solution is called **Sharding** (FSDP), and that is what we will cover next.
