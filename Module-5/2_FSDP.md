# FSDP

## 1. The Philosophy: Shard, Don't Copy

In the previous chapter, we saw that **DDP** failed because it tried to force the entire 70B parameter model onto every single GPU.

**FSDP (Fully Sharded Data Parallel)** changes the rules. Instead of replicating the model, it **shards** (slices) it across all available hardware (see the [PyTorch FSDP overview](https://pytorch.org/docs/stable/fsdp.html)).

### The Encyclopedia Analogy

Imagine a classroom of 8 students (GPUs) trying to study a massive, 8,000-page Encyclopedia (The Model).

* **The DDP Way (Replication):** The teacher forces every student to have their own copy of the 8,000-page book on their tiny desk.
  * *Result:* The desks collapse. There isn't enough room.
* **The FSDP Way (Sharding):** The teacher tears the book apart.
  * Student 1 holds pages 1–1,000.
  * Student 2 holds pages 1,001–2,000.
  * ...
  * Student 8 holds pages 7,001–8,000.
  * *Result:* Everyone sits comfortably. The "desk usage" (VRAM) is low.

---

## 2. How it Works: The "Accordion" Motion

You might be asking: *"If GPU 1 only has the first 10% of the model, how can it calculate the answer for the whole model?"*

The magic is in **Communication**. The GPUs constantly pass data back and forth in a synchronized dance. This is often called the **Accordion Motion** because the memory usage expands and contracts instantly (also illustrated in the [Meta/FAIR FSDP blog](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)).

### The Step-by-Step Process

Let's say we are training a Neural Network with 100 Layers.

#### 1. The Resting State (Sharded)

At the start, GPU 1 only holds the "shards" (pieces) of Layer 1, Layer 2... Layer 100. It cannot do any math yet because it doesn't have the full picture.

#### 2. The All-Gather (Expand)

We need to compute **Layer 1**.

* GPU 1 shouts: *"I have piece A of Layer 1!"*
* GPU 2 shouts: *"I have piece B of Layer 1!"*
* All GPUs instantly share their pieces.
* **Now:** Every GPU temporarily has the **Full Layer 1** in memory.

#### 3. The Computation

Every GPU calculates the math for Layer 1 using its specific batch of data.

#### 4. The Scatter/Discard (Contract)

* **Crucial Step:** As soon as the math for Layer 1 is done, the GPUs **delete** the Full Layer 1 from memory.
* They go back to just holding their tiny shard.
* Memory usage drops back to near zero.

#### 5. Repeat for Layer 2

They All-Gather Layer 2, compute, and delete Layer 2.

---

## 3. The Math of Memory Savings

The beauty of FSDP is that memory savings scale linearly. If you buy more GPUs, your memory burden decreases perfectly.

$$Memory_{FSDP} \approx \frac{Memory_{DDP}}{N_{GPUs}}$$

### A Concrete Example (Llama 70B)

Let's look at the 70 Billion Parameter model that requires **840 GB** of VRAM to train.

* **Scenario A: 1 GPU (H100 - 80GB)**
  * Required: 840 GB.
  * Available: 80 GB.
  * Result: **Crash.**

* **Scenario B: 8 GPUs (H100s) with DDP**
  * DDP replicates the model.
  * Required per GPU: 840 GB.
  * Result: **Crash.**

* **Scenario C: 8 GPUs (H100s) with FSDP**
  * We divide the load by 8.
  * $\frac{840 \text{ GB}}{8} = \mathbf{105 \text{ GB}}$.
  * *Result:* Still slightly too big for an 80GB card! But we are close.

* **Scenario D: 16 GPUs (H100s) with FSDP**
  * $\frac{840 \text{ GB}}{16} = \mathbf{52.5 \text{ GB}}$.
  * *Result:* **Success!** We fit comfortably under the 80GB limit.

### The Trade-off

There is no free lunch.

* **What you gain:** Infinite Memory (just add more GPUs).
* **What you pay:** Network Traffic. Because the GPUs are constantly shouting "All-Gather!" and exchanging weights, you need very fast cables (NVLink or InfiniBand) connecting the GPUs (see NVIDIA's [NVLink overview](https://www.nvidia.com/en-us/data-center/nvlink/)). If your cables are slow, FSDP will be slow.
