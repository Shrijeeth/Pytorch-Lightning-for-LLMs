# The Scalable Trainer (FSDP)

We have discussed the theory of breaking the model into pieces (Sharding). Now, let's implement it. In raw PyTorch, this takes 100+ lines of complex configuration. In **PyTorch Lightning**, it takes about 5 lines.

## The Auto-Wrap Policy (The Most Important Concept)

FSDP works by "sharding" (cutting up) the model. But **where** should it cut?

If you cut the model into tiny pieces (e.g., every single Linear Layer), the GPUs have to communicate thousands of times per second to exchange these tiny pieces. This creates network traffic jams.

**The Solution:** We cut the model at the **Block Level**.

* **Analogy:** Instead of mailing a book page-by-page (too many envelopes), we mail it **chapter-by-chapter**.
* **In Code:** We tell FSDP, *"Treat the `GPT2Block` (or `LlamaDecoderLayer`) as one solid unit."*

### The Code Implementation

Create a file named `module5_fsdp.py`. Notice how we import the specific block class from the model architecture.

```python
import lightning as L
import torch
from lightning.pytorch.strategies import FSDPStrategy

# 1. Import the specific Layer class we want to wrap
# If using Llama, this would be LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from module2_data import LLMDataModule
from module3_model import BabyGPT

def train_fsdp():
    # 2. Define the Strategy
    fsdp_strategy = FSDPStrategy(
        # THE CRITICAL LINE:
        # We tell FSDP to keep each GPT2Block intact on a single GPU during calculation.
        # This minimizes the "Accordion" motion to manageable chunks.
        auto_wrap_policy={GPT2Block},

        # "FULL_SHARD": Shard parameters, gradients, AND optimizer states.
        # This gives maximum memory savings.
        sharding_strategy="FULL_SHARD",

        # Keep this False for speed. Set True only if you are desperate for memory.
        cpu_offload=False,
    )

    # 3. Trainer Config
    trainer = L.Trainer(
        accelerator="gpu",
        devices=2, # FSDP requires at least 2 GPUs to work!
        strategy=fsdp_strategy,

        # RULE: Always use bf16 with FSDP.
        # fp16 can cause instability when sharding gradients.
        precision="bf16-mixed",

        max_epochs=3,
        log_every_n_steps=10
    )

    # 4. Data & Model
    # We try a larger model (gpt2-medium) to prove we can fit it.
    dm = LLMDataModule(model_name="gpt2-medium", batch_size=4)
    model = BabyGPT(model_name="gpt2-medium")

    # 5. Fit
    print("Starting FSDP Training... 🌶️")
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    train_fsdp()
```

-----

## Comparison: DDP vs. FSDP

Why would you ever use DDP again?

* **DDP** is simple and fast for *small* models (e.g., ResNet, BERT).
* **FSDP** is mandatory for *large* models (e.g., Llama, GPT).

| Feature | DDP (Standard) | FSDP (Sharded) |
| :--- | :--- | :--- |
| **Philosophy** | **Replication:** Every GPU has a copy of the book. | **Sharding:** Every GPU holds a few chapters. |
| **Memory Limit** | Limited by the VRAM of **1 GPU**. (If 1 GPU is 16GB, your model must be \<16GB). | Limited by the **Sum** of all GPUs. (If you have 8 x 16GB cards, you have 128GB total space). |
| **Communication** | **Low:** Only syncs gradients at the end of a step. | **High:** Syncs weights constantly (The Accordion). Needs fast cables. |
| **Speed** | 🏎️ Fastest for small models. | 🐢 Slower for small models (overhead), but the *only* way to run big ones. |

### Visualizing the Optimizer State (The Hidden Saver)

The biggest memory killer is the **Optimizer (Adam)**.

* **In DDP:** GPU 1 holds the momentum for the *entire* model. GPU 2 holds the momentum for the *entire* model. This is redundant.
* **In FSDP:** GPU 1 *only* tracks the momentum for the specific weights it owns. There is zero redundancy.

-----

## Advanced Tip: `cpu_offload=True`

What if you have a massive 70B parameter model, but you only have 2 small GPUs (24GB each)? Even FSDP might not be enough.

You can use the **Emergency Brake**: CPU Offloading.

```python
fsdp_strategy = FSDPStrategy(..., cpu_offload=True)
```

### How it works

1. **VRAM (GPU Memory):** The "Kitchen Counter." Fast, but small.
2. **RAM (System Memory):** The "Pantry." Slow, but huge (you probably have 64GB+).

With `cpu_offload=True`:

* FSDP keeps the model in **System RAM** (Pantry).
* When Layer 1 is needed, it grabs it from RAM, puts it on GPU, computes, and throws it back to RAM.
* **The Cost:** It is painfully slow because data has to travel over the PCIe bus (the cable connecting GPU to Motherboard).
* **The Benefit:** You can fine-tune massive models on cheap hardware.

-----

## 🛠️ Homework & Experiments

1. **Simulate it (Colab):**
    You cannot run FSDP on a single GPU. Open Google Colab, go to `Runtime > Change Runtime Type`, and select **T4 GPU (x2)** if available (requires Pro), or use Kaggle which often provides dual GPUs (T4 x2).

2. **The "Wrap" Test:**
    Modify the code. Remove `auto_wrap_policy={GPT2Block}`.

      * *Observation:* Training will likely become extremely slow.
      * *Why?* Without the policy, FSDP breaks the model into tiny pieces (individual Linear layers). The GPUs spend more time talking ("Send me the bias for neuron 5\!") than doing math.

3. **Scale Up:**
    Try to load `gpt2-xl` (1.5 Billion parameters).

      * Try loading it with standard DDP (or just `BabyGPT(strategy="auto")`). It will likely crash on a T4 (16GB).
      * Try loading it with the `FSDPStrategy` code above. It should fit\!
