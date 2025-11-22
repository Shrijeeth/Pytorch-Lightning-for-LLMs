# Baby GPT Trainer

Now that we have the fuel (the `LLMDataModule` from Module 2), we need to build the engine. We will wrap a Hugging Face model inside a PyTorch Lightning system.

We call this **BabyGPT** because we are keeping it simple, but the architecture is identical to the models used by OpenAI and Meta.

## Prerequisites

Make sure you have the libraries:

```bash
pip install torch lightning transformers
```

-----

## Part 1: The Model Code (`model.py`)

This file contains the "Brain." We will break down the code into 3 logical parts: **The Setup**, **The Loop**, and **The Optimizer**.

### A. The Setup (`__init__` & `forward`)

This looks standard, but there is one "Pro Move" here: **Gradient Checkpointing**.

```python
import torch
import lightning as L
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoConfig

class BabyGPT(L.LightningModule):
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", lr=2e-4):
        super().__init__()
        self.save_hyperparameters() # Saves 'lr' and 'model_name' to a file for later

        # 1. Load the Brain
        # AutoModelForCausalLM downloads the architecture AND the pre-trained weights.
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # 2. Gradient Checkpointing (The VRAM Saver)
        # Normal training remembers every step of the math to calculate gradients later.
        # This eats massive RAM. Checkpointing forgets the middle steps and re-calculates
        # them only when needed. It makes training 20% slower but saves 50% VRAM.
        self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # We act as a "Passthrough" to the Hugging Face model
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
```

### B. The Loop (`training_step` & `validation_step`)

Here we define what happens during a single step of training.

```python
    def training_step(self, batch, batch_idx):
        # 'batch' contains input_ids, attention_mask, AND labels (thanks to the Collator)

        # The ** syntax unpacks the dictionary.
        # self(**batch) is the same as self(input_ids=..., attention_mask=...)
        outputs = self(**batch)

        # Hugging Face models calculate the Loss automatically if you provide labels.
        loss = outputs.loss

        # Log to the progress bar so we can see it live
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss = outputs.loss

        # Calculate Perplexity (The "Confusion" Score)
        # Loss is Logarithmic (e.g., 2.5). Perplexity is Linear (e.g., 12.1).
        # We convert by using exponent (e^loss)
        perplexity = torch.exp(val_loss)

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_perplexity", perplexity, prog_bar=True)
```

### C. The Optimizer (`configure_optimizers`)

**This is the most critical part of the code.**
We are using **AdamW** (Adam with Weight Decay).

* **Weight Decay** pushes weights towards zero to prevent overfitting.
* **The Rule:** You should only push "MatMul" weights (2D matrices) towards zero. You should **never** push Biases or LayerNorms towards zero, or the model becomes unstable.

<!-- end list -->

```python
    def configure_optimizers(self):
        # 1. Filter parameters into two groups
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # The Logic:
            # If it's a Bias, or a LayerNorm, or a 1D vector -> NO DECAY
            # If it's a big Weight Matrix (Linear Layer) -> DECAY
            if param.dim() < 2 or "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # 2. Create the groups
        optim_groups = [
            {"params": decay_params, "weight_decay": 0.1}, # Shrink these!
            {"params": no_decay_params, "weight_decay": 0.0}, # Leave these alone!
        ]

        # 3. Initialize AdamW
        optimizer = AdamW(optim_groups, lr=self.hparams.lr, betas=(0.9, 0.95))

        # 4. The Scheduler (Cosine Decay)
        # LLM training usually starts with a high LR and slowly lowers it to 0.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000 # In real training, set this to total_training_steps
        )

        return [optimizer], [scheduler]
```

-----

## Part 2: Deep Dive - The "Magic" of Labels

You might be wondering: *"Where did we define the targets?"*
In standard Deep Learning, you manually split data into $X$ (Input) and $Y$ (Target).
In Hugging Face Causal LM, **the Input is the Target.**

The model automatically shifts the tokens internally.

* **Input Sequence:** `[A, B, C, D]`
* **Internal Input:** `[A, B, C]`
* **Internal Target:** `[B, C, D]`

The model asks: "Given **A**, did I predict **B**?"
This happens automatically inside `outputs = self(**batch)` as long as the `labels` key exists in the batch.

-----

## Part 3: Integration (`train.py`)

Now we glue Module 2 (Data) and Module 3 (Model) together.

```python
# train.py
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

# Import the classes we just wrote
# (Assuming you saved them in files named module2_data.py and module3_model.py)
from module2_data import LLMDataModule 
from module3_model import BabyGPT

def train():
    # 1. Data Fuel
    dm = LLMDataModule(model_name="gpt2", batch_size=8)

    # 2. The Engine
    # We use GPT-2 for this demo because it fits on almost any laptop.
    # To use a "Real" LLM, change model_name to "TinyLlama/TinyLlama-1.1B..."
    model = BabyGPT(model_name="gpt2", lr=3e-4)

    # 3. The Autosave (Checkpointing)
    # Saves the model every time the Validation Loss gets better.
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="babygpt-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    # 4. The Manager (Trainer)
    trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto", # Auto-detects Mac (MPS), CUDA, or CPU
        devices=1,
        # Mixed Precision: Runs math in 16-bit (fast), keeps weights in 32-bit (stable).
        # Huge speedup for free!
        precision="16-mixed",
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )

    # 5. Fit
    print("Starting Training... 🚀")
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    train()
```

-----

## 🛠️ Homework & Experiments

1. **Run the script:** Copy the code into files and run `train.py`.
2. **Monitor Perplexity:** Watch the `val_perplexity` in the progress bar.
      * *Start:* It will be high (maybe \~50-100 if using pre-trained weights, or \~50,000 if training from scratch).
      * *Goal:* See if it drops. If it stays above 1000, your model isn't learning.
3. **Break it (The Educational Crash):**
      * Go to `configure_optimizers`.
      * Remove the logic that separates `decay_params` and `no_decay_params`.
      * Set `weight_decay=0.1` for **everything**.
      * *Hypothesis:* The validation loss will likely become unstable or get worse because you are shrinking the biases and LayerNorms, which breaks the internal statistics of the network.
