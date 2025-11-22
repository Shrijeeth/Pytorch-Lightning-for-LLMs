# Benchmarking BabyGPT

We are done with the theory. Now, we are going to act like scientists. We won't just *assume* Mixed Precision is faster; we are going to write a script to **prove it**.

## Step 1: Update BabyGPT for Schedulers

First, we need to upgrade our Model class. In Module 3, we used a basic implementation. Now, we will add the "Warmup + Cosine Decay" logic using a helper tool from Hugging Face.

**Open `module3_model.py` and update `configure_optimizers`:**

```python
# module3_model.py
import torch
import lightning as L
from transformers import get_cosine_schedule_with_warmup

# ... (The rest of the class remains the same) ...

    def configure_optimizers(self):
        # 1. Create AdamW
        # We start with the learning rate defined in __init__ (e.g., 3e-4)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        # 2. Calculate the Timeline
        # A scheduler needs to know: "How long is the race?" so it knows when to slow down.
        # Lightning calculates this for us: (Total Samples / Batch Size) * Epochs
        total_steps = self.trainer.estimated_stepping_batches

        # We want to warm up for the first 10% of the training
        warmup_steps = int(total_steps * 0.1)

        # 3. Create The Scheduler
        # This function creates the curve we saw in the theory graph
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # 4. Return to Lightning
        # We return a specific dictionary format.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # CRITICAL: We update the LR every 'step' (batch), not every 'epoch'.
                "interval": "step",
            }
        }
```

-----

## Step 2: The Benchmark Script (`benchmark.py`)

Now for the experiment. We will run the exact same model 3 times with different settings to see how **Precision** and **Accumulation** affect speed and memory.

**Create a new file `benchmark.py`:**

```python
import time
import lightning as L
import torch
from module2_data import LLMDataModule
from module3_model import BabyGPT

# --- The Experiment Configs ---
configs = [
    # Control Group: The default safe, slow way
    {"name": "Baseline (FP32)", "precision": "32-true", "accum": 1, "batch": 4},

    # Experiment A: Mixed Precision (Should be faster & lighter)
    # Note: We can double the batch size because memory usage drops!
    {"name": "Mixed Precision (FP16)", "precision": "16-mixed", "accum": 1, "batch": 8},

    # Experiment B: The "Pro" Setup (BF16 + Accumulation)
    # Effective Batch Size = 8 * 4 = 32. This simulates a high-end training run.
    {"name": "BF16 + Accumulation", "precision": "bf16-mixed", "accum": 4, "batch": 8},
]

def run_benchmark():
    print(f"{'Config Name':<25} | {'Time/Epoch':<10} | {'Peak Mem':<10}")
    print("-" * 55)

    for conf in configs:
        # 1. Setup Data
        dm = LLMDataModule(model_name="gpt2", batch_size=conf['batch'])

        # 2. Setup Model
        model = BabyGPT(model_name="gpt2")

        # 3. Setup Trainer
        # limit_train_batches=50 -> We only run 50 steps. We don't need to finish
        # training to measure speed; we just need a sample.
        trainer = L.Trainer(
            max_epochs=1,
            limit_train_batches=50,
            precision=conf['precision'],
            accumulate_grad_batches=conf['accum'],
            enable_checkpointing=False, # Don't save files, just test speed
            logger=False,               # Don't log to file
            enable_progress_bar=False,  # Keep console clean
            accelerator="auto",
            devices=1
        )

        # 4. The Measurement
        torch.cuda.reset_peak_memory_stats() # Reset memory counter
        start_time = time.time()

        trainer.fit(model, datamodule=dm)

        end_time = time.time()
        # Convert bytes to Gigabytes (GB)
        memory_used = torch.cuda.max_memory_allocated() / 1e9

        print(f"{conf['name']:<25} | {end_time - start_time:.2f}s      | {memory_used:.2f} GB")

if __name__ == "__main__":
    run_benchmark()
```

-----

## 5\. Analysis & Results

When you run `python benchmark.py`, you will see a table generated in your terminal. Here is what typical results look like on a standard GPU (like a Tesla T4 or RTX 3060):

| Config Name | Time (50 steps) | Peak Mem | Analysis |
| :--- | :--- | :--- | :--- |
| **Baseline (FP32)** | 45.2s | 6.2 GB | **The Control:** Slow and heavy. It eats VRAM quickly. |
| **Mixed (FP16)** | **28.1s** | **3.8 GB** | **The Winner:** It is almost 2x faster (thanks to Tensor Cores) and uses \~40% less memory. |
| **BF16 + Accum** | 30.5s | 3.9 GB | **The Pro:** Slightly slower than FP16 (due to the overhead of stopping to accumulate gradients), but allows for a massive Effective Batch Size. |

**Important Note for Mac Users:**
Apple Silicon (M1/M2/M3) uses `MPS` (Metal Performance Shaders).

* MPS **does not** currently support `bf16` well.
* If you are on a Mac, change the config to `precision="16-mixed"` or stick to `32-true`.

-----

## 6\. Reference Cheat Sheet

Here are the flags you should memorize for PyTorch Lightning.

| Feature | Lightning Flag | Why use it? |
| :--- | :--- | :--- |
| **Mixed Precision** | `precision="16-mixed"` | **Speed:** 2x faster training. **Memory:** 0.5x usage. |
| **BF16** | `precision="bf16-mixed"` | **Stability:** Prevents "NaN" errors. Requires NVIDIA Ampere (30-series) or newer. |
| **Grad Accumulation** | `accumulate_grad_batches=K` | **Virtual RAM:** Simulates having a massive GPU on a laptop. |
| **Gradient Clip** | `gradient_clip_val=1.0` | **Safety:** If a single bad batch produces a massive gradient, this caps it at 1.0 so it doesn't break the model. |

-----

## 🛠️ Homework

1. **Check your Hardware:**
    Run this Python one-liner to see if your GPU supports the "Gold Standard" (BF16):

    ```python
    import torch
    print(f"BF16 Supported: {torch.cuda.is_bf16_supported()}")
    ```

      * *If True:* Use `bf16-mixed` forever.
      * *If False:* Use `16-mixed`.

2. **Find the Limit (Stress Test):**
    Modify `benchmark.py`. Increase the `batch_size` in the FP32 config until the program crashes with an **OOM (Out Of Memory)** error.

      * *Example:* Maybe FP32 crashes at Batch 16.
      * Now try FP16. You will likely reach Batch 32 or 48 before crashing.

3. **The Loss Curve:**
    Train the model for 5 full epochs using the new Scheduler. Look at the loss curve (using TensorBoard or just logging).

      * *Observation:* Does the loss go down smoother than the "Constant Learning Rate" version from Module 3? (It should dive quickly at the start and then flatten out smoothly).
