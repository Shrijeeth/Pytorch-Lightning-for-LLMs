# The "Vanilla" PyTorch Way (The Problem)

Now, let's look at how we translate those 5 math steps into Python code using standard ("Vanilla") [PyTorch](https://pytorch.org/docs/stable/index.html).

## The Task

Read the code below. Notice how the **Model Logic** (the research) is completely tangled with the **Engineering Logic** (loops, device management, logging).

> ✳️ Need a refresher on core data utilities? Check the PyTorch docs for [TensorDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset) and [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

```python
# vanilla_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- 1. The Data (Engineering) ---
# Creating dummy data for illustration
X = torch.randn(100, 10) # 100 samples, 10 features
y = torch.randn(100, 1)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10)

# --- 2. The Model (Research/Math) ---
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

# --- 3. The "Boilerplate" Loop (The Problem) ---
def train():
    # A. Hardware Management (Fragile!)
    # If you forget this, your code crashes on GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We have to manually move the model to the device
    model = SimpleModel().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    model.train()

    # B. The Loop (Repetitive)
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(dataloader):
            # !!! BOILERPLATE ALERT !!!
            # You must manually move every batch to the GPU
            data, target = data.to(device), target.to(device)

            # Step 1: Forward
            output = model(data)

            # Step 2: Loss
            loss = criterion(output, target)

            # Step 5: Zero Grad (Crucial reset)
            optimizer.zero_grad()

            # Step 3: Backward
            loss.backward()

            # Step 4: Step
            optimizer.step()

            # Logging (Clutters the loop)
            if batch_idx % 2 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
```

-----

## Why is this code dangerous?

The code above works, but it is **fragile**.

* **The "Device" Trap:** Notice `data.to(device)`. If you forget this on line 42, or if you try to run this on a multi-GPU setup, the code crashes or fails silently. You are hard-coding hardware logic into your math loop, and the [CUDA semantics docs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-semantics) show just how easy it is to make mistakes here.
* **The "Refactoring" Nightmare:** Imagine you want to add **Gradient Accumulation** (updating weights only every 4 steps to simulate a larger batch size). You would have to manually alter the `for` loop logic—compare with the Hugging Face guide on [gradient accumulation](https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-accumulation).
* **The "FP16" Headache:** If you want to use Mixed Precision (to save memory), you have to wrap the forward pass and backward pass in special "Scalers." The loop becomes twice as long and harder to read, as highlighted in the [Automatic Mixed Precision guide](https://pytorch.org/docs/stable/amp.html).

-----

## The Lightning Solution

PyTorch Lightning was created to solve this. It asks you to answer two questions (see the [Lightning docs](https://lightning.ai/docs/pytorch/stable/) for a deeper overview):

1. **What is the model?** (The Math)
2. **What is the data?** (The Math)

It then says: *"I will handle the loops, the GPUs, the saving, and the loading. You just focus on the math."*

In the next tutorials, we will refactor this messy loop into a clean Lightning Module.
