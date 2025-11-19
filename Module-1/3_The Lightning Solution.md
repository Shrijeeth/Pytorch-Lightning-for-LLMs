# The Lightning Solution

[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) forces you to organize your code into a specific Class structure called a `LightningModule`. It abstracts away the **Engineering** (loops, devices, logging), leaving you with the **Mathematics** (architecture, loss, optimization).

## The Mental Mapping

Think of this as moving from a "freestyle" coding approach to a "structured" template.

| Concept | Vanilla PyTorch | PyTorch Lightning ([`pl.LightningModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)) |
| :--- | :--- | :--- |
| **Architecture** | `class Model(nn.Module)` | `def __init__` & `def forward` |
| **The Math (Loss)** | Inside the `for` loop | `def training_step` |
| **Optimization** | `optim.SGD(...)` | `def configure_optimizers` |
| **The Loop/Device** | Manually written loops | Handled by the `Trainer` |

-----

## Interactive Exercise: Refactor the Code

Your task is to take the logic we discussed in the "Vanilla" section and place it into the Lightning slots.

> 📓 Prefer running this walkthrough interactively? Open the companion notebook: [3_The Lightning Solution.ipynb](notebooks/3_The%20Lightning%20Solution.ipynb).

### Step 1: Install Lightning

If you haven't already, install the library (see the [official installation guide](https://lightning.ai/docs/pytorch/stable/starter/installation.html)):

```bash
pip install lightning
```

### Step 2: The Template (Your Assignment)

Here is the skeleton (adapted from the [LightningModule API reference](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)). I have left **TODO** comments where you need to fill in the blanks.

**Copy this code and try to fill in the `training_step`.**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl

# --- 1. The Lightning Module ---
class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # The Architecture (Same as Vanilla)
        self.layer = nn.Linear(10, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # The Prediction
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        # THE MAGIC HAPPENS HERE
        # Lightning handles device placement automatically.
        # 'batch' is already on the GPU if you selected one.
        x, y = batch

        # TODO: 1. Calculate prediction (y_hat) using self(x)
        # ---------------------------------------------------

        # TODO: 2. Calculate loss using self.criterion
        # ---------------------------------------------------

        # TODO: 3. Return the loss tensor
        # Note: You DO NOT need to call backward(), step(), or zero_grad().
        # Lightning does this automatically if you return the loss.
        pass

    def configure_optimizers(self):
        # Define the optimizer (Same as Vanilla)
        return optim.SGD(self.parameters(), lr=0.01)

# --- 2. Execution ---
def run_lightning():
    # Data setup (Same as Vanilla)
    dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
    dataloader = DataLoader(dataset, batch_size=10)

    # Model
    model = LitModel()

    # The "Manager" (The Engineering)
    # accelerator="auto" will automatically find your GPU/MPS/TPU
    trainer = pl.Trainer(max_epochs=5, accelerator="auto")  # Trainer docs: https://lightning.ai/docs/pytorch/stable/common/trainer.html

    # The Loop
    trainer.fit(model=model, train_dataloaders=dataloader)

if __name__ == "__main__":
    run_lightning()
```

-----

### 🛠️ The Solution & Explanation

Did you try it? Here is what the completed `training_step` should look like (compare with the [Lightning logging guide](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#logging)).

```python
    def training_step(self, batch, batch_idx):
        x, y = batch

        # 1. Forward Propagation (The Guess)
        y_hat = self(x)

        # 2. Loss Calculation (The Error)
        loss = self.criterion(y_hat, y)

        # Logging (Optional, but easy in Lightning)
        self.log("train_loss", loss)

        # 3. Return the result
        return loss
```

## Why is this better?

1. **The "Invisible" Loop:** Notice that `backward()`, `optimizer.step()`, and `zero_grad()` are **gone**. Lightning calls these for you automatically behind the scenes as soon as you `return loss`.
2. **Device Agnostic:** Notice we never wrote `.to(device)`. The `Trainer` checks your hardware. If you have a GPU, it moves the batch to the GPU before passing it to `training_step`. If you only have a CPU, it keeps it there. You don't change a single line of code ([accelerator docs](https://lightning.ai/docs/pytorch/stable/common/trainer.html#precision-and-accelerators)).
3. **The Trainer:** The line `trainer.fit()` replaces the nested `for epoch` and `for batch` loops. This safeguards you against writing a broken loop, just as outlined in the [Trainer overview](https://lightning.ai/docs/pytorch/stable/common/trainer.html#fit).

**Observation:** By using Lightning, we have successfully separated the **Math** (defined in `LitModel`) from the **Engineering** (handled by `Trainer`).
