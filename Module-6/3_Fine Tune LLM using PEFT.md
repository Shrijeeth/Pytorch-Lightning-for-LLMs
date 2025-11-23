# Fine Tune LLM using PEFT

## 1\. Prerequisites (The Toolbelt)

We need a specific set of libraries to pull this off.

* **`transformers`**: To load the model architecture (see the [Transformers docs](https://huggingface.co/docs/transformers/index)).
* **`bitsandbytes`**: The "Compressor." This library allows us to load models in **4-bit mode**, shrinking them by 4x (see [bitsandbytes docs](https://github.com/TimDettmers/bitsandbytes)).
* **`peft`**: The "Adapter." This library handles the LoRA logic (injecting the sticky notes) (see the [PEFT docs](https://huggingface.co/docs/peft/index)).

<!-- end list -->

```bash
pip install torch lightning transformers peft bitsandbytes
```

-----

## 2\. The Code: `lora_model.py`

This is a production-ready template. I will explain exactly what is happening inside the code in the breakdown below.

```python
import lightning as L
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class LocalLoRA(L.LightningModule):
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B", r=16, alpha=32):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # --- 1. The Compression (4-Bit Quantization) --- (QLoRA: https://arxiv.org/abs/2305.14314)
        # This config tells the model: "Don't use 32 bits per number. Use 4 bits."
        # This reduces the model size from ~16GB to ~5GB.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", # A special 4-bit format optimized for AI
            bnb_4bit_compute_dtype=torch.bfloat16 # We do the math in bf16 for stability
        )

        # --- 2. Load the Frozen Base Model ---
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" # Automatically puts layers on GPU
        )

        # This helper function freezes the layers and ensures 
        # LayerNorms are in float32 (required for stability in 4-bit training)
        self.base_model = prepare_model_for_kbit_training(self.base_model)

        # --- 3. Define the Adapter (The Sticky Notes) --- (per the LoRA paper: https://arxiv.org/abs/2106.09685)
        peft_config = LoraConfig(
            r=r,                  # Rank: How "complex" the adapter is
            lora_alpha=alpha,     # Alpha: How "loud" the adapter is
            lora_dropout=0.05,    # Randomly turn off neurons to prevent overfitting
            bias="none",
            task_type="CAUSAL_LM",
            # We only attach adapters to the Attention mechanism (Query and Value matrices)
            # This is the most efficient place to learn new styles.
            target_modules=["q_proj", "v_proj"] 
        )

        # --- 4. Inject the Adapters ---
        # The self.model variable now holds the Base Model wrapped in the Adapter.
        self.model = get_peft_model(self.base_model, peft_config)

        # This prints a summary: "Trainable params: 0.05% of total"
        self.model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Pass data through the LoRA-wrapped model
        return self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # CRITICAL: We only pass `self.model.parameters()` to the optimizer.
        # Because we used PEFT, this ONLY includes the tiny Adapter weights.
        # The massive base model weights are ignored by AdamW.
        return torch.optim.AdamW(self.model.parameters(), lr=2e-4)
```

-----

## 3\. Hyperparameters: $r$ and $\alpha$

Students often stare at the `LoraConfig` and guess numbers. Let's make this concrete.

### Rank ($r$): The Complexity (as discussed in the [PEFT conceptual guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora))

Rank determines the size of the matrices $A$ and $B$. It represents the "capacity" to learn new things.

* **$r=8$ (Low):** Good for simple tasks like **Style Transfer**. (e.g., "Take this text and rewrite it like Shakespeare"). The model doesn't need to learn new facts, just a new tone.
* **$r=64$ (High):** Good for **Complex Reasoning** or **New Languages**. If you are teaching the model to speak Hindi or solve medical equations, you need more "brain space."
* **The Trade-off:** Higher $r$ = More VRAM usage and slower training.

### Alpha ($\alpha$): The Volume Knob (see the LoRA paper's scaling discussion: [LoRA paper](https://arxiv.org/abs/2106.09685))

Alpha is a scaling factor. The update to the weights is calculated as:
$$\Delta W = (B \times A) \cdot \frac{\alpha}{r}$$

* If $\frac{\alpha}{r} = 1$, the adapter has "standard" influence.
* If $\frac{\alpha}{r} = 2$, the adapter's signal is doubled.
* **Rule of Thumb:** A standard heuristic is to set $\alpha = 2 \times r$ (e.g., if $r=16$, set $\alpha=32$). This helps stabilize the gradients during training.

-----

## 4\. Merging: The Final Step

After training, you have two things:

1. **The Base Model:** 16GB (or 5GB quantized).
2. **The Adapter:** \~50MB file (containing matrices $A$ and $B$).

To serve this efficiently to users, we usually **Merge** them permanently.
$$W_{final} = W_{base} + (B \times A) \cdot \frac{\alpha}{r}$$

### The "4-Bit Trap" (explained in QLoRA/PEFT docs)

If you trained using 4-bit quantization (QLoRA), you **cannot** simply merge the adapter back into the 4-bit model directly without losing accuracy.
**The Workflow:**

1. Reload the Base Model in **FP16** (requires high RAM for a moment).
2. Load your Adapter.
3. Run `model.merge_and_unload()`.
4. Save the new, full-sized model.

-----

## 5\. Homework & Experiments

1. **Expand the Target:**
    In the code, we used `target_modules=["q_proj", "v_proj"]`.

      * *Task:* Change it to `["q_proj", "k_proj", "v_proj", "o_proj"]`.
      * *Observation:* You are now training adapters on *all* parts of the attention mechanism. The "Trainable Parameters" count will double. Does the model learn faster?

2. **Catastrophic Forgetting (The Overfitting Test):**

      * *Task:* Set `r=128` (huge capacity) and train on a tiny dataset (e.g., 20 sentences) for 10 epochs.
      * *Observation:* Ask the model a normal question like "What is the capital of France?" It might answer with gibberish or hallucinate facts related to your tiny dataset. You have "overwritten" its brain with your specific data.

3. **Portability Check:**

      * *Task:* Use `model.model.save_pretrained("my_adapter")`.
      * *Observation:* Check the file size of `adapter_model.bin`. It should be incredibly small (\~20MB to \~100MB). This is why LoRA is great—you can email a "fine-tuned model" to a friend\!
