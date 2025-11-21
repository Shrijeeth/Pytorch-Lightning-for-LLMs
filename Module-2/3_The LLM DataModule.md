# The LLMDataModule

We are going to build a **LightningDataModule**. This is a self-contained class that handles (see the [Lightning docs](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)):

1. Downloading the raw text (WikiText).
2. Tokenizing it (turning text into numbers).
3. Batching it with **Dynamic Padding** (efficiency).

## Prerequisites

Before running this code, ensure you have the libraries installed:

```bash
pip install torch lightning transformers datasets
```

-----

## Step 1: The Tokenizer & Collate Function

We need two tools before we build the module.

1. **The Tokenizer:** The dictionary that translates words to numbers (see [AutoTokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer)).
2. **The Collator:** The tool that groups samples into a batch and handles the padding (the [`DataCollatorForLanguageModeling`](https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForLanguageModeling)).

<!-- end list -->

```python
import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

# 1. Load the Tokenizer
# We use GPT-2, a classic standard for Causal LLMs.
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# CRITICAL FIX for GPT-2:
# GPT-2 was trained without a "pad" token.
# If we don't manually assign one, the code will crash when we try to pad unequal sentences.
# We tell it: "Use the End-Of-Sentence token as the Pad token."
tokenizer.pad_token = tokenizer.eos_token

# 2. The Magic Component: DataCollator
# This function runs EVERY time we fetch a batch.
# It checks the longest sentence in the batch and pads the others to match it.
# mlm=False means "Masked Language Modeling = False".
# We are doing Causal LM (Next Token Prediction), not BERT-style masking.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

-----

## Step 2: The Lightning Data Module

This is the core artifact. Read the comments carefully; they explain *where* the code runs (CPU vs GPU)—mirroring Lightning's recommended structure.

```python
import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader
import multiprocessing

class LLMDataModule(L.LightningDataModule):
    def __init__(self, model_name="gpt2", batch_size=32, max_length=128):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        # Performance Tip: Set num_workers to your CPU count to load data faster.
        self.num_workers = multiprocessing.cpu_count()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # The fix we discussed earlier
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_data(self):
        """
        STEP A: DOWNLOAD
        This method runs ONCE per computer (node).
        If you have 8 GPUs, this only runs on GPU 0.
        Strictly for downloading data to disk so we don't corrupt files.
        """
        # We use 'wikitext-2' (a small dataset) for this demo.
        load_dataset('wikitext', 'wikitext-2-raw-v1')

    def setup(self, stage=None):
        """
        STEP B: PROCESS
        This method runs on EVERY GPU.
        Here we load the data from disk and apply the tokenizer.
        """
        # 1. Load raw data (Hugging Face Datasets API)
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

        # 2. Define the tokenizer logic
        def tokenize_function(examples):
            # We truncate here to ensure no sequence exceeds our max memory
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length
            )

        # 3. Apply tokenization (Map)
        # We remove the 'text' column because the model only needs numbers (input_ids).
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        # 4. Split for training phases
        if stage == 'fit' or stage is None:
            self.train_dataset = tokenized_datasets["train"]
            self.val_dataset = tokenized_datasets["validation"]

        if stage == 'test':
            self.test_dataset = tokenized_datasets["test"]

    def train_dataloader(self):
        """
        STEP C: DELIVER
        This wraps the data in a loader that pumps batches to the GPU.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True, # Always shuffle training data!
            num_workers=self.num_workers,
            # This is where Dynamic Padding happens:
            collate_fn=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),  # Dynamic padding per batch
            pin_memory=True # Speed boost for data transfer to GPU
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            pin_memory=True
        )
```

-----

## 4\. Analysis: What just happened?

### The `setup()` Logic: `dataset.map`

Notice we used `dataset.map`. This applies the tokenizer to the **entire dataset** before training starts.

* **Pros:** Training is fast because the CPU doesn't have to tokenize every single step; it just grabs pre-calculated numbers.
* **Cons:** If you have 1 Terabyte of text, this `setup` step might take 5 hours and crash your RAM.
* *Advanced Note:* For massive datasets (like training Llama 3), we don't use `.map`. We use `IterableDataset` to stream and tokenize on the fly.

### The `collate_fn`

This is the secret sauce. In `train_dataloader`, we passed `DataCollatorForLanguageModeling`.
Imagine a batch with 3 sentences:

1. Length 10
2. Length 14
3. Length 8

The **Collator** sees this batch. It finds the max length (14).

* It pads sentence 1 with 4 zeros.
* It pads sentence 3 with 6 zeros.
* **Result:** A clean (3x14) matrix.
* **Savings:** If we didn't use this, we would have padded everything to `max_length` (128), wasting huge amounts of compute on zeros.

-----

## 5\. Testing the Pipeline (Verification)

**Never assume your data loader works.** Always look at the first batch before you start a 3-day training run.

Copy and run this function to debug your module:

```python
def debug_datamodule():
    # Initialize the module
    dm = LLMDataModule(batch_size=4)

    # Manually run the steps usually handled by Trainer
    dm.prepare_data()
    dm.setup()

    # Get a single batch from the loader
    dataloader = dm.train_dataloader()
    batch = next(iter(dataloader))

    print("Keys available:", batch.keys())
    # Expected: dict_keys(['input_ids', 'attention_mask', 'labels'])

    print("Input Shape:", batch['input_ids'].shape)
    # Expected: torch.Size([4, <dynamic_length>])

    # Verify the data makes sense (Decode back to text)
    decoded = dm.tokenizer.decode(batch['input_ids'][0])
    print(f"\n--- Sample Text (Decoded) ---\n{decoded[:100]}...")

    # Check for Labels
    # In Causal LM, the 'labels' are usually just the 'input_ids' shifted by one.
    # The DataCollator creates this 'labels' key for us automatically!
    print("\nLabels included?", 'labels' in batch)

if __name__ == "__main__":
    debug_datamodule()
```

-----

## 6\. Homework & Challenge

1. **Run the code:** Ensure you can download the data and see the "Sample Text" print out.
2. **Observe Memory:** Change `max_length` to 1024. Run the debug script. Does the `input_ids` shape change? (It shouldn't, because of Dynamic Padding\! It will only be as long as the longest sentence in that specific batch).
3. **Challenge:** Research "Sequence Packing."
      * *Current method:* `[Sentence A, Pad, Pad]` and `[Sentence B]`
      * *Packed method:* `[Sentence A, Sentence B, Sentence C]` (All concatenated to fill the context window).
      * This removes padding entirely and is how top-tier LLMs are trained.
