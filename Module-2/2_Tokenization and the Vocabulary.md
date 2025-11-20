# Tokenization & The Vocabulary

Before we can batch data into the "fuel" for our GPU, we have to solve a fundamental problem: **Computers cannot read text.** They can only do math on numbers, which is why every modern NLP stack starts with [tokenization](https://huggingface.co/docs/transformers/main/en/tokenizer_summary).

We need a system to translate "Hello World" into `[15496, 995]`. This process is called **Tokenization** (see a deeper primer in the [Transformers docs](https://huggingface.co/course/chapter6/5?fw=pt)).

## 1. The "Goldilocks" Problem of Tokenization

How should we break a sentence into numbers? We have three options, but only one works well for LLMs.

### Option A: Word-Level (Too Sparse)

We assign a number to every word in the dictionary.

* **Example:** "The" = 1, "Cat" = 2, "Sat" = 3.
* **The Problem:** The English language is infinite. What about "Microbiology"? "TikTok"? "Rizz"? Your vocabulary size ($V$) becomes millions. The model will constantly encounter "Unknown" words (the classic [OOV problem](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture01-wordvecs1.pdf)).

### Option B: Character-Level (Too Long)

We assign a number to every letter.

* **Example:** "T" = 1, "h" = 2, "e" = 3.
* **The Problem:** The sequence length ($S$) explodes. The sentence "The cat sat" becomes 11 tokens. LLMs struggle to remember relationships over very long distances (dependencies), so making sequences 5x longer makes the model dumber.

### Option C: Subword Tokenization (Just Right)

This is what modern LLMs (GPT-4, Llama 3, BERT) use. It uses algorithms like **BPE (Byte Pair Encoding)**, originally popularized by [Sennrich et al.](https://aclanthology.org/P16-1162/).

* **The Logic:** It keeps common words whole ("The", "apple") but breaks rare words into meaningful chunks, exactly as shown in the [SentencePiece](https://github.com/google/sentencepiece#subword-sampling) and BPE implementations.
* **Example:** "Unbreakable" $\rightarrow$ `["Un", "break", "able"]`.
* **The Benefit:** It keeps the Vocabulary size ($V$) small (usually ~32k to ~100k) while keeping the Sequence length ($S$) manageable.

---

## 2. The Math: The Vocabulary Trade-off

Designing a tokenizer is a balancing act between Memory and Logic.

Let:

* $V$: Vocabulary Size (e.g., 50,000 unique tokens).
* $S$: Sequence Length (e.g., 1024 tokens in a row).
* $d_{model}$: The "width" of the neural network (e.g., 768).

The very first layer of an LLM is the **Embedding Matrix**. Its size is huge: $V \times d_{model}$ (see the [Transformers architecture overview](https://huggingface.co/docs/transformers/main/en/model_summary)).

**The Trade-off:**

1. **Small Vocabulary ($V$):**
    * *Pros:* The Embedding Matrix is small (saves RAM).
    * *Cons:* You have to break words into tiny pieces. Your Sequence Length ($S$) gets longer. The model has to work harder to understand the meaning.
2. **Large Vocabulary ($V$):**
    * *Pros:* Most words are single tokens. Sequences are short and information-dense.
    * *Cons:* The Embedding Matrix explodes. If $V=1,000,000$ and $d=4096$, your embedding layer alone is 4GB of VRAM!

---

## 3. The Padding Problem (The Enemy of Efficiency)

Once we have tokens, we group them into **Batches**.
In computing, a Batch must be a perfect rectangle (a Matrix). But sentences have different lengths!

**The Problem (Static Padding):** Neural Networks require inputs to be the same shape (squares/rectangles). If you have a short sentence ("Hi.") and a long sentence ("The quick brown fox..."), you usually add zeros (padding) to the short one to make them match (see [Hugging Face's padding guide](https://huggingface.co/docs/transformers/pad_truncation)).

To put these in the same batch, we must add "Padding" (usually zeros) to Sentence A so it matches Sentence B.

### Why is Padding "Evil"? ($O(N^2)$ Complexity)

Transformers use a mechanism called **Self-Attention**. The math of attention looks at *every token's relationship with every other token* (review the original [Attention Is All You Need](https://arxiv.org/abs/1706.03762)). For more information on self-attention complexity, see the [Transformers documentation](https://huggingface.co/docs/transformers/model_doc/bert#attention-mechanism).

* **Complexity:** $O(N^2)$ (Quadratic).

If you double the sequence length ($N$), the work doesn't double; it **quadruples**.

If you pad a sentence of length 5 to length 512 (to match the longest sentence in your dataset), your GPU is spending 99% of its energy calculating the relationship between the word "Hi" and a bunch of empty Zeros. This is wasted electricity and time (Lightning highlights this in their [dynamic padding tips](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#dynamic-padding)).

---

## 4. The Solution: Dynamic Padding

We stop treating the entire dataset as one fixed size. Instead, we resize the "box" for every single batch.

**Strategy A: Static Padding (The "Lazy" Way)**
You pick a safe max number (e.g., 512). *Every* sentence, no matter how short, gets padded to 512.

* *Result:* Massive waste.

**Strategy B: Dynamic Padding (The "Smart" Way)**
You look at the current batch of 4 sentences. You find the longest one *in that group* (identical to the [dynamic batching strategy](https://huggingface.co/docs/transformers/perf_train_gpu_one#efficient-dataloaders)).

* **Batch 1:** Longest sentence is 8 tokens. We pad everyone to 8. (Fast!)
* **Batch 2:** Longest sentence is 100 tokens. We pad everyone to 100. (Slower, but necessary).

| Strategy | Batch Shape | Wasted Compute | Speed |
| :--- | :--- | :--- | :--- |
| **Static Padding** | Fixed (e.g., 1024) | **High** (Calculating 90% zeros) | 🐢 Slow |
| **Dynamic Padding** | Variable (e.g., 12) | **Low** (Only necessary padding) | 🐇 Fast |
