# The Problem: The "Full Fine-Tune" Cost

When people say "I want to train an AI," they usually imagine "Full Fine-Tuning" (see Hugging Face's [guide on fine-tuning strategies](https://huggingface.co/blog/stackllama)).
**Full Fine-Tuning** means we treat every single parameter in the model as "trainable." If the model makes a mistake, we blame every single neuron and try to adjust all of them slightly.

This creates a massive memory crisis called **Optimizer State Explosion** (described in Microsoft's [ZeRO paper](https://arxiv.org/abs/1910.02054)).

## The Math: Why one matrix costs so much

Let's look at a single weight matrix inside the Attention Layer of a standard model (like Llama 2 or Mistral).

* **Dimensions:** $4096 \times 4096$
* **Total Parameters:** $16,777,216$ (~16.7 Million).

To update this **one** matrix using the **Adam Optimizer**, you need to store much more than just the matrix itself (see the original [Adam paper](https://arxiv.org/abs/1412.6980)).

| Component | What is it? | Count (Floats) | Memory (Bytes) |
| :--- | :--- | :--- | :--- |
| **The Weight ($W$)** | The actual brain. | 16.7 Million | **33.5 MB** (FP16) |
| **The Gradient ($\nabla W$)** | "How much should we change?" | 16.7 Million | **67.1 MB** (FP32) |
| **Optimizer State 1 ($M$)** | Momentum (Velocity of change). | 16.7 Million | **67.1 MB** (FP32) |
| **Optimizer State 2 ($V$)** | Variance (Acceleration of change). | 16.7 Million | **67.1 MB** (FP32) |
| **TOTAL for 1 Matrix** | | **~67 Million** | **~235 MB** |

**The Shocking Truth:**
To train a 33 MB matrix, you need **235 MB** of VRAM.
The "training overhead" is **7x larger** than the model weight itself!

## Scaling to the Full Model (7B)

Now, imagine doing this for a 7 Billion parameter model (e.g., [Llama 2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)).

* **Model Weights (FP16):** 14 GB.
* **Training Overhead:** $\approx 14 \text{ GB} \times 7 = \mathbf{98 \text{ GB}}$.

**The Result:**
You need **~112 GB** of VRAM to full fine-tune a tiny 7B model.

* **RTX 4090 (24 GB):** Crash. 💥
* **A100 (80 GB):** Crash. 💥

This is why, until recently, only Big Tech companies could fine-tune LLMs. They had to string together 4-8 A100s just to update the weights of a small model (see the [DeepSpeed case studies](https://www.microsoft.com/en-us/research/project/deepspeed/)).

## The Solution Teaser: Freezing

What if, instead of updating all 7 Billion parameters, we **froze** 99% of them and only updated a tiny "adapter" layer on top?
This is the philosophy behind **PEFT (Parameter-Efficient Fine-Tuning)** and **LoRA**, which we will cover next (see [PEFT docs](https://huggingface.co/docs/peft/index) and the original [LoRA paper](https://arxiv.org/abs/2106.09685)).
