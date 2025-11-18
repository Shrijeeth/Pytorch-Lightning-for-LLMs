# Neural Networks as Linear Regression

We often treat "Neural Networks" as magic boxes, but at their core, they are just fancy versions of Linear Regression.[^1]

To make this concrete, let's model the basketball shot mathematically using the simplest equation in algebra: the line equation ($y = mx + b$).

## The Scenario

> You want to build a model that predicts how much **Power** ($y$) you need to use based on your **Distance** from the hoop ($x$).

Here is the comparison of the conceptual "Basketball Shot" vs. the mathematical "Linear Regression" model.

-----

## The 5 Steps: Basketball vs. Linear Regression

### 1\. Forward Propagation (The Guess)

* **The Basketball Analogy:** You look at the hoop 10 meters away ($x$). Your brain guesses: "For 10 meters, I need 50 units of power." You shoot.
* **The Linear Regression Math ($y=wx+b$):**
  * **Input ($x$):** 10 (Distance)
  * **Weights ($w$):** 4 (Slope)
  * **Bias ($b$):** 5 (Base arm strength)
  * **Calculation:** $\hat{y} = (4 \times 10) + 5 = 45$
  * The model predicts **45** units of power.

### 2\. Loss Calculation (The Error)

* **The Basketball Analogy:** The ball falls short\! It hits the front rim. You needed 50 units of power, but you only used 45. **The Miss:** -5 units.
* **The Linear Regression Math:**
  * **Ground Truth ($y$):** 50 (Actual power needed)
  * **Prediction ($\hat{y}$):** 45
  * **Loss ($J$):** $J = (y - \hat{y})^2 = (50 - 45)^2 = 25$
  * We square the error to make the math easier later.

### 3\. Backward Propagation (The Blame)

* **The Basketball Analogy:** Your brain analyzes: "Why was I short? Was it my base strength ($b$) or did I misjudge the distance multiplier ($w$)?" You realize you need to increase the multiplier.
* **The Linear Regression Math:**
  * **Gradients:** We calculate the derivative. We ask: "If I increase $w$, does the error go down?"[^3]
    $\frac{\partial J}{\partial w} = -2(y - \hat{y})x$
  * The math tells us exactly how much $w$ and $b$ contributed to the error of 25.[^2]

### 4\. Optimization (The Adjustment)

* **The Basketball Analogy:** You adjust your muscle memory. You decide: "Next time I see a target at 10 meters, I will push a little harder per meter."
* **The Linear Regression Math:**
  * **Update Rule:**
    $w_{new} = w_{old} - \text{learning\_rate} \times \text{gradient}$
  * We nudge $w$ from 4.0 up to **4.1**.
  * We nudge $b$ from 5.0 up to **5.2**.

### 5\. Zero Gradients (The Reset)

* **The Basketball Analogy:** You shake out your arms. You don't want the adjustments you made for the 10-meter shot to accidentally mess up your next shot from 3 meters. Reset focus.
* **The Linear Regression Math:**
  * **Reset:**

    ```python
    w.grad = 0
    b.grad = 0
    ```

  * We delete the stored gradients so the next training step (a new distance $x$) starts with a clean slate.

-----

## How this simple concept becomes an LLM

You might be thinking, "Okay, but how does $y = mx + b$ turn into ChatGPT?"

The magic is that the math does not change. The only thing that changes is the scale and the complexity of the function.[^6]

### **1. From 2 Parameters to Billions**

* **In our basketball linear regression**, we had two parameters to optimize: $w$ (Slope) and $b$ (Bias).
* **In an LLM (like GPT-4 or Llama 3)**, we don't just have $w$ and $b$. We have billions of them ($w_1, w_2, ... w_{70B}$).
* Instead of a simple line, the function looks like a massive, multi-dimensional web of matrix multiplications. But we still update them using the exact same Optimization step: $\theta_{new} = \theta_{old} - \eta \cdot \text{gradient}$.

### **2. From "Distance" to "Tokens"**

* **Basketball Input:** A number representing distance (e.g., "10 meters").
* **LLM Input:** A sequence of numbers representing words (tokens). e.g., "The cat sat on the..." might be represented as `[104, 22, 908, 11]`.[^4]

### **3. From "Power" to "Probability"**

* **Basketball Output:** A number representing Force.
* **LLM Output:** A probability distribution over all possible next words.[^5]
* **Forward Prop calculates:** "Given 'The cat sat on the...', what is the likely next word?"
* **Prediction ($\hat{y}$):** "Mat" (90%), "Hat" (5%), "Dog" (1%).

### **4. The "Loss" is Reading Comprehension**

* **Basketball Loss:** Distance from the hoop.
* **LLM Loss:** Did the model predict the actual next word correctly?
* If the text was "The cat sat on the mat", and the model predicted "dog", the Loss is high.
* **Backpropagation** goes back through the billions of parameters to find which specific neurons were responsible for thinking "dog" instead of "mat" and nudges them down.

> **Summary:** An LLM is just a basketball player taking billions of shots (reading billions of sentences) and adjusting its muscle memory (weights) slightly every time it guesses the next word wrong.

## References

[^1]: Scikit-learn Developers. "Linear Regression." [https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
[^2]: PyTorch Core Team. "Autograd Mechanics." [https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
[^3]: Google MLCC. "Gradient Descent." [https://developers.google.com/machine-learning/crash-course/reducing-loss/gradient-descent](https://developers.google.com/machine-learning/crash-course/reducing-loss/gradient-descent)
[^4]: Hugging Face. "Tokenization." [https://huggingface.co/docs/transformers/tokenizer_summary](https://huggingface.co/docs/transformers/tokenizer_summary)
[^5]: Stanford CS224n. "Language Modeling." [https://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture02-wordvecs1.pdf](https://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture02-wordvecs1.pdf)
[^6]: Vaswani et al. "Attention Is All You Need." [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
