# Phase 1 Deep Dive: Deep Learning Core
### A full course module — Neural Networks → Backpropagation & Gradient Descent → CNNs → RNN/LSTM/GRU

> This expands Phase 1 of the master roadmap into complete, from-first-principles teaching content: intuition, math derivations, worked numerical examples, runnable code, pitfalls, fully-answered interview questions, and a project checklist per module. Work through it in order — each module leans on the previous one.

---

## MODULE 1.1 — Neural Networks (Perceptron → MLP)

### 1.1.1 Intuition first
A neural network is a function approximator built from simple, repeated units. Each unit takes a weighted vote of its inputs, adds a bias, and passes the result through a nonlinear "squashing" function. Stack enough of these units in layers, and the network can approximate extremely complex functions — this is not a metaphor, it's a proven mathematical result (the Universal Approximation Theorem, §1.1.5).

Think of a single neuron as a tiny decision-maker: "given these input signals, weighted by how much I trust each one, plus my own bias/threshold, do I fire strongly or weakly?"

### 1.1.2 The single neuron — math
For inputs `x = [x₁, x₂, ..., xₙ]`, weights `w = [w₁, w₂, ..., wₙ]`, and bias `b`:

```
z = w·x + b = Σ(wᵢxᵢ) + b        (the "pre-activation")
a = f(z)                          (the "activation" — nonlinear function)
```

`f` is the **activation function**. Without it, `a` is just a linear combination of inputs — no matter how many neurons you chain, the whole network collapses to a single linear function (proof in §1.1.5). The activation is what gives the network its expressive power.

### 1.1.3 Activation functions — deep dive

| Function | Formula | Range | Notes |
|---|---|---|---|
| Sigmoid | `1/(1+e⁻ᶻ)` | (0,1) | Saturates at extremes → vanishing gradients. Used for binary output probabilities, rarely in hidden layers now. |
| Tanh | `(eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ)` | (-1,1) | Zero-centered (better than sigmoid for hidden layers) but still saturates. |
| ReLU | `max(0, z)` | [0,∞) | Default choice for decades: cheap, no saturation for z>0. Problem: "dying ReLU" — neurons stuck outputting 0 get zero gradient forever. |
| Leaky ReLU | `z if z>0 else αz` (α≈0.01) | (-∞,∞) | Fixes dying ReLU by allowing a small negative slope. |
| GELU | `z·Φ(z)` (Φ = Gaussian CDF) | (-∞,∞) | Smooth approximation of ReLU; standard in Transformers (GPT, BERT) — smoother gradients help optimization at scale. |
| Swish/SiLU | `z·sigmoid(z)` | (-∞,∞) | Similar motivation to GELU; used in some modern architectures (e.g., LLaMA uses SwiGLU, a gated variant). |

**Why this matters for later phases**: when you get to Transformers in Phase 3, you'll see GELU/SwiGLU in the feed-forward sublayers — this is not a new concept, just the same activation-function decision you're learning here.

### 1.1.4 From one neuron to a network (MLP)
A Multi-Layer Perceptron stacks **layers** of neurons. Layer `l`'s output becomes layer `l+1`'s input:

```
z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾        a⁽¹⁾ = f(z⁽¹⁾)
z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾      a⁽²⁾ = f(z⁽²⁾)
...
ŷ = z⁽ᴸ⁾  (or softmax(z⁽ᴸ⁾) for classification)
```

`W⁽ˡ⁾` is now a **matrix** (one row per neuron in layer `l`, one column per input from layer `l-1`) — this is exactly the matrix multiplication from Phase 0's linear algebra. A layer with 128 input features and 64 neurons has `W⁽ˡ⁾ ∈ ℝ^(64×128)` — 8,192 weights, plus 64 biases.

```
Input Layer      Hidden Layer(s)         Output Layer
   x₁ ●─┐      ┌─● a₁⁽¹⁾─┐      ┌─● 
        ├─W⁽¹⁾─┤          ├─W⁽²⁾─┤
   x₂ ●─┤      └─● a₂⁽¹⁾─┘      └─● ŷ
        │      ┌─● a₃⁽¹⁾─┐
   x₃ ●─┘      └─● a₄⁽¹⁾─┘
   (3 in)      (4 hidden)        (1 out)
```

### 1.1.5 Why nonlinearity matters (proof sketch)
If `f` were the identity function (no nonlinearity), then:
```
ŷ = W⁽²⁾(W⁽¹⁾x + b⁽¹⁾) + b⁽²⁾ = (W⁽²⁾W⁽¹⁾)x + (W⁽²⁾b⁽¹⁾ + b⁽²⁾) = W'x + b'
```
That's just **one** linear layer in disguise — depth adds zero expressive power. Nonlinear activations are what let each additional layer carve genuinely new decision boundaries.

**Universal Approximation Theorem** (Cybenko 1989, Hornik 1991): a feed-forward network with a single hidden layer of finite width, using a non-constant, bounded, continuous activation, can approximate any continuous function on a compact input domain to arbitrary precision — *given enough neurons*. This is why MLPs are theoretically powerful; in practice, **depth** (more layers of smaller width) is dramatically more parameter-efficient than width alone, which is part of why deep learning works better than "wide shallow learning."

### 1.1.6 Loss functions — deep dive

**Mean Squared Error (regression)**:
```
L = (1/N) Σ(ŷᵢ - yᵢ)²
```
Penalizes large errors quadratically; sensitive to outliers.

**Binary Cross-Entropy (binary classification)**:
```
L = -(1/N) Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```
Derived from **Maximum Likelihood Estimation**: if you assume `ŷ` is the model's predicted probability that `y=1`, minimizing cross-entropy is *exactly equivalent* to maximizing the likelihood of the true labels under the model. This is the same principle that underlies next-token prediction loss in LLMs (Phase 3/4) — cross-entropy over a probability distribution.

**Categorical Cross-Entropy (multi-class)**:
```
L = -(1/N) Σᵢ Σₖ yᵢₖ log(ŷᵢₖ)      where ŷ = softmax(z)
softmax(z)ₖ = e^(zₖ) / Σⱼ e^(zⱼ)
```
Softmax converts raw scores ("logits") into a probability distribution that sums to 1. This is the exact mechanism an LLM uses to turn its final layer output into "probability of each token in the vocabulary being next."

**Why cross-entropy over MSE for classification**: MSE's gradient w.r.t. the output shrinks as predictions saturate near 0 or 1 (combined with sigmoid, this causes very slow learning). Cross-entropy's gradient does not have this problem — it stays large exactly when the model is confidently wrong, which is when you need the largest correction.

### 1.1.7 Worked example: the XOR problem
XOR is the canonical example showing why nonlinearity/hidden layers are necessary — no single line can separate XOR's classes:

| x₁ | x₂ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Plot these four points: (0,0)→0 and (1,1)→0 are one class; (0,1)→1 and (1,0)→1 are the other. No straight line separates them (a single-layer perceptron *provably cannot* solve XOR — this was a famous historical objection to neural networks, Minsky & Papert 1969). A hidden layer with just 2 neurons using ReLU/sigmoid can solve it, by first bending the input space into a form where the classes *are* linearly separable, then classifying with a final linear layer. This is the entire point of hidden layers: each layer reshapes the representation space to make the next layer's job easier.

### 1.1.8 Code: NumPy from-scratch MLP (forward pass + training loop)
```python
import numpy as np

class MLP:
    def __init__(self, layer_sizes, lr=0.1, seed=42):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W, self.b = [], []
        for i in range(len(layer_sizes) - 1):
            # He initialization (good default for ReLU networks — see Module 1.2.8)
            fan_in = layer_sizes[i]
            self.W.append(rng.normal(0, np.sqrt(2.0 / fan_in), (layer_sizes[i], layer_sizes[i+1])))
            self.b.append(np.zeros(layer_sizes[i+1]))

    @staticmethod
    def relu(z): return np.maximum(0, z)
    @staticmethod
    def relu_grad(z): return (z > 0).astype(float)
    @staticmethod
    def sigmoid(z): return 1 / (1 + np.exp(-z))

    def forward(self, X):
        self.zs, self.activations = [], [X]
        a = X
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            a = self.relu(z)
            self.zs.append(z); self.activations.append(a)
        # output layer: sigmoid for binary classification
        z_out = a @ self.W[-1] + self.b[-1]
        a_out = self.sigmoid(z_out)
        self.zs.append(z_out); self.activations.append(a_out)
        return a_out

    def compute_loss(self, y_pred, y_true, eps=1e-8):
        y_pred = np.clip(y_pred, eps, 1 - eps)  # avoid log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# --- XOR dataset ---
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

net = MLP(layer_sizes=[2, 4, 1], lr=0.5)
pred = net.forward(X)
print("Initial loss:", net.compute_loss(pred, y))
# Full backprop training loop is in Module 1.2.9 — this section is forward-pass only by design,
# so you fully separate "what is a forward pass" from "how do we learn the weights."
```

### 1.1.9 Code: PyTorch equivalent
```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])

model = MLP(2, 4, 1)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(2000):
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()      # autograd computes all gradients — this is Module 1.2 automated
    optimizer.step()
    if epoch % 500 == 0:
        print(f"epoch {epoch}, loss {loss.item():.4f}")

print("Final predictions:", model(X).detach().numpy().round(2))
# Should converge to ~[0, 1, 1, 0]
```
**Why show both**: the NumPy version forces you to see every matrix multiply; the PyTorch version shows what `.backward()` and `optimizer.step()` are actually automating. Once you've done it by hand once, trusting the framework is no longer "magic."

### 1.1.10 Common pitfalls & debugging tips
- **All-zero or identical-row weight init** → every neuron in a layer learns the same thing ("symmetry problem"). Always use random init.
- **Forgetting to normalize inputs** → some features dominate gradients purely due to scale, slowing/destabilizing training. Standardize inputs (zero mean, unit variance).
- **Wrong loss/activation pairing** → e.g., using MSE with a softmax output, or forgetting sigmoid before BCE loss (or using `BCEWithLogitsLoss` which combines both — a common and useful PyTorch pattern).
- **Learning rate too high** → loss oscillates or diverges to NaN. Too low → painfully slow convergence. Start with a known-good default (Adam, lr=1e-3) and adjust.
- **Not shuffling training data** → the model overfits to the *order* of examples, especially with class-grouped data.

### 1.1.11 Interview Q&A (with answers)
**Q: Why do we need nonlinear activations?**
A: Without them, any depth of stacked linear layers collapses mathematically to a single linear transformation (see §1.1.5 proof) — the network could only ever learn linearly separable functions, which is far too limited for real-world data.

**Q: What happens with an all-linear deep network?**
A: The composition of linear functions is itself linear (`W₂(W₁x) = (W₂W₁)x`), so no matter how many layers you stack, the network has the same expressive power as a single linear layer.

**Q: ReLU vs sigmoid vs GELU — tradeoffs?**
A: Sigmoid saturates at both tails (gradients → 0 for large |z|), causing vanishing gradients in deep networks — mostly avoided in hidden layers today. ReLU is cheap and avoids saturation for z>0, but can "die" (stuck at 0, zero gradient) if a neuron's weights push it permanently negative. GELU is a smooth, probabilistic version of ReLU that gives non-zero gradient even for slightly negative inputs, which empirically helps optimization in very deep/large models — hence its use in Transformers.

**Q: Why is cross-entropy preferred over MSE for classification?**
A: Cross-entropy, paired with sigmoid/softmax, produces gradients that remain large when the model is confidently wrong — exactly when you need big corrections. MSE's gradient shrinks as the sigmoid output saturates near 0/1, causing very slow learning in exactly those cases. Cross-entropy is also the theoretically correct loss under a maximum-likelihood interpretation of classification.

### 1.1.12 Hands-on project
**Build**: An MLP from scratch in NumPy (forward + manual backprop — you'll write backprop in Module 1.2, then return here) trained on MNIST digit classification, then reimplement in PyTorch.
**Checklist**: (1) Load/normalize MNIST. (2) Forward pass implemented and unit-tested against a PyTorch forward pass on identical weights (should match to floating-point precision). (3) After Module 1.2, add manual backprop and confirm gradients match `torch.autograd`. (4) Achieve >95% test accuracy. (5) Plot a confusion matrix and inspect 5 misclassified examples — what do they have in common?

---

## MODULE 1.2 — Backpropagation & Gradient Descent

### 1.2.1 Intuition: "assigning blame"
After a forward pass, the network is wrong by some amount (the loss). Backpropagation answers: **"which weights, and by how much, caused this error?"** It's blame-assignment, computed exactly and efficiently via the chain rule — not an approximation, not a heuristic.

### 1.2.2 Computational graphs
Every forward pass can be drawn as a graph where nodes are operations and edges are data flow. Backprop pushes gradients **backward** through this same graph, applying the chain rule at every node:

```
x ──┐
    ├─[× W]──► z ──[ReLU]──► a ──[× W₂]──► z₂ ──[Loss]──► L
b ──┘
Forward:  compute left→right, cache every intermediate value (z, a, z₂...)
Backward: compute right→left, each node receives ∂L/∂(its output),
          computes ∂L/∂(its inputs) via the local derivative × the incoming gradient
```
This "cache forward, replay backward" strategy is exactly what `autograd` (PyTorch/TensorFlow) automates — it dynamically builds this graph as your forward code executes, then walks it backward when you call `.backward()`.

### 1.2.3 Chain rule derivation — full worked example (2-layer network)
Network: `z₁ = W₁x + b₁`, `a₁ = ReLU(z₁)`, `z₂ = W₂a₁ + b₂`, `ŷ = σ(z₂)`, `L = BCE(ŷ, y)`.

We want `∂L/∂W₁`, `∂L/∂W₂`, `∂L/∂b₁`, `∂L/∂b₂`. Work backward, one step at a time:

```
Step 1 — Loss w.r.t. output:
  ∂L/∂ŷ = -(y/ŷ) + (1-y)/(1-ŷ)

Step 2 — Output w.r.t. pre-activation z₂ (sigmoid derivative):
  ∂ŷ/∂z₂ = ŷ(1-ŷ)
  Combined (this combination famously simplifies to):
  ∂L/∂z₂ = ŷ - y                    ← the elegant BCE+sigmoid gradient

Step 3 — z₂ w.r.t. W₂, b₂, a₁:
  ∂L/∂W₂ = (∂L/∂z₂) · a₁ᵀ
  ∂L/∂b₂ = ∂L/∂z₂
  ∂L/∂a₁ = W₂ᵀ · (∂L/∂z₂)          ← this is what flows backward into layer 1

Step 4 — a₁ w.r.t. z₁ (ReLU derivative):
  ∂L/∂z₁ = ∂L/∂a₁ ⊙ ReLU'(z₁)       (⊙ = elementwise; ReLU'(z)=1 if z>0 else 0)

Step 5 — z₁ w.r.t. W₁, b₁:
  ∂L/∂W₁ = (∂L/∂z₁) · xᵀ
  ∂L/∂b₁ = ∂L/∂z₁
```
Notice the pattern: **each layer only ever needs the gradient handed to it from the layer above, plus its own local derivative.** This is why backprop is `O(1)` extra work per layer relative to the forward pass — it never needs to re-derive anything from scratch, which is precisely why it's tractable for networks with billions of parameters.

### 1.2.4 Gradient descent variants
```
Batch GD:      use ALL training data per update    → stable but slow, memory-heavy
Stochastic GD: use 1 example per update             → fast, noisy, can escape shallow local minima
Mini-batch GD: use a small batch (e.g. 32–512)       → the practical default: balances stability and speed,
                                                        and maps efficiently onto GPU parallelism
```
Update rule (vanilla SGD): `W ← W - lr · ∂L/∂W`

### 1.2.5 Optimizers deep dive
- **SGD**: the basic update above. Struggles with ravines (steep in one dimension, shallow in another) and saddle points.
- **SGD + Momentum**: `v ← βv + ∂L/∂W; W ← W - lr·v`. Accumulates a "velocity" across steps, smoothing out oscillations and accelerating through consistent-direction gradients — like a ball rolling downhill gaining speed.
- **RMSProp**: divides the learning rate per-parameter by a running average of that parameter's recent squared gradients — parameters with large, volatile gradients get automatically smaller effective steps.
- **Adam**: combines momentum (first moment, mean of gradients) + RMSProp-style adaptive scaling (second moment, variance of gradients), with bias correction for early training steps. This is the default optimizer for the vast majority of deep learning today.
- **AdamW**: Adam with **decoupled weight decay** — Adam's original weight-decay implementation was subtly wrong (it interacted badly with the adaptive learning rate); AdamW fixes this and is the standard optimizer used for training LLMs.

### 1.2.6 Learning rate scheduling
A fixed learning rate is rarely optimal across an entire training run:
- **Warmup**: start the LR near zero and linearly ramp up over the first N steps. Early in training, gradients from randomly-initialized weights can be huge/unstable — warmup prevents an early destructive update.
- **Cosine decay**: after warmup, decay the LR following a cosine curve down to ~0 by the end of training — smooth, well-tuned exploration-then-refinement.
- **Why LLM pretraining universally uses "warmup + cosine decay"**: at trillion-token scale, an unstable early step can waste enormous compute; a graceful decay at the end lets the model settle into a sharper minimum. You'll see this exact schedule again when you fine-tune models in Phase 5.

### 1.2.7 Vanishing/exploding gradients
**Cause**: in a deep network, the gradient at layer 1 is a *product* of many per-layer derivative terms (chain rule, §1.2.3). If those terms are consistently <1 (e.g., sigmoid's max derivative is 0.25), the product shrinks exponentially with depth → **vanishing gradient**, early layers barely learn. If terms are consistently >1, the product explodes → **NaN losses**.

**Fixes**:
1. **Better activations** (ReLU/GELU instead of sigmoid/tanh in hidden layers — their derivative doesn't shrink toward 0 for positive inputs).
2. **Proper weight initialization** (§1.2.8) — keeps the variance of activations/gradients roughly constant across layers.
3. **Normalization layers** (BatchNorm/LayerNorm) — explicitly re-center/re-scale activations at every layer, preventing runaway growth or shrinkage.
4. **Residual/skip connections** (`output = F(x) + x`, from ResNet, also core to Transformers) — provide a gradient "shortcut" straight back to earlier layers, bypassing the multiplicative chain entirely. This is arguably the single most important architectural trick for training very deep networks.
5. **Gradient clipping** — cap the gradient norm at a threshold before the optimizer step, directly preventing exploding updates. Standard practice in RNN and LLM training.

### 1.2.8 Weight initialization
Random initialization must be scaled correctly, or you reintroduce vanishing/exploding signal from step 1:
- **Xavier/Glorot init** (`Var(W) = 1/fan_in`, or `2/(fan_in+fan_out)`): designed for sigmoid/tanh, keeps activation variance roughly constant layer to layer.
- **He init** (`Var(W) = 2/fan_in`): designed for ReLU — accounts for the fact that ReLU zeroes out ~half the activations, so needs a larger variance to compensate. This is what the NumPy MLP in §1.1.8 uses.

### 1.2.9 Code: manual backprop vs PyTorch autograd (verification)
```python
import numpy as np
import torch

# ---------- Manual NumPy implementation (from the math in §1.2.3) ----------
np.random.seed(0)
X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = np.array([[0.],[1.],[1.],[0.]])

W1 = np.random.randn(2,4) * np.sqrt(2/2)
b1 = np.zeros(4)
W2 = np.random.randn(4,1) * np.sqrt(2/4)
b2 = np.zeros(1)

def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)
def sigmoid(z): return 1/(1+np.exp(-z))

# Forward
z1 = X @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
yhat = sigmoid(z2)

# Backward (implementing §1.2.3 exactly)
dL_dz2 = yhat - y                        # elegant BCE+sigmoid combined gradient
dL_dW2 = a1.T @ dL_dz2
dL_db2 = dL_dz2.sum(axis=0)
dL_da1 = dL_dz2 @ W2.T
dL_dz1 = dL_da1 * relu_grad(z1)
dL_dW1 = X.T @ dL_dz1
dL_db1 = dL_dz1.sum(axis=0)

print("Manual dL/dW2:\n", dL_dW2)

# ---------- PyTorch autograd on identical weights ----------
Xt = torch.tensor(X); yt = torch.tensor(y)
W1t = torch.tensor(W1, requires_grad=True); b1t = torch.tensor(b1, requires_grad=True)
W2t = torch.tensor(W2, requires_grad=True); b2t = torch.tensor(b2, requires_grad=True)

z1t = Xt @ W1t + b1t
a1t = torch.relu(z1t)
z2t = a1t @ W2t + b2t
yhat_t = torch.sigmoid(z2t)
loss = torch.nn.functional.binary_cross_entropy(yhat_t, yt)
loss.backward()

print("Autograd dL/dW2:\n", W2t.grad.numpy())
# These two should match to floating-point precision — this is the single most
# valuable verification exercise in this entire module.
```

### 1.2.10 Interview Q&A (with answers)
**Q: Derive backprop for a 2-layer network.**
A: Walk through §1.2.3 — start from `∂L/∂ŷ`, apply the chain rule backward through the sigmoid, then the second linear layer, then ReLU, then the first linear layer, at each step multiplying the incoming upstream gradient by the local derivative of that operation.

**Q: SGD vs Adam — what does Adam add?**
A: Adam adds two things over vanilla SGD: (1) momentum — an exponential moving average of past gradients, smoothing the update direction; (2) adaptive per-parameter learning rates via an exponential moving average of squared gradients (like RMSProp), so parameters with noisy/large gradients get automatically damped. It also applies bias correction to these moving averages, since they start at zero and would otherwise be biased low in early steps.

**Q: What is a learning rate warmup and why do LLMs use it?**
A: Warmup linearly increases the learning rate from ~0 to its target value over the first N steps. At the very start of training, weights are randomly initialized and gradients can be large/unstable; taking a full-sized step immediately risks a destructive update the model never recovers from. This risk is especially costly at LLM pretraining scale, where an unstable early step wastes enormous compute.

**Q: Vanishing vs exploding gradients — causes and fixes?**
A: Both stem from the chain rule multiplying many per-layer derivative terms together across depth (§1.2.7). If those terms are consistently <1, the product shrinks exponentially (vanishing); if consistently >1, it grows exponentially (exploding). Fixes: better activations (ReLU/GELU), proper initialization (He/Xavier), normalization layers (BatchNorm/LayerNorm), residual connections (the most impactful), and gradient clipping (specifically for exploding gradients).

**Q: Why does Adam sometimes generalize worse than SGD+momentum despite converging faster?**
A: This is a known empirical phenomenon (particularly noted in some CV literature) — Adam's adaptive per-parameter scaling can converge to sharper minima that generalize slightly worse than the flatter minima SGD tends to find. In practice for large-scale LLM training, AdamW remains the default because its faster/more stable convergence outweighs this effect at scale, but it's a legitimate "it depends" answer worth knowing.

### 1.2.11 Hands-on project
**Build**: Gradient descent from scratch to fit a line (`y = mx + b`) to noisy synthetic data — no ML library. Then extend it to the manual-backprop MLP verification exercise above.
**Checklist**: (1) Generate noisy linear data. (2) Implement MSE loss and its gradient by hand. (3) Run gradient descent, plot the loss curve, confirm convergence to the true `m, b`. (4) Complete §1.2.9's verification and confirm manual gradients match autograd to ~1e-6 precision. (5) Deliberately set the learning rate too high and observe/plot divergence — build the intuition for what "unstable training" looks like.

---

## MODULE 1.3 — CNNs (Convolutional Neural Networks)

### 1.3.1 Intuition: why not just flatten images into an MLP?
A 224×224 RGB image has 150,528 input values. A fully-connected first layer with even 1,000 neurons needs 150 million weights — for *one layer*, before the network has learned anything about images specifically (like the fact that a cat's ear looks the same whether it's in the top-left or bottom-right of the photo). CNNs exploit two structural facts about images: **locality** (nearby pixels are related) and **translation invariance** (a feature detector useful in one location is useful everywhere) — and bake both directly into the architecture via **parameter sharing**.

### 1.3.2 The convolution operation — math
A small learnable filter/kernel (e.g., 3×3) slides across the image. At each position, it computes an elementwise multiply-and-sum with the underlying patch of pixels:

```
Input (5×5)              Kernel (3×3)         Output (3×3)
1 1 1 0 0                1 0 1                4 3 4
0 1 1 1 0        *       0 1 0        =       2 4 3
0 0 1 1 1                1 0 1                2 3 4
0 0 1 1 0
0 1 1 0 0

Output[0,0] = (1·1 + 1·0 + 1·1) + (0·0 + 1·1 + 1·0) + (0·1 + 0·0 + 1·1) = 4
              ↑ this same 9-weight kernel is reused at every position — parameter sharing.
```
- **Stride**: how far the kernel moves each step (stride 2 = skip every other position → smaller output, cheaper compute).
- **Padding**: adding zeros around the input border so the kernel can be centered on edge pixels ("same" padding preserves spatial size; "valid" padding shrinks it, as shown above).
- **Dilation**: spacing out the kernel's sample points to grow the receptive field without adding parameters.

### 1.3.3 Feature maps, channels, and the parameter-sharing win
A single filter produces one **feature map** (a 2D grid of "how strongly did this pattern appear here"). A conv layer learns *many* filters in parallel (e.g., 64), producing 64 stacked feature maps (channels). Compare parameter counts directly:

```
Fully-connected layer, 224×224×3 input → 64 outputs:
  150,528 × 64 ≈ 9.6 million weights

Conv layer, 3×3 kernel, 3 input channels → 64 output channels:
  (3×3×3) × 64 = 1,728 weights  — regardless of image size!
```
This is the entire efficiency argument for CNNs in one comparison: the same 1,728 weights are reused at every spatial position, versus a unique weight for every single pixel-output pairing.

### 1.3.4 Pooling
**Max pooling** (most common): slide a small window (e.g., 2×2) and keep only the maximum value, discarding the rest. **Purpose**: (1) downsamples spatial resolution, reducing compute for later layers; (2) adds a small amount of translation invariance (a feature shifted by 1 pixel still triggers the same max, most of the time); (3) reduces overfitting by discarding precise spatial detail the network doesn't need.

### 1.3.5 Receptive field growth
Each successive conv layer's neurons "see" a larger region of the *original* input image, even though each individual filter is small (e.g., 3×3). Stack three 3×3 conv layers, and a single output neuron has an effective receptive field of 7×7 pixels of the original image — with far fewer parameters than one 7×7 conv layer, and an extra two nonlinearities (more expressive power) along the way. This insight (many small filters > few large filters) drove the design of VGG and every architecture since.

### 1.3.6 Classic architectures — why each mattered
- **LeNet (1998)**: proved conv+pool+FC works for digit recognition — the template every later architecture builds on.
- **AlexNet (2012)**: scaled up LeNet's idea with ReLU + dropout + GPU training, won ImageNet by a huge margin, and is widely credited with kicking off the modern deep learning boom.
- **VGG (2014)**: showed depth (many stacked small 3×3 convs) systematically improves accuracy.
- **ResNet (2015)**: solved the **degradation problem** — beyond a certain depth, plain stacked networks got *worse* training accuracy (not just overfitting — the optimization itself broke down, because gradients had to survive a longer multiplicative chain, §1.2.7). ResNet's fix: **residual/skip connections** — `output = F(x) + x` — so each block only has to learn a *residual correction* to the identity function, and gradients have a direct additive path back to earlier layers, bypassing the vanishing-gradient chain. This single idea (`+x` skip connections) is now foundational far beyond CNNs — it's a core component of every Transformer block you'll build in Phase 3.

### 1.3.7 Batch Normalization
Normalizes each layer's activations (zero mean, unit variance) across the mini-batch during training, then applies a learned scale/shift. **Why it helps**: keeps activation distributions stable across layers and training steps ("reduces internal covariate shift" — the original justification, though the exact mechanism is still debated), which in practice allows higher learning rates and faster, more stable convergence. Note: at inference time, BatchNorm uses running statistics accumulated during training (not the current batch) — a common source of subtle train/eval bugs if you forget to call `model.eval()`.

### 1.3.8 Worked example: manual convolution on a tiny matrix
Using the 5×5 input and 3×3 kernel from §1.3.2, verify `Output[1,1]` (center) by hand:
```
Patch (rows 1-3, cols 1-3):
1 1 1
0 1 1
0 1 1

Kernel:
1 0 1
0 1 0
1 0 1

Elementwise product sum: (1·1+1·0+1·1) + (0·0+1·1+1·0) + (0·1+1·0+1·1)
                        = (1+0+1) + (0+1+0) + (0+0+1) = 4
```
Matches the `Output[1,1] = 4` shown earlier — do this by hand once; it demystifies every conv layer you'll ever use.

### 1.3.9 Code: CNN training on CIFAR-10 (PyTorch)
```python
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 8 * 8, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 32x32 -> 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 16x16 -> 8x8
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

model = SimpleCNN()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(5):
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: avg loss {total_loss/len(train_loader):.4f}")
```

### 1.3.10 Forward reference: CNNs inside diffusion models
When you reach Phase 9, you'll meet the **U-Net** — the backbone of Stable Diffusion's denoising step. It's fundamentally a CNN: a downsampling path (conv+pool, shrinking spatial size while growing channels — exactly what you just built above) followed by an upsampling path, with skip connections between matching resolutions. Everything in this module — convolution, pooling, feature maps, skip connections — transfers directly.

### 1.3.11 Interview Q&A (with answers)
**Q: Why convolution instead of fully-connected for images?**
A: Convolution exploits locality (nearby pixels are related) and translation invariance (a useful feature detector is useful everywhere in the image) via parameter sharing — the same small kernel is reused at every spatial position, giving orders-of-magnitude fewer parameters than a fully-connected layer over raw pixels (see §1.3.3's concrete comparison), while also generalizing better since a learned pattern works regardless of where it appears in the image.

**Q: What does a pooling layer do?**
A: Downsamples the spatial resolution of feature maps (reducing compute for subsequent layers), adds a degree of local translation invariance, and reduces overfitting by discarding precise spatial detail that isn't needed for the task.

**Q: What is a receptive field?**
A: The region of the *original input* that a given neuron's output is influenced by. It grows with network depth — stacking small filters (e.g., three 3×3 layers) achieves a large effective receptive field (7×7) with fewer parameters and more nonlinearity than a single large-kernel layer covering the same area.

**Q: Why did ResNet's skip connections matter so much?**
A: Beyond a certain depth, plain CNNs suffered a "degradation problem" — training accuracy itself got worse, not just generalization — because gradients had to survive an increasingly long multiplicative chain back through many layers (§1.2.7). Skip connections (`F(x) + x`) give gradients a direct additive shortcut to earlier layers, letting each block learn only a residual correction rather than a full transformation, which made training networks with 50, 100+ layers actually work.

### 1.3.12 Hands-on project
**Build**: Train the CNN above on CIFAR-10, then visualize what it learned.
**Checklist**: (1) Achieve >75% test accuracy (a well-tuned simple CNN can reach 80%+). (2) Visualize first-layer conv filters directly (they should look like edge/color detectors). (3) Visualize feature maps for a sample image at each layer — confirm early layers detect edges/textures, later layers detect more abstract patterns. (4) Try removing BatchNorm and compare training stability/speed. (5) Optional: implement a small residual block and confirm it trains a deeper version of the network more stably than the plain version.

---

## MODULE 1.4 — RNN, LSTM, GRU (Sequence Models)

### 1.4.1 Intuition: sequences and memory
Language, audio, and time series share a property images don't: **order matters, and context accumulates**. "The bank raised rates" vs "I sat by the river bank" — the same word means different things depending on what came before. RNNs process a sequence one element at a time while maintaining a **hidden state** — a running summary of everything seen so far — updated at every step.

### 1.4.2 Vanilla RNN — math and unrolling
```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y
```
The *same* weight matrices (`W_hh`, `W_xh`) are reused at every timestep — this is parameter sharing across time, analogous to CNN's parameter sharing across space.

```
x₁      x₂      x₃
 │       │       │
 ▼       ▼       ▼
[h₀]──►[h₁]──►[h₂]──►[h₃]──► ...
         │       │       │
         ▼       ▼       ▼
        y₁      y₂      y₃
```
Training uses **Backpropagation Through Time (BPTT)**: "unroll" the recurrence into an equivalent deep feed-forward graph (one "layer" per timestep) and apply standard backprop across it. A 100-token sequence becomes, effectively, a 100-layer-deep network for gradient purposes — which is exactly why long sequences are so problematic (§1.4.3).

### 1.4.3 Why vanilla RNNs fail on long sequences
The gradient of the loss at step `T` with respect to the hidden state at step `t` involves a product of `(T-t)` Jacobian terms (one per timestep in between) — structurally identical to the vanishing/exploding gradient problem from §1.2.7, but now the "depth" is the *sequence length*, which can be hundreds or thousands of steps. In practice, vanilla RNNs effectively "forget" anything beyond roughly 10-20 steps back — hopeless for a paragraph, let alone a document.

### 1.4.4 LSTM — full architecture
LSTM (Long Short-Term Memory) solves this with a **separate cell state** `C_t` that flows across timesteps with only *additive*, gated interactions (no repeated matrix multiply + nonlinearity chain crushing the gradient) — directly analogous to ResNet's `+x` skip connection, but for time instead of depth.

```
Forget gate:   f_t = σ(W_f·[h_{t-1}, x_t] + b_f)      "what to erase from memory"
Input gate:    i_t = σ(W_i·[h_{t-1}, x_t] + b_i)      "what new info to consider"
Candidate:     C̃_t = tanh(W_C·[h_{t-1}, x_t] + b_C)   "the new info itself"
Cell update:   C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t         "erase some, add some — ADDITIVE"
Output gate:   o_t = σ(W_o·[h_{t-1}, x_t] + b_o)      "what part of memory to expose"
Hidden state:  h_t = o_t ⊙ tanh(C_t)
```
Each gate is a sigmoid (outputs 0-1, acting as a learned "how much to let through" valve). The critical design choice is the **additive** cell state update (`f_t ⊙ C_{t-1} + i_t ⊙ C̃_t`) — gradients flowing backward through `C_t` don't get crushed by repeated multiplication the way vanilla RNN's `h_t` does.

```
        ┌─────────────────────────────────────┐
C_{t-1}─┤──⊗(forget)──────⊕──────────────────► C_t
        │              ▲   │
        │           i_t⊙C̃_t │
        │              │   ▼
h_{t-1}─┼─►[f_t][i_t][C̃_t][o_t]      h_t = o_t ⊙ tanh(C_t) ──► h_t
   x_t──┘   (4 gates, each sees h_{t-1} and x_t)
```

### 1.4.5 GRU — simplified alternative
GRU (Gated Recurrent Unit) merges the cell and hidden state into one, and uses only 2 gates instead of LSTM's 3-4:
```
Update gate:  z_t = σ(W_z·[h_{t-1}, x_t])      "how much of the past to keep"
Reset gate:   r_t = σ(W_r·[h_{t-1}, x_t])      "how much past to forget when computing candidate"
Candidate:    h̃_t = tanh(W·[r_t ⊙ h_{t-1}, x_t])
New state:    h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t     "interpolate old vs new — also additive"
```
**GRU vs LSTM tradeoff**: GRU has fewer parameters (faster to train, less prone to overfitting on small datasets) and often performs comparably; LSTM's separate cell state gives it slightly more representational capacity, which can matter on very long sequences or larger datasets. In practice today, neither matters much — Transformers have replaced both for most NLP tasks (§1.4.10) — but the additive-gating principle you're learning here reappears conceptually in Transformer residual connections.

### 1.4.6 Bidirectional RNNs
Run two RNNs over the sequence — one left-to-right, one right-to-left — and concatenate their hidden states at each position. Useful when the *entire* sequence is available at once (e.g., classifying a complete sentence) and future context helps interpret earlier tokens — not usable for autoregressive generation, where future tokens don't exist yet at generation time.

### 1.4.7 Seq2seq (Encoder-Decoder RNN) — the direct predecessor to Transformers
```
Encoder RNN: consumes the full input sequence, producing a final hidden state
             (a fixed-size "summary" of the entire input)
Decoder RNN: initialized with that hidden state, generates the output sequence
             one token at a time, autoregressively (each output token fed back in as
             the next input)
```
This architecture (Sutskever et al. 2014) powered early neural machine translation. Its critical weakness — the **entire input sequence gets compressed into one fixed-size vector**, regardless of length — is exactly what the attention mechanism (Phase 2) was invented to fix, by letting the decoder look back at *all* encoder hidden states rather than just the final compressed one.

### 1.4.8 Worked numerical trace: one LSTM step
Simplified 1-dimensional example, `h_{t-1}=0.5`, `C_{t-1}=0.8`, `x_t=1.0`, all weights=1, all biases=0 (toy values purely to trace the mechanics):
```
concat input [h_{t-1}, x_t] = [0.5, 1.0]

f_t = σ(0.5+1.0) = σ(1.5) ≈ 0.82     (keep ~82% of old cell state)
i_t = σ(1.5) ≈ 0.82                   (let in ~82% of new candidate)
C̃_t = tanh(1.5) ≈ 0.90                (candidate new info)
C_t = 0.82×0.8 + 0.82×0.90 ≈ 0.66 + 0.74 = 1.39
o_t = σ(1.5) ≈ 0.82
h_t = 0.82 × tanh(1.39) ≈ 0.82 × 0.883 ≈ 0.72
```
Trace this by hand once with real numbers — it turns the gate equations from abstract symbols into a concrete "erase 18%, add this much new info" operation you can reason about.

### 1.4.9 Code: char-level LSTM text generator
```python
import torch, torch.nn as nn

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden

# --- toy training setup ---
text = "hello world hello there hello everyone"
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text])

seq_len = 8
def get_batch(bs=16):
    ix = torch.randint(0, len(data) - seq_len - 1, (bs,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y

model = CharLSTM(vocab_size=len(chars))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

for step in range(500):
    x, y = get_batch()
    logits, _ = model(x)
    loss = criterion(logits.view(-1, len(chars)), y.view(-1))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if step % 100 == 0:
        print(f"step {step}, loss {loss.item():.3f}")

# --- generate ---
@torch.no_grad()
def generate(model, start="h", length=30):
    model.eval()
    idx = torch.tensor([[stoi[start]]])
    hidden = None
    out = start
    for _ in range(length):
        logits, hidden = model(idx, hidden)
        probs = torch.softmax(logits[0, -1], dim=0)
        next_id = torch.multinomial(probs, 1).item()   # sampling — same idea as LLM decoding (Phase 4)
        out += itos[next_id]
        idx = torch.tensor([[next_id]])
    return out

print(generate(model))
```
Note the `hidden` state being threaded through `generate()` step by step — this is the exact same autoregressive, one-token-at-a-time generation loop you'll use for LLMs in Phase 4, just with an LSTM's recurrence instead of a Transformer's attention doing the context-tracking.

### 1.4.10 Bridge to Transformers — every limitation that motivated them
This is the single most important list in this module — internalize it before Phase 3:
1. **Sequential computation** — RNN step `t` can't start until step `t-1` finishes. This is fundamentally unparallelizable across the sequence dimension, making training on long sequences slow regardless of hardware. Transformers process all positions simultaneously via self-attention.
2. **Long-range dependency loss** — even with LSTM/GRU's gating, information from far back in a very long sequence degrades. Self-attention gives every token a *direct* path to every other token, regardless of distance — no degradation with distance in principle.
3. **Fixed-size bottleneck in seq2seq** — the encoder's final hidden state has to compress an entire input sequence into one vector. Attention (Phase 2 bridge, then full self-attention in Phase 3) lets the decoder access all encoder states directly.
4. **Limited effective context** — practically, RNNs struggle to use information from more than a few dozen steps back, no matter the gating mechanism. Transformers' context window is limited by compute/memory, not by gradient degradation over distance.

Everything you now understand about gating (LSTM's forget/input gates as learned "how much information to let through" valves) and additive state updates (avoiding the vanishing-gradient multiplicative chain) will resurface conceptually when you study residual connections and gating variants (like SwiGLU) inside Transformer blocks.

### 1.4.11 Interview Q&A (with answers)
**Q: Why do plain RNNs suffer from vanishing gradients?**
A: BPTT unrolls the recurrence into an effectively very deep network (one "layer" per timestep), and the gradient back to early timesteps is a product of many per-step Jacobian terms — structurally the same problem as vanishing gradients in very deep feed-forward nets (§1.2.7), except here "depth" equals sequence length, which can be far larger than any practical feed-forward network.

**Q: How do LSTM gates solve this?**
A: The cell state `C_t` is updated additively (`f_t⊙C_{t-1} + i_t⊙C̃_t`) rather than through a repeated matrix-multiply-then-nonlinearity chain. Gradients flowing backward through the cell state don't get multiplicatively crushed the way vanilla RNN hidden states do — the forget gate can learn to stay near 1 for information that needs to persist over long distances, essentially creating a near-uninterrupted gradient highway back through time. This is conceptually identical to ResNet's residual connections solving the same problem across network depth.

**Q: Why were Transformers a breakthrough over RNNs for parallelization?**
A: RNNs must process a sequence strictly step by step because each hidden state depends on the previous one — this can't be parallelized across the time dimension, no matter how much hardware you throw at it. Self-attention computes relationships between all pairs of positions simultaneously via matrix multiplication, so the entire sequence can be processed in parallel on a GPU — the primary reason Transformers can be trained on today's massive datasets in feasible time.

**Q: GRU vs LSTM — when would you choose one over the other?**
A: GRU has fewer parameters (2 gates vs LSTM's effectively 3-4, and no separate cell state), so it trains faster and can generalize better on smaller datasets; LSTM's separate cell state gives slightly more representational capacity, which can help on longer sequences or larger datasets. In practice the difference is usually small and secondary to whether you should be using attention/Transformers instead.

### 1.4.12 Hands-on project
**Build**: The char-level LSTM text generator above, extended to a real corpus.
**Checklist**: (1) Train on a larger corpus (e.g., a public-domain book, tinyshakespeare) rather than the toy string. (2) Implement both temperature-scaled sampling and greedy (argmax) decoding — compare output diversity/quality. (3) Plot training loss and confirm it decreases smoothly (if it doesn't, revisit §1.2's optimizer/LR guidance). (4) Swap LSTM for GRU with the same hyperparameters and compare training speed and final loss. (5) Reflect exercise (no code): write out, in your own words, all four points from §1.4.10 — this is your checkpoint before Phase 3.

---

## Phase 1 Completion Checklist
Before moving to Phase 2 (Attention) and Phase 3 (Transformers), you should be able to, without looking anything up:
- [ ] Derive the forward pass of an MLP and explain why nonlinearity is mathematically required for depth to matter.
- [ ] Derive backpropagation for a 2-layer network from the chain rule, from memory.
- [ ] Explain Adam's two components (momentum + adaptive scaling) and why LLM training uses warmup + cosine decay.
- [ ] Explain vanishing/exploding gradients and name all 5 mitigations from §1.2.7.
- [ ] Explain convolution's parameter-sharing efficiency argument with actual numbers (§1.3.3).
- [ ] Explain why ResNet's skip connections solved the degradation problem — and connect this to LSTM's additive cell-state update from Module 1.4 (same underlying principle, different domain).
- [ ] Recite, unprompted, all four RNN limitations that motivated the Transformer (§1.4.10) — this is the single most interview-relevant list in this document.
- [ ] Have working, from-scratch code for: an MLP (NumPy), manual backprop (verified against autograd), a CNN (PyTorch, trained on CIFAR-10), and an LSTM text generator (PyTorch).

## What's Next
**Phase 2** (Attention, pre-Transformer bridge) is short (1-2 weeks) and directly resolves the seq2seq bottleneck from §1.4.7 — you'll implement Bahdanau attention on top of the seq2seq pattern you just learned. **Phase 3** (Transformers) is where all of Phase 1's building blocks — matrix multiplication, residual connections, normalization, gating intuition, and the autoregressive generation loop from §1.4.9 — recombine into the architecture powering every modern LLM. You're closer than it feels; Phase 1 is the hardest phase precisely because it's the least glamorous, and everything after this reuses it constantly.
