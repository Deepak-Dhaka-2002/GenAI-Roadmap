# Phase 2 Deep Dive: The Attention Mechanism — Bridge to Transformers
### A full course module — from the seq2seq bottleneck to the exact equation Transformers scale up

> This expands Phase 2 of the master roadmap. The roadmap listed one topic here ("Attention Mechanism, originally added to RNNs") — I've broken it into 13 modules, including a few additions beyond the original list (marked **[Addition]**) that don't cost much time now but save real confusion when you hit self-attention in Phase 3. This phase is short by design (1-2 weeks) — it's a hinge, not a destination.

---

## MODULE 2.1 — Recap: the exact problem this phase solves

From Phase 1 (§1.4.7, seq2seq), recall the architecture:
```
Encoder RNN: reads x₁...x_T  →  produces one final hidden state h_T (a fixed-size vector)
Decoder RNN: initialized with h_T  →  generates y₁...y_M, one token at a time
```
The entire input sequence — whether 5 words or 500 — gets compressed into that **one fixed-size vector** `h_T`. The decoder never sees the individual encoder states again; it only ever sees the encoder's final summary. In practice, translation quality measurably degrades as input sentences get longer, because more and more information is being forced through the same fixed-size bottleneck (Cho et al. 2014 documented this directly). This module fixes exactly that, and only that — everything below is elaboration on one idea: **let the decoder look back at every encoder state, not just the last one.**

---

## MODULE 2.2 — The core idea, stated as a reusable recipe

Every attention mechanism you will ever encounter — Bahdanau attention, Luong attention, Transformer self-attention, cross-attention, multi-head attention — is the same three-step recipe, applied to different things:

```
1. SCORE:   compare a "query" against every "key" → a raw compatibility number per key
2. NORMALIZE: softmax the scores → a probability distribution that sums to 1 ("attention weights")
3. COMBINE: weighted sum of "values" using those weights → one output vector ("context vector")
```

In this phase: **query** = the decoder's current state (what am I trying to generate right now?), **keys** = every encoder hidden state (what information is available?), **values** = usually the same encoder hidden states again (what do I actually retrieve?). Memorize this recipe now in the abstract — in Phase 3, `Q`, `K`, `V` will just be three learned projections of token embeddings, and the same three steps apply unchanged.

---

## MODULE 2.3 — Bahdanau (additive) attention — full derivation

From Bahdanau, Cho & Bengio, 2014 ("Neural Machine Translation by Jointly Learning to Align and Translate") — the paper that introduced this whole idea.

**Setup**: bidirectional encoder RNN produces hidden states `h₁, h₂, ..., h_T` (one per input position). Decoder RNN is about to produce output token `i`, and currently holds state `s_{i-1}` (its state *before* this step).

```
Step 1 — Score (additive/concat form):
  e_{ij} = vₐᵀ · tanh(Wₐs_{i-1} + Uₐh_j)      for every encoder position j

  Wₐ, Uₐ, vₐ are learned parameters. Concatenating s_{i-1} and h_j (implicitly, via
  the sum of two separate linear projections) and passing through a small feed-forward
  network is why this is called "additive" — it's literally a 1-hidden-layer MLP scoring
  compatibility between the decoder state and each encoder state.

Step 2 — Normalize:
  α_{ij} = softmax_j(e_{ij}) = exp(e_{ij}) / Σₖ exp(e_{ik})
  (α_{i1}, ..., α_{iT}) is a probability distribution over encoder positions — the
  "attention weights" for this decoding step, summing to 1.

Step 3 — Combine:
  c_i = Σⱼ α_{ij} · h_j                        ("context vector" for this decoding step)

Step 4 — Decode:
  s_i = RNN(s_{i-1}, [y_{i-1}; c_i])            decoder update, now informed by c_i
  y_i = output_layer([s_i; c_i])                 prediction, also informed by c_i
```
**Critical property**: `c_i` is recomputed **fresh at every decoding step** — a different weighted combination of the same encoder states, depending on what the decoder is currently trying to produce. This is the entire fix: the bottleneck is gone because the decoder is no longer limited to one static summary vector.

---

## MODULE 2.4 — Luong (multiplicative) attention — full math and score variants

From Luong, Pham & Manning, 2015 ("Effective Approaches to Attention-based Neural Machine Translation") — a simplification that turned out to matter enormously for what came next.

Luong proposed three alternative score functions, all replacing Bahdanau's feed-forward scorer:

```
dot:      score(s_i, h_j) = s_iᵀ h_j                    (plain dot product — cheapest)
general:  score(s_i, h_j) = s_iᵀ Wₐ h_j                  (one learned bilinear matrix)
concat:   score(s_i, h_j) = vₐᵀ tanh(Wₐ[s_i; h_j])        (≈ Bahdanau's form)
```

**Why "dot" matters more than it looks**: it requires *zero* extra parameters beyond the states themselves, and it's a single matrix multiplication — no nonlinearity, no extra feed-forward network evaluated at every (decoder step × encoder position) pair. This makes it dramatically cheaper to compute at scale, at the cost of requiring the decoder and encoder states to live in geometrically comparable spaces (dot product only measures compatibility meaningfully if the vectors' relative directions are informative — the "general" variant's `Wₐ` matrix exists precisely to first re-project one space into alignment with the other).

**This is not a footnote — it's the direct ancestor of the Transformer.** Vaswani et al. 2017's self-attention formula is:
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```
Strip away the `Q`/`K`/`V` naming and the `/√d_k` scaling term, and this is *exactly* Luong's "dot" score, generalized from "one decoder query against many encoder keys" to "every position against every position." The scaling factor `1/√d_k` is the one genuinely new ingredient — and it exists to solve a problem Luong's formulation quietly has too: **dot products grow in magnitude as dimensionality `d` increases** (the dot product of two random `d`-dimensional vectors has variance proportional to `d`), which pushes the softmax into a saturated regime (a few huge scores, everything else near zero) where gradients vanish. Dividing by `√d_k` keeps the variance of the scores roughly constant regardless of dimension, keeping softmax in a well-behaved range. You will re-derive this exact justification in Phase 3 — it isn't new there, it's this same fix, formalized.

---

## MODULE 2.5 — Global vs. local attention

Luong's paper also distinguished two attention *scopes*:

- **Global attention** (what §2.3–2.4 describe by default): attend over **every** encoder position at every decoding step. Simple, but cost grows linearly with input length per decoding step.
- **Local attention**: attend only over a small window of encoder positions centered at a predicted alignment point `p_t`, rather than the whole sequence — cheaper, and works well when alignment is roughly monotonic (e.g., similar word order between source/target languages, or speech recognition where audio-to-text alignment is naturally close to monotonic).
  ```
  Predictive local attention: p_t = S · sigmoid(vₚᵀ tanh(Wₚs_t))     (S = source length, p_t is learned)
  Monotonic local attention:  p_t = t                                  (assumes alignment ≈ position)
  ```
  Attention weights are then computed (via the same score/softmax/weighted-sum recipe) only over positions near `p_t`, typically within a fixed window `D`.

**Why this matters beyond this phase [Addition]**: full Transformer self-attention is, by default, global — every token attends to every other token, which is exactly why it costs `O(n²)` (Module 2.10). The *local* attention idea you just learned resurfaces later, in Phase 10's territory, as the core idea behind efficient long-context attention variants (sliding-window attention, sparse attention) that make very long contexts computationally tractable. You're not learning a historical dead end here — you're learning the pattern that gets reused when Transformers hit their own scaling wall.

---

## MODULE 2.6 — Worked numerical example (Luong dot-product attention, by hand)

Toy setup: 3 encoder hidden states (2-dimensional, for hand-computability), 1 decoder state.
```
h₁ = [1.0, 0.0]      h₂ = [0.0, 1.0]      h₃ = [1.0, 1.0]
s  = [1.0, 0.5]       (current decoder state — the "query")

Step 1 — Score (dot product):
  e₁ = s·h₁ = (1.0)(1.0) + (0.5)(0.0) = 1.0
  e₂ = s·h₂ = (1.0)(0.0) + (0.5)(1.0) = 0.5
  e₃ = s·h₃ = (1.0)(1.0) + (0.5)(1.0) = 1.5

Step 2 — Softmax:
  exp(1.0)=2.718,  exp(0.5)=1.649,  exp(1.5)=4.482
  sum = 2.718+1.649+4.482 = 8.849
  α₁ = 2.718/8.849 = 0.307
  α₂ = 1.649/8.849 = 0.186
  α₃ = 4.482/8.849 = 0.506
  (check: 0.307+0.186+0.506 = 0.999 ≈ 1 ✓)

Step 3 — Weighted sum (context vector):
  c = 0.307·[1.0,0.0] + 0.186·[0.0,1.0] + 0.506·[1.0,1.0]
    = [0.307, 0.0] + [0.0, 0.186] + [0.506, 0.506]
    = [0.813, 0.692]
```
`h₃` got the highest weight (0.506) because it had the highest dot-product similarity with the query `s` — the decoder is "looking mostly at position 3, partly at position 1, least at position 2" for this generation step. Trace this by hand once; it's the entire mechanism, laid bare with real numbers instead of matrix notation hiding what's happening.

---

## MODULE 2.7 — Code: seq2seq with Bahdanau attention, from scratch (PyTorch)

Toy task: **sequence reversal** (`[3,1,4,1,5] → [5,1,4,1,3]`) — chosen deliberately because the correct attention alignment is a clean anti-diagonal, which makes the heatmap in Module 2.8 immediately legible as a sanity check.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

VOCAB_SIZE = 10   # digits 0-9
SOS, EOS = 10, 11
VOCAB_SIZE += 2

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.reduce = nn.Linear(hidden_dim * 2, hidden_dim)  # merge bidirectional states

    def forward(self, x):
        embedded = self.embed(x)
        outputs, (h, c) = self.lstm(embedded)          # outputs: [batch, seq_len, hidden*2]
        outputs = torch.tanh(self.reduce(outputs))      # -> [batch, seq_len, hidden]  (the "keys/values")
        h_cat = torch.tanh(self.reduce(torch.cat([h[0], h[1]], dim=-1)))  # initial decoder state
        return outputs, h_cat

class BahdanauAttention(nn.Module):
    """Implements §2.3 exactly: e_ij = v^T tanh(W_a s + U_a h_j)"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)   # applied to decoder state
        self.U_a = nn.Linear(hidden_dim, hidden_dim, bias=False)   # applied to encoder states
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        # decoder_state: [batch, hidden]   encoder_outputs: [batch, seq_len, hidden]
        query = self.W_a(decoder_state).unsqueeze(1)         # [batch, 1, hidden]
        keys = self.U_a(encoder_outputs)                      # [batch, seq_len, hidden]
        scores = self.v_a(torch.tanh(query + keys)).squeeze(-1)  # [batch, seq_len]  <- Step 1: SCORE
        weights = F.softmax(scores, dim=-1)                   # [batch, seq_len]     <- Step 2: NORMALIZE
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # <- Step 3: COMBINE
        return context, weights   # returning weights lets us plot the heatmap in §2.8

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(hidden_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim * 2, vocab_size)

    def forward_step(self, input_token, h, c, encoder_outputs):
        embedded = self.embed(input_token)                    # [batch, embed_dim]
        context, weights = self.attention(h, encoder_outputs)  # uses PREVIOUS decoder state as query
        lstm_input = torch.cat([embedded, context], dim=-1)
        h, c = self.lstm_cell(lstm_input, (h, c))
        logits = self.out(torch.cat([h, context], dim=-1))
        return logits, h, c, weights

# --- toy data: random digit sequences and their reversal ---
def make_batch(batch_size=32, seq_len=6):
    src = torch.randint(0, 10, (batch_size, seq_len))
    tgt = src.flip(dims=[1])
    return src, tgt

encoder = Encoder(VOCAB_SIZE)
decoder = Decoder(VOCAB_SIZE)
optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

for step in range(2000):
    src, tgt = make_batch()
    encoder_outputs, h = encoder(src)
    c = torch.zeros_like(h)
    loss = 0
    input_token = torch.full((src.size(0),), SOS, dtype=torch.long)
    for t in range(tgt.size(1)):
        logits, h, c, _ = decoder.forward_step(input_token, h, c, encoder_outputs)
        loss += F.cross_entropy(logits, tgt[:, t])
        input_token = tgt[:, t]   # teacher forcing
    loss = loss / tgt.size(1)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if step % 400 == 0:
        print(f"step {step}, loss {loss.item():.4f}")
```
Notice `forward_step` returns `weights` — that's the attention distribution for this single decoding step, and it's what Module 2.8 visualizes.

---

## MODULE 2.8 — Visualizing attention: alignment heatmaps

The single most useful debugging/interpretability tool this phase gives you: plot attention weights as a heatmap, decoder steps on one axis, encoder positions on the other. For the reversal task above, a correctly-trained model should show a clean **anti-diagonal** (output position 1 attends almost entirely to input position `seq_len`, output position 2 to `seq_len-1`, etc.) — if your heatmap doesn't show this pattern, the model hasn't actually learned the alignment, regardless of what the loss curve says.

```python
import matplotlib.pyplot as plt

@torch.no_grad()
def generate_with_attention(src, max_len=6):
    encoder.eval(); decoder.eval()
    encoder_outputs, h = encoder(src.unsqueeze(0))
    c = torch.zeros_like(h)
    input_token = torch.tensor([SOS])
    all_weights = []
    outputs = []
    for _ in range(max_len):
        logits, h, c, weights = decoder.forward_step(input_token, h, c, encoder_outputs)
        input_token = logits.argmax(dim=-1)
        outputs.append(input_token.item())
        all_weights.append(weights.squeeze(0).numpy())
    return outputs, all_weights

src = torch.tensor([3,1,4,1,5,9])
outputs, weights = generate_with_attention(src)

plt.imshow(weights, cmap='viridis', aspect='auto')
plt.xlabel('Encoder position (input)')
plt.ylabel('Decoder step (output)')
plt.xticks(range(len(src)), src.tolist())
plt.yticks(range(len(outputs)), outputs)
plt.colorbar(label='attention weight')
plt.title('Attention alignment: reversal task')
plt.savefig('attention_heatmap.png')
```
This exact visualization — an alignment matrix — is what made the original Bahdanau paper convincing: for machine translation, the heatmaps showed the model learning sensible word-to-word alignments (including reordering between languages) purely from data, with no alignment supervision ever provided.

---

## MODULE 2.9 — [Addition] From attention to self-attention: the conceptual leap you need before Phase 3

Everything above is **cross-attention**: the query comes from one sequence (the decoder), the keys/values come from a *different* sequence (the encoder). There's one more conceptual step to Transformers, and it's smaller than it looks:

**What if a sequence attended to itself?** — i.e., query, key, *and* value are all derived from the *same* sequence. Instead of "decoder position asks: which encoder positions are relevant to me," it becomes "token 5 asks: which *other tokens in this same sentence* (including itself) are relevant to understanding me." This is **self-attention**, and it lets a single sequence build context-aware representations of itself — a word like "bank" can gather information from "river" or "rates" appearing elsewhere in the same sentence, resolving ambiguity without needing any second sequence at all.

Formally, generalize Module 2.4's dot-product score:
```
Cross-attention (this phase):     Q = decoder states,  K = V = encoder states  (different sequences)
Self-attention (Phase 3):         Q = K = V = (learned projections of) the SAME sequence

Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```
where `Q = XWQ`, `K = XWK`, `V = XWV` — three separate learned linear projections of the same input `X`. Notice this is a direct, almost mechanical generalization: replace "one decoder state at a time" with "a whole matrix of query vectors, one per token, computed in parallel," and replace Luong's raw `sᵀh` dot product with the scaled version `QKᵀ/√d_k` from Module 2.4's discussion. You already understand every piece of this equation — Phase 3 is about what changes when you apply it to a sequence attending to itself, stack it into multi-head form, and remove the RNN entirely.

---

## MODULE 2.10 — [Addition] Complexity analysis: why this matters for what comes later

For cross-attention here: at each of `T_dec` decoding steps, you score against `T_enc` encoder positions — total cost `O(T_dec · T_enc · d)` across a full sequence generation. Manageable, since RNN decoding is already sequential and this adds a constant-factor overhead per step.

For **self-attention** (Phase 3): every one of `n` tokens attends to every other token in the *same* sequence of length `n` — total cost `O(n² · d)`. This quadratic-in-sequence-length cost is the defining computational characteristic of Transformers, and it's the direct reason:
- Context windows were historically limited (2K, 4K, 8K tokens) before optimization work made longer contexts practical.
- **FlashAttention** and other efficient-attention implementations (Phase 10 territory) exist specifically to make this `O(n²)` computation fast and memory-efficient on real hardware, without changing the math.
- **Sparse/local/sliding-window attention** variants (the modern descendants of Module 2.5's local attention) trade exhaustive all-pairs attention for a cheaper approximation when context windows grow very large.

You now understand *why* this tradeoff exists, not just that it exists — the local-vs-global distinction you learned in Module 2.5 for RNN attention is the exact same tradeoff being fought at a much larger scale in every modern long-context LLM.

---

## MODULE 2.11 — Common pitfalls

- **Forgetting to mask padded positions before softmax.** When batching variable-length sequences, padded positions must be set to `-inf` *before* the softmax (never after) — otherwise the model learns to attend to meaningless padding tokens, and gradients leak through them. This exact masking mechanism, generalized, becomes **causal masking** in Transformer decoders (Phase 3) — preventing a token from attending to future positions. Learn the mechanic here; the stakes are just higher later.
- **Confusing which axis softmax normalizes over.** Attention weights sum to 1 *across keys, for a fixed query* — not across queries for a fixed key. Getting this backward silently produces a mathematically valid but meaningless distribution.
- **Using unscaled dot-product scores with high-dimensional states.** Even in this phase (before Transformers formalize the `√d_k` fix), if your hidden dimension is large, raw dot-product scores can push softmax into a near-one-hot regime, effectively killing gradient flow to all but one position. If training seems to collapse onto attending to a single position early and never recovers, this is a likely cause — try Luong's "general" (bilinear) score, or scale manually.
- **Evaluating only on loss, not on attention quality.** A model can achieve low loss with degenerate or nonsensical attention patterns on simple/short-sequence tasks (it can partially "cheat" via the decoder's own recurrence). Always sanity-check the alignment heatmap, not just the loss curve — Module 2.8 exists specifically because of this failure mode.

---

## MODULE 2.12 — Interview Q&A (with answers)

**Q: What problem does attention solve versus vanilla seq2seq?**
A: Vanilla seq2seq compresses the entire input sequence into one fixed-size vector (the encoder's final hidden state), which becomes an information bottleneck that worsens as input length grows. Attention lets the decoder access *all* encoder hidden states at every decoding step, computing a fresh weighted combination (the context vector) relevant to what it's currently generating, rather than relying on one static summary.

**Q: What's the difference between additive (Bahdanau) and multiplicative (Luong) attention?**
A: Bahdanau's score function is a small feed-forward network — concatenate (via separate linear projections) the decoder state and an encoder state, apply tanh, then a linear layer to a scalar. Luong's simplest variant is a raw dot product between the two state vectors — no extra parameters, no nonlinearity, just a matrix multiply. Luong's dot-product form is cheaper to compute and is the direct mathematical ancestor of Transformer self-attention's `QKᵀ` term.

**Q: Why does dividing by √d_k matter (even though it's technically introduced in the Transformer paper)?**
A: Dot products between random vectors grow in variance proportional to their dimensionality `d`. As `d` grows, raw dot-product scores can become large in magnitude, pushing the softmax into a saturated regime where one score dominates and the gradient to all others vanishes. Dividing by `√d_k` keeps the variance of the scores roughly constant regardless of dimension, keeping softmax well-behaved and gradients flowing.

**Q: Global vs. local attention — when would you use local?**
A: Global attention scores every encoder position at every decoding step — simple but linear-cost per step. Local attention restricts scoring to a small window around a predicted (or assumed monotonic) alignment position, cutting compute — useful when the source/target alignment is roughly monotonic (e.g., similar word order languages, or speech-to-text) and when input sequences are long enough that full global scoring becomes expensive.

**Q: How does self-attention differ from the attention studied in this phase?**
A: The attention in this phase is cross-attention: the query comes from one sequence (the decoder) and the keys/values come from a different sequence (the encoder). Self-attention uses the *same* sequence to derive the query, key, and value (via three separate learned projections), letting a sequence build context-aware representations of itself without needing a second sequence at all — this is the mechanism inside every Transformer block.

**Q: What does an attention weight actually represent, probabilistically?**
A: For a fixed query (decoding step), the attention weights form a probability distribution over key positions (summing to 1) — interpretable as "how much of this position's information should be included in the context vector, relative to every other position," learned end-to-end from the task loss with no explicit alignment supervision.

---

## MODULE 2.13 — Hands-on project

**Build**: The full seq2seq + Bahdanau attention pipeline from §2.7, extended and evaluated properly.
**Checklist**:
1. Train the reversal-task model from §2.7 to near-perfect accuracy (it should be an easy task once attention is working — if it isn't converging, check for the masking/scaling pitfalls in §2.11).
2. Generate the alignment heatmap (§2.8) and confirm the anti-diagonal pattern — this is your primary correctness signal, more informative than loss alone.
3. **Ablation**: train a second model with the *same* architecture but attention removed (decoder only ever sees the encoder's final hidden state, like plain seq2seq). Compare accuracy on short (length 5) vs. long (length 15+) sequences — you should see the attention model's advantage widen as sequence length grows, directly demonstrating the bottleneck problem from §2.1.
4. Swap Bahdanau's additive scorer for Luong's dot-product scorer (§2.4) in the same pipeline; compare training speed (should be faster — fewer parameters, simpler computation) and final accuracy (should be comparable on this task).
5. **[Addition]** Implement basic padding masking: train on variable-length sequences within a batch, mask padded positions before softmax, and confirm (via the heatmap) that attention weight on padding positions is ~0.

---

## Phase 2 Completion Checklist
Before moving to Phase 3 (Transformers), you should be able to, without looking anything up:
- [ ] State the three-step attention recipe (score → normalize → combine) from memory and explain how it applies to Bahdanau attention specifically.
- [ ] Derive Bahdanau's additive score function and Luong's three score variants (dot, general, concat).
- [ ] Explain *why* Luong's dot-product score is the direct mathematical ancestor of `QKᵀ` in Transformer self-attention.
- [ ] Explain why dividing by `√d_k` is necessary as dimensionality grows — the softmax-saturation argument, not just "the paper says so."
- [ ] Explain the difference between cross-attention (this phase) and self-attention (next phase) in one sentence each.
- [ ] Explain why self-attention costs `O(n²)` and why that specific fact drives a meaningful chunk of later engineering work (FlashAttention, long-context optimization, sparse attention).
- [ ] Have working code that trains a seq2seq+attention model and produces a legible alignment heatmap.

## What's Next
**Phase 3** takes the self-attention formula from Module 2.9 — `softmax(QKᵀ/√d_k)V` — and builds an entire architecture around it: multi-head attention (running several of these in parallel with different learned projections), position-wise feed-forward layers, residual connections + normalization (both concepts you already have from Phase 1's ResNet and LSTM discussions), and positional embeddings (since removing recurrence also removes any inherent sense of token order — attention alone is permutation-invariant). Every mechanical piece has now been introduced somewhere in Phase 1 or 2; Phase 3 is where they get assembled into the architecture that changed everything.
