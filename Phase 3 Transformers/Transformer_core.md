# Phase 3 Deep Dive: Tokenization, Embeddings, and the Transformer Architecture
### A full course module — THE CORE of the entire roadmap

> This expands Phase 3 into complete teaching content. The roadmap listed three topics (Tokenization, Embeddings, Transformer Architecture) — I've broken them into 11 modules, and added several **[Addition]** modules covering things every practicing GenAI engineer needs but that don't get taught in the standard "Attention Is All You Need" walkthrough: RoPE (the positional encoding actually used in LLaMA/Mistral/Qwen/Gemma), GQA/MQA (the attention variant actually used in production models, not vanilla multi-head), Pre-LN vs Post-LN (a real training-stability decision), and KV-caching (how inference actually works). Everything in Phase 1 (residual connections, LayerNorm, gating) and Phase 2 (the attention recipe, `QKᵀ`, causal masking's ancestor) gets used here without re-derivation — if anything below feels unfamiliar, that's the signal to revisit those modules first.

---

## MODULE 3.1 — Tokenization

### 3.1.1 Why tokenization, precisely
Neural networks consume numbers. Text must become integers before anything in Phase 1 (matrix multiplication, embeddings) can touch it. Two naive options and why both fail at scale:
- **Character-level**: tiny vocabulary (~100 symbols), but sequences become very long (a 500-word essay is ~2,500 characters), and the model must learn spelling *and* meaning from scratch — wasteful use of context length and model capacity.
- **Word-level**: short sequences, but the vocabulary is enormous (hundreds of thousands of words) and still fails on any word not seen during training (out-of-vocabulary/OOV), including typos, new slang, and most non-English morphology.

**Subword tokenization** is the compromise that won: common words stay whole ("the", "cat"), rare/complex words split into meaningful pieces ("unhappiness" → "un"+"happi"+"ness"), and *any* string can be represented (worst case, fall back to individual bytes/characters) — no OOV problem, ever.

### 3.1.2 Byte-Pair Encoding (BPE) — full algorithm, worked example
BPE (adapted from a 1994 data-compression algorithm, applied to NLP by Sennrich et al. 2016) builds a vocabulary bottom-up: start with individual characters, iteratively merge the most frequent adjacent pair into a new symbol, repeat until you hit a target vocabulary size.

**Toy corpus** (word: frequency), each word split into characters plus an end-of-word marker `</w>`:
```
"lower"  ×3  →  l o w e r </w>
"lowest" ×2  →  l o w e s t </w>
"newer"  ×4  →  n e w e r </w>
"wider"  ×1  →  w i d e r </w>
```

**Iteration 1 — count every adjacent symbol pair, weighted by word frequency:**
```
(l,o): 3+2=5    (o,w): 3+2=5    (w,e): 3+2+4=9   ← most frequent
(e,r): 3+4+1=8  (r,</w>): 3+4+1=8   (e,s): 2   (s,t): 2   (t,</w>): 2
(n,e): 4        (e,w): 4        (w,i): 1   (i,d): 1   (d,e): 1
```
Merge `(w,e) → "we"` (highest count, 9). Corpus becomes:
```
l o we r </w>   ×3      l o we s t </w>   ×2
n e we r </w>   ×4      w i d e r </w>    ×1
```

**Iteration 2 — recount pairs on the updated corpus:**
```
(we,r): 3+4=7  ← now most frequent    (l,o): 5   (o,we): 5   (e,r): 1 (only wider left)
(n,e): 4   (we,s): 2   (s,t): 2   ... etc.
```
Merge `(we,r) → "wer"`. Corpus becomes:
```
l o wer </w>  ×3    l o we s t </w>  ×2    n e wer </w>  ×4    w i d e r </w>  ×1
```
**This continues** for a fixed number of merges (or until reaching a target vocab size, e.g., 50,257 for GPT-2, ~128,000 for LLaMA-3) — each merge is recorded, in order, as a **merge rule**. To tokenize new text later, you apply the exact same merge rules, in the exact same order, to whatever raw text comes in.

**Key insight from just these two steps**: frequent morphemes ("wer" — as in lower/newer/wider's shared ending) get their own token, while everything stays decomposable into characters as a fallback — nothing is ever truly OOV.

### 3.1.3 WordPiece, SentencePiece, and byte-level BPE
| Method | Used by | Key difference from vanilla BPE |
|---|---|---|
| **BPE** | GPT-2/3/4 (via `tiktoken`) | Merges by raw frequency count, as above. |
| **WordPiece** | BERT | Merges by *likelihood* improvement to a language model, not raw frequency — chooses the pair that most increases the training data's likelihood under a unigram LM, not just the most common pair. |
| **SentencePiece** | LLaMA, T5, Mistral | A *framework*, not a distinct algorithm — implements BPE or Unigram LM tokenization but operates directly on raw text (treating spaces as an ordinary symbol, often `▁`) rather than pre-splitting on whitespace first. This makes it language-agnostic — critical for languages like Japanese/Chinese with no whitespace word boundaries. |
| **Byte-level BPE** | GPT-2 onward | Runs BPE over raw UTF-8 *bytes* (256 possible base symbols) instead of Unicode characters. **[Addition, but essential]**: this guarantees the tokenizer can represent literally any input — any Unicode character, emoji, or even malformed text — as some sequence of the 256 byte values, with zero possibility of an unrepresentable character. Pure character-level BPE can still fail on rare Unicode; byte-level BPE structurally cannot. |

### 3.1.4 [Addition] Special tokens and chat templates
Every production tokenizer reserves specific token IDs for structural markers, not natural text:
```
<bos> / <s>       beginning of sequence
<eos> / </s>      end of sequence — the model learns to emit this to signal "stop generating"
<pad>             padding, used to batch variable-length sequences together (must be masked out of loss/attention)
<unk>             historically "unknown token" — largely obsolete with byte-level BPE, since nothing is truly OOV anymore
```
Modern chat-tuned LLMs add further structural tokens to mark conversation turns — e.g., something functionally like `<|im_start|>system`, `<|im_start|>user`, `<|im_start|>assistant` (exact tokens vary by model family). This is the **chat template**: a fixed string format wrapping the raw conversation into the exact token sequence the model was fine-tuned to expect. **Why this matters practically**: using the wrong chat template (or none at all) with an instruction-tuned model is one of the most common real-world bugs — the model wasn't trained to see raw unformatted text as "the user's turn," so outputs degrade in ways that look like a model quality problem but are actually a formatting bug. You'll use this concept directly in Phase 4/5.

### 3.1.5 Tokenization artifacts — why LLMs have specific, predictable blind spots
Several well-known LLM quirks trace directly back to tokenization, not reasoning ability:
- **Struggling to count letters in a word** (e.g., "how many r's in strawberry") — the model never sees the letters "s-t-r-a-w-b-e-r-r-y" individually; it sees whatever subword tokens BPE happened to split that word into (e.g., "straw"+"berry" or similar), so letter-counting requires the model to have implicitly learned spelling from statistical co-occurrence, not direct character access.
- **Arithmetic errors on multi-digit numbers** — tokenizers often split numbers inconsistently (e.g., "1234" might become one token, while "12345" splits into "123"+"45"), so the model doesn't see digits with consistent positional/place-value structure the way a human reading them does. (Some modern tokenizers now special-case digit-by-digit splitting specifically to reduce this.)
- **Cost/context asymmetry across languages** — languages with less pretraining-corpus representation often tokenize far less efficiently (more tokens per "unit of meaning"), meaning the same sentence can cost 2-3x more tokens (and therefore more of the context window and more API cost) in some languages than in English.

### 3.1.6 Code: BPE from scratch
```python
import re
from collections import defaultdict, Counter

def get_vocab(corpus):
    """corpus: dict of {word: frequency}. Returns dict of {tuple-of-symbols: frequency}."""
    vocab = {}
    for word, freq in corpus.items():
        symbols = tuple(word) + ('</w>',)
        vocab[symbols] = freq
    return vocab

def get_pair_counts(vocab):
    pairs = defaultdict(int)
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    new_vocab = {}
    bigram = pair[0] + " " + pair[1]
    pattern = re.compile(r'(?<!\S)' + re.escape(bigram) + r'(?!\S)')
    for symbols, freq in vocab.items():
        word_str = " ".join(symbols)
        new_word_str = pattern.sub(pair[0] + pair[1], word_str)
        new_vocab[tuple(new_word_str.split())] = freq
    return new_vocab

def train_bpe(corpus, num_merges):
    vocab = get_vocab(corpus)
    merges = []
    for i in range(num_merges):
        pairs = get_pair_counts(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)
        merges.append(best_pair)
        print(f"Merge {i+1}: {best_pair} (count={pairs[best_pair]})")
    return vocab, merges

corpus = {"lower": 3, "lowest": 2, "newer": 4, "wider": 1}
final_vocab, merges = train_bpe(corpus, num_merges=6)
print("\nFinal tokenization:", final_vocab)
# Compare this output against the hand-worked §3.1.2 example — the first two merges should match exactly.
```

### 3.1.7 Code: comparing against a production tokenizer (`tiktoken`)
```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4-family tokenizer

samples = ["unhappiness", "strawberry", "GenAI engineering", "12345 + 67890"]
for s in samples:
    tokens = enc.encode(s)
    print(f"{s!r} -> {len(tokens)} tokens: {[enc.decode([t]) for t in tokens]}")

# Try this with a non-English sentence and compare token count to an equivalent English
# sentence of similar meaning — you'll directly observe the cost asymmetry from §3.1.5.
```

### 3.1.8 Common pitfalls
- **Assuming 1 token ≈ 1 word.** English averages roughly 0.75 words/token, but this varies enormously by language and content type (code, rare technical terms, and non-English text tokenize far less efficiently). Never hardcode this ratio for cost/context estimates — always tokenize the actual text.
- **Forgetting tokenizer ≠ model-agnostic.** Every model family (GPT, LLaMA, Mistral, Claude) has its own tokenizer/vocabulary. Token counts, and even the token *boundaries themselves*, differ across models for identical text — code written against one model's token-counting logic will silently misestimate for another.
- **Mismatched chat template.** Covered in §3.1.4 — this produces subtly degraded outputs that look like a capability problem but are a formatting bug.

### 3.1.9 Interview Q&A
**Q: Why subword tokenization over word-level or char-level?**
A: Word-level vocabularies are enormous and still fail on unseen words (OOV); char-level avoids OOV but produces long sequences and forces the model to relearn spelling/morphology from scratch, wasting context and capacity. Subword tokenization (BPE/WordPiece/SentencePiece) keeps common words whole and decomposes rare/complex words into meaningful, frequently-reused pieces — balancing vocabulary size against sequence length while guaranteeing every input is representable.

**Q: How does BPE build its vocabulary?**
A: Start from individual characters (or bytes); repeatedly find the most frequent adjacent symbol pair across the training corpus and merge it into a new symbol; record each merge as a rule, in order; repeat until reaching a target vocabulary size. New text is tokenized by applying the same merge rules in the same order.

**Q: Why do LLMs struggle with counting letters or doing precise arithmetic on digits?**
A: Both trace back to tokenization, not reasoning capacity — the model operates on whatever subword/digit-grouping tokens BPE happened to produce, not on individual characters or digits with consistent positional meaning. Letter-counting requires the model to have implicitly inferred spelling from training statistics rather than reading characters directly; inconsistent digit-grouping across different numbers disrupts the place-value structure a human reader relies on.

**Q: What's the difference between token count and word count, and why does it matter?**
A: Token count depends on the specific tokenizer's vocabulary and merge rules, and varies by language, content type (code vs. prose), and even the specific model family — it is not a fixed ratio to word count. This matters directly for context-window budgeting and API cost estimation, both of which are billed/limited in tokens, not words.

### 3.1.10 Hands-on project
**Build**: BPE tokenizer trained from scratch on a real corpus (e.g., a public-domain book).
**Checklist**: (1) Train to a vocab size of ~1,000 merges and inspect the learned merges — do they look like sensible morphemes? (2) Tokenize the same paragraph with your tokenizer vs. `tiktoken` vs. a SentencePiece-based tokenizer (e.g., LLaMA's) — compare token counts and where the boundaries differ. (3) Feed your tokenizer a string in a language very different from your training corpus (or an emoji-heavy string) and observe degraded (over-fragmented) tokenization — this is the OOV-adjacent "unfamiliar domain" effect in action. (4) Implement byte-level fallback and confirm literally any input string round-trips (encode→decode) losslessly.

---

## MODULE 3.2 — Embeddings & Vector Representations

### 3.2.1 From one-hot to dense
A one-hot vector for a 50,000-word vocabulary is 50,000-dimensional, almost entirely zeros, and carries **zero** semantic information — the vector for "cat" is exactly as different from "dog" as it is from "bicycle" (all pairwise distances are identical). An **embedding** replaces this with a dense, low-dimensional (e.g., 768, 4096) vector, *learned* such that semantically related tokens end up geometrically close.

### 3.2.2 The embedding matrix — mechanics
```
E ∈ ℝ^(vocab_size × d)      one row per token, d = embedding dimension

Looking up token id 4521:  embedding = E[4521]      (just a row index — no matrix multiply needed)
```
In practice this row-lookup is implemented as a matrix multiply against a one-hot vector for autodiff convenience (`nn.Embedding` in PyTorch), but conceptually it's simply indexing. `E` is a learned parameter matrix, updated by backprop exactly like every weight matrix in Phase 1 — gradients flow back through whichever rows were used in a given batch.

### 3.2.3 word2vec — full derivation (historically important, still the cleanest way to build the intuition)
**Skip-gram objective**: given a center word, predict its surrounding context words within a window. For center word `c` and context word `o`:
```
P(o|c) = exp(u_oᵀv_c) / Σ_w exp(u_wᵀv_c)         (softmax over the ENTIRE vocabulary — expensive)
```
`v_c` = "center" embedding of word c, `u_o` = "context" embedding of word o (two separate embedding matrices are learned; typically only `v` is kept afterward as "the" word embedding).

**The problem**: that softmax denominator sums over the whole vocabulary (hundreds of thousands of terms) — computing it for every training example is prohibitively slow.

**Negative sampling (the practical fix)**: instead of a full softmax, turn this into `k+1` independent binary classification problems — "is `o` really a context word of `c`" (positive) vs. `k` randomly sampled words that are *not* actual context words (negatives):
```
L = log σ(u_oᵀv_c) + Σᵢ₌₁ᵏ log σ(-u_{negᵢ}ᵀv_c)
```
This turns an expensive `|V|`-way softmax into `k+1` cheap sigmoid evaluations (`k` is typically 5-20) — the optimization that made word2vec practical to train on billion-word corpora, and the direct conceptual ancestor of contrastive learning objectives used in modern embedding models (including CLIP, Phase 9).

### 3.2.4 Geometric properties of embedding spaces
Because embeddings are trained so that co-occurring words end up nearby, the resulting space captures relationships as *directions*, not just proximity — the famous illustrative example:
```
embedding("king") - embedding("man") + embedding("woman") ≈ embedding("queen")
```
This works because the *difference vector* `king - man` ends up encoding something like "royalty," roughly consistently across many such pairs (`king-man ≈ queen-woman ≈ actor-actress`), a property that emerges purely from co-occurrence statistics with no explicit supervision about "gender" or "royalty." **Cosine similarity** (not Euclidean distance) is the standard way to measure closeness:
```
cos_sim(a, b) = (a·b) / (‖a‖ ‖b‖)
```
Using the *angle* between vectors rather than raw distance matters because embedding magnitude often correlates with word frequency, not semantic content — cosine similarity normalizes this away, comparing direction only. This exact operation — cosine similarity between two embeddings — is precisely what powers semantic search and RAG retrieval in Phase 6.

### 3.2.5 Static vs. contextual embeddings — the critical distinction going into Transformers
word2vec/GloVe embeddings are **static**: the word "bank" has exactly one embedding vector, regardless of whether it appears in "river bank" or "savings bank" — the training process averages over all uses. This is a real limitation: polysemy (multiple meanings per word) isn't captured.

**Transformer-based embeddings are contextual**: the vector representing "bank" at a specific position in a specific sentence is computed *through self-attention*, incorporating information from the surrounding words at that specific occurrence — so "bank" in "river bank" and "bank" in "savings bank" end up as genuinely different vectors. This is arguably the single biggest representational leap from pre-Transformer to Transformer-based NLP, and it's a direct consequence of the self-attention mechanism you're about to formalize in Module 3.4 — each token's representation is recomputed as a function of its context at every layer, not looked up once and left static.

### 3.2.6 Code: train skip-gram word2vec from scratch, visualize
```python
import torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
import random

text = "the quick brown fox jumps over the lazy dog the dog barks at the fox".split()
vocab = sorted(set(text))
stoi = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

def make_pairs(text, window=2):
    pairs = []
    for i, center in enumerate(text):
        for j in range(max(0, i-window), min(len(text), i+window+1)):
            if i != j:
                pairs.append((stoi[center], stoi[text[j]]))
    return pairs

pairs = make_pairs(text)

class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, embed_dim=16):
        super().__init__()
        self.v_embed = nn.Embedding(vocab_size, embed_dim)   # center embeddings
        self.u_embed = nn.Embedding(vocab_size, embed_dim)   # context embeddings

    def forward(self, center, context, negatives):
        v_c = self.v_embed(center)                     # [batch, dim]
        u_o = self.u_embed(context)                     # [batch, dim]
        u_neg = self.u_embed(negatives)                 # [batch, k, dim]
        pos_score = torch.sum(v_c * u_o, dim=-1)                      # [batch]
        neg_score = torch.bmm(u_neg, v_c.unsqueeze(-1)).squeeze(-1)   # [batch, k]
        pos_loss = F.logsigmoid(pos_score)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=-1)
        return -(pos_loss + neg_loss).mean()             # negative sampling loss from §3.2.3

model = SkipGramNS(V, embed_dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    random.shuffle(pairs)
    centers = torch.tensor([p[0] for p in pairs])
    contexts = torch.tensor([p[1] for p in pairs])
    negatives = torch.randint(0, V, (len(pairs), 5))       # k=5 negative samples
    loss = model(centers, contexts, negatives)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

# Inspect nearest neighbors by cosine similarity
def nearest(word, k=3):
    idx = stoi[word]
    vecs = model.v_embed.weight.detach()
    sims = F.cosine_similarity(vecs[idx].unsqueeze(0), vecs)
    top = sims.argsort(descending=True)[1:k+1]
    return [(vocab[i], sims[i].item()) for i in top]

print(nearest("fox"))
print(nearest("dog"))
```

### 3.2.7 Interview Q&A
**Q: How is an embedding different from a one-hot vector?**
A: A one-hot vector is high-dimensional, sparse, and carries no semantic information — all pairs of distinct words are equally "different." An embedding is a dense, low-dimensional, *learned* vector where geometric proximity (typically measured via cosine similarity) reflects semantic/distributional similarity, learned end-to-end from data.

**Q: Why does negative sampling make word2vec practical?**
A: The original skip-gram objective requires a softmax over the entire vocabulary for every training example, which is computationally prohibitive at scale. Negative sampling reframes the problem as k+1 binary classifications (one true context word vs. k random negative samples), replacing one expensive |V|-way softmax with k+1 cheap sigmoid evaluations per example.

**Q: What's the practical difference between static and contextual embeddings?**
A: Static embeddings (word2vec/GloVe) assign exactly one vector per word regardless of context, failing to capture polysemy. Contextual embeddings (produced by self-attention in Transformers) compute a token's representation as a function of its surrounding context at that specific occurrence, so the same word gets different representations in different contexts — a direct consequence of self-attention's mechanism, not a separate technique bolted on.

**Q: Why cosine similarity instead of Euclidean distance for comparing embeddings?**
A: Embedding vector magnitude often correlates with factors like word/document frequency rather than semantic content. Cosine similarity measures only the angle between vectors, normalizing away magnitude differences and comparing direction (semantic content) alone — which is why it's the standard metric for semantic search and RAG retrieval.

### 3.2.8 Hands-on project
**Build**: Train the skip-gram model above on a larger corpus (a few thousand sentences), then visualize.
**Checklist**: (1) Train and confirm `nearest()` returns semantically sensible neighbors for several test words. (2) Reduce embeddings to 2D with t-SNE or UMAP and plot — look for clusters of related words. (3) Attempt an analogy (`king - man + woman`) on your trained embeddings and check the nearest neighbor to the result — small corpora often won't reproduce this cleanly; note *why* (insufficient co-occurrence statistics) as part of the exercise. (4) Compare your embeddings' nearest-neighbors against a pretrained embedding model (e.g., `sentence-transformers`) on the same test words — the pretrained model's neighbors should be noticeably more coherent, demonstrating the value of large-scale pretraining data.

---

## MODULE 3.3 — Positional Encoding

### 3.3.1 [Addition — this deserves its own module] Why attention needs this at all
Self-attention (Module 3.4) computes a weighted sum over all other tokens based purely on content similarity (`QKᵀ`) — nothing in that formula references *where* a token sits in the sequence. Formally, self-attention is **permutation-invariant**: shuffle the input tokens, and (aside from which token attends to which) the *set* of outputs is identical — the model has no innate way to distinguish "the cat sat on the mat" from "mat the on sat cat the." Positional information must be injected explicitly, or the entire notion of word order — which RNNs got "for free" via their sequential structure — is lost. This single fact is why positional encoding exists as its own major design decision in every Transformer.

### 3.3.2 Sinusoidal positional encoding (the original 2017 approach)
Add (not concatenate) a position-dependent vector to each token embedding before the first layer:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
`pos` = position in the sequence, `i` = dimension index, `d` = embedding dimension. Each dimension pair oscillates at a different frequency (low dimensions oscillate fast, high dimensions slowly) — like a binary-clock encoding of position, but smooth and continuous. **Why sin/cos specifically**: this construction gives a useful mathematical property — `PE(pos+k)` can be expressed as a *linear function* of `PE(pos)` (via trigonometric angle-addition identities), meaning relative positional relationships are, in principle, learnable by a linear operation on the encodings — and it extrapolates smoothly to sequence lengths not seen during training (no learned parameters to run out of).

### 3.3.3 Learned positional embeddings
Simpler alternative: just add a learned embedding matrix indexed by position, exactly like the token embedding matrix (§3.2.2), trained by backprop like everything else. Used in GPT-2/early GPT-3. **Weakness**: since it's a lookup table with one row per position, it can *only* represent positions up to whatever maximum length was seen during training — it cannot extrapolate to longer sequences at all (unlike sinusoidal encoding, which is a formula, not a table).

### 3.3.4 [Addition — this is what modern LLMs actually use] RoPE — Rotary Positional Embeddings
Used in LLaMA, Mistral, Qwen, Gemma, and most current open-weight LLMs. Neither sinusoidal nor learned encoding is what powers the models you'll fine-tune in Phase 5 — RoPE (Su et al., 2021) is, and it works differently enough to deserve full treatment.

**Core idea**: instead of *adding* a positional vector to the embedding, **rotate** the query and key vectors by an angle proportional to their position, applied in 2D pairs of dimensions:
```
For a 2D pair (x₁, x₂) of a Q or K vector at position m, rotate by angle m·θ:

[x₁'] = [cos(mθ)  -sin(mθ)] [x₁]
[x₂']   [sin(mθ)   cos(mθ)] [x₂]
```
This rotation is applied independently to every consecutive pair of dimensions in the head, each pair using a different base frequency `θᵢ = 10000^(-2i/d)` — deliberately similar in spirit to sinusoidal encoding's frequency schedule, but applied as a *rotation of Q/K* rather than an *addition to the embedding*.

**Why this specific mechanism — the elegant part**: because rotation matrices compose (`R_m R_n = R_{m+n}`... more precisely, `R_mᵀR_n = R_{n-m}`), the dot product between a rotated query at position `m` and a rotated key at position `n` depends *only on their relative distance* `(n-m)`, not on their absolute positions:
```
(R_m q)ᵀ(R_n k) = qᵀ R_mᵀ R_n k = qᵀ R_{n-m} k
```
This gives the model **relative position awareness baked directly into the attention score geometry**, with no extra parameters and — critically — better extrapolation behavior to sequence lengths beyond what was seen in training than either sinusoidal or learned absolute encodings (though RoPE-based models still typically need explicit techniques, like positional interpolation, to extrapolate well far beyond training length — it's better, not unlimited).

### 3.3.5 [Addition] ALiBi — a simpler alternative worth knowing
ALiBi (Attention with Linear Biases, Press et al. 2021) skips rotation entirely: it adds a fixed, non-learned linear penalty directly to the attention scores, proportional to the distance between positions:
```
score(q_m, k_n) = qᵀk - λ·|m-n|          (λ = a fixed, head-specific slope, not learned)
```
No positional embedding is added to the input at all — position only ever enters through this direct penalty on attention scores, which increases with distance (encoding "closer tokens matter more" as a direct, structural bias, rather than something the model must learn from sinusoidal patterns). Notably good extrapolation to longer sequences than trained on, with an even simpler mechanism than RoPE, though RoPE remains more widely adopted in current frontier open-weight models.

### 3.3.6 Code: visualize and compare positional encodings
```python
import numpy as np
import matplotlib.pyplot as plt

def sinusoidal_pe(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

def rope_rotation_angles(seq_len, dim, base=10000):
    """Returns the rotation angle applied to each (position, dim-pair)."""
    theta = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
    positions = np.arange(seq_len)
    angles = np.outer(positions, theta)     # [seq_len, dim/2]
    return angles

pe = sinusoidal_pe(seq_len=50, d_model=64)
plt.figure(figsize=(10, 4))
plt.imshow(pe.T, cmap='RdBu', aspect='auto')
plt.xlabel('Position'); plt.ylabel('Embedding dimension')
plt.title('Sinusoidal positional encoding — note the varying frequency per dimension')
plt.colorbar()
plt.savefig('sinusoidal_pe.png')

# Verify RoPE's relative-position property numerically:
angles = rope_rotation_angles(seq_len=20, dim=8)
def rotate(vec, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([vec[0]*c - vec[1]*s, vec[0]*s + vec[1]*c])

q = np.array([1.0, 0.5])
k = np.array([0.3, 0.8])
for (m, n) in [(2, 5), (10, 13), (0, 3)]:   # different absolute positions, SAME relative distance (3)
    q_rot = rotate(q, angles[m, 0])
    k_rot = rotate(k, angles[n, 0])
    print(f"pos ({m},{n}), rel dist {n-m}: dot product = {np.dot(q_rot, k_rot):.4f}")
# All three should print nearly identical dot products — confirming the score depends
# only on relative distance (3), not on the absolute positions, exactly as §3.3.4 proves.
```

### 3.3.7 Interview Q&A
**Q: Why do Transformers need positional encoding at all?**
A: Self-attention computes outputs as a weighted combination of values based purely on content similarity between queries and keys — it's mathematically permutation-invariant, with no inherent sense of sequence order. Positional information must be injected explicitly (added to embeddings, or built into the attention computation itself via RoPE/ALiBi), or the model cannot distinguish different orderings of the same tokens.

**Q: Compare RoPE vs learned vs sinusoidal positional embeddings.**
A: Learned absolute embeddings are simplest but cannot generalize beyond the maximum sequence length seen during training (a fixed-size lookup table). Sinusoidal encoding is a formula (not a table), extrapolates more gracefully, and is added directly to token embeddings once at the input. RoPE instead rotates query/key vectors by a position-dependent angle *inside* the attention computation at every layer, giving attention scores that depend mathematically only on relative position between tokens rather than absolute position — generally better extrapolation and the current standard in most open-weight LLMs (LLaMA, Mistral, Qwen).

**Q: What does it mean for RoPE to encode "relative" rather than "absolute" position?**
A: Because rotation matrices compose additively in their angles, the dot product between a rotated query at position m and a rotated key at position n mathematically reduces to a function of only `(n-m)` — the relative distance — not the specific values of m and n individually. Two token pairs at the same relative distance but different absolute positions in the sequence produce essentially the same attention-score contribution from position.

**Q: Why might ALiBi or RoPE extrapolate to longer sequences better than learned absolute position embeddings?**
A: Learned absolute embeddings are a lookup table with one entry per training-time position — there's simply no representation defined for positions beyond that table's size. RoPE and ALiBi are both defined by a formula (rotation angle, or linear distance penalty) that's well-defined for any position value, including ones far beyond what was seen in training — though in practice, extrapolation quality still degrades at very long distances and often needs additional techniques (e.g., positional interpolation/scaling) to work well in production.

---

## MODULE 3.4 — Self-Attention, Formalized

### 3.4.1 Bridge from Phase 2
Recall Module 2.9: cross-attention has `Q` from one sequence and `K`/`V` from another; self-attention sets `Q = K = V` = (different learned projections of) the *same* sequence. Nothing else about the three-step recipe (score → normalize → combine) changes.

### 3.4.2 The full self-attention equation, term by term
```
Q = XW_Q     K = XW_K     V = XW_V         X ∈ ℝ^(n×d): n tokens, d-dim embeddings
                                            W_Q, W_K, W_V ∈ ℝ^(d×d_k): learned projections

Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```
- `X` — the input sequence (token embeddings + positional info from Module 3.3), shape `[n, d]`.
- `W_Q, W_K, W_V` — three separately learned weight matrices. Every token gets projected three different ways: as "what am I looking for" (query), "what do I offer for matching" (key), and "what do I actually contain" (value) — three distinct *roles* the same token representation plays simultaneously.
- `QKᵀ` — an `[n, n]` matrix of raw compatibility scores between every pair of tokens (`n²` scores — this is Module 2.10's `O(n²)` cost, made concrete).
- `/√d_k` — the scaling fix from Module 2.4, preventing softmax saturation as `d_k` grows.
- `softmax(...)` — row-wise normalization, so each token's outgoing attention weights (across all tokens it attends to) sum to 1.
- `... V` — the weighted combination of values, producing one output vector per input token, each now infused with context from every other (allowed — see §3.4.4) token.

### 3.4.3 Worked numerical example (self-attention, causal masking included)
Three tokens ("I", "love", "AI"), toy 2-dimensional Q/K/V vectors (imagine these are already the result of applying `W_Q, W_K, W_V` — skipping the projection arithmetic to keep this hand-traceable, exactly as Phase 2's worked example did):
```
Token 1 ("I"):    Q1=[1,0]  K1=[1,0]  V1=[1,2]
Token 2 ("love"): Q2=[0,1]  K2=[0,1]  V2=[3,1]
Token 3 ("AI"):   Q3=[1,1]  K3=[1,1]  V3=[0,4]
```
**Output for token 3** ("AI"), causally allowed to attend to tokens 1, 2, and 3 (itself):
```
Scores:  Q3·K1 = 1·1+1·0 = 1     Q3·K2 = 1·0+1·1 = 1     Q3·K3 = 1·1+1·1 = 2
Scaled (÷√2≈1.414):  0.707, 0.707, 1.414
Softmax: exp(0.707)=2.028, exp(0.707)=2.028, exp(1.414)=4.113 → sum=8.169
         weights = 0.248, 0.248, 0.503
Output = 0.248·V1 + 0.248·V2 + 0.503·V3
       = 0.248·[1,2] + 0.248·[3,1] + 0.503·[0,4]
       = [0.248,0.496] + [0.744,0.248] + [0,2.012]
       = [0.992, 2.756]
```
**Output for token 1** ("I"), causally allowed to attend to *only itself* (nothing comes before position 1):
```
Score: Q1·K1 = 1     Scaled: 0.707     Softmax over a single value: weight = 1.0
Output = 1.0 · V1 = [1, 2]                (trivially just its own value — nothing else was visible)
```
This second calculation makes causal masking's effect concrete: early tokens genuinely have less context available, by construction — not a training artifact, a structural fact about autoregressive generation.

### 3.4.4 Causal masking — the mechanism
For a decoder-only model (GPT-style, autoregressive generation), token `i` must **never** attend to tokens `j > i` — otherwise the model could "cheat" during training by looking at the very token it's supposed to predict. This is enforced by masking the score matrix *before* the softmax:
```
scores_masked[i,j] = scores[i,j]     if j ≤ i
scores_masked[i,j] = -∞              if j > i        (so softmax(-∞) → 0, contributing nothing)
```
This produces the characteristic **lower-triangular** attention pattern — visualize the `[n,n]` score matrix, and every entry above the diagonal is zeroed out after softmax. This is the direct generalization of Module 2.11's padding-mask mechanic from Phase 2 — same "set to `-∞` before softmax, never after" principle, now applied to enforce autoregressive order rather than to ignore padding.

### 3.4.5 Code: self-attention from scratch (matrix form, batched)
```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, causal=True):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_k, bias=False)
        self.d_k = d_k
        self.causal = causal

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        Q = self.W_Q(x)    # [batch, seq_len, d_k]
        K = self.W_K(x)
        V = self.W_V(x)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)   # [batch, seq_len, seq_len]

        if self.causal:
            seq_len = x.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))    # §3.4.4's masking, vectorized

        weights = torch.softmax(scores, dim=-1)                 # row-wise normalize
        output = weights @ V                                    # [batch, seq_len, d_k]
        return output, weights

# Sanity-check against §3.4.3's hand-worked example
x = torch.eye(3).unsqueeze(0)  # trivial stand-in input; in practice x carries real embeddings
attn = SelfAttention(d_model=3, d_k=2, causal=True)
out, weights = attn(x)
print("Attention weights (causal, lower-triangular):\n", weights[0])
# Row 0 should have weight 1.0 in column 0 and 0 elsewhere — matching §3.4.3's token-1 result.
```

### 3.4.6 Complexity — why this specific formula shapes everything downstream
`QKᵀ` is an `[n,d_k] × [d_k,n]` matrix multiply → `O(n²·d_k)` time, and the resulting `[n,n]` score matrix itself takes `O(n²)` **memory**. Doubling sequence length quadruples both attention compute and the score matrix's memory footprint. This is precisely why: context windows were historically capped; FlashAttention (Module 3.11) exists to make this computation IO-efficient without changing the math; and MQA/GQA (Module 3.6) exist to reduce the *other* major cost driver at inference time — not attention compute itself, but the memory needed to cache `K`/`V` across a long generation.

### 3.4.7 Interview Q&A
**Q: Walk through self-attention math step by step.**
A: Project the input `X` into three learned spaces — queries `Q=XW_Q`, keys `K=XW_K`, values `V=XW_V`. Compute raw compatibility scores between every pair of tokens via `QKᵀ`, scale by `1/√d_k` to prevent softmax saturation, apply softmax row-wise so each token's attention weights over all other tokens sum to 1, then compute the output as that weighted combination of value vectors: `softmax(QKᵀ/√d_k)V`.

**Q: Why divide by √d_k?**
A: Dot products between random vectors grow in variance proportional to dimensionality. Without scaling, as `d_k` grows, raw scores become large in magnitude, pushing softmax into a saturated regime where one score dominates and gradients to the rest vanish. Dividing by `√d_k` keeps score variance roughly constant regardless of dimension.

**Q: Explain causal masking.**
A: Before the softmax, set every score where a token would attend to a *future* position to negative infinity, so after softmax those positions receive exactly zero weight. This enforces that token `i`'s representation can only depend on tokens at positions `≤ i`, which is required for autoregressive next-token-prediction training to be valid — otherwise the model could trivially "cheat" by attending directly to the token it's being trained to predict.

**Q: What's the time/space complexity of self-attention with respect to sequence length, and why does it matter?**
A: Both the compute (`QKᵀ`) and the memory footprint of the resulting score matrix scale as `O(n²)` in sequence length `n`. This quadratic cost is the direct reason long-context models require specialized engineering — memory-efficient attention implementations (FlashAttention) to make the `O(n²)` computation fast without materializing the full score matrix in slow memory, and architectural changes (sparse/local attention, GQA/MQA) to control the other cost drivers at both training and inference time.

---

## MODULE 3.5 — Multi-Head Attention

### 3.5.1 Why one attention computation isn't enough
A single self-attention operation learns *one* notion of "relevance" between tokens. But language has many simultaneous relationship types a sentence might need — syntactic (subject-verb agreement), coreference (which noun a pronoun refers to), semantic (topical relatedness) — and a single set of `Q/K/V` projections, optimized jointly for all of these, tends to average them together into a compromise. **Multi-head attention** runs several independent attention computations in parallel, each with its *own* learned `W_Q, W_K, W_V`, so different heads are free to specialize in different types of relationships.

### 3.5.2 The math — split, attend, concatenate, project
```
head_i = Attention(XW_Q^i, XW_K^i, XW_V^i)          for i = 1...h  (h heads)
MultiHead(X) = Concat(head_1, ..., head_h) W_O
```
Critically, the *total* parameter count is kept roughly comparable to a single large-dimension attention head: if the model dimension is `d_model=512` and there are `h=8` heads, each head typically operates in a *reduced* dimension `d_k = d_model/h = 64`, so the combined compute across all 8 heads is comparable to one full-dimension attention computation — you're not paying 8x the cost, you're *splitting* the same budget across 8 parallel, differently-specialized subspaces.

### 3.5.3 What different heads empirically learn
Interpretability studies on trained Transformers (probing attention patterns directly) have found heads that specialize surprisingly cleanly — some heads attend almost entirely to the immediately previous token (positional/local patterns), some attend to syntactically related words (e.g., a verb attending to its subject), and some attend to rare "outlier" heads that seem to perform more global aggregation. No individual head's specialization is guaranteed or designed — it emerges purely from gradient descent optimizing the training objective, but it emerges consistently enough across trained models that this is now a standard interpretability finding, not a one-off curiosity.

### 3.5.4 Code: multi-head attention from scratch
```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, causal=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        # Project, then reshape into [batch, num_heads, seq_len, d_k]
        Q = self.W_Q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)   # [batch, heads, seq_len, seq_len]

        if self.causal:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        attn_out = weights @ V                                    # [batch, heads, seq_len, d_k]

        # Concatenate heads back together
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.W_O(attn_out), weights

mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)   # batch=2, seq_len=10
out, weights = mha(x)
print(out.shape)      # torch.Size([2, 10, 512])
print(weights.shape)  # torch.Size([2, 8, 10, 10])  <- one attention map per head
```

### 3.5.5 Interview Q&A
**Q: Why multi-head instead of one large attention head?**
A: A single attention computation learns one notion of token relevance, forcing many different relationship types (syntactic, semantic, positional) to share one representation. Multiple heads, each with independently learned projections, let the model specialize different heads for different relationship types in parallel, typically at comparable total compute to one large head by splitting the model dimension across heads rather than multiplying it.

**Q: Does splitting into more heads always help?**
A: No — each head operates in a smaller subspace (`d_model/h` dimensions), so beyond a certain point, more heads means each individual head has too little representational capacity to be useful, and returns diminish or reverse. The head count is a hyperparameter tuned empirically per model scale, not a "more is always better" knob.

**Q: What does `W_O` do in multi-head attention, and why is it needed?**
A: After concatenating all heads' outputs back into a single vector, `W_O` is a learned linear projection that mixes information across heads before passing it to the next layer — without it, each head's output would remain in its own separate "slot" of the concatenated vector with no way for downstream layers to combine information across heads.

---

## MODULE 3.6 — [Addition] Efficient Attention Variants: MQA and GQA

### 3.6.1 Why this module exists — the problem vanilla multi-head attention causes at inference
This isn't covered in the original "Attention Is All You Need" paper, but essentially every production LLM you'll deploy (Phase 5, Phase 10) uses one of these variants — skipping this module leaves a real gap between "I understand Transformers" and "I understand why LLaMA-2-70B's config file looks the way it does." The issue is **KV-cache memory** (fully explained in Module 3.9) — during autoregressive generation, the keys and values for every previous token must be kept in memory for every attention head, and with standard multi-head attention, that means `num_heads` separate full-size K/V caches. For large models with many heads and long contexts, this becomes the dominant memory cost of inference — often exceeding the memory used by the model's weights themselves.

### 3.6.2 Multi-Query Attention (MQA)
Proposed by Shazeer, 2019. Radical simplification: keep separate learned `Q` projections per head (preserving the specialization benefit from Module 3.5), but make **all heads share a single K and V projection**:
```
Standard MHA:  h separate (Q,K,V) triples          → h separate KV caches
MQA:           h separate Q, but ONE shared (K,V)   → 1 KV cache, shared across all heads
```
This cuts KV-cache memory by a factor of `h` (e.g., 32x for a 32-head model) — but forces every head to attend using the *same* keys/values, which measurably hurts quality/expressiveness compared to full MHA, especially at larger model scales.

### 3.6.3 Grouped-Query Attention (GQA)
Proposed by Ainslie et al., 2023 (used in LLaMA-2-70B, LLaMA-3, Mistral, and most current production LLMs) — the middle ground that won out in practice. Instead of *all* heads sharing one K/V (MQA) or *each* head having its own (MHA), heads are split into **groups**, and each group shares one K/V pair:
```
Example: 32 query heads, grouped into 8 groups (4 query heads per group)
→ 8 shared (K,V) pairs instead of 32 (MHA) or 1 (MQA)
→ KV-cache memory reduced 4x vs. MHA, while retaining far more of MHA's quality than MQA does
```
GQA is explicitly a tunable interpolation: set the group count to `h` and you recover standard MHA exactly; set it to 1 and you recover MQA exactly. Model designers choose a group count balancing the KV-cache memory budget against measured quality degradation — empirically, GQA with a moderate number of groups captures most of MHA's quality at a fraction of MQA's problems.

### 3.6.4 Which real models use what
| Model | Attention variant |
|---|---|
| Original Transformer (2017), GPT-2/3, BERT | Standard Multi-Head Attention (MHA) |
| PaLM, early efficiency-focused models | Multi-Query Attention (MQA) |
| LLaMA-2-70B, LLaMA-3 (all sizes), Mistral-7B, Qwen2.5, Gemma 2 | Grouped-Query Attention (GQA) |

**Practical takeaway for Phase 5/10**: when you inspect an open model's config (`num_attention_heads` vs. `num_key_value_heads` in a Hugging Face config file), a mismatch between these two numbers tells you it's using GQA — `num_key_value_heads < num_attention_heads` means fewer K/V groups than query heads, exactly as described above. This is directly relevant when you calculate expected memory usage for serving a model in Phase 10.

### 3.6.5 Interview Q&A
**Q: Why did production LLMs move away from standard multi-head attention?**
A: Standard MHA requires a separate KV-cache per attention head during autoregressive generation. For large models with many heads and long context windows, this KV-cache memory becomes a dominant (sometimes the dominant) cost of serving the model — often larger than the model weights themselves — directly limiting achievable batch size and context length at inference time.

**Q: MQA vs GQA — what's the tradeoff, and why did GQA become the standard rather than MQA?**
A: MQA shares one K/V pair across *all* query heads, maximizing KV-cache savings (divide by the full head count) but measurably hurting model quality since every head is forced to attend using identical keys/values. GQA shares K/V within smaller groups of heads rather than globally, capturing most of MQA's memory savings while retaining significantly more of standard MHA's representational quality — an empirically better point on the memory/quality tradeoff curve, which is why it's the current default in most production open-weight models.

**Q: How would you tell from a model's config whether it uses MHA, MQA, or GQA?**
A: Compare `num_attention_heads` to `num_key_value_heads` (naming varies by codebase, but this is the standard HF convention). Equal values → standard MHA. `num_key_value_heads = 1` → MQA. Any value strictly between 1 and `num_attention_heads` → GQA, with the ratio telling you the group size.

---

## MODULE 3.7 — The Full Transformer Block

### 3.7.1 The feed-forward sublayer (FFN)
After attention mixes information *across* token positions, each position independently passes through a position-wise feed-forward network — the exact MLP architecture from Phase 1, applied identically (same weights) to every token position:
```
FFN(x) = activation(xW₁ + b₁)W₂ + b₂          typically expands then contracts:
                                                d_model → 4×d_model → d_model
```
**Why the 4x expansion**: gives the sublayer more capacity to compute a richer nonlinear transformation of each token's (now context-infused) representation, before compressing back to `d_model` for the residual connection to work dimensionally. **Activation choice**: original Transformer used ReLU; GPT/BERT-family models widely adopted GELU (Module 1.1.3 — smoother gradients help at scale); LLaMA/PaLM/Mistral use **SwiGLU** — a *gated* variant:
```
SwiGLU(x) = (Swish(xW₁) ⊙ xV) W₂        (⊙ = elementwise multiply; Swish/SiLU from Module 1.1.3)
```
The extra gating multiplication (an additional learned projection `V` elementwise-multiplying the activated branch) empirically outperforms plain GELU/ReLU FFNs at matched compute budgets — this is now the standard choice in most current open-weight LLMs, worth recognizing when you read a model's architecture config in Phase 5/11.

### 3.7.2 Residual connections — direct reuse from Phase 1
Exactly the ResNet mechanism from Module 1.3.6/1.2.7: each sublayer's output is *added back* to its input (`output = x + Sublayer(x)`), giving gradients a direct additive path through the network, bypassing the multiplicative vanishing-gradient chain — essential for training the 32, 80, or more stacked layers of a real LLM.

### 3.7.3 [Addition/expanded] Normalization: LayerNorm vs. RMSNorm, and Pre-LN vs. Post-LN
**LayerNorm** (used in the original Transformer, GPT-2/3, BERT): normalizes across the feature dimension for each token independently (unlike BatchNorm, Module 1.3.7, which normalizes across the batch — a Transformer needs per-token normalization since sequence lengths vary and batching is along a different axis):
```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β        μ, σ² computed across the feature dimension
```
**RMSNorm** (Zhang & Sennrich, 2019; used in LLaMA, T5, Mistral, most current models): a simplification that skips mean-centering entirely, only rescaling by the root-mean-square:
```
RMSNorm(x) = γ · x / √(mean(x²) + ε)              no mean subtraction, no bias term β
```
Empirically performs comparably to LayerNorm while being computationally cheaper (fewer operations, no need to compute and subtract the mean) — the efficiency motivation for its adoption at scale.

**Pre-LN vs. Post-LN — a real, practically important design decision**:
```
Post-LN (original Transformer, 2017):   output = LayerNorm(x + Sublayer(x))
Pre-LN  (GPT-2 onward, LLaMA, most modern models): output = x + Sublayer(LayerNorm(x))
```
This ordering might look like a minor detail, but it has a major effect on training stability. In Post-LN, the residual stream itself gets normalized at every layer, which means the "clean" identity path that residual connections are supposed to provide (Module 1.2.7's whole justification) is disrupted — normalization sits *in* the gradient's shortcut path. Post-LN Transformers are empirically harder to train at depth, often requiring careful learning-rate warmup (Module 1.2.6) to avoid divergence in early training. Pre-LN normalizes only the *input* to each sublayer, leaving the residual stream itself untouched by normalization — a genuinely clean additive gradient highway all the way through the network — which makes deep Pre-LN Transformers substantially easier to train stably, at a small empirical cost in final model quality relative to a (harder-to-train) equivalent Post-LN model. This tradeoff — easier training vs. a small quality ceiling — is exactly why virtually every large modern LLM uses Pre-LN: reliably training a 70B+ parameter model at all matters more than a marginal quality gap.

### 3.7.4 Assembling the full block
```
Decoder block (GPT-style):
  x = x + MultiHeadSelfAttention(LayerNorm(x))     ← Pre-LN, causal masking (§3.4.4)
  x = x + FFN(LayerNorm(x))                         ← Pre-LN, position-wise
  (repeat this block N times — N=12 for GPT-2 small, N=96 for GPT-3, etc.)
```
Every single piece of this two-line block has now been derived from first principles somewhere in Phase 1, 2, or 3: matrix multiplication (0.3a), residual connections (1.2.7/1.3.6), gating intuition (1.4.4), the attention recipe (2.2), scaled dot-product attention (2.4/3.4.2), multi-head splitting (3.5.2), and normalization (3.7.3). Nothing here is "new machinery" — it's a specific, carefully-tuned assembly of parts you already understand.

### 3.7.5 Code: full Pre-LN Transformer decoder block
```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, causal=True)   # from §3.5.4
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(self.ln1(x))            # Pre-LN: normalize BEFORE the sublayer
        x = x + self.dropout(attn_out)                    # residual add — clean gradient highway
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)                      # residual add again
        return x

block = TransformerBlock(d_model=512, num_heads=8)
x = torch.randn(2, 20, 512)
out = block(x)
print(out.shape)   # torch.Size([2, 20, 512]) — same shape in and out, ready to stack N blocks
```

### 3.7.6 Interview Q&A
**Q: Why residual connections in a Transformer, specifically?**
A: Transformers stack many layers (dozens to over a hundred in large models); without residual connections, gradients would need to survive the same multiplicative chain-rule problem covered in Module 1.2.7, now across Transformer depth instead of RNN sequence length. Residual connections provide a direct additive gradient path, making very deep Transformer stacks trainable.

**Q: Pre-LN vs Post-LN — why does the ordering matter?**
A: Post-LN normalizes the residual stream itself after each sublayer's output is added, which disrupts the clean, unnormalized gradient shortcut that residual connections are meant to provide — making deep Post-LN Transformers harder to train stably, often requiring careful warmup schedules. Pre-LN normalizes only the sublayer's *input*, leaving the residual stream itself untouched, giving a genuinely clean additive path through the whole network and substantially easier, more stable training at depth — at a small cost in peak achievable quality versus a well-tuned Post-LN model, which is why nearly all large modern LLMs use Pre-LN.

**Q: Why do LLaMA-family models use RMSNorm instead of LayerNorm?**
A: RMSNorm skips LayerNorm's mean-centering step, normalizing only by the root-mean-square of the activations — computationally cheaper (fewer operations) while performing comparably to LayerNorm in practice, an efficiency win that matters at the scale these models are trained and served.

**Q: What does the feed-forward sublayer add that attention alone doesn't provide?**
A: Attention only mixes information *across* token positions via a weighted sum — it's a fundamentally linear recombination of value vectors (softmax weights aside) at each position. The FFN sublayer applies an independent nonlinear transformation to *each* position's (now context-mixed) representation, adding the actual representational/computational capacity — exactly the same role nonlinearity plays in an MLP (Module 1.1.5) — that attention's linear recombination doesn't provide on its own.

---

## MODULE 3.8 — Encoder-Only, Decoder-Only, and Encoder-Decoder Architectures

### 3.8.1 Encoder-only (BERT-style) — bidirectional, Masked Language Modeling
No causal mask — every token attends to *every* other token, both before and after it, in both directions. Can't be used for left-to-right autoregressive generation (there's no notion of "only see the past"), so it's trained differently: **Masked Language Modeling (MLM)** — randomly mask ~15% of input tokens, train the model to predict the masked tokens using full bidirectional context. **Best suited for**: understanding/classification tasks where the full input is available upfront — sentiment classification, named entity recognition, sentence similarity, extractive Q&A. **Examples**: BERT, RoBERTa.

### 3.8.2 Decoder-only (GPT-style) — causal, next-token prediction
Causal masking throughout (§3.4.4) — every token can only attend to itself and earlier tokens. Trained via straightforward **next-token prediction**: given tokens `1...i`, predict token `i+1`, for every position simultaneously in one forward pass (this is why training is efficient despite generation being sequential — during *training*, the entire sequence's loss is computed in parallel via the causal mask; only *generation/inference* is inherently sequential). **Best suited for**: open-ended generation — exactly what modern LLM chat/completion products need. **Examples**: the GPT family, LLaMA, Mistral, Claude's underlying architecture family, virtually every modern general-purpose LLM.

### 3.8.3 Encoder-Decoder (T5-style) — the original 2017 architecture, and where cross-attention lives
Two stacks: a bidirectional encoder (like §3.8.1) processes the full input, then a causal decoder (like §3.8.2) generates output — but the decoder has an *additional* attention sublayer beyond self-attention: **cross-attention**, where queries come from the decoder but keys/values come from the encoder's output. **This is Phase 2's exact mechanism, formalized as a Transformer sublayer** — the same `Q` from one sequence, `K`/`V` from another pattern you learned by hand in Module 2.6/2.7, now living inside a Transformer block instead of an RNN:
```
Decoder block (encoder-decoder variant):
  x = x + SelfAttention(LN(x), causal=True)                    ← attends to decoder's own prior tokens
  x = x + CrossAttention(LN(x), K=encoder_out, V=encoder_out)   ← attends to the FULL encoder output
  x = x + FFN(LN(x))
```
**Best suited for**: tasks with a clear "transform this input into that output" structure where the output isn't simply a continuation of the input — translation, summarization. **Examples**: the original Transformer paper's own task (translation), T5, BART.

### 3.8.4 Comparison table
| | Encoder-only | Decoder-only | Encoder-Decoder |
|---|---|---|---|
| Attention direction | Bidirectional | Causal (unidirectional) | Encoder: bidirectional; Decoder: causal + cross-attention |
| Pretraining objective | Masked Language Modeling | Next-token prediction | Varies (e.g., span corruption in T5) |
| Natural fit | Classification, understanding | Open-ended generation, chat | Input→output transformation tasks |
| Examples | BERT, RoBERTa | GPT-family, LLaMA, Mistral | T5, BART, original Transformer |
| Dominant today for GenAI? | No (superseded for generation) | **Yes — the standard for modern LLMs** | Niche (specific transformation tasks) |

**Why decoder-only won for general-purpose LLMs**: a single unified next-token-prediction objective scales cleanly to enormous, uncurated internet-scale text (no need to construct input/output pairs the way encoder-decoder training does), and — perhaps counterintuitively — a decoder-only model *can* still perform translation/summarization-style tasks by simply framing them as continuation ("Translate to French: [text] →"), just less architecturally specialized for it. This flexibility, combined with pretraining simplicity at scale, is a large part of why the field converged on decoder-only architectures for foundation models (Phase 4), even though encoder-decoder is architecturally "more correct" for pure transformation tasks.

### 3.8.5 Interview Q&A
**Q: Encoder-only vs decoder-only vs encoder-decoder — when to use which?**
A: Encoder-only (bidirectional, MLM-trained) suits understanding/classification tasks where the full input is available at once. Decoder-only (causal, next-token-trained) suits open-ended generation and is the dominant choice for general-purpose LLMs today, since next-token prediction scales cleanly to massive uncurated text and a single model can still be prompted to perform transformation-style tasks. Encoder-decoder suits tasks with a clear input→output transformation structure (translation, summarization) where cross-attention lets the decoder condition generation on the full, bidirectionally-processed input.

**Q: Where does cross-attention live in a Transformer, and how does it relate to Phase 2's material?**
A: Cross-attention is an attention sublayer, typically inserted between the causal self-attention and FFN sublayers of an encoder-decoder's decoder block, where queries come from the decoder's own sequence but keys/values come from the encoder's output. It's structurally identical to the Bahdanau/Luong attention mechanism from Phase 2 — one sequence querying another — just implemented as a Transformer sublayer (scaled dot-product form) instead of bolted onto an RNN.

**Q: Why can a decoder-only model perform tasks like translation without an explicit encoder?**
A: By framing the task as continuation — e.g., feeding "Translate to French: [source text] →" as a prompt and letting the causal model generate the continuation — a sufficiently capable decoder-only model learns, from broad pretraining exposure to such patterns in text, to perform the transformation as part of general next-token prediction, without needing an architecturally separate encoder/cross-attention mechanism.

---

## MODULE 3.9 — [Addition] KV-Caching: How Inference Actually Works

### 3.9.1 The naive-generation waste problem
Autoregressive generation produces one token at a time, each conditioned on all previous tokens. Naively, generating token `n+1` means re-running the *entire* forward pass over tokens `1...n` from scratch — recomputing `Q`, `K`, `V` for every position, even though positions `1...n-1`'s `K` and `V` vectors are **identical** to what they were on the previous generation step (only `Q` at the new position and the new token's own `K`/`V` are actually new). Recomputing them repeatedly is pure waste: generating `n` tokens naively costs `O(n²)` total work purely from this redundant recomputation, on top of attention's own `O(n²)` cost from Module 3.4.6.

### 3.9.2 The KV-cache mechanism
Cache every layer's `K` and `V` vectors as they're computed, for every token generated so far. At each new generation step:
```
1. Compute Q, K, V for ONLY the new token (not the whole sequence again)
2. Append the new K, V to the cache (K_cache, V_cache now cover all tokens so far)
3. Compute attention: new Q attends over the FULL cached K/V (all previous tokens + the new one)
4. Generate the next token; repeat
```
This reduces per-step cost from re-running the full sequence to processing just the one new token (plus an attention lookup over the growing cache) — turning naive `O(n²)` redundant computation into the `O(n)`-per-step, `O(n²)`-total cost that's inherent to attention itself (Module 3.4.6), with no *extra* redundant work layered on top.

### 3.9.3 Memory cost — why this connects directly back to Module 3.6
The cache must store `K` and `V` for every layer, every head, every token generated so far:
```
KV-cache size ≈ 2 (K and V) × num_layers × num_kv_heads × d_head × seq_len × batch_size × bytes_per_value
```
For a large model with many layers, many heads, and long sequences/large batches, this becomes enormous — often exceeding the memory footprint of the model's own weights, especially for long-context or high-throughput serving. This is *precisely* the cost that GQA (Module 3.6.3) reduces by cutting `num_kv_heads` — every KV-cache byte saved by using fewer K/V head-groups multiplies across every layer and every token in the cache, which is why the MQA/GQA design decision matters so much more in practice than it might seem from the architecture diagram alone.

### 3.9.4 Code: minimal KV-cache implementation
```python
import torch
import torch.nn as nn
import math

class CachedSelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_k, bias=False)
        self.d_k = d_k

    def forward(self, x_new, cache=None):
        # x_new: [batch, 1, d_model] — just the newest token
        Q = self.W_Q(x_new)     # [batch, 1, d_k]
        K_new = self.W_K(x_new)
        V_new = self.W_V(x_new)

        if cache is not None:
            K = torch.cat([cache['K'], K_new], dim=1)   # append to existing cache
            V = torch.cat([cache['V'], V_new], dim=1)
        else:
            K, V = K_new, V_new

        new_cache = {'K': K, 'V': V}

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)   # [batch, 1, seq_len_so_far]
        weights = torch.softmax(scores, dim=-1)
        output = weights @ V                                       # [batch, 1, d_k]
        return output, new_cache

# Simulate generating 5 tokens one at a time, reusing the cache
attn = CachedSelfAttention(d_model=16, d_k=16)
cache = None
for step in range(5):
    x_new = torch.randn(1, 1, 16)                # the newly generated token's embedding
    out, cache = attn(x_new, cache)
    print(f"step {step}: cache K shape = {cache['K'].shape}")   # grows by 1 each step
# Compare against re-running full self-attention over the whole sequence from scratch each step —
# same final result, but this version never recomputes K/V for earlier tokens.
```

### 3.9.5 Interview Q&A
**Q: Why is KV-caching necessary for efficient LLM inference?**
A: Without it, generating each new token would require recomputing keys and values for every previous token from scratch, even though those values are identical to what was already computed on prior steps — pure redundant work. Caching K/V as they're computed means each generation step only needs to process the newest token and attend over the cached history, avoiding that redundancy.

**Q: What's the memory cost of a KV-cache, and how does it scale?**
A: Roughly proportional to `num_layers × num_kv_heads × head_dim × sequence_length × batch_size` (times 2, for both K and V, times bytes per value). It grows linearly with sequence length and batch size, and for large models/long contexts frequently becomes the dominant memory consumer at inference time — often larger than the model weights themselves.

**Q: How does GQA specifically help with KV-cache costs, mechanically?**
A: `num_kv_heads` is a direct multiplicative factor in the KV-cache size formula. GQA reduces the number of distinct K/V head-groups relative to the number of query heads, directly shrinking that term — e.g., going from 32 KV heads (full MHA) to 8 KV-head-groups (a common GQA configuration) cuts KV-cache memory by 4x, with the savings compounding across every layer and every cached token.

---

## MODULE 3.10 — Capstone: Build a GPT From Scratch

This assembles every module above into a complete, trainable, generating language model — deliberately in the spirit of Andrej Karpathy's nanoGPT, built here from the pieces you've just derived rather than imported as a black box.

### 3.10.1 Full architecture
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=4,
                 max_seq_len=256, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)          # §3.2.2
        self.pos_embed = nn.Embedding(max_seq_len, d_model)            # §3.3.3 (learned; simplest to implement)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout=dropout)      # §3.7.5
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.lm_head.weight = self.token_embed.weight    # weight tying: input/output embeddings
                                                            # share parameters — reduces param count
                                                            # and is standard practice (GPT-2 onward)

    def forward(self, idx, targets=None):
        batch, seq_len = idx.shape
        positions = torch.arange(seq_len, device=idx.device).unsqueeze(0)

        x = self.token_embed(idx) + self.pos_embed(positions)   # §3.2.2 + §3.3.3, added together
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)                                          # §3.7.4's two-line block, N times
        x = self.ln_final(x)
        logits = self.lm_head(x)                                   # [batch, seq_len, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # Cross-entropy over the vocabulary — Module 1.1.6's softmax+cross-entropy,
            # now predicting the NEXT TOKEN at every position simultaneously (§3.8.2)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.pos_embed.num_embeddings:]   # truncate to max context
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature               # only need the LAST position's logits
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # sampling — Phase 4 formalizes this
            idx = torch.cat([idx, next_token], dim=1)
        return idx
```
**Note on `generate()`**: this is the naive (non-KV-cached) version for clarity — it recomputes the full forward pass at every step, exactly the inefficiency Module 3.9 identified. As an extension exercise (§3.10.4), rewire this to use the `CachedSelfAttention` pattern from §3.9.4 and confirm identical outputs with dramatically fewer redundant computations for long generations.

### 3.10.2 Training loop
```python
# --- character-level tokenizer on a small text corpus (swap in a real corpus, e.g. tinyshakespeare) ---
text = open('input.txt').read()   # any plain-text file
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

block_size = 128    # context length
batch_size = 32

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])   # targets = inputs shifted by 1 (§3.8.2)
    return x, y

model = GPT(vocab_size=vocab_size, d_model=128, num_heads=4, num_layers=4, max_seq_len=block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)   # AdamW — Module 1.2.5

# Cosine LR schedule with warmup — Module 1.2.6, exactly the schedule real LLM pretraining uses
total_steps = 5000
warmup_steps = 100
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

for step in range(total_steps):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # gradient clipping — Module 1.2.7
    optimizer.step()
    scheduler.step()

    if step % 500 == 0:
        model.eval()
        with torch.no_grad():
            val_x, val_y = get_batch('val')
            _, val_loss = model(val_x, val_y)
        print(f"step {step}: train loss {loss.item():.4f}, val loss {val_loss.item():.4f}, "
              f"lr {scheduler.get_last_lr()[0]:.6f}")
        model.train()
```

### 3.10.3 Generation
```python
model.eval()
context = torch.tensor([[stoi[c] for c in "The "]], dtype=torch.long)
generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)
print(''.join([itos[i] for i in generated[0].tolist()]))
```

### 3.10.4 Extension exercises (in increasing difficulty)
1. Swap the learned positional embedding for RoPE (§3.3.4) — this requires applying the rotation inside the attention computation itself, not adding a separate positional embedding at the input.
2. Rewire `generate()` to use `CachedSelfAttention` (§3.9.4) instead of full recomputation; benchmark generation speed for both versions at `max_new_tokens=500` and confirm the speedup.
3. Replace standard multi-head attention with GQA (§3.6.3) — group your heads and confirm the KV-cache shrinks proportionally.
4. Swap the GELU FFN for SwiGLU (§3.7.1) and compare validation loss at matched parameter count.

---

## MODULE 3.11 — [Addition] FlashAttention: a brief, conceptual preview

You don't need the engineering depth here yet (that's Phase 10 territory), but understanding *what problem it solves* now makes Phase 10 land immediately rather than requiring you to re-derive the motivation from scratch. Module 3.4.6 established that self-attention's `[n,n]` score matrix costs `O(n²)` **memory**. The naive implementation materializes this entire matrix in GPU memory (HBM — high-bandwidth memory, but still far slower to access repeatedly than on-chip SRAM), then reads it back for the softmax and the weighted sum — lots of slow memory traffic. **FlashAttention** (Dao et al., 2022) computes mathematically *identical* attention output, but restructures the computation to process the score matrix in small blocks that fit in fast on-chip SRAM, using a numerically stable "online softmax" technique to avoid ever materializing the full `[n,n]` matrix in slow memory. The result: dramatically faster and more memory-efficient attention, with **zero change to the mathematical output** — it's a systems/hardware optimization of an unchanged formula, not a new attention variant. This is why it's a drop-in replacement in virtually every modern training/inference stack, and why you'll see `attn_implementation="flash_attention_2"` as a simple config flag rather than an architecture decision when you fine-tune models in Phase 5.

---

## Common Pitfalls Across Phase 3
- **Confusing `d_model` and `d_k`.** `d_model` is the full embedding dimension; `d_k = d_model / num_heads` is each individual head's dimension. Mixing these up is the most common source of shape-mismatch bugs when implementing multi-head attention from scratch.
- **Masking after softmax instead of before.** Covered in both Module 2.11 and 3.4.4, but worth repeating because it's the single most common self-attention implementation bug — masking must happen on raw scores, before the softmax, using `-inf`, not by zeroing weights after.
- **Forgetting weight tying.** GPT-style models typically share weights between the input token embedding and the output projection (`lm_head`) — skipping this (as many first implementations do) still works, but uses more parameters and often trains slightly worse at small scale.
- **Applying dropout/LayerNorm in the wrong order relative to residual connections.** This is precisely the Pre-LN vs. Post-LN distinction (§3.7.3) — get this backward and training stability suffers in ways that are easy to misdiagnose as a learning-rate or data problem.
- **Testing generation without ever inspecting attention weights.** Exactly as Phase 2 warned (Module 2.11) — a model can produce fluent-looking output while attending in degenerate or nonsensical patterns internally, especially early in training. Visualize attention maps as a sanity check, not just final output quality.

---

## Phase 3 Completion Checklist
Before moving to Phase 4 (LLMs, Foundation Models, Prompt Engineering), you should be able to, without looking anything up:
- [ ] Explain BPE's algorithm and trace through 2 merge steps on a small example by hand.
- [ ] Explain why byte-level BPE guarantees no OOV tokens, ever.
- [ ] Derive the full self-attention equation term by term, and compute a small worked example (2-3 tokens) by hand, including causal masking.
- [ ] Explain why positional encoding is necessary (the permutation-invariance argument) and compare sinusoidal, learned, and RoPE approaches.
- [ ] Explain RoPE's relative-position property and why it matters for long-context extrapolation.
- [ ] Explain multi-head attention's motivation and mechanics, including what `W_O` does.
- [ ] Explain MQA and GQA, why they exist (KV-cache memory), and identify which real production models use which.
- [ ] Assemble a full Transformer block from memory: Pre-LN, self-attention, residual, Pre-LN, FFN, residual — and explain why Pre-LN specifically won out over Post-LN.
- [ ] Compare encoder-only, decoder-only, and encoder-decoder architectures, and explain why decoder-only dominates modern general-purpose LLMs.
- [ ] Explain KV-caching's mechanism and why it's necessary for efficient inference.
- [ ] Have a working, from-scratch GPT implementation that trains on a text corpus and generates coherent (if small-scale) text.

## What's Next
**Phase 4** takes this exact architecture — now fully understood down to the matrix-multiplication level — and scales it up conceptually: pretraining on trillions of tokens, instruction tuning (SFT), and RLHF/DPO alignment (previewed in Phase 1's roadmap, covered in depth in Phase 8) turn this raw next-token predictor into a model that follows instructions and holds a conversation. You'll also start prompt engineering — and every technique there (few-shot examples, chain-of-thought) works precisely *because* of the autoregressive, attention-based mechanism you just built by hand in Module 3.10. There is no more foundational phase than this one; everything from here forward is built directly on top of it.
