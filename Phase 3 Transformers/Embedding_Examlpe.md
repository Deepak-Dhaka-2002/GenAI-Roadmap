```markdown
This is **the next major concept in Phase 3**. If tokenization answers:

> **"What pieces of text does the model see?"**
then **embeddings** answer:

> **"How does the model represent those pieces numerically so a neural network can process them?"**

---

# Step 1: Computers Don't Understand Words
Suppose our sentence is:

```
I love pizza

```
After tokenization:

```
["I", "love", "pizza"]

```
The neural network cannot process strings like `"love"`.

It only understands numbers.

---

# Step 2: Convert Tokens to IDs
The tokenizer has a vocabulary.

Example:
```
Vocabulary

```
"I"      → 15
"love"   → 892
"pizza"  → 5312
```
So the sentence becomes:

```
[15, 892, 5312]

```
These are **token IDs**.

But...

---

# Token IDs Have No Meaning
Imagine:

```
cat  → 25
dog  → 26
car  → 27
```
Does this imply

```
dog = cat + 1 ?
```
Of course not.

These numbers are just indexes.

Think of them like employee IDs:
```
Alice → Employee #101

Bob → Employee #102
```
Employee #102 isn't "more Bob" than #101.

IDs carry no semantic meaning.

---

# So We Need Something Better
Instead of representing a word by

```
pizza = 5312
```
represent it by **hundreds or thousands of learned numbers.**
Example:

```
pizza

↓

[0.82,
 -1.41,
 0.17,
 2.93,
 ...
 0.44]
```
This list is called an **embedding vector**.

---
# Visual
Instead of

```
pizza

↓

5312
```
we have

```
pizza

↓
┌──────────────────────────────┐
│ 0.82                         │
│ -1.41                        │
│ 0.17                         │
│ 2.93                         │
│ ...                          │
│ 0.44                         │
└──────────────────────────────┘
```
This could have:

- 128 numbers
- 512 numbers
- 768 numbers
- 4096 numbers
depending on the model.

---
# What Do These Numbers Mean?
Individually...

Nothing.
Dimension 347 isn't

```
"animalness"
```
Dimension 812 isn't

```
"foodness"
```
Instead...
**The whole vector together represents the meaning.**

Think of it like GPS coordinates.

---

# Similar Words Have Similar Vectors
Suppose we plot only 2 dimensions (real models use hundreds or thousands).

```
		  Animal

dog •

cat •

					 car •

							truck •

--------------------------------------------

				Vehicle
```
Notice:
- dog close to cat
- car close to truck
Similar meanings become **nearby vectors**.
This is called **vector space**.

---
# Why Is It Called a Vector?
A vector is simply an ordered list of numbers.

For example:
```
[2, 5]
```
is a 2-dimensional vector.

```
[1.3, -0.7, 9.8]
```
is a 3-dimensional vector.
LLMs use very high-dimensional vectors, such as:

```
[0.12, -0.55, 1.03, ..., 0.41]
```
with hundreds or thousands of dimensions.

---

# Where Do These Vectors Come From?
They're **learned during training**.

Initially:
```
pizza

↓
[0.02,
-0.01,
0.03,
...]
```
Random.
As training progresses:

```
pizza

↓
[0.91,
-1.44,
0.72,
...]
```
The values change through gradient descent so that words used in similar contexts end up with similar embeddings.

---

# Embedding Matrix
Instead of storing one vector...

Store one vector for **every token**.
Imagine a vocabulary of 50,000 tokens.

```
Vocabulary
Token

I
love
pizza
dog
cat
...
```
Each has its own vector.

Visually:
```
Embedding Matrix

			Dim1   Dim2   Dim3   ... Dim768
I            0.2    1.1   -0.8
love        -0.5    0.7    1.2
pizza        1.9   -0.3    0.1
dog         -0.7    0.8    0.6
cat         -0.6    0.9    0.5
...
```
This huge table is called the **embedding matrix**.

---
# Converting IDs to Embeddings
Input:

```
[15, 892, 5312]
```
Lookup:
```
Embedding Matrix

Row 15
↓

Vector for "I"
Row 892

↓
Vector for "love"
Row 5312

↓
Vector for "pizza"
```
Output:

```
[
Embedding(I),
Embedding(love),
Embedding(pizza)
]
```
This is simply a table lookup.

No computation yet.

---

# Entire Pipeline

```
Text

↓
```
I love pizza

↓
Tokenizer

↓
["I","love","pizza"]

↓
Token IDs

↓
[15,892,5312]

↓
Embedding Lookup

↓
e₁
e₂
e₃

↓
Transformer
```
This is the **actual input** to the Transformer.

---
# But There's a Problem...
Notice:

```
I
love
pizza
```
and

```
pizza
love
I
```
contain the **same three tokens**.

Their embeddings are the same:

```
Embedding(I)
Embedding(love)
Embedding(pizza)
```
So how does the model know **which token came first**?

It doesn't.

Embeddings encode **what** the token is, but **not where it appears**.

That missing information is the motivation for the **next concept in Phase 3: Positional Encoding (or Positional Embeddings)**, which tells the Transformer the order of tokens in the sequence. This is the next piece needed before we can fully understand self-attention in Transformers.
```
