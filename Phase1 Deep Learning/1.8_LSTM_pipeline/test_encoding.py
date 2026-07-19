# Always build the vocabulary from the training data only.
# The model learns only from the training set.
# Validation and test sets are used only for evaluation.

from load_dataset import train_text, val_text, test_text

chars = sorted(list(set(train_text)))

#  We'll also need the mapping later when generating text. (char to int)
char_to_idx = {char: idx for idx, char in enumerate(chars)}

# We'll also need the reverse mapping later when generating text. (int to char)
idx_to_char = {idx: char for idx, char in enumerate(chars)}

# Now replace every character with its integer ID.
encoded_train = [char_to_idx[ch] for ch in train_text]

vocab_size = len(chars)


# print("Unique characters:", len(chars))
# print("Vocabulary Size:", vocab_size)
# print("First 20 Encoded Values:", encoded_train[:20])
# decoded = ''.join(idx_to_char[i] for i in encoded_train[:100])
# print("\nDecoded Text:\n")
# print(decoded)