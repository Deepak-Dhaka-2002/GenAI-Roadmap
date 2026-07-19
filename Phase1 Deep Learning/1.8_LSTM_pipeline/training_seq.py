# Instead of learning from one huge text string, 
# it learns from thousands of overlapping sequences, each teaching it to predict the next character.

import numpy as np
from test_encoding import encoded_train, vocab_size
from test_encoding import idx_to_char

sequence_length = 100  # Define the length of each input sequence 

inputs = []
targets = []

for i in range(len(encoded_train) - sequence_length):
    input_seq = encoded_train[i:i + sequence_length]
    target_seq = encoded_train[i + 1:i + sequence_length + 1]

    inputs.append(input_seq)
    targets.append(target_seq)

print("Number of sequences:", len(inputs))    ##check how many seq we have created

inputs = np.array(inputs)
targets = np.array(targets)
# print("Input shape :", inputs.shape)
# print("Target shape:", targets.shape)

# sample = 0

# print("INPUT\n")
# print("".join(idx_to_char[i] for i in inputs[sample]))

# print("\nTARGET\n")
# print("".join(idx_to_char[i] for i in targets[sample]))