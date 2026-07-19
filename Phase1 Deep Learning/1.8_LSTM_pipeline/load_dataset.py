import numpy as np
import pandas as pd

# | Check                    | Status                        |
# | ------------------------ | ----------------------------- |
# | CSV loaded               | ✅                             |
# | Text column found        | ✅                             |
# | Missing values checked   | ✅                             |
# | Text extracted as string | ⏳ (do this small improvement) |
# | Ready for preprocessing  | ✅                             |

train_df = pd.read_csv(r"D:\Courses\Vscode\Agentic_AI\data\LSTM\test.csv")
val_df = pd.read_csv(r"D:\Courses\Vscode\Agentic_AI\data\LSTM\validation.csv")
test_df = pd.read_csv(r"D:\Courses\Vscode\Agentic_AI\data\LSTM\test.csv")

# print("Training Set:", train_df.shape)
# print("Validation Set:", val_df.shape)
# print("Test Set:", test_df.shape)

# print(train_df.info())

# print(train_df.isnull().sum())

train_df["length"] = train_df["text"].str.len()
# print(train_df["length"].describe())
longest = train_df.loc[train_df["length"].idxmax()]
# print(longest["text"])


train_text = train_df.loc[0, "text"]
val_text = val_df.loc[0, "text"]
test_text = test_df.loc[0, "text"]

# print(type(train_text))
# print(len(train_text))
# print(train_text[:200])

