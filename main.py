import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("gender_submission.csv")

print("=== Train ===")
print(train.shape)
print(train.head())
print(train.info())

print("\n=== Test ===")
print(test.shape)
print(test.head())
print(test.info())

print("\n=== Sample Submission ===")
print(sample_submission.shape)
print(sample_submission.head())
