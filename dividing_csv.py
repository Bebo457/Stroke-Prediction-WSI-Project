import pandas as pd
from sklearn.model_selection import train_test_split

# Wczytanie danych
df = pd.read_csv("dataset/dataset_prep_PCA.csv")

# Najpierw wydzielamy train (70%) i resztę (30%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    shuffle=True
)

# Z pozostałych 30% robimy validation (15%) i test (15%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    shuffle=True
)

# Zapis do CSV
train_df.to_csv("dataset/train.csv", index=False)
val_df.to_csv("dataset/validation.csv", index=False)
test_df.to_csv("dataset/test.csv", index=False)

print("Podział zakończony:")
print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")