import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data_V2.csv")

print(df["type"].value_counts())

# Wyświetlenie wykresu słupkowego
df["type"].value_counts().plot(kind='bar', title='Rozkład klas TYPE')
plt.xlabel("Klasa")
plt.ylabel("Liczność")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Wyświetlenie procentowego udziału klas
print(df["type"].value_counts(normalize=True) * 100)


# Zbiór jest niezbalansowany
# istnieją klasy dominujące: Humanoid, Beast

af = pd.read_csv("clean_data.csv", dtype=float)

# statystyki opisowe
print(af.describe())

# wizualizacja

print(af.dtypes)

df["type"].value_counts().plot(kind='bar', title="Rozkład klas")
plt.xlabel("Typ")
plt.ylabel("Liczba próbek")
plt.show()

sns.heatmap(af.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Macierz korelacji")
plt.show()

af.hist(bins=20, figsize=(15,10))
plt.tight_layout()
plt.show()