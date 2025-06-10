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

af.describe().to_csv("opis_danych.csv")
# wizualizacja

print(af.dtypes)

df["type"].value_counts().plot(kind='bar', title="Rozkład klas")
plt.xlabel("Typ")
plt.ylabel("Liczba próbek")
plt.show()

corr = af.corr(numeric_only=True)

plt.figure(figsize=(12, 10))

ax = sns.heatmap(
    corr,
    annot=False,
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

# Wymuś pokazanie wszystkich etykiet na osiach
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, ha='right')

ax.set_yticks(range(len(corr.index)))
ax.set_yticklabels(corr.index, rotation=0)

plt.title("Macierz korelacji", fontsize=14)
plt.tight_layout()
plt.show()

af.hist(bins=20, figsize=(15,10))
plt.tight_layout()
plt.show()