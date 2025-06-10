import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

np.random.seed(12345)

data = np.loadtxt('clean_data.csv', dtype = 'object', delimiter = ',')

newData = np.delete(data, 0, 0)

y = newData[:, 1].astype(int)
X = newData[:, [0] + list(range(2, newData.shape[1]))].astype(float)

desired_n = 50
current_counts = {
    10: 189, 
    2: 105, 
    8: 89, 
    11: 87,
    14: 50, 
    5: 47, 
    6: 38, 
    1: 37,
    9: 27, 
    13: 24, 
    4: 24, 
    7: 21,
    12: 8, 
    3: 8, 
    15: 8
}

sampling_strategy = {label: desired_n for label, count in current_counts.items() if count < desired_n}

smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=2, random_state=12345)
X_resampled, y_resampled = smote.fit_resample(X, y)

# print(Counter(y_resampled))

X_manual = X_resampled.copy()
X_rfe = X_resampled.copy()
y_manual = y_resampled.copy()
y_rfe = y_resampled.copy()


selected_columns = [3, 2, 4, 5, 15, 17, 19, 21, 23, 25]
X_manual = X_manual[:, selected_columns]


estimator = RandomForestClassifier(n_estimators=100, random_state=12345)
selector = RFE(estimator, n_features_to_select=11, step=1)
X_rfe = selector.fit_transform(X_rfe, y_rfe)


selected_rfe_columns = selector.get_support(indices=True)
# print("Wybrane kolumny przez RFE:", selected_rfe_columns)



test_size = 0.2
k = 5  # liczba sąsiadów w KNN

# =====================
# 1️⃣ RĘCZNA SELEKCJA
# =====================
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_manual, y_manual, test_size=test_size, stratify=y_manual, random_state=12345)

clf_manual = KNeighborsClassifier(n_neighbors=k)
clf_manual.fit(X_train_m, y_train_m)
y_pred_m = clf_manual.predict(X_test_m)

print("\n📘 Wyniki – RĘCZNA selekcja")
print("Accuracy:", accuracy_score(y_test_m, y_pred_m))
print("Macro F1:", f1_score(y_test_m, y_pred_m, average='macro'))
cm_manual = confusion_matrix(y_test_m, y_pred_m)

disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_manual)
disp1.plot(cmap='Blues')
plt.title("Macierz pomyłek – ręczna selekcja")
plt.show()

# =====================
# 2️⃣ RFE SELEKCJA
# =====================
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_rfe, y_rfe, test_size=test_size, stratify=y_rfe, random_state=12345)

clf_rfe = KNeighborsClassifier(n_neighbors=k)
clf_rfe.fit(X_train_r, y_train_r)
y_pred_r = clf_rfe.predict(X_test_r)

print("\n📗 Wyniki – RFE selekcja")
print("Accuracy:", accuracy_score(y_test_r, y_pred_r))
print("Macro F1:", f1_score(y_test_r, y_pred_r, average='macro'))
cm_rfe = confusion_matrix(y_test_r, y_pred_r)

disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_rfe)
disp2.plot(cmap='Greens')
plt.title("Macierz pomyłek – RFE")
plt.show()

# ========================
# Eksperyment nr 2
# ========================

X_train, X_test, y_train, y_test = train_test_split(X_rfe, y_rfe, test_size=test_size, stratify=y_rfe, random_state=12345)

# ========================
# Klasyfikator KNN
# ========================
clf_knn = KNeighborsClassifier(n_neighbors=k)
clf_knn.fit(X_train, y_train)
y_pred_knn = clf_knn.predict(X_test)

print("\n🔹 KNN")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Macro F1:", f1_score(y_test, y_pred_knn, average='macro'))
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap='Blues')
plt.title("Macierz pomyłek – KNN")
plt.show()

# ========================
# Klasyfikator MLP
# ========================
clf_mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=12345)
clf_mlp.fit(X_train, y_train)
y_pred_mlp = clf_mlp.predict(X_test)

print("\n🔸 MLP")
print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("Macro F1:", f1_score(y_test, y_pred_mlp, average='macro'))
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp)
disp_mlp.plot(cmap='Oranges')
plt.title("Macierz pomyłek – MLP")
plt.show()

# ========================
# Klasyfikator Random Forest
# ========================
clf_rf = RandomForestClassifier(n_estimators=100, random_state=12345)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

print("\n🔹 Random Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Macro F1:", f1_score(y_test, y_pred_rf, average='macro'))
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp_rf.plot(cmap='Greens')
plt.title("Macierz pomyłek – Random Forest")
plt.show()
