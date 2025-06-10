import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold
from statistics import mean

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
print("Wybrane kolumny przez RFE:", selected_rfe_columns)



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

X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_rfe, y_rfe, test_size=test_size, stratify=y_rfe, random_state=12345)


classifiers = {
    'KNN': KNeighborsClassifier(n_neighbors=k),
    'MPL': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=12345),
    'Random Forest' : RandomForestClassifier(n_estimators=100, random_state=12345)
}

rkfss = RepeatedKFold(n_splits=2, n_repeats=5)
results = np.zeros((10, len(classifiers), 2))
confusion_matrices = {clf_name: [] for clf_name in classifiers}
for i, (train_index, test_index) in enumerate(rkfss.split(X, y)):
    #pętla ta przechodzi przez klasyfikatory
    for clf_idx, (clf_name, clf) in enumerate(classifiers.items()):

        # trening klasyfikatora
        clf.fit(X_train_cv, y_train_cv)
            
        # predykcja dla danego klasyfikatora
        y_pred_cv = clf.predict(X_test_cv)
                    
        # obliczenia dokładnośći dla danego klasyfikatora i zapisanie wyniku w macierzy wyników
        results[i, clf_idx, 0] = accuracy_score(y_test_cv, y_pred_cv)
        results[i, clf_idx, 1] = f1_score(y_test_cv, y_pred_cv, average='macro')
        cm = confusion_matrix(y_test_cv, y_pred_cv, labels=np.unique(y))
        confusion_matrices[clf_name].append(cm)



print("\n🔹 KNN")
print("Accuracy:", mean(results[:,0,0]))
print("Macro F1:", mean(results[:,0,1]))


print("\n🔸 MLP")
print("Accuracy:", mean(results[:,1,0]))
print("Macro F1:", mean(results[:,1,1]))


print("\n🔹 Random Forest")
print("Accuracy:", mean(results[:,2,0]))
print("Macro F1:", mean(results[:,2,1]))

for clf_name, cm_list in confusion_matrices.items():
    avg_cm = np.mean(cm_list, axis=0)
    avg_cm = np.round(avg_cm).astype(int)

    disp = ConfusionMatrixDisplay(confusion_matrix=avg_cm)
    disp.plot(cmap='Blues')
    plt.title(f"Macierz pomyłek – {clf_name}")
    plt.show()