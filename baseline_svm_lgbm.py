import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# LightGBM (pip install lightgbm)
from lightgbm import LGBMClassifier

# === Twoja ścieżka do CSV ===
CSV = r"D:\Projekt_Przejsciowy\CSV\models_data.csv"

# === Kolumny w Twoim CSV ===
FEATURE_COLUMNS = ["D_med_mm", "W_med_mm", "C_med_mm", "A_med_deg"]
LABEL_COLUMN    = "ModelClassID_0_15"
ID_COLUMN       = "Model_ID"

# === Wczytanie CSV (obsługa , lub ;) ===
df = pd.read_csv(CSV, sep=None, engine="python", encoding="utf-8-sig")

# Walidacja kolumn
missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN, ID_COLUMN] if c not in df.columns]
if missing:
    raise ValueError(f"Brakuje kolumn w CSV: {missing}\nDostępne: {list(df.columns)}")

# Usunięcie braków
df = df.dropna(subset=FEATURE_COLUMNS + [LABEL_COLUMN, ID_COLUMN]).copy()

# Dane
X   = df[FEATURE_COLUMNS].to_numpy(dtype=float)
# Szybka diagnostyka cech (czy są prawie stałe / mało unikalnych wartości)
print("Diagnostyka cech (nunique):")
print(df[FEATURE_COLUMNS].nunique().to_string())
y   = df[LABEL_COLUMN].astype(int).to_numpy()
ids = df[ID_COLUMN].astype(str).to_numpy()

N_CLASSES = 16

# Rozkład klas
counts = pd.Series(y).value_counts().sort_index()
print("Rozkład klas (label -> liczność):")
print(counts.to_string())

# === Modele: SVM + LightGBM ===
models = {
    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            C=2.0,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            probability=False,
            random_state=42
        ))
    ]),
    "lgbm": LGBMClassifier(
        objective="multiclass",
        num_class=N_CLASSES,
        # bardziej "miękkie" ustawienia pod małą liczbę cech / próbek:
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=5,          # pozwól dzielić liście przy mniejszej liczbie próbek
        min_split_gain=0.0,           # nie blokuj splitów progiem gain
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1                  # wycisza spam ostrzeżeń
    )
}

# K-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === Walidacja CV ===
for name, clf in models.items():
    print(f"\n=== {name} (5-fold CV) ===")
    accs, cms_sum = [], None
    last_va_idx, last_yp = None, None

    for tr_idx, va_idx in skf.split(X, y):
        clf.fit(X[tr_idx], y[tr_idx])
        yp = clf.predict(X[va_idx])
        accs.append(accuracy_score(y[va_idx], yp))

        cm = confusion_matrix(y[va_idx], yp, labels=np.arange(N_CLASSES))
        cms_sum = cm if cms_sum is None else (cms_sum + cm)

        last_va_idx, last_yp = va_idx, yp

    print(f"ValAcc mean={np.mean(accs):.3f}  std={np.std(accs):.3f}")
    print("Raport (ostatni fold):")
    print(classification_report(y[last_va_idx], last_yp, digits=3))

# === OOF (out-of-fold) dla obu modeli ===
print("\n[OOF] Liczę predykcje OOF...")
y_pred_svm  = cross_val_predict(models["svm"],  X, y, cv=skf,                method="predict")
y_pred_lgbm = cross_val_predict(models["lgbm"], X, y, cv=skf, n_jobs=-1,      method="predict")

# Zapis OOF
oof = pd.DataFrame({"Model_ID": ids, "y_true": y, "y_svm": y_pred_svm, "y_lgbm": y_pred_lgbm})
out_dir  = Path(CSV).parent
fig_dir  = out_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
oof_path = out_dir / "oof_baselines.csv"
oof.to_csv(oof_path, index=False, encoding="utf-8-sig")
print(f"[OOF] Zapisano: {oof_path}")

# === Metryki OOF ===
acc_svm  = accuracy_score(y, y_pred_svm)
acc_lgbm = accuracy_score(y, y_pred_lgbm)

rep_svm  = classification_report(y, y_pred_svm,  labels=np.arange(N_CLASSES),
                                 output_dict=True, zero_division=0)
rep_lgbm = classification_report(y, y_pred_lgbm, labels=np.arange(N_CLASSES),
                                 output_dict=True, zero_division=0)

cm_svm   = confusion_matrix(y, y_pred_svm,  labels=np.arange(N_CLASSES))
cm_lgbm  = confusion_matrix(y, y_pred_lgbm, labels=np.arange(N_CLASSES))

# Pomocnicze: pobierz metryki per klasa (0..N_CLASSES-1)
def per_class(report_dict, n_classes=N_CLASSES):
    prec, rec, f1, sup = [], [], [], []
    for k in range(n_classes):
        k = str(k)
        d = report_dict.get(k, {"precision":0, "recall":0, "f1-score":0, "support":0})
        prec.append(float(d.get("precision", 0)))
        rec.append(float(d.get("recall", 0)))
        f1.append(float(d.get("f1-score", 0)))
        sup.append(int(d.get("support", 0)))
    return np.array(prec), np.array(rec), np.array(f1), np.array(sup)

def plot_confusion(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig = plt.figure(figsize=(7, 7))
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca(), colorbar=True)
    plt.title("Confusion Matrix (Validation)")
    plt.xlabel("Predicted label [class]")
    plt.ylabel("True label [class]")
    try:
        disp.im_.colorbar.set_label('Count [#]')
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

prec_svm,  rec_svm,  f1_svm,  sup_svm  = per_class(rep_svm,  N_CLASSES)
prec_lgbm, rec_lgbm, f1_lgbm, sup_lgbm = per_class(rep_lgbm, N_CLASSES)

# === Wykresy ===
classes = np.arange(N_CLASSES)
cls_labels = [str(i) for i in classes]

def save_bar(values, title, ylabel, xticklabels, path, ylim=None):
    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), xticklabels, rotation=0)
    plt.xlabel("Klasa")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# 1) Rozkład klas (prawdziwych)
save_bar(
    counts.reindex(classes, fill_value=0).values,
    "Rozkład klas (y_true)",
    "Liczność [szt.]",
    cls_labels,
    fig_dir / "class_distribution.png"
)

# 2) Accuracy SVM vs LightGBM (pojedynczy wykres z 2 słupkami)
plt.figure(figsize=(5,4))
plt.bar([0,1], [acc_svm*100, acc_lgbm*100])
plt.xticks([0,1], ["SVM (RBF)", "LightGBM"])
plt.ylabel("Accuracy [%]")
plt.title("Accuracy OOF (SVM vs LightGBM)")
plt.tight_layout()
plt.savefig(fig_dir / "accuracy_svm_vs_lgbm.png", dpi=150)
plt.close()

# 3) SVM: precision/recall/f1/support per klasa
save_bar(prec_svm*100, "SVM – Precision per klasa", "Precision [%]", cls_labels,
         fig_dir / "svm_precision_per_class.png", ylim=(0,100))
save_bar(rec_svm*100,  "SVM – Recall per klasa",    "Recall [%]",    cls_labels,
         fig_dir / "svm_recall_per_class.png",    ylim=(0,100))
save_bar(f1_svm*100,   "SVM – F1-score per klasa",  "F1-score [%]",  cls_labels,
         fig_dir / "svm_f1_per_class.png",        ylim=(0,100))
save_bar(sup_svm,      "SVM – Support per klasa",   "Liczność [szt.]", cls_labels,
         fig_dir / "svm_support_per_class.png")

# 4) LightGBM: precision/recall/f1/support per klasa
save_bar(prec_lgbm*100, "LightGBM – Precision per klasa", "Precision [%]", cls_labels,
         fig_dir / "lgbm_precision_per_class.png", ylim=(0,100))
save_bar(rec_lgbm*100,  "LightGBM – Recall per klasa",    "Recall [%]",    cls_labels,
         fig_dir / "lgbm_recall_per_class.png",    ylim=(0,100))
save_bar(f1_lgbm*100,   "LightGBM – F1-score per klasa",  "F1-score [%]",  cls_labels,
         fig_dir / "lgbm_f1_per_class.png",        ylim=(0,100))
save_bar(sup_lgbm,      "LightGBM – Support per klasa",   "Liczność [szt.]", cls_labels,
         fig_dir / "lgbm_support_per_class.png")

# 5) Macierze pomyłek
class_names = [f"{(i>>3)&1}_{(i>>2)&1}_{(i>>1)&1}_{i&1}" for i in range(N_CLASSES)]
plot_confusion(y, y_pred_svm,  class_names, fig_dir / "svm_confusion_matrix.png")
plot_confusion(y, y_pred_lgbm, class_names, fig_dir / "lgbm_confusion_matrix.png")

print(f"\n[FIGURES] Zapisano wykresy do: {fig_dir}")
print(f"Accuracy SVM={acc_svm:.3f}  LightGBM={acc_lgbm:.3f}")
print(f"Macro-F1 SVM={f1_score(y, y_pred_svm, average='macro'):.3f}  "

f"LightGBM={f1_score(y, y_pred_lgbm, average='macro'):.3f}")
