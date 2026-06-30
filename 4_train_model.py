"""
╔══════════════════════════════════════════════════════════════════╗
║  FILE 4 — PELATIHAN MODEL SVM                                    ║
║  Skripsi: Analisis Sentimen Review Game PUBG pada Steam          ║
║  Muhammad Rafly Badru Tamam — NIM 15220026                       ║
║  Universitas Bina Sarana Informatika, 2026                       ║
╚══════════════════════════════════════════════════════════════════╝

FUNGSI:
  1. Membagi data: 80% latih / 20% uji (stratified)
  2. Terapkan SMOTE pada data latih saja (anti data leakage)
  3. GridSearchCV untuk tuning hyperparameter C
  4. Latih DUA model:
       a) Model Baseline  → SVM + TF-IDF saja
       b) Model Usulan    → SVM + TF-IDF + Playtime
  5. Simpan keempat objek model & data ke file .pkl

JALANKAN:
  python 4_train_model.py

INPUT  : features.pkl
OUTPUT :
  - split_data.pkl           (X/y train & test untuk baseline & gabungan)
  - model_baseline.pkl       (model SVM trained baseline)
  - model_gabungan.pkl       (model SVM trained gabungan)
  - smote_info.pkl           (info distribusi sebelum & sesudah SMOTE)
"""

import pickle
import warnings
import numpy as np
from collections import Counter

from sklearn.model_selection  import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm              import LinearSVC
from sklearn.exceptions       import ConvergenceWarning
from imblearn.over_sampling   import SMOTE

RANDOM_STATE = 42

# ─── Fungsi utilitas ─────────────────────────────────────────────
def dist_str(y) -> str:
    c = Counter(y)
    total = len(y)
    return (f"Positif(1)={c[1]} ({c[1]/total*100:.1f}%) | "
            f"Negatif(0)={c[0]} ({c[0]/total*100:.1f}%)")


# ─── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  FILE 4 — PELATIHAN MODEL SVM")
    print("=" * 60)

    # ── Muat fitur ───────────────────────────────────────────────
    print("\n[INFO] Memuat features.pkl...")
    with open("features.pkl", "rb") as f:
        feat = pickle.load(f)

    X_baseline = feat["X_baseline"]   # TF-IDF saja
    X_gabungan = feat["X_gabungan"]   # TF-IDF + Playtime
    y          = feat["y"]

    print(f"  Dimensi X_baseline : {X_baseline.shape}")
    print(f"  Dimensi X_gabungan : {X_gabungan.shape}")
    print(f"  Total sampel       : {len(y)}")
    print(f"  Distribusi         : {dist_str(y)}")

    # ── Pembagian Data 80:20 (stratified) ───────────────────────
    print("\n[INFO] Membagi data 80% latih / 20% uji (stratified)...")

    (X_bl_train, X_bl_test,
     X_gb_train, X_gb_test,
     y_train,    y_test) = train_test_split(
        X_baseline, X_gabungan, y,
        test_size    = 0.20,
        random_state = RANDOM_STATE,
        stratify     = y
    )

    print(f"  Data latih  : {X_bl_train.shape[0]} sampel | {dist_str(y_train)}")
    print(f"  Data uji    : {X_bl_test.shape[0]} sampel  | {dist_str(y_test)}")

    # ── SMOTE pada data latih saja ───────────────────────────────
    print("\n[INFO] Menerapkan SMOTE pada data latih (k_neighbors=5)...")
    print("  CATATAN: SMOTE HANYA diterapkan pada data latih → anti data leakage")

    smote = SMOTE(k_neighbors=5, random_state=RANDOM_STATE)

    # Baseline
    print("\n  → Baseline (TF-IDF saja):")
    print(f"    Sebelum SMOTE: {dist_str(y_train)}")
    X_bl_train_sm, y_bl_train_sm = smote.fit_resample(X_bl_train, y_train)
    print(f"    Sesudah SMOTE: {dist_str(y_bl_train_sm)}")

    # Gabungan
    print("\n  → Gabungan (TF-IDF + Playtime):")
    print(f"    Sebelum SMOTE: {dist_str(y_train)}")
    X_gb_train_sm, y_gb_train_sm = smote.fit_resample(X_gb_train, y_train)
    print(f"    Sesudah SMOTE: {dist_str(y_gb_train_sm)}")

    smote_info = {
        "before": dist_str(y_train),
        "after_baseline": dist_str(y_bl_train_sm),
        "after_gabungan": dist_str(y_gb_train_sm),
    }

    # ── GridSearchCV — Tuning Hyperparameter C ───────────────────
    print("\n[INFO] GridSearchCV — Tuning hyperparameter C...")
    print("  Nilai C yang diuji: [1, 10, 100]")
    print("  Strategi CV: StratifiedKFold (5-fold), scoring=f1_macro")
    print("  (Proses ini memakan waktu beberapa menit...)\n")

    param_grid = {"C": [1, 10, 100]}
    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Tuning pada Model Gabungan (lebih representatif)
    # ConvergenceWarning disuppress karena berasal dari kandidat C yang
    # tidak terpilih — model final (best_C) dilatih dengan max_iter=100000
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        gs = GridSearchCV(
            LinearSVC(max_iter=100000, random_state=RANDOM_STATE),
            param_grid,
            cv      = cv,
            scoring = "f1_macro",
            n_jobs  = -1,
            verbose = 1,
        )
        gs.fit(X_gb_train_sm, y_gb_train_sm)

    best_C = gs.best_params_["C"]
    print(f"\n  Best C ditemukan   : {best_C}")
    print(f"  Best F1-macro (CV) : {gs.best_score_:.4f}")

    # ── Latih Model Baseline ─────────────────────────────────────
    print(f"\n[INFO] Melatih MODEL BASELINE (SVM + TF-IDF, C={best_C})...")
    model_baseline = LinearSVC(C=best_C, max_iter=100000, random_state=RANDOM_STATE)
    model_baseline.fit(X_bl_train_sm, y_bl_train_sm)
    print("  ✔ Model Baseline selesai dilatih.")

    # ── Latih Model Gabungan ─────────────────────────────────────
    print(f"\n[INFO] Melatih MODEL USULAN (SVM + TF-IDF + Playtime, C={best_C})...")
    model_gabungan = LinearSVC(C=best_C, max_iter=100000, random_state=RANDOM_STATE)
    model_gabungan.fit(X_gb_train_sm, y_gb_train_sm)
    print("  ✔ Model Usulan selesai dilatih.")

    # ── Simpan semua objek ───────────────────────────────────────
    print("\n[INFO] Menyimpan model dan data uji ke file .pkl...")

    with open("split_data.pkl", "wb") as f:
        pickle.dump({
            "X_bl_test" : X_bl_test,
            "X_gb_test" : X_gb_test,
            "y_test"    : y_test,
            "best_C"    : best_C,
        }, f)

    with open("model_baseline.pkl", "wb") as f:
        pickle.dump(model_baseline, f)

    with open("model_gabungan.pkl", "wb") as f:
        pickle.dump(model_gabungan, f)

    with open("smote_info.pkl", "wb") as f:
        pickle.dump(smote_info, f)

    print("  ✔ split_data.pkl")
    print("  ✔ model_baseline.pkl")
    print("  ✔ model_gabungan.pkl")
    print("  ✔ smote_info.pkl")

    print("\n✅ SELESAI. Lanjutkan ke: python 5_evaluate_model.py")
