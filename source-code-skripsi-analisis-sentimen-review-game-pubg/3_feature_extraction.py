"""
╔══════════════════════════════════════════════════════════════════╗
║  FILE 3 — EKSTRAKSI FITUR & PENGGABUNGAN FITUR WAKTU BERMAIN    ║
║  Skripsi: Analisis Sentimen Review Game PUBG pada Steam          ║
║  Muhammad Rafly Badru Tamam — NIM 15220026                       ║
║  Universitas Bina Sarana Informatika, 2026                       ║
╚══════════════════════════════════════════════════════════════════╝

FUNGSI:
  1. Membaca pubg_reviews_clean.csv
  2. Encode label sentimen → 1 (positif) / 0 (negatif)
  3. Ekstraksi fitur teks dengan TF-IDF Vectorizer
     (max_features=10000, ngram_range=(1,2), min_df=2)
  4. Normalisasi fitur numerik waktu bermain (StandardScaler)
  5. Penggabungan (hstack) TF-IDF + playtime → X_gabungan
  6. Simpan semua objek ke file .pkl untuk digunakan file berikutnya

JALANKAN:
  python 3_feature_extraction.py

INPUT  : pubg_reviews_clean.csv
OUTPUT :
  - features.pkl      (X_baseline, X_gabungan, y, nama fitur)
  - tfidf_vectorizer.pkl
  - scaler_playtime.pkl
"""

import pandas as pd
import numpy  as np
import pickle
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing           import StandardScaler, LabelEncoder

# ─── Konfigurasi TF-IDF ──────────────────────────────────────────
TFIDF_CONFIG = dict(
    max_features = 10_000,   # 10.000 kata/frasa teratas
    ngram_range  = (1, 2),   # unigram + bigram
    min_df       = 2,        # abaikan token yang hanya muncul 1x
    sublinear_tf = True,     # log-scaling TF → reduksi dominasi kata sangat umum
)


# ─── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    INPUT_CSV = "pubg_reviews_clean.csv"

    print("=" * 60)
    print("  FILE 3 — EKSTRAKSI FITUR")
    print("=" * 60)

    # ── Muat data ────────────────────────────────────────────────
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"[INFO] Dataset dimuat: {len(df)} baris")

    # Pastikan tidak ada nilai kosong
    df.dropna(subset=["review_clean", "sentiment", "playtime_at_review_hours"], inplace=True)
    df["review_clean"] = df["review_clean"].astype(str)
    print(f"[INFO] Baris valid untuk ekstraksi fitur: {len(df)}")

    # ── Encode Label ─────────────────────────────────────────────
    # positif → 1, negatif → 0
    df["label"] = (df["sentiment"] == "positif").astype(int)
    y = df["label"].values
    print(f"\n[INFO] Distribusi label:")
    print(f"  Positif (1): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
    print(f"  Negatif (0): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")

    # ── Fitur Teks: TF-IDF ───────────────────────────────────────
    print(f"\n[INFO] Menjalankan TF-IDF Vectorizer...")
    print(f"  Konfigurasi: {TFIDF_CONFIG}")

    tfidf = TfidfVectorizer(**TFIDF_CONFIG)
    X_tfidf = tfidf.fit_transform(df["review_clean"])

    print(f"  Dimensi matriks TF-IDF : {X_tfidf.shape}")
    print(f"  Jumlah vocabulary      : {len(tfidf.vocabulary_)}")

    # Tampilkan 10 token dengan IDF terendah (paling umum)
    idf_scores = pd.Series(tfidf.idf_, index=tfidf.get_feature_names_out())
    print(f"  10 token IDF terendah (paling umum): "
          f"{list(idf_scores.nsmallest(10).index)}")

    # ── Fitur Numerik: Waktu Bermain ─────────────────────────────
    print(f"\n[INFO] Normalisasi fitur waktu bermain...")

    playtime_raw = df["playtime_at_review_hours"].values.reshape(-1, 1)
    scaler = StandardScaler()
    playtime_scaled = scaler.fit_transform(playtime_raw)

    print(f"  Mean   sebelum scaling: {playtime_raw.mean():.2f} jam")
    print(f"  Std    sebelum scaling: {playtime_raw.std():.2f} jam")
    print(f"  Mean   setelah scaling: {playtime_scaled.mean():.4f}")
    print(f"  Std    setelah scaling: {playtime_scaled.std():.4f}")

    # Konversi ke sparse matrix agar bisa di-hstack dengan TF-IDF
    X_playtime_sparse = sp.csr_matrix(playtime_scaled)

    # ── Penggabungan Fitur (hstack) ──────────────────────────────
    print(f"\n[INFO] Menggabungkan fitur TF-IDF + Playtime...")
    X_gabungan = sp.hstack([X_tfidf, X_playtime_sparse])
    print(f"  Dimensi matriks Baseline (TF-IDF saja) : {X_tfidf.shape}")
    print(f"  Dimensi matriks Gabungan (TF-IDF+Ptime): {X_gabungan.shape}")
    print(f"  Fitur tambahan: +1 (waktu bermain, sudah dinormalisasi)")

    # ── Simpan ke file .pkl ──────────────────────────────────────
    print(f"\n[INFO] Menyimpan objek ke file .pkl...")

    # Simpan matriks fitur dan label
    with open("features.pkl", "wb") as f:
        pickle.dump({
            "X_baseline": X_tfidf,
            "X_gabungan": X_gabungan,
            "y"         : y,
            "df_index"  : df.index.tolist(),
        }, f)

    # Simpan vectorizer (dibutuhkan untuk prediksi baru)
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    # Simpan scaler (dibutuhkan untuk prediksi baru)
    with open("scaler_playtime.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("  ✔ features.pkl")
    print("  ✔ tfidf_vectorizer.pkl")
    print("  ✔ scaler_playtime.pkl")

    print("\n✅ SELESAI. Lanjutkan ke: python 4_train_model.py")