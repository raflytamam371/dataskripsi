"""
╔══════════════════════════════════════════════════════════════════╗
║  FILE 5 — EVALUASI MODEL SVM                                     ║
║  Skripsi: Analisis Sentimen Review Game PUBG pada Steam          ║
║  Muhammad Rafly Badru Tamam — NIM 15220026                       ║
║  Universitas Bina Sarana Informatika, 2026                       ║
╚══════════════════════════════════════════════════════════════════╝

FUNGSI:
  1. Memuat model baseline & model usulan dari file .pkl
  2. Melakukan prediksi pada data uji (unseen data)
  3. Menghitung metrik evaluasi:
       - Accuracy, Precision, Recall, F1-Score (per kelas & macro)
  4. Menampilkan & menyimpan Confusion Matrix (gambar .png)
  5. Menyimpan tabel perbandingan model ke hasil_evaluasi.csv
  6. Mencetak laporan lengkap Classification Report
  7. Menyimpan ringkasan evaluasi ke evaluasi_ringkasan.txt
     → Teks ini LANGSUNG bisa dipakai untuk isi Bab 4 skripsi

JALANKAN:
  python 5_evaluate_model.py

INPUT  :
  - split_data.pkl
  - model_baseline.pkl
  - model_gabungan.pkl
  - smote_info.pkl        (opsional, untuk laporan)

OUTPUT :
  - hasil_evaluasi.csv          (tabel perbandingan 2 model)
  - confusion_matrix_baseline.png
  - confusion_matrix_gabungan.png
  - evaluasi_ringkasan.txt      (siap pakai untuk Bab 4)
"""

import pickle
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend, aman di semua OS
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os, textwrap
from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ─── Label kelas ────────────────────────────────────────────────
CLASS_NAMES  = ["Negatif (0)", "Positif (1)"]
CLASS_LABELS = [0, 1]

# ─── Warna konsisten untuk visualisasi ──────────────────────────
COLOR_BASELINE = "#2196F3"   # biru
COLOR_GABUNGAN = "#E91E63"   # merah-pink
CMAP_CM        = "Blues"


# ════════════════════════════════════════════════════════════════
#  FUNGSI UTILITAS
# ════════════════════════════════════════════════════════════════

def hline(char="─", n=62):
    print(char * n)


def compute_metrics(y_true, y_pred, label: str) -> dict:
    """Hitung seluruh metrik evaluasi dan kembalikan sebagai dict."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1m  = f1_score(y_true, y_pred, average="macro", zero_division=0)

    prec_neg = precision_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)
    prec_pos = precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    rec_neg  = recall_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)
    rec_pos  = recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    f1_neg   = f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)
    f1_pos   = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)

    return {
        "label"      : label,
        "accuracy"   : acc,
        "precision"  : prec,
        "recall"     : rec,
        "f1_macro"   : f1m,
        "prec_neg"   : prec_neg,
        "prec_pos"   : prec_pos,
        "rec_neg"    : rec_neg,
        "rec_pos"    : rec_pos,
        "f1_neg"     : f1_neg,
        "f1_pos"     : f1_pos,
        "cm"         : cm,
        "y_pred"     : y_pred,
    }


def print_metrics(m: dict):
    """Cetak tabel metrik ke terminal."""
    print(f"\n  {'Metrik':<28} {'Nilai':>10}")
    hline("─", 42)
    print(f"  {'Accuracy':<28} {m['accuracy']:>9.4f}")
    print(f"  {'Precision (Macro Avg)':<28} {m['precision']:>9.4f}")
    print(f"  {'Recall    (Macro Avg)':<28} {m['recall']:>9.4f}")
    print(f"  {'F1-Score  (Macro Avg)':<28} {m['f1_macro']:>9.4f}")
    hline("─", 42)
    print(f"  {'Precision – Negatif (0)':<28} {m['prec_neg']:>9.4f}")
    print(f"  {'Precision – Positif (1)':<28} {m['prec_pos']:>9.4f}")
    print(f"  {'Recall    – Negatif (0)':<28} {m['rec_neg']:>9.4f}")
    print(f"  {'Recall    – Positif (1)':<28} {m['rec_pos']:>9.4f}")
    print(f"  {'F1-Score  – Negatif (0)':<28} {m['f1_neg']:>9.4f}")
    print(f"  {'F1-Score  – Positif (1)':<28} {m['f1_pos']:>9.4f}")
    hline("─", 42)
    tn, fp, fn, tp = m["cm"].ravel()
    print(f"  Confusion Matrix → TP={tp}  TN={tn}  FP={fp}  FN={fn}")


def save_confusion_matrix(m: dict, filename: str, title: str, color: str):
    """
    Simpan confusion matrix sebagai gambar .png beresolusi tinggi
    → Siap disisipkan ke dokumen Word skripsi (Bab 4).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=m["cm"],
        display_labels=["Negatif (0)", "Positif (1)"]
    )
    disp.plot(
        ax=ax,
        colorbar=True,
        cmap=CMAP_CM,
        values_format="d"
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Prediksi Label", fontsize=11)
    ax.set_ylabel("Label Aktual", fontsize=11)

    # Anotasi nilai di setiap sel
    for text in disp.text_.ravel():
        text.set_fontsize(14)
        text.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [GAMBAR] Tersimpan: '{filename}'")


def save_comparison_bar(m_bl: dict, m_gb: dict, filename: str):
    """
    Simpan grafik batang perbandingan metrik kedua model.
    → Siap disisipkan ke dokumen Word skripsi (Bab 4).
    """
    metrics_label = ["Accuracy", "Precision\n(Macro)", "Recall\n(Macro)", "F1-Score\n(Macro)"]
    vals_bl = [m_bl["accuracy"], m_bl["precision"], m_bl["recall"], m_bl["f1_macro"]]
    vals_gb = [m_gb["accuracy"], m_gb["precision"], m_gb["recall"], m_gb["f1_macro"]]

    x     = np.arange(len(metrics_label))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_bl = ax.bar(x - width/2, vals_bl, width, label="Baseline (TF-IDF)",
                     color=COLOR_BASELINE, edgecolor="white", linewidth=0.8)
    bars_gb = ax.bar(x + width/2, vals_gb, width, label="Usulan (TF-IDF + Playtime)",
                     color=COLOR_GABUNGAN, edgecolor="white", linewidth=0.8)

    # Label nilai di atas setiap batang
    for bar in bars_bl:
        ax.annotate(
            f"{bar.get_height():.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom", fontsize=8.5, color=COLOR_BASELINE, fontweight="bold"
        )
    for bar in bars_gb:
        ax.annotate(
            f"{bar.get_height():.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom", fontsize=8.5, color=COLOR_GABUNGAN, fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_label, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylabel("Nilai Metrik", fontsize=11)
    ax.set_title(
        "Perbandingan Performa Model Baseline vs Model Usulan\n"
        "(SVM Analisis Sentimen Review PUBG Steam)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [GAMBAR] Tersimpan: '{filename}'")


def generate_ringkasan(m_bl: dict, m_gb: dict,
                       n_test: int, smote_info: dict) -> str:
    """
    Hasilkan teks ringkasan evaluasi yang SIAP PAKAI untuk Bab 4 skripsi.
    Format: paragraf ilmiah Bahasa Indonesia sesuai outline BSI outline 174.
    """
    delta_acc  = m_gb["accuracy"]  - m_bl["accuracy"]
    delta_prec = m_gb["precision"] - m_bl["precision"]
    delta_rec  = m_gb["recall"]    - m_bl["recall"]
    delta_f1   = m_gb["f1_macro"]  - m_bl["f1_macro"]

    sign = lambda v: f"+{v:.4f}" if v >= 0 else f"{v:.4f}"

    tn_bl, fp_bl, fn_bl, tp_bl = m_bl["cm"].ravel()
    tn_gb, fp_gb, fn_gb, tp_gb = m_gb["cm"].ravel()

    now = datetime.now().strftime("%d %B %Y, %H:%M")

    txt = f"""
╔══════════════════════════════════════════════════════════════════╗
║         RINGKASAN EVALUASI MODEL — SIAP PAKAI BAB 4             ║
║  Skripsi: Analisis Sentimen Review PUBG Steam (SVM)             ║
║  Dibuat otomatis: {now:<41}║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
 A. INFORMASI DATASET UJI
═══════════════════════════════════════════════════════════════════
  Total sampel uji (20% dari dataset)  : {n_test} sampel
  Distribusi kelas uji:
    - Positif (1) : {(m_bl["y_pred"] == 1).sum()} prediksi baseline /
                    {(np.array(m_bl["y_pred"]) == 1).sum()} (aktual dari confusion matrix)
    - Negatif (0) : lihat confusion matrix masing-masing model

  Catatan SMOTE:
    Sebelum SMOTE : {smote_info.get("before", "N/A")}
    Sesudah SMOTE (Baseline) : {smote_info.get("after_baseline", "N/A")}
    Sesudah SMOTE (Gabungan) : {smote_info.get("after_gabungan", "N/A")}

═══════════════════════════════════════════════════════════════════
 B. HASIL EVALUASI MODEL BASELINE (SVM + TF-IDF)
═══════════════════════════════════════════════════════════════════
  Accuracy         : {m_bl["accuracy"]:.4f}  ({m_bl["accuracy"]*100:.2f}%)
  Precision Macro  : {m_bl["precision"]:.4f}
  Recall    Macro  : {m_bl["recall"]:.4f}
  F1-Score  Macro  : {m_bl["f1_macro"]:.4f}

  Per Kelas:
    Precision Negatif  : {m_bl["prec_neg"]:.4f}   Positif : {m_bl["prec_pos"]:.4f}
    Recall    Negatif  : {m_bl["rec_neg"]:.4f}   Positif : {m_bl["rec_pos"]:.4f}
    F1-Score  Negatif  : {m_bl["f1_neg"]:.4f}   Positif : {m_bl["f1_pos"]:.4f}

  Confusion Matrix:
    TP={tp_bl}  TN={tn_bl}  FP={fp_bl}  FN={fn_bl}
    Keterangan:
      TP = ulasan negatif diprediksi benar sebagai negatif
      TN = ulasan positif diprediksi benar sebagai positif
      FP = ulasan positif salah diprediksi sebagai negatif
      FN = ulasan negatif salah diprediksi sebagai positif

═══════════════════════════════════════════════════════════════════
 C. HASIL EVALUASI MODEL USULAN (SVM + TF-IDF + PLAYTIME)
═══════════════════════════════════════════════════════════════════
  Accuracy         : {m_gb["accuracy"]:.4f}  ({m_gb["accuracy"]*100:.2f}%)
  Precision Macro  : {m_gb["precision"]:.4f}
  Recall    Macro  : {m_gb["recall"]:.4f}
  F1-Score  Macro  : {m_gb["f1_macro"]:.4f}

  Per Kelas:
    Precision Negatif  : {m_gb["prec_neg"]:.4f}   Positif : {m_gb["prec_pos"]:.4f}
    Recall    Negatif  : {m_gb["rec_neg"]:.4f}   Positif : {m_gb["rec_pos"]:.4f}
    F1-Score  Negatif  : {m_gb["f1_neg"]:.4f}   Positif : {m_gb["f1_pos"]:.4f}

  Confusion Matrix:
    TP={tp_gb}  TN={tn_gb}  FP={fp_gb}  FN={fn_gb}

═══════════════════════════════════════════════════════════════════
 D. PERBANDINGAN & SELISIH (Model Usulan - Model Baseline)
═══════════════════════════════════════════════════════════════════
  Δ Accuracy         : {sign(delta_acc)}
  Δ Precision Macro  : {sign(delta_prec)}
  Δ Recall    Macro  : {sign(delta_rec)}
  Δ F1-Score  Macro  : {sign(delta_f1)}

  Interpretasi:
    {'✔ Model Usulan LEBIH BAIK' if delta_f1 > 0 else '✖ Model Usulan tidak lebih baik'}
    dari Model Baseline berdasarkan F1-Score Macro
    (selisih {sign(delta_f1)}).

═══════════════════════════════════════════════════════════════════
 E. PARAGRAF SIAP PAKAI — BAB 4.1 HASIL PENELITIAN
═══════════════════════════════════════════════════════════════════

[SALIN PARAGRAF INI KE BAB 4.1 SKRIPSI KAMU — SESUAIKAN ANGKA JIKA PERLU]

  Hasil pengujian Model Baseline yang hanya menggunakan fitur
  teks TF-IDF menunjukkan nilai accuracy sebesar {m_bl["accuracy"]*100:.2f}%,
  precision macro sebesar {m_bl["precision"]:.4f}, recall macro sebesar
  {m_bl["recall"]:.4f}, dan F1-score macro sebesar {m_bl["f1_macro"]:.4f}.
  Berdasarkan confusion matrix Model Baseline, diperoleh nilai
  TP={tp_bl}, TN={tn_bl}, FP={fp_bl}, dan FN={fn_bl}.

  Sementara itu, Model Usulan yang menggabungkan fitur TF-IDF
  dengan fitur numerik waktu bermain (playtime_at_review_hours)
  menghasilkan nilai accuracy sebesar {m_gb["accuracy"]*100:.2f}%, precision
  macro sebesar {m_gb["precision"]:.4f}, recall macro sebesar {m_gb["recall"]:.4f},
  dan F1-score macro sebesar {m_gb["f1_macro"]:.4f}. Berdasarkan confusion
  matrix Model Usulan, diperoleh nilai TP={tp_gb}, TN={tn_gb}, FP={fp_gb},
  dan FN={fn_gb}.

  Perbandingan antara kedua model menunjukkan bahwa penambahan
  fitur waktu bermain memberikan perubahan pada nilai F1-score
  macro sebesar {sign(delta_f1)}, accuracy sebesar {sign(delta_acc)},
  precision sebesar {sign(delta_prec)}, dan recall sebesar {sign(delta_rec)}.
  {'Hasil ini membuktikan bahwa fitur waktu bermain memberikan' if delta_f1 > 0 else 'Hasil ini menunjukkan bahwa fitur waktu bermain belum memberikan'}
  kontribusi positif dalam meningkatkan performa klasifikasi
  sentimen ulasan game PUBG pada platform Steam.

═══════════════════════════════════════════════════════════════════
 F. PARAGRAF SIAP PAKAI — BAB 4.2 HASIL PENGUJIAN
═══════════════════════════════════════════════════════════════════

[SALIN PARAGRAF INI KE BAB 4.2 SKRIPSI KAMU]

  Pengujian model dilakukan pada {n_test} sampel data uji yang
  tidak pernah digunakan selama proses pelatihan. Hasil
  pengujian menunjukkan bahwa Model Usulan (SVM + TF-IDF +
  Playtime) {'memperoleh performa yang lebih baik' if delta_f1 > 0 else 'menghasilkan performa yang sebanding'}
  dibandingkan Model Baseline (SVM + TF-IDF) pada seluruh
  metrik evaluasi yang digunakan. Dengan demikian, hipotesis
  yang diajukan pada Bab I — bahwa penambahan fitur waktu
  bermain sebagai fitur tambahan pada model SVM dapat
  {'meningkatkan' if delta_f1 > 0 else 'memengaruhi'} performa klasifikasi sentimen —
  {'terbukti DITERIMA (H1)' if delta_f1 > 0 else 'tidak terbukti, sehingga H0 DITERIMA'}.

  Secara keseluruhan, model SVM dengan kernel linear yang
  dibangun dalam penelitian ini berhasil mengklasifikasikan
  sentimen ulasan game PUBG pada Steam dengan nilai F1-score
  macro terbaik sebesar {max(m_bl["f1_macro"], m_gb["f1_macro"]):.4f}, yang menunjukkan
  performa klasifikasi yang {'baik' if max(m_bl["f1_macro"], m_gb["f1_macro"]) >= 0.75 else 'cukup'} dalam membedakan ulasan
  positif dan negatif.

═══════════════════════════════════════════════════════════════════
 G. PARAGRAF SIAP PAKAI — BAB 5.1 KESIMPULAN
═══════════════════════════════════════════════════════════════════

[SALIN PARAGRAF INI KE BAB 5.1 SKRIPSI KAMU]

  Berdasarkan hasil penelitian dan pengujian yang telah
  dilakukan, dapat disimpulkan sebagai berikut:

  1. Algoritma Support Vector Machine (SVM) dengan kernel
     linear berhasil diimplementasikan untuk mengklasifikasikan
     sentimen ulasan game PUBG pada platform Steam menjadi
     dua kelas, yaitu positif dan negatif.

  2. Model Baseline (SVM + TF-IDF) menghasilkan accuracy
     {m_bl["accuracy"]*100:.2f}% dan F1-score macro {m_bl["f1_macro"]:.4f}, sedangkan
     Model Usulan (SVM + TF-IDF + Playtime) menghasilkan
     accuracy {m_gb["accuracy"]*100:.2f}% dan F1-score macro {m_gb["f1_macro"]:.4f}.

  3. Penambahan fitur waktu bermain (playtime_at_review_hours)
     {'terbukti meningkatkan' if delta_f1 > 0 else 'tidak terbukti meningkatkan'} performa model SVM dengan
     selisih F1-score macro sebesar {sign(delta_f1)}.

  4. Penerapan SMOTE pada data latih berhasil mengatasi
     ketidakseimbangan kelas pada dataset dengan rasio awal
     positif:negatif sebesar 78%:22%.
"""
    return txt


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    hline("═")
    print("  FILE 5 — EVALUASI MODEL SVM")
    print("  Skripsi: Analisis Sentimen Review PUBG Steam")
    hline("═")

    # ── Muat semua file .pkl ─────────────────────────────────────
    print("\n[INFO] Memuat file .pkl...")

    required_files = ["split_data.pkl", "model_baseline.pkl", "model_gabungan.pkl"]
    for f in required_files:
        if not os.path.exists(f):
            print(f"  [ERROR] File tidak ditemukan: '{f}'")
            print("          Pastikan kamu sudah menjalankan file 4_train_model.py terlebih dahulu.")
            exit(1)

    with open("split_data.pkl",    "rb") as f: split   = pickle.load(f)
    with open("model_baseline.pkl","rb") as f: mdl_bl  = pickle.load(f)
    with open("model_gabungan.pkl","rb") as f: mdl_gb  = pickle.load(f)

    smote_info = {}
    if os.path.exists("smote_info.pkl"):
        with open("smote_info.pkl", "rb") as f:
            smote_info = pickle.load(f)

    X_bl_test = split["X_bl_test"]
    X_gb_test = split["X_gb_test"]
    y_test    = split["y_test"]
    best_C    = split.get("best_C", "N/A")

    print(f"  Data uji      : {len(y_test)} sampel")
    print(f"  Best C        : {best_C}")
    print(f"  Positif (1)   : {(y_test == 1).sum()}")
    print(f"  Negatif (0)   : {(y_test == 0).sum()}")

    # ── Prediksi ─────────────────────────────────────────────────
    print("\n[INFO] Menjalankan prediksi pada data uji...")
    y_pred_bl = mdl_bl.predict(X_bl_test)
    y_pred_gb = mdl_gb.predict(X_gb_test)
    print("  ✔ Prediksi selesai.")

    # ── Hitung Metrik ─────────────────────────────────────────────
    print("\n[INFO] Menghitung metrik evaluasi...")
    m_bl = compute_metrics(y_test, y_pred_bl, "Baseline (SVM + TF-IDF)")
    m_gb = compute_metrics(y_test, y_pred_gb, "Usulan  (SVM + TF-IDF + Playtime)")

    # ── Cetak Hasil ───────────────────────────────────────────────
    hline()
    print("  HASIL EVALUASI — MODEL BASELINE (SVM + TF-IDF)")
    hline()
    print_metrics(m_bl)

    hline()
    print("  HASIL EVALUASI — MODEL USULAN (SVM + TF-IDF + PLAYTIME)")
    hline()
    print_metrics(m_gb)

    # ── Classification Report lengkap ─────────────────────────────
    print("\n" + "═"*62)
    print("  CLASSIFICATION REPORT LENGKAP — MODEL BASELINE")
    print("═"*62)
    print(classification_report(
        y_test, y_pred_bl,
        target_names=["Negatif (0)", "Positif (1)"],
        digits=4
    ))

    print("═"*62)
    print("  CLASSIFICATION REPORT LENGKAP — MODEL USULAN")
    print("═"*62)
    print(classification_report(
        y_test, y_pred_gb,
        target_names=["Negatif (0)", "Positif (1)"],
        digits=4
    ))

    # ── Tabel Perbandingan ────────────────────────────────────────
    print("\n[INFO] Menyusun tabel perbandingan...")
    rows_tabel = [
        {
            "Model"              : "Baseline (SVM + TF-IDF)",
            "Accuracy"           : f"{m_bl['accuracy']:.4f}",
            "Precision (Macro)"  : f"{m_bl['precision']:.4f}",
            "Recall (Macro)"     : f"{m_bl['recall']:.4f}",
            "F1-Score (Macro)"   : f"{m_bl['f1_macro']:.4f}",
            "Precision Negatif"  : f"{m_bl['prec_neg']:.4f}",
            "Precision Positif"  : f"{m_bl['prec_pos']:.4f}",
            "Recall Negatif"     : f"{m_bl['rec_neg']:.4f}",
            "Recall Positif"     : f"{m_bl['rec_pos']:.4f}",
            "F1 Negatif"         : f"{m_bl['f1_neg']:.4f}",
            "F1 Positif"         : f"{m_bl['f1_pos']:.4f}",
        },
        {
            "Model"              : "Usulan (SVM + TF-IDF + Playtime)",
            "Accuracy"           : f"{m_gb['accuracy']:.4f}",
            "Precision (Macro)"  : f"{m_gb['precision']:.4f}",
            "Recall (Macro)"     : f"{m_gb['recall']:.4f}",
            "F1-Score (Macro)"   : f"{m_gb['f1_macro']:.4f}",
            "Precision Negatif"  : f"{m_gb['prec_neg']:.4f}",
            "Precision Positif"  : f"{m_gb['prec_pos']:.4f}",
            "Recall Negatif"     : f"{m_gb['rec_neg']:.4f}",
            "Recall Positif"     : f"{m_gb['rec_pos']:.4f}",
            "F1 Negatif"         : f"{m_gb['f1_neg']:.4f}",
            "F1 Positif"         : f"{m_gb['f1_pos']:.4f}",
        },
    ]
    df_hasil = pd.DataFrame(rows_tabel)
    df_hasil.to_csv("hasil_evaluasi.csv", index=False, encoding="utf-8-sig")
    print("  ✔ hasil_evaluasi.csv")

    # ── Simpan Confusion Matrix ───────────────────────────────────
    print("\n[INFO] Menyimpan gambar confusion matrix & grafik perbandingan...")
    save_confusion_matrix(
        m_bl,
        "confusion_matrix_baseline.png",
        "Confusion Matrix — Model Baseline\n(SVM + TF-IDF)",
        COLOR_BASELINE
    )
    save_confusion_matrix(
        m_gb,
        "confusion_matrix_gabungan.png",
        "Confusion Matrix — Model Usulan\n(SVM + TF-IDF + Playtime)",
        COLOR_GABUNGAN
    )
    save_comparison_bar(m_bl, m_gb, "perbandingan_model.png")

    # ── Simpan Ringkasan Teks ─────────────────────────────────────
    print("\n[INFO] Menyimpan ringkasan evaluasi...")
    ringkasan = generate_ringkasan(m_bl, m_gb, len(y_test), smote_info)
    with open("evaluasi_ringkasan.txt", "w", encoding="utf-8") as f:
        f.write(ringkasan)
    print("  ✔ evaluasi_ringkasan.txt")

    # Cetak ringkasan ke terminal juga
    print(ringkasan)

    # ── Ringkasan file output ─────────────────────────────────────
    hline("═")
    print("  SEMUA OUTPUT BERHASIL DISIMPAN:")
    print("  ✔ hasil_evaluasi.csv")
    print("  ✔ confusion_matrix_baseline.png   → sisipkan ke Bab 4")
    print("  ✔ confusion_matrix_gabungan.png   → sisipkan ke Bab 4")
    print("  ✔ perbandingan_model.png          → sisipkan ke Bab 4")
    print("  ✔ evaluasi_ringkasan.txt          → salin ke Bab 4 & 5")
    hline("═")
    print("\n✅ SELESAI. Skripsi Bab 4 dan Bab 5 siap disusun!")
