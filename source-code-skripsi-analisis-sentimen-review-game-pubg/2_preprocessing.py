"""
╔══════════════════════════════════════════════════════════════════╗
║  FILE 2 — PRA-PEMROSESAN TEKS (TEXT PREPROCESSING)              ║
║  Skripsi: Analisis Sentimen Review Game PUBG pada Steam          ║
║  Muhammad Rafly Badru Tamam — NIM 15220026                       ║
║  Universitas Bina Sarana Informatika, 2026                       ║
╚══════════════════════════════════════════════════════════════════╝

FUNGSI:
  Membaca pubg_reviews.csv, lalu menjalankan 6 tahapan preprocessing:
    1. Case Folding
    2. Pembersihan Simbol & Karakter Khusus
    3. Normalisasi Bahasa Informal / Slang Gamer
    4. Tokenisasi
    5. Penghapusan Stopword (EN + ID)
    6. Stemming (Porter Stemmer EN + Sastrawi ID)

JALANKAN:
  python 2_preprocessing.py

INPUT  : pubg_reviews.csv
OUTPUT : pubg_reviews_clean.csv  (digunakan oleh 3_feature_extraction.py)
"""

import re
import csv
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords
from nltk.stem     import PorterStemmer

# Download resource NLTK (hanya perlu sekali)
nltk.download("punkt",            quiet=True)
nltk.download("punkt_tab",        quiet=True)
nltk.download("stopwords",        quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# Sastrawi untuk Bahasa Indonesia
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    _sastrawi_ok = True
except ImportError:
    print("[PERINGATAN] Sastrawi tidak terinstall. "
          "Jalankan: pip install PySastrawi\n"
          "Stemming & stopword Bahasa Indonesia akan dilewati.")
    _sastrawi_ok = False

# ─── Inisialisasi stemmer & stopword ─────────────────────────────
porter  = PorterStemmer()
sw_en   = set(stopwords.words("english"))

if _sastrawi_ok:
    sw_factory  = StopWordRemoverFactory()
    sw_id_list  = set(sw_factory.get_stop_words())
    stem_factory = StemmerFactory()
    id_stemmer   = stem_factory.create_stemmer()
else:
    sw_id_list = set()
    id_stemmer = None

# ─── Kamus Normalisasi Slang ──────────────────────────────────────
# Format: {"slang / singkatan": "bentuk baku"}
SLANG_DICT = {
    # Slang game / komunitas
    "fps"      : "frame per second",
    "sbmm"     : "skill based matchmaking",
    "br"       : "battle royale",
    "bg"       : "battlegrounds",
    "pubg"     : "playerunknowns battlegrounds",
    "op"       : "overpowered",
    "nerf"     : "melemahkan",
    "buff"     : "menguatkan",
    "tp"       : "teleport",
    "hp"       : "health point",
    "dmg"      : "damage",
    "rng"      : "random number generator",
    "dc"       : "disconnect",
    "afk"      : "away from keyboard",
    "gg"       : "good game",
    "wp"       : "well played",
    "noob"     : "pemula",
    "bot"      : "pemain bot",
    "hackers"  : "hacker",
    "cheat"    : "kecurangan",
    "lag"      : "latensi tinggi",
    "lagging"  : "latensi tinggi",
    "laggy"    : "latensi tinggi",
    "ping"     : "latensi jaringan",
    "crash"    : "error aplikasi",
    "crashing" : "error aplikasi",
    "update"   : "pembaruan",
    "patch"    : "pembaruan",
    "dlc"      : "konten tambahan",
    "p2w"      : "pay to win",
    "f2p"      : "free to play",
    "pvp"      : "player versus player",
    "pve"      : "player versus environment",
    "squad"    : "tim",
    "duo"      : "dua pemain",
    "solo"     : "satu pemain",
    "loot"     : "item rampasan",
    "looting"  : "mengambil item",
    "airdrop"  : "kiriman udara",
    "zone"     : "zona aman",
    "circle"   : "lingkaran aman",
    "camper"   : "pemain bersembunyi",
    "camping"  : "bersembunyi",
    "spray"    : "menembak otomatis",
    "sniper"   : "penembak jitu",
    "rush"     : "menyerang cepat",
    "rushing"  : "menyerang cepat",
    "knock"    : "menjatuhkan",
    "downed"   : "terjatuh",
    "revive"   : "menghidupkan kembali",
    "matchmaking": "pencarian lawan",
    "ranked"   : "peringkat",
    "unranked" : "tidak berperingkat",
    # Slang bahasa Indonesia informal
    "gue"      : "saya",
    "gw"       : "saya",
    "lo"       : "kamu",
    "lu"       : "kamu",
    "yg"       : "yang",
    "dgn"      : "dengan",
    "utk"      : "untuk",
    "tdk"      : "tidak",
    "ga"       : "tidak",
    "gak"      : "tidak",
    "nggak"    : "tidak",
    "ngga"     : "tidak",
    "bgt"      : "banget",
    "bgt"      : "sangat",
    "bngt"     : "banget",
    "emg"      : "memang",
    "emang"    : "memang",
    "kayak"    : "seperti",
    "kaya"     : "seperti",
    "udah"     : "sudah",
    "udh"      : "sudah",
    "sdh"      : "sudah",
    "blm"      : "belum",
    "belom"    : "belum",
    "jgn"      : "jangan",
    "dpt"      : "dapat",
    "skrg"     : "sekarang",
    "trus"     : "terus",
    "terus"    : "terus",
    "abis"     : "habis",
    "msh"      : "masih",
    "org"      : "orang",
    "hrs"      : "harus",
    "sampe"    : "sampai",
    "gimana"   : "bagaimana",
    "gmn"      : "bagaimana",
    "knp"      : "kenapa",
    "krn"      : "karena",
    "karna"    : "karena",
    "soalnya"  : "karena",
    "tp"       : "tetapi",
    "tapi"     : "tetapi",
    "aja"      : "saja",
    "doang"    : "saja",
    "sih"      : "",       # partikel → hapus
    "deh"      : "",
    "dong"     : "",
    "lah"      : "",
    "kah"      : "",
    "pake"     : "pakai",
    "pkai"     : "pakai",
    "mulu"     : "terus",
    "bener"    : "benar",
    "bnr"      : "benar",
    "bner"     : "benar",
    "makin"    : "semakin",
    "bikin"    : "membuat",
    "ngerti"   : "mengerti",
    "ngriti"   : "mengerti",
    "pengen"   : "ingin",
    "pgn"      : "ingin",
    "nyoba"    : "mencoba",
    "nyobain"  : "mencoba",
    "cobain"   : "mencoba",
    "mantap"   : "bagus",
    "mantul"   : "bagus",
    "asik"     : "menyenangkan",
    "seru"     : "menyenangkan",
    "jelek"    : "buruk",
    "parah"    : "buruk",
    "ancur"    : "hancur",
    "sampah"   : "buruk",
    "bagus"    : "bagus",
}


# ─── Fungsi Preprocessing ────────────────────────────────────────
def case_folding(text: str) -> str:
    """Tahap 1: Ubah seluruh teks ke huruf kecil."""
    return text.lower()


def clean_text(text: str) -> str:
    """
    Tahap 2: Bersihkan simbol dan karakter khusus.
    - Hapus URL
    - Hapus simbol sensor Steam (♥♥♥♥♥)
    - Hapus tanda baca, angka, dan karakter non-alfanumerik
    - Hapus spasi ganda
    """
    text = re.sub(r"http\S+|www\S+", "", text)           # URL
    text = re.sub(r"[♥♦♣♠★☆©®™]", "", text)             # simbol sensor Steam & unicode umum
    text = re.sub(r"[^a-z\s]", " ", text)                # hanya huruf & spasi (angka ikut hilang)
    text = re.sub(r"\s+", " ", text).strip()             # normalisasi spasi
    return text


def normalize_slang(text: str) -> str:
    """
    Tahap 3: Normalisasi bahasa informal dan slang gamer
    menggunakan kamus SLANG_DICT.
    """
    tokens = text.split()
    normalized = []
    for token in tokens:
        replacement = SLANG_DICT.get(token, token)
        if replacement:                   # jika replacement kosong, token dihapus
            normalized.append(replacement)
    return " ".join(normalized)


def tokenize(text: str) -> list[str]:
    """Tahap 4: Tokenisasi menggunakan NLTK word_tokenize."""
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Tahap 5: Hapus stopword Bahasa Inggris (NLTK)
    dan Bahasa Indonesia (Sastrawi).
    """
    combined_sw = sw_en | sw_id_list
    return [t for t in tokens if t not in combined_sw and len(t) > 1]


def stem_tokens(tokens: list[str]) -> list[str]:
    """
    Tahap 6: Stemming.
    - Porter Stemmer untuk token Bahasa Inggris
    - Sastrawi Stemmer untuk token Bahasa Indonesia
    Deteksi bahasa: jika token ada di stopword Bahasa Indonesia
    atau kamus Sastrawi, gunakan Sastrawi; otherwise Porter.
    Catatan sederhana: kita stem semua dengan Porter dulu,
    lalu jika ada Sastrawi, re-stem token yang berakhiran
    imbuhan Indonesia (me-, di-, ke-, -kan, -an, -i).
    """
    stemmed = []
    id_affixes = re.compile(r"^(me|di|ke|pe|ber|ter|se)|(\bkan\b|\ban\b|\bi\b)$")

    for token in tokens:
        if _sastrawi_ok and re.search(r"(me|di|ke|pe|ber|ter|se).+|.+(kan|an|nya)$", token):
            stemmed.append(id_stemmer.stem(token))
        else:
            stemmed.append(porter.stem(token))
    return stemmed


def full_preprocess(text: str) -> str:
    """Pipeline lengkap preprocessing satu ulasan."""
    text   = case_folding(text)
    text   = clean_text(text)
    text   = normalize_slang(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return " ".join(tokens)


# ─── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    INPUT_CSV  = "pubg_reviews.csv"
    OUTPUT_CSV = "pubg_reviews_clean.csv"

    print("=" * 60)
    print("  FILE 2 — PRA-PEMROSESAN TEKS")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"[INFO] Dataset dimuat: {len(df)} baris")
    print(f"[INFO] Distribusi sentimen:\n{df['sentiment'].value_counts().to_string()}\n")

    # Hapus baris dengan review kosong / NaN
    before = len(df)
    df.dropna(subset=["review"], inplace=True)
    df = df[df["review"].str.strip() != ""]
    print(f"[INFO] Baris setelah hapus review kosong: {len(df)} (dihapus: {before - len(df)})")

    # Jalankan preprocessing
    print("[INFO] Memulai preprocessing... (harap tunggu)")
    df["review_clean"] = df["review"].astype(str).apply(full_preprocess)

    # Hapus ulasan yang setelah diproses menjadi kosong
    before2 = len(df)
    df = df[df["review_clean"].str.strip() != ""]
    print(f"[INFO] Baris setelah hapus review kosong post-preprocessing: {len(df)} (dihapus: {before2 - len(df)})")

    # Simpan
    cols_out = [
        "recommendationid", "language", "sentiment",
        "review", "review_clean",
        "playtime_at_review_hours", "timestamp_created"
    ]
    df[cols_out].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n[SAVED] '{OUTPUT_CSV}' — {len(df)} baris")
    print(f"\n[CONTOH HASIL PREPROCESSING]")
    sample = df[["review", "review_clean"]].sample(3, random_state=42)
    for _, row in sample.iterrows():
        print(f"  ASLI   : {row['review'][:80]}")
        print(f"  BERSIH : {row['review_clean'][:80]}")
        print()

    print("✅ SELESAI. Lanjutkan ke: python 3_feature_extraction.py")