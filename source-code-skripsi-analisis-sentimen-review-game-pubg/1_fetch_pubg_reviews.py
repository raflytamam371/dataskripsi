"""
╔══════════════════════════════════════════════════════════════════╗
║  FILE 1 — PENGUMPULAN DATA                                       ║
║  Steam Review Fetcher - PUBG (AppID: 578080)                     ║
║  Skripsi: Analisis Sentimen Review Game PUBG pada Steam          ║
║  Muhammad Rafly Badru Tamam — NIM 15220026                       ║
║  Universitas Bina Sarana Informatika, 2026                       ║
╚══════════════════════════════════════════════════════════════════╝

FUNGSI:
  Mengambil ulasan game PUBG dari Steam Web API secara otomatis,
  filter bahasa English & Indonesian, periode 1 Jan - 15 Apr 2026,
  lalu menyimpannya ke file pubg_reviews.csv.

JALANKAN:
  python 1_fetch_pubg_reviews.py

OUTPUT:
  pubg_reviews.csv  (digunakan oleh file 2_preprocessing.py)
"""

import requests
import json
import csv
import time
from datetime import datetime, timezone

# ─── Konfigurasi ─────────────────────────────────────────────────
BASE_URL   = "https://store.steampowered.com/appreviews/578080"
DATE_START = datetime(2026, 1,  1, tzinfo=timezone.utc)
DATE_END   = datetime(2026, 4, 15, 23, 59, 59, tzinfo=timezone.utc)
OUTPUT_CSV = "pubg_reviews.csv"

BASE_PARAMS = {
    "json"         : 1,
    "filter"       : "recent",
    "review_type"  : "all",
    "num_per_page" : 100,
    "purchase_type": "all",
}

CSV_COLUMNS = [
    "recommendationid",
    "language",
    "sentiment",
    "review",
    "timestamp_created",
    "playtime_at_review_hours",
    "steam_purchase",
    "voted_up",
]


# ─── Helpers ─────────────────────────────────────────────────────
def ts_to_dt(ts):
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        return None


def ts_to_str(ts):
    dt = ts_to_dt(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None


def format_review(review: dict) -> dict:
    author   = review.get("author", {})
    voted_up = review.get("voted_up", False)
    return {
        "recommendationid"       : review.get("recommendationid"),
        "language"               : review.get("language"),
        "sentiment"              : "positif" if voted_up else "negatif",
        "voted_up"               : voted_up,
        "review"                 : review.get("review", "").strip(),
        "timestamp_created"      : ts_to_str(review.get("timestamp_created")),
        "playtime_at_review_hours": round(author.get("playtime_at_review", 0) / 60, 1),
        "steam_purchase"         : review.get("steam_purchase"),
    }


def fetch_by_language(language: str) -> list[dict]:
    results, cursor, page = [], "*", 1
    stopped_early = False

    print(f"\n  [LANG={language.upper()}] Mulai fetch...")

    while True:
        params = {**BASE_PARAMS, "language": language, "cursor": cursor}

        try:
            resp = requests.get(BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"    [ERROR] {e}")
            break
        except json.JSONDecodeError:
            print("    [ERROR] Response bukan JSON valid.")
            break

        if data.get("success") != 1:
            print("    [ERROR] success != 1.")
            break

        if page == 1:
            qs = data.get("query_summary", {})
            print(f"    Total tersedia di Steam : {qs.get('total_reviews', 'N/A')} ulasan")

        reviews = data.get("reviews", [])
        if not reviews:
            print("    → Tidak ada ulasan lagi.")
            break

        batch = 0
        for r in reviews:
            created_dt = ts_to_dt(r.get("timestamp_created"))
            if created_dt and created_dt > DATE_END:
                continue
            if created_dt and created_dt < DATE_START:
                print(f"    → Melewati batas awal ({DATE_START.date()}). Berhenti.")
                stopped_early = True
                break
            results.append(format_review(r))
            batch += 1

        print(f"    [Page {page:>4}] +{batch:>3} lolos | Total: {len(results):>5}")

        if stopped_early:
            break

        next_cursor = data.get("cursor")
        if not next_cursor or next_cursor == cursor:
            print("    → Cursor tidak berubah. Selesai.")
            break

        cursor = next_cursor
        page  += 1
        time.sleep(0.3)   # sopan ke API Steam

    pos = sum(1 for r in results if r["sentiment"] == "positif")
    neg = len(results) - pos
    print(f"    Selesai: {len(results)} ulasan | 👍 {pos} | 👎 {neg}")
    return results


def save_csv(reviews: list[dict]):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reviews)
    print(f"\n[SAVED] '{OUTPUT_CSV}' — {len(reviews)} baris")


def print_summary(reviews: list[dict]):
    total    = len(reviews)
    pos      = sum(1 for r in reviews if r["sentiment"] == "positif")
    neg      = total - pos
    id_count = sum(1 for r in reviews if r["language"] == "indonesian")
    en_count = sum(1 for r in reviews if r["language"] == "english")

    print(f"\n{'='*60}")
    print(f"  RINGKASAN AKHIR")
    print(f"  Periode : {DATE_START.date()} s/d {DATE_END.date()}")
    print(f"{'='*60}")
    print(f"  Total ulasan      : {total}")
    print(f"  ├─ Indonesian     : {id_count}")
    print(f"  └─ English        : {en_count}")
    if total:
        print(f"  Sentimen")
        print(f"  ├─ 👍 Positif     : {pos} ({pos/total*100:.1f}%)")
        print(f"  └─ 👎 Negatif     : {neg} ({neg/total*100:.1f}%)")
    print(f"{'='*60}")


# ─── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PUBG Steam Review Fetcher — Dataset Skripsi SVM")
    print(f"  Periode : {DATE_START.date()} s/d {DATE_END.date()}")
    print("=" * 60)

    en_reviews = fetch_by_language("english")
    id_reviews = fetch_by_language("indonesian")
    all_reviews = en_reviews + id_reviews

    # Hapus duplikat berdasarkan recommendationid
    seen, unique = set(), []
    for r in all_reviews:
        rid = r["recommendationid"]
        if rid not in seen:
            seen.add(rid)
            unique.append(r)

    if unique:
        print_summary(unique)
        save_csv(unique)
        print("\n✅ SELESAI. Lanjutkan ke: python 2_preprocessing.py")
    else:
        print("[INFO] Tidak ada ulasan dalam periode yang ditentukan.")