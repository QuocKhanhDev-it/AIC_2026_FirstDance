"""
27_do_tra_loi_qa.py — Trả lời Q&A từ OCR/ASR đúng được bao nhiêu phần trăm?

Q&A là 42/115 câu tập dev và 3/24 gói đề mẫu, và tới giờ **chắc chắn 0 điểm** ở
cả bốn lượt nộp (A24). `src/tra_loi_ocr.py` mở đường rẻ nhất: ba câu Q&A của đề
mẫu đều hỏi **chữ hiện trên hình**, mà chữ đó đã nằm trong `ocr_asr.parquet`.

    python scripts/27_do_tra_loi_qa.py --backend gemini --so-cau 42
    python scripts/27_do_tra_loi_qa.py --backend ollama        # local, miễn phí

ĐO Ở KHUNG ĐÚNG — ĐÂY LÀ TRẦN TRÊN, KHÔNG PHẢI ĐIỂM THI
========================================================

Script chấm trên **khung đáp án của tập dev**, không phải khung do truy hồi trả
về. Tức nó trả lời đúng một câu hỏi: *"cho đúng khung rồi thì có rút ra được đáp
án không?"*

Vì sao tách bạch: điểm Q&A thi thật là **tích của hai thứ** — truy hồi ra đúng
khung VÀ đáp án đúng. Trộn hai thứ vào một con số thì lúc nó thấp sẽ không biết
phải sửa bên nào. Trần trên này chặn từ trên: nếu ở đây chỉ đúng 30% thì dù truy
hồi hoàn hảo, Q&A cũng không vượt 30%.

BA MỨC KHỚP, VÌ BTC TỰ MÂU THUẪN
=================================

Quy định ghi *"so sánh chính xác về mặt NGỮ NGHĨA"* (tr.2) và *"so sánh dưới
dạng CHUỖI CHÍNH XÁC"* (tr.8) — xem C7. Chưa hỏi được BTC thì báo cáo cả ba:

    chuỗi chính xác   khớp từng ký tự sau khi chuẩn hoá khoảng trắng
    không phân biệt   thêm: bỏ hoa/thường và dấu câu
    chứa nhau        thêm: một bên nằm trong bên kia (đáp án dài như câu thơ)

Con số thật nằm đâu đó giữa mức 1 và mức 3. Đừng chỉ báo mức dễ nhất.
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import tap_dev                                        # noqa: E402
from tra_loi_ocr import tra_loi_tu_ocr, van_ban_quanh  # noqa: E402


def chuan(s: str) -> str:
    return " ".join(str(s or "").split())


def khong_dau_hoa(s: str) -> str:
    s = unicodedata.normalize("NFC", chuan(s).lower())
    return re.sub(r"[^\w\s]", "", s).strip()


def main():
    ap = argparse.ArgumentParser(description="do do dung cua tra loi Q&A tu OCR")
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--backend", default="gemini", choices=("gemini", "ollama"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--so-khung", type=int, default=5)
    ap.add_argument("--so-cau", type=int, default=0, help="0 = tất cả")
    ap.add_argument("--in-sai", action="store_true", help="in các câu trả sai")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    bang = pd.read_parquet(a.index / "ocr_asr.parquet")
    cau = [c for c in tap_dev.doc() if c.loai == "QA" and c.dap_an]
    if a.so_cau:
        cau = cau[:a.so_cau]
    print(f"{len(cau)} câu Q&A | backend {a.backend} | "
          f"{a.so_khung} khung quanh khung đáp án\n")

    dung1 = dung2 = dung3 = rong = 0
    sai = []
    for i, c in enumerate(cau, 1):
        r = c.row_id_dung[0]
        r = r[0] if isinstance(r, list) else r
        vb = van_ban_quanh(master, bang, int(r), so_khung=a.so_khung)
        try:
            tra = tra_loi_tu_ocr(c.cau_hoi, vb, backend=a.backend, model=a.model)
        except Exception as e:
            print(f"  ⚠️  {c.id}: {type(e).__name__} {e}")
            tra = ""
        if not tra or khong_dau_hoa(tra) in ("khong ro", ""):
            rong += 1
        g, t = chuan(c.dap_an), chuan(tra)
        gk, tk = khong_dau_hoa(g), khong_dau_hoa(t)
        m1 = g == t
        m2 = gk == tk and gk != ""
        m3 = gk != "" and tk != "" and (gk in tk or tk in gk)
        dung1 += m1
        dung2 += m2
        dung3 += m3
        if not m3:
            sai.append((c.id, c.cau_hoi[:60], c.dap_an, tra))
        if i % 10 == 0:
            print(f"  {i}/{len(cau)}")

    n = len(cau)
    print(f"\n{'mức khớp':<26}{'đúng':>7}{'tỷ lệ':>10}")
    print("-" * 44)
    for ten, d in (("chuỗi chính xác", dung1), ("không phân biệt hoa/dấu câu",
                                                dung2), ("chứa nhau", dung3)):
        print(f"{ten:<26}{d:>5}/{n}{d / n * 100:>9.1f}%")
    print(f"\ntrả 'không rõ' hoặc rỗng: {rong}/{n} ({rong / n * 100:.1f}%)")
    print("\n⚠️  Đây là TRẦN TRÊN — chấm ở khung đáp án, không phải khung do "
          "truy hồi\n    trả về. Điểm Q&A thi thật là tích của hai thứ.")

    if a.in_sai and sai:
        print(f"\n{len(sai)} câu chưa khớp ở mức nào:")
        for cid, hoi, g, t in sai[:20]:
            print(f"  {cid:16} {hoi}")
            print(f"      đáp án: {g!r}")
            print(f"      trả về: {t!r}")


if __name__ == "__main__":
    main()
