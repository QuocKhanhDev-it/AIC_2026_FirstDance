"""
123_san_cau_kho.py — Săn một câu BÍ bằng ba tín hiệu độc lập, cạnh nhau.

    python scripts/123_san_cau_kho.py --de dev/SOTUYEN3-bo-de-thi --goi query-p2-11-kis
    python scripts/123_san_cau_kho.py --de ... --goi ... --them "sợi trắng nở phồng"

VÌ SAO CẦN, VÀ NÓ KHÁC UI Ở CHỖ NÀO

UI cho xem **bể ứng viên của bài nộp** — tức kênh 1 + kênh 3 đã hợp nhất, đúng
cấu hình sẽ nộp. Đó là thứ cần khi soát bài. Nhưng khi một câu **không ra gì**
thì việc cần làm khác hẳn: tìm xem *có nguồn nào trong kho biết về cảnh này
không*, kể cả nguồn không dùng để nộp.

Script này bày ba tín hiệu **độc lập** cạnh nhau:

    kênh 1  — ảnh (SigLIP2). Thứ bài nộp dùng.
    OCR/ASR — chữ trên màn hình + lời thoại. Bắt được TÊN MÓN, TÊN ĐỊA DANH.
    caption — mô tả HÌNH ẢNH do VLM sinh, phủ 100% kho.

**Caption là thứ UI không có.** Kênh 5 bị bác khỏi bài nộp (A73/A90: ✅ tệ hơn
khi hợp nhất), nhưng bác nó với vai trò *kênh xếp hạng* không có nghĩa là nó vô
dụng với vai trò *công cụ tra cứu*. Truy vấn KIS mô tả **hình ảnh**, mà caption
cũng mô tả **hình ảnh** — hai thứ cùng một ngôn ngữ, nên tra thẳng thường ra
những thứ kênh 1 bỏ sót.

⚠️ `--them` để thử cách diễn đạt KHÁC cho hai kênh văn bản. Kênh 1 thì KHÔNG
thử được cách diễn đạt mới trên máy này — nó chạy bằng vector mã hoá sẵn, mà
mã hoá câu mới thì phải nạp model (máy 7,7 GB không nạp nổi). Đó là lý do hai
kênh văn bản đáng giá gấp đôi khi đang bí.

⚠️ RAM: dựng BM25 trên 177.321 tài liệu tốn đáng kể. Mặc định chỉ dựng caption
(nhẹ hơn, trung vị 280 ký tự); thêm `--ocr` nếu còn RAM.
"""

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from rrf import hop_nhat                              # noqa: E402


def bang(ten, uv, master, cap, van, n):
    print(f"\n{'=' * 78}\n{ten}\n{'=' * 78}")
    if not uv:
        print("  (không có gì)")
        return
    for i, c in enumerate(uv[:n], 1):
        r = c.row_id
        print(f"{i:>2}. {c.video_id}  f{c.frame_idx:<7} "
              f"{master.pts_time.iloc[r]:>6.0f}s  row {r}")
        if cap is not None:
            print(f"     ẢNH: {cap.get(r, '')[:150]}")
        t = str(van.get(r, "")).strip()
        if t:
            print(f"     CHỮ: {t[:130]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--de", type=Path, required=True)
    ap.add_argument("--goi", required=True)
    ap.add_argument("--them", action="append", default=[],
                    help="cách diễn đạt khác, thử trên hai kênh văn bản")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--ocr", action="store_true", help="dựng thêm BM25 OCR/ASR")
    ap.add_argument("--khong-anh", action="store_true")
    a = ap.parse_args()

    f = a.de / f"{a.goi}.txt"
    if not f.exists():
        raise SystemExit(f"❌ không có {f}")
    q = f.read_text("utf-8").strip()
    master = pd.read_parquet(a.index / "master.parquet")

    print(f"\n{a.goi}  ({len(q.split())} từ)\n{q}\n")
    me = R.tach_truy_van(q)
    print(f"tách thành {len(me)} mệnh đề:")
    for x in me:
        # Mệnh đề quá ngắn là NHIỄU có phiếu ngang: A51 hợp nhất mệnh đề bằng
        # RRF hạng, nên một mảnh 3 từ vô nghĩa vẫn bỏ phiếu bằng mệnh đề 40 từ.
        canh = "   ⚠️ QUÁ NGẮN — phiếu ngang mệnh đề chính" if len(x.split()) <= 4 else ""
        print(f"  ({len(x.split()):>2} từ) {x[:110]}{canh}")

    van = {}
    p = a.index / "ocr_asr.parquet"
    if p.exists():
        b = pd.read_parquet(p, columns=["row_id", "text"])
        van = dict(zip(b.row_id.values, b.text.fillna("").values))
        del b
        gc.collect()

    cap_df = pd.read_parquet(a.index / "caption.parquet")
    cap = dict(zip(cap_df.row_id.values, cap_df.caption.fillna("").values))

    if not a.khong_anh:
        from dense import KenhAnhCache
        k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                          matrix="clip_gopt.npy")
        thieu = k1.co_du(me)
        if thieu:
            print(f"\n⚠️ {len(thieu)} mệnh đề CHƯA mã hoá — kênh 1 bỏ qua câu này")
        else:
            uv = hop_nhat([k1.tim(m, k=200) for m in me]) if len(me) > 1 \
                else k1.tim(me[0], k=200)
            bang("KÊNH 1 — ẢNH (thứ bài nộp dùng)", uv, master, cap, van, a.n)
        del k1
        gc.collect()

    print("\ndựng BM25 trên caption…", flush=True)
    k5 = KenhVanBan.tu_bang_khung(master, cap_df, cot="caption", ten="caption")
    del cap_df
    gc.collect()
    for t in [q] + a.them:
        bang(f"CAPTION — mô tả HÌNH ẢNH · «{t[:60]}»",
             k5.tim(t, k=a.n), master, cap, van, a.n)
    del k5
    gc.collect()

    if a.ocr and p.exists():
        print("\ndựng BM25 trên OCR/ASR…", flush=True)
        k3 = KenhVanBan.tu_bang_khung(
            master, pd.read_parquet(p), cot="text", ten="ocr_asr")
        for t in [q] + a.them:
            bang(f"OCR/ASR — chữ + lời thoại · «{t[:60]}»",
                 k3.tim(t, k=a.n), master, cap, van, a.n)

    print("\n" + "-" * 78)
    print("ĐỌC BẢNG: video nào xuất hiện ở HAI tín hiệu trở lên thì đáng mở UI")
    print("xem trước. Ba tín hiệu này độc lập nhau, nên trùng lặp là bằng chứng")
    print("thật, không phải một kênh tự khẳng định mình.")


if __name__ == "__main__":
    main()
