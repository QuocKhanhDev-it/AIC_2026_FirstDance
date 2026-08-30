"""
37_do_neo_mo_rong.py — "Neo bằng từ hiếm rồi đi bộ thời gian" (Anchor & Expand).

CÂU HỎI ĐANG ĐO
===============

Hai ca khó nhất nhóm giải được bằng tay đều đi cùng một đường, và **không phải
đường của SigLIP2**:

* `p1-12-kis` — OCR bắt từ hiếm *"mazut"* → khoanh còn 13 khung → soi ra 4 tài xế
* `p1-17-qa`  — bám *"sạt lở" + "đèo"* → còn 10 video → đọc lời bình → ra Tà Pứa

Cả hai: **mỏ neo văn bản trước, thị giác sau.** Ghi chép của nhóm còn rút ra
*"chuỗi suy luận NGẮN thì ít lệch"* — bám thẳng mỏ neo ngắn hơn hẳn quét ngữ
nghĩa mù mờ toàn kho.

Script này hỏi: quy đường đó ra thuật toán thì được mấy điểm.

    python scripts/37_do_neo_mo_rong.py --cache index/truy_van.npz

HAI BIẾN THỂ, VÀ VÌ SAO PHẢI ĐO CẢ HAI
=======================================

A27/A28 đo được một luật rất chặt:

> **Kênh yếu chỉ được XẾP LẠI thứ kênh mạnh đã chọn — không được THÊM ứng viên
> mới, không được THAY thế.** Xếp lại trong bể: **+1,6 điểm**. Thay bằng ứng
> viên mới: **−0,4 điểm**.

Nhưng cách người soát tay làm ở `p1-12` là **THÊM** — họ lấy khung OCR khớp
*mazut* rồi soi, chứ không xếp lại danh sách SigLIP2. Hai thứ mâu thuẫn nhau,
nên đo cả hai chứ không chọn trước:

| biến thể | làm gì | quan hệ với A27/A28 |
| --- | --- | --- |
| `xep_lai` | chỉ **đảo thứ tự** trong top-100 của SigLIP2 | tuân thủ |
| `chen` | **chèn** khung neo lên đầu, kể cả khung SigLIP2 không có | vi phạm |

Nếu `chen` thắng thì luật A27/A28 có ngoại lệ, và ngoại lệ đó là **mỏ neo có
độ chính xác cao** — khác hẳn "trộn một kênh yếu vào".

MỎ NEO LÀ GÌ — ĐỊNH NGHĨA ĐO ĐƯỢC, KHÔNG PHẢI CẢM TÍNH
=======================================================

Một token là mỏ neo khi nó **xuất hiện trong kho OCR nhưng rất hiếm**: tần suất
tài liệu `1 <= df <= --nguong-df`. Có mặt (df≥1) nên tra được; hiếm (df nhỏ) nên
tra ra ít và sắc.

Cố ý KHÔNG dùng danh sách thực thể/địa danh soạn tay: danh sách đó phải bảo trì,
và nó là chỗ dễ vá theo đề mẫu rồi hụt trên đề thật — đúng cái bẫy A40.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                        # noqa: E402
import tap_dev                                         # noqa: E402
from bm25 import KenhVanBan, bo_dau, tach              # noqa: E402
from cham_diem import MOC_DUNG_SAI, bao_cao_do_nhay    # noqa: E402
from dense import KenhAnhCache                         # noqa: E402
from schema import Candidate                           # noqa: E402


def tim_neo(cau: str, kenh: KenhVanBan, nguong_df: int) -> list[tuple[str, int]]:
    """Token của truy vấn vừa CÓ trong kho OCR vừa HIẾM. `[(token, df), ...]`."""
    ra = {}
    for chi_muc, c in ((kenh.co_dau.chi_muc, cau),
                       (kenh.khong_dau.chi_muc, bo_dau(cau))):
        for t in set(tach(c, bigram=False)):
            hit = chi_muc.get(t)
            if hit is not None and 1 <= len(hit[0]) <= nguong_df:
                ra[t] = min(ra.get(t, 10 ** 9), len(hit[0]))
    return sorted(ra.items(), key=lambda x: x[1])


def khung_neo(cau: str, kenh: KenhVanBan, neo: list, master,
              truoc: float, sau: float) -> set:
    """`row_id` của mọi khung nằm trong cửa sổ quanh một lần khớp mỏ neo.

    "Đi bộ thời gian": mỏ neo cho biết *lúc nào*, không cho biết *khung nào*.
    Chữ chạy trên màn hình thường xuất hiện TRƯỚC hoặc SAU cảnh nó nói tới —
    `p1-9-qa` là ví dụ: đáp án nằm ở khung SAU cảnh xe lội nước.

    ⚠️ Cửa sổ dùng `pts_time`, KHÔNG dùng số khung. Kho có 4 giá trị fps.
    """
    tai_lieu = set()
    for chi_muc, c in ((kenh.co_dau.chi_muc, cau),
                       (kenh.khong_dau.chi_muc, bo_dau(cau))):
        for t, _ in neo:
            hit = chi_muc.get(t)
            if hit is not None:
                tai_lieu.update(int(x) for x in hit[0])

    goc = set()
    for i in tai_lieu:
        goc.update(int(r) for r in kenh.khoa_dong[i])
    if not goc:
        return set()

    vid = master.video_id.values
    pts = master.pts_time.values
    ra = set()
    for r in goc:
        v, t0 = vid[r], pts[r]
        # Nở trong CÙNG video. Bảng cái sắp theo (video_id, frame_idx) nên chỉ
        # cần đi hai phía từ r cho tới khi đổi video hoặc ra ngoài cửa sổ.
        for buoc in (-1, 1):
            i = r
            while 0 <= i < len(vid) and vid[i] == v:
                d = pts[i] - t0
                if d < -truoc or d > sau:
                    break
                ra.add(int(i))
                i += buoc
    return ra


def _the(rid: int, diem: float, master, nguon: str) -> Candidate:
    r = master.iloc[rid]
    return Candidate(row_id=int(rid), video_id=str(r.video_id),
                     frame_idx=int(r.frame_idx), score=float(diem), source=nguon,
                     meta={"pts_time": float(r.pts_time), "fps": float(r.fps),
                           "kf_n": int(r.kf_n), "title": r.title})


def main():
    ap = argparse.ArgumentParser(description="do Anchor & Expand")
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--matrix", default="clip_siglip2.npy")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--nguong-df", type=int, default=60,
                    help="token hiếm = có mặt ở <= ngần này khung OCR")
    ap.add_argument("--truoc", type=float, default=15.0)
    ap.add_argument("--sau", type=float, default=30.0)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    k1 = KenhAnhCache(str(a.index), str(a.cache), matrix=a.matrix)
    ocr = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    cau = [c for c in tap_dev.doc() if c.loai in ("KIS", "QA")]
    thieu = k1.co_du([m for c in cau for m in R.tach_truy_van(c.cau_hoi)])
    if thieu:
        raise SystemExit(f"cache thiếu {len(thieu)} mệnh đề, ví dụ {thieu[0][:80]!r}")

    # Dựng MỘT LẦN: bể SigLIP2 và tập khung neo của từng câu. Dựng lại theo
    # biến thể thì khác biệt đo được có thể đến từ nhiễu của kênh.
    print(f"Dựng bể ứng viên cho {len(cau)} câu...", flush=True)
    be, neo_cua, thong_ke = {}, {}, []
    for i, c in enumerate(cau, 1):
        be[c.id] = k1.tim(R.tach_truy_van(c.cau_hoi), k=a.k)
        neo = tim_neo(c.cau_hoi, ocr, a.nguong_df)
        neo_cua[c.id] = khung_neo(c.cau_hoi, ocr, neo, master,
                                  a.truoc, a.sau) if neo else set()
        thong_ke.append((c.id, len(neo), len(neo_cua[c.id]),
                         [t for t, _ in neo[:3]]))
        if i % 30 == 0:
            print(f"  {i}/{len(cau)}", flush=True)

    co_neo = [x for x in thong_ke if x[1]]
    print(f"\n{len(co_neo)}/{len(cau)} câu CÓ mỏ neo "
          f"(df <= {a.nguong_df}), cửa sổ [-{a.truoc:g}s, +{a.sau:g}s]")
    if co_neo:
        m = int(np.median([x[2] for x in co_neo]))
        print(f"  trung vị {m} khung neo/câu")
        for cid, n, k, ts in co_neo[:6]:
            print(f"    {cid:16} {n} neo, {k:4d} khung  {ts}")
    print()

    def moc(c):
        return be[c.id]

    def xep_lai(c):
        """Đảo thứ tự TRONG bể — tuân thủ A27/A28."""
        n = neo_cua[c.id]
        if not n:
            return be[c.id]
        trong = [x for x in be[c.id] if x.row_id in n]
        ngoai = [x for x in be[c.id] if x.row_id not in n]
        return trong + ngoai

    def chen(c):
        """Chèn khung neo lên đầu, kể cả khung SigLIP2 không có — VI PHẠM A27/A28."""
        n = neo_cua[c.id]
        if not n:
            return be[c.id]
        da_co = {x.row_id for x in be[c.id]}
        them = [_the(r, 9.0, master, "neo") for r in sorted(n) if r not in da_co]
        trong = [x for x in be[c.id] if x.row_id in n]
        ngoai = [x for x in be[c.id] if x.row_id not in n]
        return (trong + them + ngoai)[:a.k]

    def duoi(c):
        """Cấp riêng SUẤT NỘP ở đuôi, không trộn vào thứ hạng đầu.

        Đây là thiết kế duy nhất còn sống sau khi đo được cơ chế: tập khung neo
        và top-100 của SigLIP2 **gần như không giao nhau** (25/38 câu giao = 0,
        trung vị 0 khung). Nên:

        * `xep_lai` là no-op — không có gì để đảo;
        * `chen` đẩy ứng viên tốt của SigLIP2 từ hạng 1–100 xuống 73–172, tức
          **rơi khỏi bài nộp**. Đo được −0,1474/−0,2316.

        Cách này không đụng vào thứ hạng đầu. Hạng 81–100 chỉ vào `R@100`, nên
        mỗi dòng ở đó đáng nhiều nhất **0,2/5 = 0,04** điểm — mặt trái bị chặn
        cứng, mặt phải là từ 0 lên 0,2 nếu mỏ neo trúng.

        Không phạt dòng sai (PHẦN C), nên 20 dòng cuối vốn là vé số; đổi vé số
        ngẫu nhiên lấy vé số có mỏ neo là phép đổi không mất gì.
        """
        n = neo_cua[c.id]
        if not n:
            return be[c.id]
        giu = be[c.id][:80]
        da_co = {x.row_id for x in giu}
        them = [_the(r, 0.0, master, "neo") for r in sorted(n) if r not in da_co]
        con = [x for x in be[c.id][80:] if x.row_id not in da_co]
        return (giu + them + con)[:a.k]

    cau_hinh = {"SigLIP2 một mình (mốc nền)": moc,
                "neo: XẾP LẠI trong bể (A27/A28)": xep_lai,
                "neo: CHÈN lên đầu (vi phạm A27/A28)": chen,
                "neo: chỉ chiếm hạng 81-100 (suất riêng)": duoi}

    chi_neo = [c for c in cau if neo_cua[c.id]]
    de_that = [c for c in cau if "-DE1-" in c.id]
    that_neo = [c for c in de_that if neo_cua[c.id]]

    for ten, bo in (("TOÀN BỘ", cau),
                    (f"⭐ chỉ {len(chi_neo)} câu CÓ MỎ NEO — nơi kỹ thuật này "
                     f"tuyên bố", chi_neo),
                    (f"chỉ {len(de_that)} câu ĐỀ THẬT", de_that),
                    (f"⭐ chỉ {len(that_neo)} câu ĐỀ THẬT CÓ MỎ NEO", that_neo)):
        if len(bo) < 2:
            print(f"### {ten} — bỏ qua, n={len(bo)}\n")
            continue
        print("=" * 70)
        print(f"### {ten}")
        print("=" * 70)
        print(bao_cao_do_nhay(bo, cau_hinh, master, MOC_DUNG_SAI, a.k))
        print()


if __name__ == "__main__":
    main()
