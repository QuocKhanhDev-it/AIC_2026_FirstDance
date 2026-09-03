"""
94_soi_cau_that_bai.py — Hai chẩn đoán trên cùng một lượt chạy kênh.

    python scripts/94_soi_cau_that_bai.py

**E — 6 câu nằm ngoài top-1000.** A54 đo được 6/49 câu có đáp án ngoài
top-1000/177.321; với chúng mọi hậu xử lý đều vô nghĩa. Câu hỏi: chúng thất
bại vì cùng một lý do, hay mỗi câu một kiểu? Nếu cùng lý do với Q&A (OCR mất
dấu, ASR viết số bằng chữ) thì A76 có thể cứu luôn một phần mà không cần
hướng riêng.

**"Cluster flooding" — top-20 bị khung lân cận chiếm chỗ?** Có đề xuất cho
rằng R@20 đứng yên ở 0,6122 qua mọi cỡ bể là vì top-20 bị các keyframe lân cận
của cùng một shot SAI chiếm hết.

⚠️ SUY LUẬN ĐÓ KHÔNG ĐỨNG VỮNG, và đó là lý do phải đo. R@20 không đổi khi bể
đi từ 100 lên 1000 chỉ nói rằng **ứng viên mới đều xếp dưới hạng 20** — nó
không nói gì về thứ ĐANG chiếm top-20. Hai chuyện khác hẳn nhau.

Nên đo thẳng: trong top-20 có bao nhiêu video khác nhau, và bao nhiêu ứng viên
là hàng xóm thời gian (cùng video, cách nhau ≤ `--gan` giây) của một ứng viên
xếp trên nó. Nếu con số đó nhỏ thì "cluster flooding" không có, và phép lọc
phi cực đại theo thời gian không có gì để dọn.

⚠️ A18 đã đo **ràng buộc đa dạng** (tối đa 2 ứng viên mỗi video) và nó **làm tệ
đi**. Lọc phi cực đại theo thời gian mềm hơn, nhưng cùng họ — nên chẩn đoán
này quyết định có đáng đo tiếp không.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import no_cua_so                       # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl",
        GOC / "dev" / "tap_de_thi_thu.jsonl"])
    ap.add_argument("--be", type=int, default=1000)
    ap.add_argument("--gan", type=float, default=4.0,
                    help="hai ứng viên cùng video cách nhau dưới ngần này giây "
                         "thì coi là hàng xóm thời gian")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    pts = master.pts_time.values
    vid = master.video_id.values
    bang = pd.read_parquet(a.index / "ocr_asr.parquet")
    van = {int(r): f"{o} {s}".strip() for r, o, s in zip(
        bang.row_id.values, bang.ocr_text.fillna("").values,
        bang.asr_text.fillna("").values)}

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(master, bang, cot="text", ten="ocr_asr")

    cau = []
    for f in a.file:
        cau += tap_dev.doc(f)
    giu = [c for c in cau
           if c.loai != "TRAKE" and not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    print(f"{len(giu)} câu KIS/QA đo được | bể {a.be} | "
          f"hàng xóm = cùng video, ≤ {a.gan:g}s\n")

    ngoai_be, chong = [], {"video": [], "hang_xom": []}
    for c in giu:
        anh = hop_nhat([k1.tim(m, k=a.be) for m in R.tach_truy_van(c.cau_hoi)])
        uv = hop_nhat([anh, k3.tim(c.cau_hoi, k=a.be)], trong_so=[1.0, W3])
        dung = no_cua_so(c.row_id_dung, master, 2.0)
        h = next((i for i, x in enumerate(uv[:a.be], 1) if x.row_id in dung),
                 None)
        if h is None:
            ngoai_be.append(c)

        top = uv[:20]
        chong["video"].append(len({x.video_id for x in top}))
        n_hx = 0
        for i, x in enumerate(top):
            if any(vid[y.row_id] == vid[x.row_id]
                   and abs(pts[y.row_id] - pts[x.row_id]) <= a.gan
                   for y in top[:i]):
                n_hx += 1
        chong["hang_xom"].append(n_hx)

    # ── cluster flooding ─────────────────────────────────────────────
    v = chong["video"]
    hx = chong["hang_xom"]
    print("TOP-20 GỒM NHỮNG GÌ")
    print(f"  video khác nhau : trung vị {sorted(v)[len(v) // 2]}/20 "
          f"(min {min(v)}, max {max(v)})")
    print(f"  hàng xóm ≤{a.gan:g}s : trung vị {sorted(hx)[len(hx) // 2]}/20 "
          f"(min {min(hx)}, max {max(hx)})")
    print(f"  câu có ≥10 hàng xóm trong top-20: "
          f"{sum(1 for x in hx if x >= 10)}/{len(hx)}")
    print()

    # ── E: câu ngoài bể ──────────────────────────────────────────────
    print(f"E — {len(ngoai_be)}/{len(giu)} CÂU CÓ ĐÁP ÁN NGOÀI TOP-{a.be}\n")
    if not ngoai_be:
        return
    print(f"{'câu':<16}{'loại':<7}{'chữ ở khung đúng':>18}"
          f"{'có dấu':>9}{'dài câu':>9}")
    print("-" * 60)
    dem = Counter()
    for c in ngoai_be:
        t = " ".join(van.get(r, "") for r in c.row_id_dung).strip()
        co_chu = bool(t)
        dem["có chữ" if co_chu else "KHÔNG có chữ"] += 1
        n_tu = len(c.cau_hoi.split())
        dem["câu dài >40 từ" if n_tu > 40 else "câu ngắn"] += 1
        print(f"{c.id:<16}{c.loai:<7}{(str(len(t)) + ' ký tự') if co_chu else '—':>18}"
              f"{('có' if any(ord(x) > 127 for x in t) else '—'):>9}{n_tu:>9}")
    print()
    for k, n in dem.most_common():
        print(f"  {k:<20}{n}")
    print("\nĐỌC: nếu phần lớn 'KHÔNG có chữ' thì chúng thất bại vì kênh 1 —")
    print("     OCR/ASR không cứu được, và A76 (VietOCR) cũng không.")


if __name__ == "__main__":
    main()
