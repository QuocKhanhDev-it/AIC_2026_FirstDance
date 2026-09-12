"""
112_tinh_hubness.py — Tính `r_K(d)` cho CSLS: độ tương đồng trung bình của mỗi keyframe với K láng giềng gần nhất.

    python scripts/112_tinh_hubness.py            # ~15-25 phút, CHẠY OFFLINE
    python scripts/112_tinh_hubness.py --K 10

Ghi ra `index/hubness_gopt.npy` — 177.321 số float32 = **709 KB**. Lúc truy vấn
chỉ là một phép trừ vector, nên chi phí online bằng không.

CSLS LÀ GÌ VÀ VÌ SAO ĐÁNG THỬ Ở ĐÂY

Không gian nhúng nhiều chiều có hiện tượng **hub**: vài điểm nằm gần *mọi thứ*
và được trả về cho mọi truy vấn. A92 đo được KIS có đáp án trong top-100 ở 81%
số câu nhưng chỉ ở hạng 1 ở **24%** — tức bài toán là XẾP HẠNG, không phải tìm
kiếm, và hub là một nghi phạm cụ thể cho kiểu hỏng đó.

    s_CSLS(q, d) = 2·cos(q, d) − λ·r_K(d)

với `r_K(d)` = trung bình cosine của `d` với K láng giềng gần nhất của nó.
Điểm nào "gần mọi thứ" thì `r_K` cao và bị trừ nhiều.

⚠️ λ LÀ THAM SỐ DÒ THẬT, KHÔNG PHẢI HẰNG SỐ. Bản CSLS gốc (Conneau 2018) cố
định λ = 1; ở đây λ được dò lưới trong `113_do_csls.py`. Viết tường minh vì
bản đề xuất ban đầu vừa ghi công thức λ=1 ẩn vừa đề nghị dò λ — hai điều đó
không thể cùng đúng.

⚠️ HAI SAI LỆCH PHẢI CHẶN, VÀ CHÚNG KHÔNG HIỂN NHIÊN

**1. Bản sao cùng video làm hub GIẢ.** A5.6 đo được **11,83% keyframe có bản
sao cùng video ở cosine ≥ 0,99** (riêng L25: **49,82%**). Một khung có năm bản
sao trong chính video của nó sẽ có `r_K ≈ 0,99` và bị phạt nặng nhất — trong
khi nó chẳng "gần mọi thứ" chút nào, nó chỉ gần chính nó. Phạt đúng khung đáp
án là hỏng ngược. Nên **loại toàn bộ láng giềng CÙNG VIDEO** khi tính `r_K`,
không chỉ loại chính nó.

**2. Đây là hub ẢNH–ẢNH, không phải hub của CSLS gốc.** CSLS gốc đo hub trong
không gian **xuyên miền** (văn bản ↔ ảnh): `r_K(d)` là độ tương đồng trung bình
của `d` với K *truy vấn* gần nhất. Ở đây dùng láng giềng ẢNH vì:

  * ước lượng từ truy vấn cần một tập truy vấn, mà tập duy nhất đang có là 52
    câu dev — **chính những câu sẽ dùng để chấm**. Dựng chỉ mục từ đầu vào
    kiểm thử là transductive: điểm sẽ đẹp lên mà **không** tổng quát sang đề
    thi thật, nơi truy vấn là câu chưa từng thấy. Đó là rò rỉ, không phải kỹ
    thuật.
  * hub ảnh–ảnh là xấp xỉ thực dụng, và phải gọi đúng tên nó là xấp xỉ.

Nếu về sau có một tập truy vấn ĐỘC LẬP đủ lớn (tiêu đề tin tức chẳng hạn) thì
nên đo lại bằng bản xuyên miền và so hai bản.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent

K_MAC_DINH = 10
KHOI_HANG = 2000        # dòng mỗi lô truy vấn
KHOI_COT = 20000        # cột mỗi lô tham chiếu


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--matrix", default="clip_gopt.npy")
    ap.add_argument("--K", type=int, default=K_MAC_DINH)
    ap.add_argument("--ra", default=None,
                    help="mặc định index/hubness_<tên ma trận>.npy")
    a = ap.parse_args()

    mat = np.load(a.index / a.matrix, mmap_mode="r")
    n = mat.shape[0]
    master = pd.read_parquet(a.index / "master.parquet")
    assert len(master) == n, f"master {len(master)} != ma trận {n}"
    # video_id -> số nguyên, để so sánh bằng phép số học chứ không phải chuỗi
    vid = pd.factorize(master.video_id.values)[0].astype(np.int32)

    ra = np.empty(n, np.float32)
    t0 = time.time()
    for i0 in range(0, n, KHOI_HANG):
        i1 = min(i0 + KHOI_HANG, n)
        q = np.asarray(mat[i0:i1], np.float32)
        vq = vid[i0:i1]
        # Giữ K điểm cao nhất đang thấy, gộp dần qua từng lô tham chiếu.
        giu = np.full((i1 - i0, a.K), -np.inf, np.float32)
        for j0 in range(0, n, KHOI_COT):
            j1 = min(j0 + KHOI_COT, n)
            s = q @ np.asarray(mat[j0:j1], np.float32).T
            # ⚠️ Loại CẢ VIDEO, không chỉ loại chính nó — xem docstring.
            s[vq[:, None] == vid[j0:j1][None, :]] = -np.inf
            k = min(a.K, s.shape[1])
            top = np.partition(s, -k, axis=1)[:, -k:]
            gop = np.concatenate([giu, top], axis=1)
            giu = np.partition(gop, -a.K, axis=1)[:, -a.K:]
            del s, top, gop
        ra[i0:i1] = giu.mean(axis=1)
        xong = i1 / n
        print(f"  {i1:>7,}/{n:,} ({xong * 100:5.1f}%) "
              f"— {time.time() - t0:6.0f}s, còn ~"
              f"{(time.time() - t0) * (1 - xong) / max(xong, 1e-9):.0f}s",
              flush=True)

    f = Path(a.ra) if a.ra else (a.index /
                                 f"hubness_{Path(a.matrix).stem}.npy")
    np.save(f, ra)
    print(f"\n✅ {f}  ({f.stat().st_size / 1024:.0f} KB)")
    print(f"r_K (K={a.K}) — trung vị {np.median(ra):.4f}, "
          f"min {ra.min():.4f}, max {ra.max():.4f}, "
          f"p99 {np.percentile(ra, 99):.4f}")
    print(f"độ lệch chuẩn {ra.std():.4f} — nếu con số này rất nhỏ thì CSLS "
          f"gần như không đổi được thứ hạng nào,\nvà bảng ở 113_ sẽ ra ⚪.")


if __name__ == "__main__":
    main()
