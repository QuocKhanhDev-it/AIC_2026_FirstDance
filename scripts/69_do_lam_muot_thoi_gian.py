"""
69_do_lam_muot_thoi_gian.py — Làm mượt vector ảnh theo trục thời gian.

    python scripts/69_do_lam_muot_thoi_gian.py

Ý TƯỞNG (Gemini gợi ý, hướng 3)

Một sự kiện trong video diễn ra qua nhiều khung liên tiếp. Vector của một khung
đứng lẻ hay bị nhiễu vì nhoè chuyển động, chớp mắt, góc quay chuyển tiếp. Cộng
thêm vector hàng xóm rồi chuẩn hoá lại là đưa ngữ cảnh trước–sau vào chính
khung đó:

    v'(t) = chuẩn_hoá( v(t) + α·v(t−1) + α·v(t+1) + α²·v(t±2) … )

CHI PHÍ LÚC THI: **BẰNG KHÔNG**. Ma trận vẫn 177.321 × 1536, chỉ đổi giá trị.
Đây là điểm khác biệt lớn nhất so với mọi hướng còn lại đang chờ.

KHÁC HẲN VỚI GOM CỤM ĐÃ BỊ BÁC (A55)

A55 bác việc **bỏ bớt** khung trùng cảnh khỏi danh sách kết quả. Ở đây không bỏ
ai cả — số ứng viên không đổi, chỉ có vector của từng khung mang thêm ngữ cảnh.
Một cái sửa ĐẦU RA, một cái sửa ĐẦU VÀO.

⚠️ CHỈ LÀM MƯỢT TRONG CÙNG MỘT VIDEO, và chỉ với hàng xóm cách dưới
`--cach-toi-da` giây. Keyframe hai bên một điểm cắt cảnh là hai nội dung khác
nhau; trộn chúng là bôi nhoè đúng thứ cần phân biệt.

⚠️ Ghi ra file TẠM, không đụng `clip_gopt.npy`. Ghi đè ma trận gốc rồi mới đo
là tự đặt mình vào thế không lùi được nếu kết quả xấu.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import bao_cao_do_nhay                 # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5
LO = 20_000
# (bán kính W, hệ số α) — α là trọng số của hàng xóm gần nhất, α^k cho xa hơn
CAU_HINH = ((1, 0.2), (1, 0.35), (1, 0.5), (2, 0.35))


def lam_muot(mat, master, w: int, alpha: float, cach_toi_da: float,
             ra: Path) -> Path:
    """Ghi ma trận đã làm mượt ra `ra`. Trả về đường dẫn."""
    n, d = mat.shape
    vid = master.video_id.values
    t = master.pts_time.values.astype(np.float32)

    moi = np.lib.format.open_memmap(ra, mode="w+", dtype=np.float16,
                                    shape=(n, d))
    dem_bo = 0
    for i in range(0, n, LO):
        j = min(i + LO, n)
        # lấy dư ±w hàng để hàng xóm ở mép lô vẫn có mặt
        lo_i, hi_i = max(0, i - w), min(n, j + w)
        M = np.asarray(mat[lo_i:hi_i], dtype=np.float32)
        acc = M[i - lo_i:j - lo_i].copy()
        for k in range(1, w + 1):
            he_so = alpha ** k
            for dau in (-k, k):
                idx = np.arange(i, j) + dau
                hop_le = (idx >= 0) & (idx < n)
                # cùng video VÀ cách dưới ngưỡng giây
                idx_an = np.clip(idx, 0, n - 1)
                hop_le &= vid[idx_an] == vid[np.arange(i, j)]
                hop_le &= np.abs(t[idx_an] - t[np.arange(i, j)]) <= cach_toi_da
                dem_bo += int((~hop_le).sum())
                lay = M[idx_an - lo_i] * hop_le[:, None]
                acc += he_so * lay
        acc /= (np.linalg.norm(acc, axis=1, keepdims=True) + 1e-9)
        moi[i:j] = acc.astype(np.float16)
    moi.flush()
    return ra


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    ap.add_argument("--be", type=int, default=100)
    ap.add_argument("--cach-toi-da", type=float, default=10.0,
                    help="giây tối đa giữa hai keyframe còn coi là hàng xóm")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)
    goc = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                       matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    giu = [c for c in cau if not goc.co_du(R.tach_truy_van(c.cau_hoi))]
    print(f"{a.file.name}: đo {len(giu)}/{len(cau)} câu | hàng xóm ≤ "
          f"{a.cach_toi_da:g}s\n")

    ocr = {}

    def lam(k1):
        nho = {}

        def g(c):
            if c.id not in nho:
                anh = hop_nhat([k1.tim(m, k=a.be)
                                for m in R.tach_truy_van(c.cau_hoi)])
                if c.id not in ocr:
                    ocr[c.id] = k3.tim(c.cau_hoi, k=a.be)
                nho[c.id] = hop_nhat([anh, ocr[c.id]], trong_so=[1.0, W3])
            return nho[c.id]
        return g

    cau_hinh = {"1. mốc: run.py (không làm mượt)": lam(goc)}
    tam = []
    for w, al in CAU_HINH:
        f = a.index / f"tam_muot_W{w}_a{al}.npy"
        print(f"dựng ma trận W={w} α={al} …", flush=True)
        lam_muot(goc.mat, master, w, al, a.cach_toi_da, f)
        tam.append(f)
        k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                          matrix=f.name)
        cau_hinh[f"2. mượt W={w} α={al:g}"] = lam(k1)

    print()
    print(bao_cao_do_nhay(giu, cau_hinh, master))
    print("\nFile tạm (xoá được): " + ", ".join(f.name for f in tam))


if __name__ == "__main__":
    main()
