"""
81_gop_bge_theo_doan_asr.py — Gộp vector BGE-M3 theo ĐOẠN ASR, rồi đo.

    python scripts/81_gop_bge_theo_doan_asr.py --dung           # dựng ma trận
    python scripts/81_gop_bge_theo_doan_asr.py --file <tap>     # đo

Ý TƯỞNG, VÀ VÌ SAO NÓ KHÔNG TỐN GPU

A69 đo được: gom khung liên tiếp cùng video theo ASR (Jaccard ≥ 0,3) cho
**9.802 đoạn, 18,1 khung/đoạn** — giảm 18 lần. Đề xuất gốc là gọi LLM tóm tắt
mỗi đoạn. Nhưng ta đã có sẵn `van_ban_bge.npz`: BGE-M3 đã nhúng từng khung.

Gộp VECTOR theo đoạn cho hai thứ mà per-frame không có, và **không cần model**:

  1. **KHỬ NHIỄU** — ASR của một khung là lát cắt vài giây, hay đứt giữa câu.
     Vector của cả đoạn là nội dung trọn vẹn của cảnh đó.
  2. **LAN TÍN HIỆU** — 22,6% khung KHÔNG có ASR. Chúng nằm giữa những khung
     có ASR trong cùng một đoạn, nên gộp đoạn cho chúng vector của đoạn.
     Kênh 6 hiện bỏ trắng những khung đó.

Điểm 2 mới là phần đáng giá: nó là thứ DUY NHẤT mà tóm tắt-bằng-LLM hứa hẹn mà
per-frame không làm được — và ở đây có được với chi phí bằng không.

⚠️ ĐỌC CÙNG A59. Kênh 6 (BGE per-frame) đứng một mình chỉ 0,1462. Gộp đoạn là
một biến đổi khác trên CÙNG nguồn tín hiệu, nên đừng kỳ vọng nó vượt xa. Nếu
gộp đoạn cũng ~0,15 thì cả nhánh "tóm tắt ASR bằng LLM" nên đóng luôn — vì LLM
tốn vài giờ để làm đúng thứ phép gộp này làm miễn phí.
"""

import argparse
import json
import sys
import unicodedata
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
from van_ban_dense import KenhVanBanDense             # noqa: E402

W3 = 0.5
NGUONG = 0.3
LO = 20_000


def bo_dau(s) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def chia_doan(master, bang, nguong=NGUONG):
    """-> mảng `doan[i]` = chỉ số đoạn của keyframe i (theo thứ tự row_id)."""
    d = master[["row_id", "video_id"]].merge(
        bang[["row_id", "asr_text"]], on="row_id", how="left")
    tok = [set(bo_dau(x).split()) for x in d.asr_text.fillna("")]
    vid = d.video_id.values
    doan = np.zeros(len(d), dtype=np.int64)
    k = 0
    for i in range(1, len(d)):
        a, b = tok[i - 1], tok[i]
        giong = (len(a & b) / len(a | b)) if (a or b) else 1.0
        if vid[i] != vid[i - 1] or giong < nguong:
            k += 1
        doan[i] = k
    return doan


def dung(index: Path, ra: Path, cach: str = "max"):
    """Gộp vector BGE theo đoạn, ghi ma trận phủ MỌI keyframe của đoạn có chữ."""
    master = pd.read_parquet(index / "master.parquet")
    bang = pd.read_parquet(index / "ocr_asr.parquet")
    doan = chia_doan(master, bang)
    print(f"{doan.max() + 1:,} đoạn / {len(master):,} khung "
          f"({len(master) / (doan.max() + 1):.1f} khung/đoạn)")

    z = np.load(index / "van_ban_bge.npz", allow_pickle=False)
    rid = np.asarray(z["row_id"], dtype=np.int64)
    vec = z["vec"]
    d = vec.shape[1]
    gc = json.loads(str(z["ghi_chu"]))

    # Tổng (hoặc max) vector theo đoạn — chỉ trên khung CÓ vector.
    tong = np.zeros((doan.max() + 1, d), dtype=np.float32)
    dem = np.zeros(doan.max() + 1, dtype=np.int64)
    for i in range(0, len(rid), LO):
        j = min(i + LO, len(rid))
        v = np.asarray(vec[i:j], dtype=np.float32)
        dj = doan[rid[i:j]]
        if cach == "max":
            np.maximum.at(tong, dj, v)
        else:
            np.add.at(tong, dj, v)
        np.add.at(dem, dj, 1)
    if cach != "max":
        tong /= np.maximum(dem, 1)[:, None]
    tong /= (np.linalg.norm(tong, axis=1, keepdims=True) + 1e-9)

    co = dem > 0                       # đoạn có ít nhất một khung có ASR
    giu = np.flatnonzero(co[doan])     # MỌI khung của những đoạn đó
    print(f"  phủ {len(giu):,}/{len(master):,} khung "
          f"({len(giu) / len(master) * 100:.1f}%) — kênh 6 gốc phủ "
          f"{len(rid):,} ({len(rid) / len(master) * 100:.1f}%)")

    np.savez(ra, vec=tong[doan[giu]].astype(np.float16),
             row_id=giu.astype(np.int64),
             ghi_chu=json.dumps({**gc, "gop_doan": cach, "nguong": NGUONG,
                                 "so_doan": int(doan.max() + 1)},
                                ensure_ascii=False))
    print(f"✅ {ra}  ({ra.stat().st_size / 1024**2:.0f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    ap.add_argument("--ra", default=GOC / "index" / "van_ban_bge_doan.npz",
                    type=Path)
    ap.add_argument("--cach", default="max", choices=("max", "tb"))
    ap.add_argument("--dung", action="store_true", help="dựng ma trận rồi thoát")
    ap.add_argument("--be", type=int, default=100)
    a = ap.parse_args()

    if a.dung:
        dung(a.index, a.ra, a.cach)
        return
    if not a.ra.exists():
        raise SystemExit(f"Chưa có {a.ra}. Chạy `--dung` trước.")

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)
    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")
    cache_bge = str(a.index / "truy_van_bge.npz")
    k6 = KenhVanBanDense(str(a.index), a.index / "van_ban_bge.npz", cache_bge)
    k6d = KenhVanBanDense(str(a.index), a.ra, cache_bge)
    print(f"kênh 6 gốc : {k6.vec.shape[0]:,} vector / {len(k6._r_duy):,} khung")
    print(f"kênh 6 đoạn: {k6d.vec.shape[0]:,} vector / {len(k6d._r_duy):,} khung"
          f" | {k6d.ghi_chu.get('so_doan'):,} đoạn\n")

    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    print(f"{a.file.name}: đo {len(giu)}/{len(cau)} câu\n")

    nen = {}

    def phan(c):
        if c.id not in nen:
            anh = hop_nhat([k1.tim(m, k=a.be) for m in R.tach_truy_van(c.cau_hoi)])
            md = R.tach_truy_van(c.cau_hoi)
            nen[c.id] = (anh, k3.tim(c.cau_hoi, k=a.be),
                         k6.tim(md, k=a.be), k6d.tim(md, k=a.be))
        return nen[c.id]

    def _nho(f):
        n = {}

        def g(c):
            if c.id not in n:
                n[c.id] = f(c)[:100]
            return n[c.id]
        return g

    cau_hinh = {
        "1. mốc: run.py": _nho(lambda c: hop_nhat(
            list(phan(c)[:2]), trong_so=[1.0, W3])),
    }
    for w in (0.25, 0.5):
        cau_hinh[f"2. + kênh 6 GỐC ({w:g})"] = _nho(
            (lambda w: lambda c: hop_nhat(
                [phan(c)[0], phan(c)[1], phan(c)[2]],
                trong_so=[1.0, W3, w]))(w))
        cau_hinh[f"3. + kênh 6 GỘP ĐOẠN ({w:g})"] = _nho(
            (lambda w: lambda c: hop_nhat(
                [phan(c)[0], phan(c)[1], phan(c)[3]],
                trong_so=[1.0, W3, w]))(w))
    cau_hinh["4. chỉ kênh 6 gốc (chẩn đoán)"] = _nho(lambda c: phan(c)[2])
    cau_hinh["5. chỉ kênh 6 GỘP ĐOẠN (chẩn đoán)"] = _nho(lambda c: phan(c)[3])

    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
