"""
van_ban_dense.py — KÊNH 6: truy hồi OCR/ASR bằng vector, không phải BM25.

Sinh dữ liệu: `scripts/55_ma_hoa_van_ban.py`.

    kenh = KenhVanBanDense('./index', 'index/van_ban_gopt.npz',
                           cache='index/truy_van_gopt.npz')
    kq = kenh.tim("người phụ nữ thái dứa", k=100)

KHÁC KÊNH 3 Ở ĐÂU

Kênh 3 (`bm25.KenhVanBan`) khớp **mặt chữ**: truy vấn "xe cứu thương" mà bản
tin viết "xe cấp cứu" thì điểm bằng 0. Kênh này nhúng cùng văn bản đó vào không
gian 1536 chiều của gopt, nên hai cách gọi nằm gần nhau.

Hai kênh **không thay nhau** — chúng hỏng ở hai chỗ khác nhau. BM25 mạnh ở tên
riêng, số hiệu, biển hiệu (khớp chính xác); vector mạnh ở cách diễn đạt khác.
Đó là lý do đáng giữ cả hai nếu đo được là có lãi.

MỘT KEYFRAME CÓ NHIỀU ĐOẠN — GỘP BẰNG MAX

Tài liệu dài bị chia thành nhiều đoạn ≤ 60 token (trần ngữ cảnh của SigLIP2 là
64). Điểm của một keyframe là **max** trên các đoạn của nó: một đoạn khớp là đủ
để khung đó đáng trả về. Lấy trung bình thì một tài liệu dài có 5 đoạn, 1 đoạn
khớp, bị dìm bởi 4 đoạn không liên quan — đúng ngược với điều ta muốn.

⚠️ KÊNH NÀY DÙNG MODEL NGOÀI PHÂN BỐ HUẤN LUYỆN

SigLIP2 học để khớp **ảnh ↔ chữ**, không phải **chữ ↔ chữ**. Truy vấn và tài
liệu đều là chữ, nên phép so này chưa từng là mục tiêu huấn luyện. Có thể tốt,
có thể không — **phải đo trên `tap_de_that.jsonl` trước khi bật**.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .schema import Candidate
except ImportError:
    from schema import Candidate


def _mo(f_vec):
    """Đọc `van_ban_gopt.npz`, HOẶC một thư mục cùng tên đã bị giải nén.

    `.npz` chỉ là một file zip chứa `vec.npy` / `row_id.npy` / `ghi_chu.npy`.
    Trên đường đi Kaggle → Google Drive → máy thành viên, nó hay bị bung thành
    thư mục — đã xảy ra thật với bản 1,32 GB đầu tiên. Ma trận vẫn nguyên vẹn,
    chỉ là hình dạng trên đĩa khác; bắt cả nhóm nén lại 1,3 GB chỉ để `np.load`
    vui lòng là phí công vô ích.

    Dạng thư mục còn HƠN dạng nén ở đây: `vec.npy` mở được bằng `mmap`, nên
    ma trận 1,32 GB nằm trên đĩa chứ không nằm trong RAM. Máy 7,7 GB của nhóm
    đã treo cứng một lần vì nạp thẳng nó cùng lúc với chỉ mục BM25 kênh 3.
    """
    p = Path(f_vec)
    if not p.exists() and p.with_suffix("").is_dir():
        p = p.with_suffix("")
    if p.is_dir():
        return {
            "vec": np.load(p / "vec.npy", mmap_mode="r"),
            "row_id": np.load(p / "row_id.npy"),
            "ghi_chu": np.load(p / "ghi_chu.npy", allow_pickle=True),
        }
    return np.load(p, allow_pickle=False)


class KenhVanBanDense:
    """Truy hồi OCR/ASR bằng vector, gộp theo `row_id` bằng max."""

    LO = 20_000                      # đoạn mỗi lô khi nhân — xem `tim()`

    def __init__(self, index_dir="./index", f_vec="index/van_ban_gopt.npz",
                 cache="index/truy_van_gopt.npz", ten="van_ban_dense"):
        d = Path(index_dir)
        self.ten = ten
        self.master = pd.read_parquet(d / "master.parquet")

        z = _mo(f_vec)
        # KHÔNG `np.asarray(..., dtype=)` — với mmap nó sao chép cả 1,32 GB
        # vào RAM, đúng thứ dạng thư mục sinh ra để tránh.
        self.vec = z["vec"]
        self.row_id = np.asarray(z["row_id"], dtype=np.int64)
        self.ghi_chu = json.loads(str(z["ghi_chu"]))
        if len(self.vec) != len(self.row_id):
            raise SystemExit(
                f"{f_vec}: {len(self.vec):,} vector nhưng {len(self.row_id):,} "
                f"row_id — file hỏng, đừng dùng.")

        # Cache vector truy vấn — cùng đường với `dense.KenhAnhCache`, và cùng
        # model. Số chiều lệch nghĩa là sai cặp file, dừng ngay.
        zc = np.load(cache, allow_pickle=False)
        gc = json.loads(str(zc["ghi_chu"]))
        v = np.asarray(zc["vec"], dtype=np.float32)
        if v.shape[1] != self.vec.shape[1]:
            raise SystemExit(
                f"Cache {v.shape[1]} chiều nhưng văn bản {self.vec.shape[1]} "
                f"chiều. Sai cặp file — cache mã hoá bằng {gc.get('model')}, "
                f"văn bản bằng {self.ghi_chu.get('model')}.")
        self._cache = {str(c): v[i] for i, c in enumerate(zc["cau"])}
        self.nguon_cache = str(cache)

        # Nhóm theo `row_id` — tính MỘT LẦN, dùng lại cho mọi truy vấn.
        self._sx = np.argsort(self.row_id, kind="stable")
        rid = self.row_id[self._sx]
        self._dau = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
        self._r_duy = rid[self._dau]

    def co_du(self, cac_cau) -> list[str]:
        """Câu CHƯA có trong cache. Gọi trước khi chạy để hỏng sớm."""
        return [c for c in cac_cau if c not in self._cache]

    def _vec_truy_van(self, cau) -> np.ndarray:
        """Một chuỗi hoặc danh sách chuỗi -> một vector (đã chuẩn hoá).

        Nhiều mệnh đề thì lấy TRUNG BÌNH rồi chuẩn hoá lại — cùng cách
        `dense.KenhAnh` làm, để hai kênh nhận cùng một vector truy vấn.
        """
        cs = [cau] if isinstance(cau, str) else list(cau)
        thieu = self.co_du(cs)
        if thieu:
            raise KeyError(
                f"Truy vấn chưa có trong cache {self.nguon_cache}:\n"
                f"  {thieu[0][:120]!r}\n"
                f"Mã hoá thêm bằng scripts/25_ma_hoa_truy_van.py")
        v = np.mean([self._cache[c] for c in cs], axis=0)
        return v / (np.linalg.norm(v) + 1e-9)

    def tim(self, cau, k: int = 100, be=None) -> list[Candidate]:
        q = self._vec_truy_van(cau).astype(np.float32)

        # Nhân theo LÔ: `vec` là mmap 1,32 GB. `self.vec @ q` một phát sẽ kéo
        # nguyên ma trận vào RAM — trên máy 7,7 GB là treo máy, không phải
        # chậm. Mỗi lô 20.000 đoạn ~123 MB ở float32.
        diem = np.empty(len(self.row_id), dtype=np.float32)
        for i in range(0, len(diem), self.LO):
            j = min(i + self.LO, len(diem))
            diem[i:j] = np.asarray(self.vec[i:j], dtype=np.float32) @ q

        if be is not None:
            # Khoá bể ứng viên: loại đoạn thuộc row_id ngoài bể.
            diem = np.where(be[self.row_id], diem, -9.0)

        # Gộp theo row_id bằng MAX. Thứ tự sắp theo `row_id` KHÔNG đổi giữa các
        # truy vấn nên tính sẵn một lần trong `__init__`; ở đây chỉ còn
        # `reduceat`, không phải `lexsort` 462k phần tử cho mỗi truy vấn.
        d_duy = np.maximum.reduceat(diem[self._sx], self._dau)
        r_duy = self._r_duy

        if k < len(r_duy):
            lay = np.argpartition(-d_duy, k)[:k]
            lay = lay[np.argsort(-d_duy[lay])]
        else:
            lay = np.argsort(-d_duy)

        ra = []
        for i in lay:
            if d_duy[i] <= -8.0:                 # bị bể ứng viên loại
                continue
            r = int(r_duy[i])
            d = self.master.iloc[r]
            ra.append(Candidate(
                row_id=r, video_id=d.video_id, frame_idx=int(d.frame_idx),
                score=float(d_duy[i]), source=self.ten,
                meta={"pts_time": float(d.pts_time), "fps": float(d.fps),
                      "kf_n": int(d.kf_n)}))
        return ra
