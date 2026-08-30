"""
dense.py — Kênh 1: truy hồi bằng vector ảnh (CLIP / SigLIP2 / bất kỳ).

Bóc lõi từ `scripts/06_tim.py` thành module gọi được, để Khánh chấm tập dev
bằng code thay vì đọc màn hình.

KHÔNG DÍNH VÀO MỘT MODEL NÀO. Số chiều đọc từ chính ma trận, tên model đọc từ
file `.json` cạnh nó (nếu có). Nhờ vậy khi máy GPU sinh ra `clip_siglip2.npy`
1152 chiều thì chỉ đổi tham số `--matrix`, không sửa dòng code nào:

    kenh = KenhAnh('./index')                              # ViT-B/32 của BTC
    kenh = KenhAnh('./index', matrix='clip_siglip2.npy')   # SigLIP2

Dùng:
    from dense import KenhAnh
    kenh = KenhAnh('./index')          # nạp model + ma trận MỘT lần
    kq = kenh.tim("a person riding a motorbike", k=100)

Chạy trực tiếp để lấy mốc cho bài test cố định:
    python src/dense.py --ghi-moc

⚠️ KHÔNG có ngưỡng cosine nào đáng tin để tự loại kết quả sai. Đo thật: truy
vấn "people playing football on a field" trả về một video ôn thi môn Toán với
cos = 0,311 — sai hoàn toàn nhưng điểm không thấp bất thường. Model LUÔN trả
về top-1 kể cả khi kho không có cảnh đó. Đừng xây logic lọc theo ngưỡng ở đây;
việc phân định để RRF và tập dev lo.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .schema import Candidate
except ImportError:                     # chạy trực tiếp: python src/dense.py
    from schema import Candidate

# Mặc định cho ma trận BTC phát. ĐỪNG đổi thành "ViT-B-32" — bẫy A6: sai biến
# thể làm cosine tụt 0,9913 -> 0,9513 mà KHÔNG ném lỗi nào.
MODEL_MAC_DINH = "ViT-B-32-quickgelu"
PRETRAINED_MAC_DINH = "openai"


# RAM tối thiểu (GB) cần CÒN TRỐNG để nạp model, tra theo số chiều ma trận.
# Số chiều là proxy tốt cho cỡ model: 512 = ViT-B/32, 1152 = SO400M.
# 1536 = ViT-gopt-16-SigLIP2-384, ~1,1 tỷ tham số — GẤP GẦN BA LẦN SO400M. Con
# số 10,0 là ƯỚC theo số tham số, chưa đo. Không có dòng này thì 1536 rơi vào
# `RAM_CAN_MAC_DINH` (6,5 — vốn là ngưỡng của SO400M), tức là chốt chống treo
# máy lại cho qua đúng model nặng nhất.
RAM_CAN = {512: 2.0, 768: 3.0, 1024: 5.0, 1152: 6.5, 1536: 10.0}
RAM_CAN_MAC_DINH = 6.5


def ram_trong_gb() -> float | None:
    """RAM vật lý còn trống, GB. `None` nếu không hỏi được."""
    try:
        import ctypes

        class _S(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = _S()
        s.dwLength = ctypes.sizeof(_S)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return s.ullAvailPhys / 1024 ** 3
    except Exception:
        pass
    try:                                     # Linux/macOS
        import os
        return (os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
                / 1024 ** 3)
    except Exception:
        return None


def kiem_ram(model_tag: str, chieu: int) -> None:
    """Dừng TRƯỚC khi nạp model nếu RAM không đủ.

    ⚠️ **Đây là chốt chống TREO MÁY, không phải chốt lịch sự.** Nạp
    `ViT-SO400M-14-SigLIP2-378` (~3,5 GB trọng số) trên máy 7,7 GB đã làm đứng
    hẳn máy **hai lần** — không phải `MemoryError` gọn gàng mà là hệ điều hành
    thrash tới mức không thao tác được, phải khởi động lại.

    Thà chết ngay với một dòng đọc được còn hơn treo máy người dùng.
    """
    can = RAM_CAN.get(chieu, RAM_CAN_MAC_DINH)
    tro = ram_trong_gb()
    if tro is None or tro >= can:
        return
    raise SystemExit(
        f"\n❌ KHÔNG ĐỦ RAM — dừng trước khi nạp model.\n\n"
        f"   Model      : {model_tag} ({chieu} chiều)\n"
        f"   Cần trống  : ~{can:.1f} GB\n"
        f"   Đang trống : {tro:.1f} GB\n\n"
        f"   Nạp tiếp gần như chắc chắn làm ĐỨNG MÁY (đã xảy ra hai lần),\n"
        f"   không phải báo lỗi gọn gàng.\n\n"
        f"   Cách đi tiếp:\n"
        f"     • Chạy trên máy >= 16 GB — xem docs/09_do_tren_may_khoe.md\n"
        f"     • Hoặc đóng bớt ứng dụng rồi thử lại\n"
        f"     • Hoặc dùng ma trận nhẹ hơn: --matrix clip.npy (512 chiều)\n"
        f"       ⚠️ nhưng CLIP được 0,0000 trên truy vấn tiếng Việt (A10)\n"
        f"     • Ép chạy bất chấp: KenhAnh(..., bo_qua_ram=True)\n")


class KenhAnh:
    """Giữ model + ma trận trong RAM, dùng lại cho mọi truy vấn.

    Nạp model tốn vài giây. `06_tim.py` nạp lại mỗi lần chạy — chấm 50 câu dev
    kiểu đó là nạp 50 lần. Lớp này nạp một lần.
    """

    def __init__(self, index_dir="./index", matrix="clip.npy",
                 model=None, pretrained=None, mmap=False, bo_qua_ram=False):
        d = Path(index_dir)
        self.master = pd.read_parquet(d / "master.parquet")
        self.mat = np.load(d / matrix, mmap_mode="r" if mmap else None)

        if len(self.master) != self.mat.shape[0]:
            raise SystemExit(
                f"master.parquet ({len(self.master)} dòng) và {matrix} "
                f"({self.mat.shape[0]} dòng) LỆCH NHAU. Dừng — mọi kết quả sau "
                f"đây sẽ trỏ nhầm keyframe.")

        # Tên model đọc từ .json cạnh ma trận nếu có (do 08_encode.py ghi ra).
        # Không có thì coi như ma trận của BTC.
        canh = d / (Path(matrix).stem + ".json")
        ghi_chu = json.loads(canh.read_text("utf-8")) if canh.exists() else {}
        self.model_tag = model or ghi_chu.get("model", MODEL_MAC_DINH)
        self.pretrained = pretrained or ghi_chu.get("pretrained", PRETRAINED_MAC_DINH)
        self.chieu = self.mat.shape[1]

        if not bo_qua_ram:
            kiem_ram(self.model_tag, self.chieu)

        import torch, open_clip
        self._torch = torch

        # `pretrained` là ĐƯỜNG DẪN FILE CỤC BỘ (không phải tag như "webli")
        # khi ma trận được encode bằng scripts/08_encode.py -> open_clip
        # KHÔNG tự lấy được cấu hình tiền xử lý ảnh (kích cỡ) đi kèm tag đó,
        # ảnh sẽ bị resize sai (mặc định 224) — đúng bẫy đã ghi ở
        # docs/08_nhat_ky_encode_GPU.md §4. Tên tag SigLIP2 của open_clip có
        # quy ước kết thúc bằng kích cỡ ảnh thật (`...SigLIP2-378` = 378px,
        # xem A10.3/A17) — khác CLIP thường (`ViT-B-32` thì 32 là patch size,
        # không phải kích cỡ ảnh). Chỉ ép khi tải từ file cục bộ VÀ đúng quy
        # ước SigLIP2, để không áp nhầm cho CLIP.
        import re
        ep_kich_co = None
        m = re.search(r"SigLIP2-(\d+)$", self.model_tag)
        if m and not str(self.pretrained).lower().startswith(
                ("webli", "openai", "laion", "datacomp", "dfn")):
            ep_kich_co = int(m.group(1))

        self.model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_tag, pretrained=self.pretrained,
            force_image_size=ep_kich_co)
        self.model.eval()
        self.tok = open_clip.get_tokenizer(self.model_tag)

        # Model và ma trận phải cùng không gian vector. Số chiều lệch là dấu
        # hiệu rõ ràng nhất, và bắt được ngay lúc khởi tạo thay vì để kết quả
        # sai âm thầm.
        thu = self.encode_text("kiểm tra số chiều")
        if thu.shape[0] != self.chieu:
            raise SystemExit(
                f"Model {self.model_tag}/{self.pretrained} sinh vector "
                f"{thu.shape[0]} chiều nhưng {matrix} có {self.chieu} chiều. "
                f"Sai cặp model/ma trận.")

    def encode_text(self, cau: str) -> np.ndarray:
        """Vector truy vấn, đã chuẩn hóa L2 (bắt buộc, để `M @ q` là cosine)."""
        with self._torch.no_grad():
            v = self.model.encode_text(self.tok([cau]))[0].numpy().astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def encode_image(self, anh: list) -> np.ndarray:
        """Vector cho một loạt ảnh PIL, đã chuẩn hóa L2 — CÙNG không gian với
        `encode_text()` nên so cosine trực tiếp được (`v @ q`).

        Dùng cho khung trích dày (`trich_day.KhungDay.anh`) — những khung này
        KHÔNG có sẵn trong ma trận `.npy` (không có `row_id` thật, xem
        docstring đầu `trich_day.py`), nên phải tự encode mới đưa vào so sánh
        được. `self._preprocess` là transform ĐÚNG kích cỡ của model này —
        khớp `force_image_size` đã ép ở `__init__` nếu có.
        """
        if not anh:
            return np.zeros((0, self.chieu), dtype=np.float32)
        with self._torch.no_grad():
            lo = self._torch.stack([self._preprocess(a) for a in anh])
            v = self.model.encode_image(lo).numpy().astype(np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    # Nhân theo lô 20.000 dòng. Số này giữ bộ đệm tạm dưới ~100 MB ở mọi số
    # chiều đang dùng, mà vẫn đủ lớn để BLAS chạy hết tốc.
    LO = 20_000

    def _nhan(self, q: np.ndarray) -> np.ndarray:
        """`mat @ q` theo LÔ, không dựng bản sao float32 của cả ma trận.

        ⚠️ `np.asarray(self.mat) @ q` (bản cũ) nổ RAM với ma trận float16 nhiều
        chiều: numpy phải nâng kiểu để nhân, tức cấp phát một bản float32 của
        TOÀN BỘ ma trận. Với `clip_siglip2.npy` (177.321 × 1152 float16) đó là
        **817 MB cho mỗi truy vấn**, trên máy 7,7 GB thì chết hoặc thrash.
        Với `clip.npy` (512 chiều, float32) thì không lộ ra vì không phải nâng
        kiểu — nên lỗi này chỉ xuất hiện đúng lúc đổi sang model mạnh hơn.
        """
        n = self.mat.shape[0]
        ra = np.empty(n, dtype=np.float32)
        for i in range(0, n, self.LO):
            j = min(i + self.LO, n)
            ra[i:j] = np.asarray(self.mat[i:j], dtype=np.float32) @ q
        return ra

    def dong_da_encode(self) -> np.ndarray:
        """Mặt nạ bool: dòng nào có vector THẬT (khác vector 0).

        Ma trận chạy thử của `08_encode.py` luôn đủ 177.321 dòng nhưng phần
        chưa encode để 0. Cần biết phần nào thật để khóa bể ứng viên khi so
        cấu hình — xem `be_chung()`.
        """
        return np.abs(np.asarray(self.mat[:, :8], dtype=np.float32)).sum(1) > 0

    def tim(self, cau, k=100, video_id=None, chi_co_anh=False,
            moi_video=None, be=None) -> list[Candidate]:
        """Trả về tối đa `k` ứng viên, điểm cao xuống thấp.

        `cau` nhận cả một chuỗi lẫn danh sách chuỗi. Nhiều biến thể của cùng
        một ý thì lấy ĐIỂM CAO NHẤT trên từng keyframe, không lấy trung bình:
        một cách diễn đạt trúng là đủ, không nên bị các cách diễn đạt trượt
        kéo xuống.

        `moi_video` là ràng buộc đa dạng của PHẦN C mục 2. Để None ở tầng kênh
        (mặc định) và áp một lần duy nhất sau khi RRF — áp sớm sẽ cắt mất ứng
        viên mà các kênh khác còn cần.

        `be` là mặt nạ bool giới hạn bể ứng viên. **BẮT BUỘC dùng khi so hai
        ma trận có độ phủ khác nhau** — xem `be_chung()`.
        """
        cac_cau = [cau] if isinstance(cau, str) else list(cau)
        sim = np.max([self._nhan(self.encode_text(c)) for c in cac_cau], axis=0)

        if be is not None:
            sim = np.where(np.asarray(be, dtype=bool), sim, -9.0)
        if video_id is not None:
            sim = np.where((self.master.video_id == video_id).values, sim, -9.0)
        if chi_co_anh:
            sim = np.where(self.master.kf_path.notna().values, sim, -9.0)

        lay = min(len(sim), k * (moi_video or 1) + 200)
        top = np.argpartition(-sim, lay - 1)[:lay]
        top = top[np.argsort(-sim[top])]
        top = top[sim[top] > -1]

        ra = [Candidate(row_id=int(i), video_id=r.video_id,
                        frame_idx=int(r.frame_idx), score=float(sim[i]),
                        source="clip",
                        meta={"pts_time": float(r.pts_time), "fps": float(r.fps),
                              "kf_n": int(r.kf_n), "title": r.title})
              for i, r in zip(top, self.master.iloc[top].itertuples())]

        if moi_video:
            dem, loc = {}, []
            for c in ra:
                if dem.get(c.video_id, 0) < moi_video:
                    dem[c.video_id] = dem.get(c.video_id, 0) + 1
                    loc.append(c)
            ra = loc
        return ra[:k]


class KenhAnhCache(KenhAnh):
    """Kênh 1 chạy KHÔNG NẠP MODEL, bằng vector truy vấn đã mã hoá sẵn.

    GỠ CHỐT RAM MÀ KHÔNG ĐỔI GÌ KHÁC
    =================================

    Máy 7,7 GB không nạp nổi `ViT-SO400M-14-SigLIP2-378` (~3,5 GB) nên mọi phép
    đo dính kênh 1 đều tắc — kể cả phép đo TRAKE và RRF(SigLIP2, OCR), hai thứ
    đáng giá nhất còn lại. Nhưng để ý kỹ: **model chỉ được dùng đúng một việc —
    biến câu truy vấn thành vector 1152 chiều.** Ma trận ảnh 177.321 × 1152 thì
    đã nằm sẵn trên đĩa.

    Mà số truy vấn là **hữu hạn và biết trước**: 24 câu của bộ đề, 115 câu tập
    dev. Mã hoá chúng MỘT LẦN trên máy khoẻ (hoặc Colab/Kaggle) ra một file vài
    trăm KB, rồi máy yếu làm toàn bộ truy hồi bằng numpy thuần.

        # máy khoẻ, chạy một lần
        python scripts/25_ma_hoa_truy_van.py --de de_thi --ra index/truy_van.npz

        # máy yếu, từ đó về sau
        kenh = KenhAnhCache('./index', 'index/truy_van.npz')
        kq = kenh.tim("người phụ nữ thái cà chua", k=100)

    ⚠️ **`tim()` KHÔNG được viết lại ở đây — nó thừa kế nguyên vẹn.** Đó là chủ
    ý: chỉ `encode_text` đổi từ "chạy model" thành "tra bảng". Viết lại `tim()`
    là mở đường cho hai nhánh code lệch nhau âm thầm, mà lệch ở tầng này thì
    mọi con số đo được trên máy yếu sẽ không so được với số đo trên máy khoẻ.

    ⚠️ Truy vấn KHÔNG có trong cache thì **ném lỗi, không đoán bừa**. Trả về
    vector 0 sẽ cho ra 100 ứng viên ngẫu nhiên trông hoàn toàn hợp lệ — đúng
    loại hỏng im lặng mà cả repo này dựng để chặn.
    """

    def __init__(self, index_dir="./index", cache="index/truy_van.npz",
                 matrix=None, mmap=True):
        import json as _json
        d = Path(index_dir)
        z = np.load(cache, allow_pickle=False)
        ghi_chu = _json.loads(str(z["ghi_chu"]))

        self.model_tag = ghi_chu.get("model", MODEL_MAC_DINH)
        self.pretrained = ghi_chu.get("pretrained", PRETRAINED_MAC_DINH)
        matrix = matrix or ghi_chu.get("matrix", "clip.npy")

        self.master = pd.read_parquet(d / "master.parquet")
        self.mat = np.load(d / matrix, mmap_mode="r" if mmap else None)
        self.chieu = self.mat.shape[1]

        if len(self.master) != self.mat.shape[0]:
            raise SystemExit(
                f"master.parquet ({len(self.master)} dòng) và {matrix} "
                f"({self.mat.shape[0]} dòng) LỆCH NHAU.")

        vec = np.asarray(z["vec"], dtype=np.float32)
        if vec.shape[1] != self.chieu:
            raise SystemExit(
                f"Cache có vector {vec.shape[1]} chiều nhưng {matrix} có "
                f"{self.chieu} chiều. Sai cặp cache/ma trận — cache này mã hoá "
                f"bằng {self.model_tag}.")
        self._cache = {str(c): vec[i] for i, c in enumerate(z["cau"])}
        self.nguon_cache = str(cache)

    def encode_text(self, cau: str) -> np.ndarray:
        v = self._cache.get(cau)
        if v is None:
            raise KeyError(
                f"Truy vấn chưa có trong cache {self.nguon_cache}:\n"
                f"  {cau[:120]!r}\n"
                f"Cache có {len(self._cache)} câu. Mã hoá thêm bằng:\n"
                f"  python scripts/25_ma_hoa_truy_van.py --them \"...\"")
        return v

    def co_du(self, cac_cau) -> list[str]:
        """Các câu CHƯA có trong cache. Gọi trước khi chạy để hỏng sớm."""
        return [c for c in cac_cau if c not in self._cache]


def be_chung(*kenh: "KenhAnh") -> np.ndarray:
    """Bể ứng viên chung: chỉ những dòng CẢ HAI ma trận đều đã encode thật.

    ⚠️ BẮT BUỘC khi so hai cấu hình có độ phủ khác nhau — ví dụ `clip.npy`
    (đủ 177.321 dòng thật) với ma trận chạy thử của `08_encode.py` (chỉ vài
    nghìn dòng thật, phần còn lại là vector 0).

    Không khóa bể là so "tìm trong 177 nghìn" với "tìm trong vài nghìn", và
    bên có bể nhỏ hơn thắng vì lý do KHÔNG liên quan gì tới chất lượng model.

    Đã đo trên tập dev 12 câu, CÙNG một ma trận `clip.npy`, chỉ đổi bể:

        bể đầy đủ 177.321 keyframe   ->  0,5167
        bể thu hẹp   2.328 keyframe  ->  0,8000      (+0,2833 THUẦN ẢO GIÁC)

    Để so sánh: toàn bộ lợi ích đội AIC'25 thu được khi thêm SigLIP2 chỉ là
    +0,07 điểm/câu (A8.2). Ảo giác này lớn gấp 4 lần thứ cần đo — không khóa
    bể thì kết luận ra ngược.

        be = be_chung(kenh_b32, kenh_siglip2)
        a  = cham(dev, lambda c: kenh_b32.tim(c.cau_hoi, k=100, be=be))
        b  = cham(dev, lambda c: kenh_siglip2.tim(c.cau_hoi, k=100, be=be))
    """
    if not kenh:
        raise ValueError("cần ít nhất một kênh")
    ra = kenh[0].dong_da_encode()
    for k in kenh[1:]:
        if len(k.master) != len(kenh[0].master):
            raise ValueError("các kênh không cùng bảng cái")
        ra &= k.dong_da_encode()
    return ra


# Truy vấn dùng làm mốc cho bài test cố định ở tests/test_dense.py.
CAU_MOC = "a person riding a motorbike on the street"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default=CAU_MOC)
    ap.add_argument("--index", default=Path("./index"), type=Path)
    ap.add_argument("--matrix", default="clip.npy")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ghi-moc", action="store_true",
                    help="ghi mốc hiện tại ra tests/moc_dense.json")
    a = ap.parse_args()

    kenh = KenhAnh(a.index, matrix=a.matrix)
    print(f"{a.matrix}: {kenh.mat.shape}  |  {kenh.model_tag} / {kenh.pretrained}\n")

    kq = kenh.tim(a.query, k=a.k)
    print(f'"{a.query}"\n')
    print(f"{'cos':>6}  {'video_id':<10} {'frame_idx':>9}  {'row_id':>7}  tiêu đề")
    print("-" * 92)
    for c in kq:
        print(f"{c.score:6.3f}  {c.video_id:<10} {c.frame_idx:>9}  {c.row_id:>7}  "
              f"{c.meta['title'][:44]}")

    if a.ghi_moc:
        moc = Path("tests/moc_dense.json")
        moc.parent.mkdir(exist_ok=True)
        dau = kenh.tim(CAU_MOC, k=1)[0]
        moc.write_text(json.dumps({
            "cau": CAU_MOC, "model": kenh.model_tag, "pretrained": kenh.pretrained,
            "chieu": kenh.chieu, "row_id": dau.row_id,
            "video_id": dau.video_id, "cos": round(dau.score, 4),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nĐã ghi mốc -> {moc}")


if __name__ == "__main__":
    main()
