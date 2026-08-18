"""
bm25.py — Bộ máy truy hồi văn bản, dùng chung cho KÊNH 2, 3 và 5.

Ba kênh còn thiếu của kế hoạch đều là **cùng một bài toán**: có một bảng
`(khóa, văn bản tiếng Việt)`, cho câu truy vấn, xếp hạng. Khác nhau đúng ở chỗ
văn bản từ đâu ra:

    kênh 2  metadata     873 video      title + description + keywords   ✅ có sẵn 100%
    kênh 3  OCR + ASR    177.321 khung  chữ trên hình + lời nói          ⬜ chờ TV4
    kênh 5  caption VLM  177.321 khung  mô tả cảnh do VLM sinh           ⬜ chờ khóa API

Viết ba lần là ba lần mắc lại cùng những lỗi tiếng Việt. Ở đây một lần.

VÌ SAO TỰ VIẾT BM25 THAY VÌ `rank_bm25`
=======================================

Cùng lý do đã bỏ Milvus ở B1: 873 tài liệu × 1.300 ký tự là bài toán nhỏ, mà
thêm một phụ thuộc là thêm một thứ phải cài trên sáu máy. Toàn bộ engine dưới
đây gọn hơn phần docstring của nó.

HAI CHUYỆN TIẾNG VIỆT PHẢI XỬ ĐÚNG
==================================

**1. Dấu.** Trong phiên thi người ta gõ nhanh và bỏ dấu — chính bài báo AIC'25
có truy vấn `"giai phong khi hidro"`. Mà `keywords` của kho cũng lẫn cả hai
kiểu trong cùng một video:

    "món ngon mỗi ngày ... mon ngon moi ngay ... thực đơn mổi ngày"

Cách làm ở đây: **dựng HAI chỉ mục** — một trên token có dấu, một trên token đã
bỏ dấu — rồi **cộng điểm cả hai**. Không có siêu tham số nào:

    khớp đúng cả dấu  ->  ăn điểm ở CẢ HAI chỉ mục
    khớp khi bỏ dấu   ->  ăn điểm ở MỘT chỉ mục

nên tài liệu khớp chính xác tự động xếp trên tài liệu chỉ khớp mờ, mà truy vấn
không dấu vẫn tìm ra được. Cách khác — nở token truy vấn ra mọi biến thể có dấu
— thì `"cho"` nở thành {chó, chợ, chờ, chỗ, cho} và kéo về đủ thứ rác.

**2. Từ ghép.** Tiếng Việt viết rời từng âm tiết: `"xe máy"` là một từ nhưng
hai token. Chỉ dùng unigram thì `"xe máy"` khớp cả video nói về `"máy xay"` và
`"xe đạp"`. Nên **thêm bigram** — `"xe_máy"` là một token riêng, hiếm hơn nhiều
nên IDF cao hơn nhiều, và nó tự động thắng. Không cần bộ tách từ
(`underthesea`/`pyvi`): thêm một phụ thuộc nặng để làm việc mà IDF đã làm.

CẢNH BÁO VỀ KÊNH 2 — ĐỌC TRƯỚC KHI TIN CON SỐ
==============================================

Metadata mô tả **cả video**, không mô tả từng khung. Kênh này chỉ ra được
*video nào*, hoàn toàn không biết *khung nào trong video*. Một video trung bình
203 khung, nên kể cả khi nó chỉ đúng video ở hạng 1 thì khung đúng vẫn nằm đâu
đó trong 203 khung ấy theo thứ tự thời gian — tức gần như ngẫu nhiên.

**Đừng đọc điểm R@1 của kênh này là năng lực của nó.** Giá trị của nó nằm ở chỗ
hợp nhất: nó thu hẹp còn vài video, kênh ảnh chọn khung. Đó đúng là việc RRF
sinh ra để làm.

Dùng:
    from bm25 import KenhVanBan
    k2 = KenhVanBan.tu_metadata(master)              # kênh 2
    k5 = KenhVanBan.tu_bang_khung(master, caption)   # kênh 5, sau khi có caption
    kq = k2.tim("người phụ nữ thái dứa trên thớt gỗ", k=100)
"""

import argparse
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .schema import Candidate
except ImportError:                     # chạy trực tiếp: python src/bm25.py
    from schema import Candidate

K1 = 1.5      # bão hòa tần suất — giá trị chuẩn của Okapi BM25
B = 0.75      # mức phạt tài liệu dài

# Bỏ ký tự không phải chữ/số. Giữ nguyên chữ có dấu (\w trong Python re đã bao
# chữ Unicode), nên không cần liệt kê bảng chữ cái tiếng Việt.
_TACH = re.compile(r"\w+", re.UNICODE)

# `#\w+` chứ KHÔNG phải `#\S+`. `\S+` chạy tới khoảng trắng gần nhất nên
# `"#amthuc,rau cu"` bị ăn luôn cả `rau`. Đo trên kho hiện tại: 0 token mất
# (hashtag ở đây luôn có khoảng trắng theo sau), nên đây là siết PHÒNG XA —
# nhưng kênh 3 sắp đẩy chữ OCR qua đúng hàm này, mà chữ OCR thì bẩn hơn nhiều.
_RAC = re.compile(r"https?://\S+|#\w+|@\w+", re.UNICODE)


def don_metadata(s: str) -> str:
    """Xóa URL, hashtag, mention khỏi văn bản trước khi đưa vào chỉ mục.

    Thay bằng `'. '` chứ không phải `' '`: dấu chấm làm `tach()` NGẮT CỤM, nên
    hai từ thật ở hai bên đoạn rác không bị nối thành bigram lai
    (`"tại https://... nấu"` -> `tại_nấu`, một cụm không có trong văn bản gốc).

    ⚠️ Đã đo: trên 873 video, đổi `' '` thành `'. '` chỉ khác **đúng 1 bigram**
    trong 14.317 token (`của_trên`). Chi tiết ở A15 — ghi lại để người sau không
    tưởng đây là chỗ đáng tinh chỉnh. Cái thật sự làm việc là VIỆC DỌN RÁC.
    """
    return _RAC.sub(". ", s or "")



def bo_dau(s: str) -> str:
    """Thường hóa, bỏ dấu thanh và dấu mũ, `đ` -> `d`.

    Giống hệt `objects.bo_dau` — cố ý lặp lại chứ không import chéo, để hai
    kênh không dính vào nhau. `đ` phải xử riêng: nó là ký tự độc lập trong
    Unicode, `NFD` không tách ra được.
    """
    s = unicodedata.normalize("NFD", s.strip().lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("đ", "d").replace("Đ", "d")


def tach(s: str, bigram: bool = True) -> list[str]:
    """Chuỗi -> danh sách token: unigram, cộng bigram nếu bật.

    Bigram nối bằng `_` để không lẫn với unigram. Chỉ nối trong CÙNG một cụm
    ngăn cách bởi dấu câu — `"xe máy. Hôm nay"` không được sinh ra
    `"máy_hôm"`, vì hai từ đó không đứng cạnh nhau về nghĩa.
    """
    ra = []
    for cum in re.split(r"[^\w\s]+", s.lower()):
        tu = _TACH.findall(cum)
        ra += tu
        if bigram:
            ra += [f"{a}_{b}" for a, b in zip(tu, tu[1:])]
    return ra


class BM25:
    """Okapi BM25 trên một danh sách chuỗi. Không phụ thuộc gì ngoài numpy."""

    def __init__(self, tai_lieu: list[str], k1: float = K1, b: float = B,
                 bigram: bool = True):
        self.n = len(tai_lieu)
        self.k1, self.b = k1, b
        self.bigram = bigram

        dai = np.zeros(self.n, dtype=np.float32)
        tho: dict[str, dict[int, int]] = {}
        for i, d in enumerate(tai_lieu):
            tok = tach(d or "", bigram)
            dai[i] = len(tok)
            for t in tok:
                tho.setdefault(t, {})
                tho[t][i] = tho[t].get(i, 0) + 1

        self.dai = dai
        self.dai_tb = float(dai.mean()) if self.n and dai.sum() else 1.0

        # Chốt sẵn phần mẫu số không phụ thuộc truy vấn: BM25 gọi hàng nghìn
        # lần khi chấm tập dev, tính lại mỗi lần là phí.
        self._chuan = k1 * (1 - b + b * dai / self.dai_tb)

        self.chi_muc: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}
        for t, m in tho.items():
            ids = np.fromiter(m.keys(), dtype=np.int32, count=len(m))
            tf = np.fromiter(m.values(), dtype=np.float32, count=len(m))
            self.chi_muc[t] = (ids, tf)
            # Dạng có +1 bên trong log: không bao giờ âm, kể cả token có mặt ở
            # quá nửa số tài liệu. Dạng cổ điển cho IDF ÂM trong trường hợp đó
            # và biến token phổ biến thành hình phạt — sai hẳn về ý nghĩa.
            self.idf[t] = math.log(1 + (self.n - len(ids) + 0.5) / (len(ids) + 0.5))

    def diem(self, cau: str) -> np.ndarray:
        """Điểm BM25 của MỌI tài liệu cho câu truy vấn. Không khớp gì -> 0."""
        ra = np.zeros(self.n, dtype=np.float32)
        for t in set(tach(cau, self.bigram)):
            hit = self.chi_muc.get(t)
            if hit is None:
                continue
            ids, tf = hit
            ra[ids] += self.idf[t] * tf * (self.k1 + 1) / (tf + self._chuan[ids])
        return ra


class KenhVanBan:
    """Một kênh truy hồi dựa trên văn bản, trả về `list[Candidate]`.

    Nhận `khoa_dong`: với mỗi tài liệu, mảng `row_id` của những keyframe mà nó
    mô tả. Kênh cấp video (kênh 2) thì một tài liệu ứng với hàng trăm keyframe;
    kênh cấp khung (kênh 3, 5) thì ứng với đúng một.
    """

    def __init__(self, van_ban: list[str], khoa_dong: list[np.ndarray],
                 master: pd.DataFrame, ten: str = "bm25", bigram: bool = True):
        if len(van_ban) != len(khoa_dong):
            raise ValueError(f"{len(van_ban)} tài liệu nhưng {len(khoa_dong)} khóa")
        self.master, self.ten = master, ten
        self.khoa_dong = khoa_dong
        self.co_dau = BM25(van_ban, bigram=bigram)
        self.khong_dau = BM25([bo_dau(v or "") for v in van_ban], bigram=bigram)

    def __len__(self):
        return len(self.khoa_dong)

    def diem_tai_lieu(self, cau: str) -> np.ndarray:
        """Điểm từng TÀI LIỆU (chưa nở ra keyframe).

        Cộng hai chỉ mục — xem phần "Dấu" ở đầu file. Khớp đúng dấu ăn điểm cả
        hai lần, khớp mờ ăn một lần.
        """
        return self.co_dau.diem(cau) + self.khong_dau.diem(bo_dau(cau))

    def tim(self, cau, k: int = 100, moi_video: int | None = None,
            be=None) -> list[Candidate]:
        """Tối đa `k` ứng viên, điểm cao xuống thấp.

        `cau` nhận cả chuỗi lẫn danh sách chuỗi; nhiều cách diễn đạt thì lấy
        ĐIỂM CAO NHẤT trên từng tài liệu — giống `dense.KenhAnh.tim`, để hai
        kênh hành xử như nhau khi đưa vào RRF.

        `moi_video` giới hạn số keyframe mỗi video. **Với kênh cấp video thì
        đây là tham số quyết định**: để None là một video khớp sẽ đổ cả 203
        keyframe của nó vào danh sách và chiếm sạch top-100. Đúng hay sai tùy
        bài — phải ĐO, xem `scripts/15_do_bm25.py`.
        """
        cac_cau = [cau] if isinstance(cau, str) else list(cau)
        d = np.max([self.diem_tai_lieu(c) for c in cac_cau], axis=0)

        ra, dem = [], {}
        for i in np.argsort(-d):
            if d[i] <= 0:
                break                       # hết tài liệu khớp; đừng bịa thêm
            rid = self.khoa_dong[i]
            if be is not None:
                rid = rid[np.asarray(be, dtype=bool)[rid]]
            for r in rid:
                v = self.master.video_id.iloc[r]
                if moi_video and dem.get(v, 0) >= moi_video:
                    continue
                dem[v] = dem.get(v, 0) + 1
                ra.append((int(r), float(d[i])))
                if len(ra) >= k:
                    break
            if len(ra) >= k:
                break

        m = self.master
        return [Candidate(row_id=r, video_id=m.video_id.iloc[r],
                          frame_idx=int(m.frame_idx.iloc[r]), score=s,
                          source=self.ten,
                          meta={"pts_time": float(m.pts_time.iloc[r]),
                                "fps": float(m.fps.iloc[r]),
                                "kf_n": int(m.kf_n.iloc[r]),
                                "title": m.title.iloc[r]})
                for r, s in ra]

    # ---- hai cách dựng kênh -------------------------------------------------

    @classmethod
    def tu_metadata(cls, master: pd.DataFrame, bigram: bool = True) -> "KenhVanBan":
        """KÊNH 2 — một tài liệu mỗi VIDEO, ghép title + description + keywords.

        `title` lặp ba lần: BM25 không có khái niệm trường nào quan trọng hơn,
        mà tiêu đề 62 ký tự thì chìm nghỉm cạnh description 954 ký tự. Lặp là
        cách tăng trọng số trường mà không phải sửa công thức — thủ thuật cũ
        của IR, và nó đo được: bật/tắt bằng `scripts/15_do_bm25.py`.

        Ghép bằng `'. '` chứ không phải `' '`: ghép bằng khoảng trắng thì
        `tach()` sinh bigram BẮC CẦU QUA BIÊN TRƯỜNG — `"…VIVU TV"` nối với
        lần lặp sau đẻ ra `tv_món`, một cụm không có trong văn bản gốc. Lỗi này
        do NgThanhDat-ne tìm ra.

        **`title` cũng phải dọn rác**, không chỉ description: 13/873 tiêu đề có
        hashtag, mà tiêu đề lặp 3 lần nên rác ở đây ăn trọng số GẤP BA.
        """
        g = master.groupby("video_id", sort=True)
        vids, van_ban, khoa = [], [], []
        for v, sub in g:
            r = sub.iloc[0]
            t = don_metadata(str(r.title or ""))
            desc = don_metadata(str(r.description or ""))
            kw = don_metadata(str(r.keywords or ""))
            van_ban.append(". ".join([t, t, t, desc, kw]))
            khoa.append(sub.index.to_numpy(dtype=np.int64))
            vids.append(v)
        k = cls(van_ban, khoa, master, ten="metadata", bigram=bigram)
        k.video_id = vids
        return k

    @classmethod
    def tu_bang_khung(cls, master: pd.DataFrame, bang: pd.DataFrame,
                      cot: str = "caption", ten: str = "caption",
                      bigram: bool = True) -> "KenhVanBan":
        """KÊNH 3 hoặc 5 — một tài liệu mỗi KEYFRAME.

        `bang` cần hai cột: `row_id` và cột văn bản. Chỉ những `row_id` có
        trong bảng mới vào chỉ mục — keyframe chưa sinh caption không tự nhiên
        biến thành tài liệu rỗng, vì tài liệu rỗng làm lệch `dai_tb` và qua đó
        lệch điểm của MỌI tài liệu khác.
        """
        thieu = {"row_id", cot} - set(bang.columns)
        if thieu:
            raise ValueError(f"bảng thiếu cột {sorted(thieu)}")
        b = bang[bang[cot].fillna("").str.strip() != ""]
        return cls(b[cot].tolist(),
                   [np.array([r], dtype=np.int64) for r in b.row_id],
                   master, ten=ten, bigram=bigram)


def main():
    ap = argparse.ArgumentParser(description="Thử kênh 2 (metadata) từ dòng lệnh")
    ap.add_argument("query", nargs="?", default="người phụ nữ thái dứa trên thớt gỗ")
    ap.add_argument("--index", default=Path("./index"), type=Path)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--moi-video", type=int, default=1)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    kenh = KenhVanBan.tu_metadata(master)
    print(f"kênh metadata: {len(kenh)} video, "
          f"{len(kenh.co_dau.chi_muc):,} token có dấu / "
          f"{len(kenh.khong_dau.chi_muc):,} không dấu\n")

    print(f'"{a.query}"\n')
    print(f"{'điểm':>7}  {'video_id':<10} {'row_id':>7}  tiêu đề")
    print("-" * 92)
    for c in kenh.tim(a.query, k=a.k, moi_video=a.moi_video):
        print(f"{c.score:7.2f}  {c.video_id:<10} {c.row_id:>7}  "
              f"{str(c.meta['title'])[:58]}")


if __name__ == "__main__":
    main()
