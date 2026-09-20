"""
hop_diem.py — Hợp nhất các kênh bằng ĐIỂM (đã chuẩn hoá), thay vì bằng THỨ HẠNG.

VÌ SAO CÓ FILE NÀY, KHI `rrf.py` ĐÃ NÓI RÕ LÝ DO KHÔNG CỘNG ĐIỂM

`rrf.py` viết: *"cosine CLIP nằm khoảng 0,25–0,40 còn BM25 không chặn trên.
Cộng thẳng thì BM25 nuốt hết; muốn cộng được phải chuẩn hoá thang điểm, mà
chuẩn hoá thế nào là một siêu tham số nữa phải dò."* Lập luận đó đúng, nhưng nó
là lý do **né**, không phải một phép đo.

A84 đo được thứ khiến chuyện này đáng đo lại: giữa kênh 1 và kênh 5, `chồng@20`
chỉ **4,3%** và Spearman **0,093**. Nghĩa là hai kênh gần như không bao giờ đề
cử cùng `row_id`. RRF chỉ cộng hưởng khi có trùng `row_id`; không trùng thì nó
**chỉ đan xen hai danh sách** — và A14 đã đo được đan xen làm TỆ ĐI, vì mỗi ứng
viên tốt của kênh mạnh bị một ứng viên của kênh yếu đẩy lùi một bậc.

Hợp nhất theo điểm không có tính chất đó: một ứng viên mạnh vượt trội ở một
kênh vẫn giữ được BIÊN ĐỘ của nó, thay vì bị quy về "hạng 1" ngang với một ứng
viên chỉ hơi nhỉnh trong danh sách của kênh yếu.

⚠️ CHƯA BẬT MẶC ĐỊNH. `run.py` vẫn dùng RRF. File này để `85_do_hop_nhat_diem.py`
đo; chỉ đổi mặc định nếu thắng trên tập đề thật ở CẢ HAI mức dung sai.

BA CHỖ DỄ SAI, ĐÃ CẮN THẬT HOẶC SUÝT CẮN

1. **Giá trị bù cho ứng viên VẮNG MẶT ở một kênh.** A60 từng gán `-1e9` cho ứng
   viên chưa được VLM chấm, và thế là mọi ứng viên chưa chấm bị đẩy xuống đáy —
   đo ra một kết luận sai. Ở đây "vắng mặt" chỉ nghĩa là *không lọt top-k của
   kênh đó*, không phải *kênh đó nói nó sai*. Nên bù bằng **giá trị thấp nhất
   kênh đó thực sự trả về** (`bu="min"`), tức "coi như kém nhất trong số đã
   thấy" — chứ không phải một hình phạt vô hạn.

2. **Chuẩn hoá theo TỪNG TRUY VẤN, không theo cả kho.** Thang điểm BM25 của câu
   3 từ khoá và câu 15 từ khoá khác hẳn nhau. Chuẩn hoá gộp là trộn hai thang.

3. **BM25 lệch nặng** — phần lớn ứng viên gần 0 vì không khớp từ nào, một nhúm
   vọt cao. z-score trên phân phối đó bị cái đuôi kéo. `truoc="log1p"` có để đo
   xem nắn đuôi trước có khác không; đừng bật vì "nghe hợp lý", đo rồi hãy bật.

VỀ "DÙNG logit_scale/logit_bias CỦA SigLIP2 ĐỂ RA XÁC SUẤT HIỆU CHỈNH"

`σ(s·τ + b)` là hàm **đơn điệu tăng** theo cosine `s`. Nó KHÔNG thêm được thông
tin hiệu chỉnh nào theo từng truy vấn — nó là một phép uốn trục, giống hệt nhau
cho mọi câu, nên **thứ hạng nội bộ một kênh không đổi một chút nào**. Thứ nó
đổi là BIÊN ĐỘ tương đối khi đem cộng với kênh khác. Và nếu thứ ta cần chỉ là
biên độ, thì `τ` **dò được như một tham số thường** — làm vậy còn mạnh hơn dùng
giá trị của checkpoint, vì giá trị đó chỉ là một điểm trong dải dò (`cach=
"sigmoid"`, `tau`). Khỏi phải mở checkpoint, mà máy thi cũng không mở nổi.
"""

import math

try:
    from .schema import Candidate
except ImportError:
    from schema import Candidate

CACH = ("z", "minmax", "sigmoid")


def chuan_hoa(ds: list[Candidate], cach: str = "z", tau: float = 1.0,
              truoc: str | None = None) -> dict[int, float]:
    """Điểm thô của MỘT kênh cho MỘT truy vấn -> `{row_id: điểm đã chuẩn hoá}`.

    * `z`       — trừ trung bình, chia độ lệch chuẩn.
    * `minmax`  — kéo về [0, 1].
    * `sigmoid` — z rồi `1/(1+e^(-tau·z))`. `tau` lớn -> gần bậc thang (ứng viên
      đầu áp đảo); `tau` nhỏ -> gần tuyến tính. Đây là chỗ "nhiệt độ" vào.
    * `truoc="log1p"` — nắn đuôi trước khi chuẩn hoá, cho BM25.

    Danh sách rỗng trả về `{}`; danh sách mọi điểm bằng nhau trả về toàn 0 (chứ
    không phải chia cho 0).
    """
    if cach not in CACH:
        raise ValueError(f"cach phải là một trong {CACH}, nhận {cach!r}")
    if not ds:
        return {}

    s = [float(c.score) for c in ds]
    if truoc == "log1p":
        s = [math.log1p(max(x, 0.0)) for x in s]
    elif truoc is not None:
        raise ValueError(f"truoc phải là None hoặc 'log1p', nhận {truoc!r}")

    n = len(s)
    if cach == "minmax":
        lo, hi = min(s), max(s)
        v = [0.0] * n if hi <= lo else [(x - lo) / (hi - lo) for x in s]
    else:
        tb = sum(s) / n
        sd = math.sqrt(sum((x - tb) ** 2 for x in s) / n)
        v = [0.0] * n if sd <= 0 else [(x - tb) / sd for x in s]
        if cach == "sigmoid":
            # chặn mũ để không tràn với tau lớn
            v = [1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, tau * x))))
                 for x in v]

    return {c.row_id: x for c, x in zip(ds, v)}


def hop_nhat_diem(cac_danh_sach: list[list[Candidate]],
                  trong_so: list[float] | None = None,
                  cach: str = "z", tau: float = 1.0,
                  truoc: list[str | None] | None = None,
                  bu: str = "min") -> list[Candidate]:
    """Gộp N danh sách bằng tổng có trọng số của điểm ĐÃ CHUẨN HOÁ.

    Cùng chữ ký tinh thần với `rrf.hop_nhat` để hai cơ chế thay nhau được trong
    script đo mà không phải sửa chỗ gọi.

    `truoc` là danh sách song song với `cac_danh_sach` (mỗi kênh một phép nắn
    riêng — BM25 cần `log1p`, cosine thì không), hoặc `None` cho tất cả.

    `bu`: giá trị gán cho ứng viên VẮNG MẶT ở một kênh.
      * `"min"`  — thấp nhất kênh đó thật sự trả về. Mặc định, xem docstring
        đầu file: vắng mặt nghĩa là "không lọt top-k", không phải "sai".
      * `"zero"` — 0. Hợp với `minmax`, nơi 0 đã là đáy thang.
    """
    if trong_so is None:
        trong_so = [1.0] * len(cac_danh_sach)
    if len(trong_so) != len(cac_danh_sach):
        raise ValueError(f"{len(cac_danh_sach)} danh sách nhưng "
                         f"{len(trong_so)} trọng số")
    if truoc is None:
        truoc = [None] * len(cac_danh_sach)
    if len(truoc) != len(cac_danh_sach):
        raise ValueError(f"{len(cac_danh_sach)} danh sách nhưng "
                         f"{len(truoc)} phép nắn")
    if bu not in ("min", "zero"):
        raise ValueError(f"bu phải là 'min' hoặc 'zero', nhận {bu!r}")

    da_chuan = [chuan_hoa(ds, cach, tau, t)
                for ds, t in zip(cac_danh_sach, truoc)]
    day = [(min(d.values()) if d and bu == "min" else 0.0) for d in da_chuan]

    goc: dict[int, Candidate] = {}
    nguon: dict[int, list[str]] = {}
    for ds in cac_danh_sach:
        for hang, c in enumerate(ds, start=1):
            goc.setdefault(c.row_id, c)
            nguon.setdefault(c.row_id, []).append(f"{c.source or '?'}@{hang}")

    diem: dict[int, float] = {}
    for rid in goc:
        diem[rid] = sum(w * d.get(rid, day[i])
                        for i, (d, w) in enumerate(zip(da_chuan, trong_so)))

    ra = []
    for rid, d in sorted(diem.items(), key=lambda x: -x[1]):
        c = goc[rid]
        ra.append(Candidate(row_id=rid, video_id=c.video_id,
                            frame_idx=c.frame_idx, score=d, source="diem",
                            meta={**c.meta, "nguon": nguon[rid]}))
    return ra
