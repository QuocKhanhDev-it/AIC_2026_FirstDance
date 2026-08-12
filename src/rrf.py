"""
rrf.py — Hợp nhất N danh sách xếp hạng bằng Reciprocal Rank Fusion.

Vì sao RRF chứ không phải cộng có trọng số: cosine CLIP nằm khoảng 0,25–0,40
còn BM25 không chặn trên. Cộng thẳng thì BM25 nuốt hết; muốn cộng được phải
chuẩn hóa thang điểm, mà chuẩn hóa thế nào là một siêu tham số nữa phải dò.
RRF chỉ nhìn THỨ HẠNG nên miễn nhiễm với chuyện đó.

    RRF_score(d) = Σ_i  1 / (k + rank_i(d))

`k = 60` là giá trị chuẩn trong bài gốc (Cormack 2009) và cũng là giá trị đội
AIC'25 dùng.

NHẬN N DANH SÁCH, KHÔNG HARDCODE SỐ KÊNH. Đây là chủ ý: hàm này dùng cho cả
hai phép hợp nhất khác nhau trong kế hoạch —

  * hợp nhất 4–5 KÊNH (CLIP, BM25 metadata, BM25 OCR, objects, caption)
  * hợp nhất 2 MODEL trong cùng kênh 1 (ViT-B/32 + SigLIP2)

nên khi máy GPU trả về ma trận thứ hai, việc thêm nó vào chỉ là truyền thêm
một danh sách. Không sửa dòng nào ở đây.

Dùng:
    from rrf import hop_nhat
    cuoi = hop_nhat([kq_clip, kq_bm25, kq_objects], k=60)
"""

from collections import defaultdict

try:
    from .schema import Candidate
except ImportError:
    from schema import Candidate

K_MAC_DINH = 60


def hop_nhat(cac_danh_sach: list[list[Candidate]], k: int = K_MAC_DINH,
             trong_so: list[float] | None = None) -> list[Candidate]:
    """Gộp nhiều danh sách ứng viên thành một, xếp theo điểm RRF giảm dần.

    Mỗi danh sách phải đã được sắp sẵn theo thứ tự tốt->kém của kênh đó; hàm
    này chỉ đọc VỊ TRÍ, không đọc `score`.

    `trong_so` để trống thì mọi kênh ngang nhau. CHỈ chỉnh khi tập dev của
    Khánh chứng minh được là có lợi — dò trọng số bằng cảm tính là cách nhanh
    nhất để tự lừa mình.

    Ứng viên trùng nhau giữa các kênh được nhận diện bằng `row_id`.
    """
    if trong_so is None:
        trong_so = [1.0] * len(cac_danh_sach)
    if len(trong_so) != len(cac_danh_sach):
        raise ValueError(f"{len(cac_danh_sach)} danh sách nhưng "
                         f"{len(trong_so)} trọng số")

    diem: dict[int, float] = defaultdict(float)
    nguon: dict[int, list[str]] = defaultdict(list)
    goc: dict[int, Candidate] = {}

    for ds, w in zip(cac_danh_sach, trong_so):
        for hang, c in enumerate(ds, start=1):
            diem[c.row_id] += w / (k + hang)
            nguon[c.row_id].append(f"{c.source or '?'}@{hang}")
            goc.setdefault(c.row_id, c)

    ra = []
    for rid, d in sorted(diem.items(), key=lambda x: -x[1]):
        c = goc[rid]
        ra.append(Candidate(row_id=c.row_id, video_id=c.video_id,
                            frame_idx=c.frame_idx, score=d, source="rrf",
                            meta={**c.meta, "nguon": nguon[rid]}))
    return ra


def gioi_han_moi_video(ung_vien: list[Candidate], moi_video: int = 2,
                       k: int | None = None) -> list[Candidate]:
    """Ràng buộc đa dạng của PHẦN C mục 2: mỗi video tối đa `moi_video` slot.

    Áp MỘT LẦN, SAU khi RRF. Áp sớm ở từng kênh sẽ cắt mất ứng viên mà kênh
    khác còn cần.

    ⚠️ Ràng buộc này tính theo `video_id` nên KHÔNG bắt được bản sao trong
    CÙNG một video — mà 11,83% keyframe có bản sao cùng video ở cosine ≥ 0,99
    (A5.6, riêng L25 tới 49,82%). Chạy `dedup.gom_ban_sao()` trước hàm này.
    """
    dem, ra = {}, []
    for c in ung_vien:
        if dem.get(c.video_id, 0) < moi_video:
            dem[c.video_id] = dem.get(c.video_id, 0) + 1
            ra.append(c)
            if k and len(ra) >= k:
                break
    return ra
