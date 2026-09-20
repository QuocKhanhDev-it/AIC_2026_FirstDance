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


def xep_video(ds: list[Candidate]) -> list[str]:
    """Danh sách ứng viên -> xếp hạng VIDEO, theo khung tốt nhất của mỗi video.

    Đây là cách một kênh CẤP KHUNG phát biểu ý kiến về video: video nào có
    khung lọt vào sớm nhất thì đứng trước. Kênh cấp video (metadata) trả sẵn
    theo thứ tự đó rồi nên hàm này không đổi gì.
    """
    thay, ra = set(), []
    for c in ds:
        if c.video_id not in thay:
            thay.add(c.video_id)
            ra.append(c.video_id)
    return ra


def hop_nhat_hai_tang(cac_danh_sach: list[list[Candidate]], k: int = K_MAC_DINH,
                      moi_video: int = 3, gioi_han: int = 100,
                      trong_so: list[float] | None = None) -> list[Candidate]:
    """RRF hai tầng: chọn VIDEO trước, rồi xếp KHUNG trong những video đó.

    VÌ SAO CẦN — A14 đo được `hop_nhat()` thô làm TỆ ĐI (−0,0144, ổn định).
    Nguyên nhân không phải RRF sai, mà là các kênh **không đồng ý ở cùng độ
    mịn**: kênh 2 và kênh 4 chung khung ở **5/97 câu**, nhưng chung video ở
    **79/97 câu**. RRF thô cộng theo `row_id` nên nó đo đúng cái 5/97 kia, và
    khi không cộng hưởng được thì nó chỉ đan xen hai danh sách — mỗi ứng viên
    tốt của kênh mạnh bị một ứng viên của kênh yếu đẩy lùi một bậc.

    Tầng 1 hợp nhất ở chỗ các kênh CÓ đồng ý. Tầng 2 để kênh cấp khung làm
    việc của nó.

    ⚠️ Tầng 1 mạnh KHÔNG tự sinh ra điểm. Trung bình 203 khung/video, nên chọn
    đúng video mà không xếp được khung thì R@1 vẫn ~0,5%. **Điểm nằm ở tầng 2**
    — và tầng 2 chỉ chạy được nếu có kênh cấp khung tốt. Đừng đọc hàm này là
    "gom hết kênh vào là xong".
    """
    if trong_so is None:
        trong_so = [1.0] * len(cac_danh_sach)
    if len(trong_so) != len(cac_danh_sach):
        raise ValueError(f"{len(cac_danh_sach)} danh sách nhưng "
                         f"{len(trong_so)} trọng số")

    # --- tầng 1: xếp hạng video --------------------------------------------
    diem_v: dict[str, float] = defaultdict(float)
    for ds, w in zip(cac_danh_sach, trong_so):
        for hang, v in enumerate(xep_video(ds), start=1):
            diem_v[v] += w / (k + hang)

    # --- tầng 2: xếp hạng khung TRONG nội bộ từng video ---------------------
    # Dùng hạng NỘI BỘ VIDEO chứ không dùng hạng toàn cục: nếu dùng hạng toàn
    # cục thì một video đứng cuối tầng 1 sẽ có mọi khung mang hạng rất lớn, và
    # thứ tự trong nội bộ nó bị nén phẳng thành như nhau.
    diem_f: dict[int, float] = defaultdict(float)
    nguon: dict[int, list[str]] = defaultdict(list)
    goc: dict[int, Candidate] = {}
    theo_video: dict[str, set[int]] = defaultdict(set)

    for ds, w in zip(cac_danh_sach, trong_so):
        dem: dict[str, int] = {}
        for c in ds:
            h = dem.get(c.video_id, 0) + 1
            dem[c.video_id] = h
            diem_f[c.row_id] += w / (k + h)
            nguon[c.row_id].append(f"{c.source or '?'}@v{h}")
            goc.setdefault(c.row_id, c)
            theo_video[c.video_id].add(c.row_id)

    # --- ghép: duyệt video theo thứ tự tầng 1, mỗi video lấy `moi_video` khung
    ra = []
    for v in sorted(diem_v, key=lambda x: -diem_v[x]):
        khung = sorted(theo_video[v], key=lambda r: -diem_f[r])[:moi_video]
        for r in khung:
            c = goc[r]
            ra.append(Candidate(row_id=r, video_id=c.video_id,
                                frame_idx=c.frame_idx,
                                score=diem_v[v] + diem_f[r], source="rrf2",
                                meta={**c.meta, "nguon": nguon[r],
                                      "diem_video": diem_v[v]}))
            if len(ra) >= gioi_han:
                return ra
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
