"""
thoi_gian.py — Xếp hạng lại theo chuỗi sự kiện (Bước 2b của TRAKE).

Cài đặt Algorithm 1 của bài báo AIC'25 (A8.7 #3). Ý tưởng: một video chứa TRỌN
chuỗi "trước -> hiện tại -> sau" thì đáng tin hơn hẳn video chỉ khớp mỗi cảnh
giữa. Cộng dồn điểm cao nhất mà mỗi video đạt được ở từng truy vấn con.

    S_cuoi(rc) = S(rc) + max{S(rp) : rp cùng video} + max{S(rn) : rn cùng video}

Độ phức tạp O(K log K) — chi phí không đáng kể so với bước truy hồi, nên chạy
được trên TOÀN KHO. Đó là điểm mạnh so với quy hoạch động ở Bước 5: dùng cái
này để THU HẸP VÙNG trước, rồi mới trích frame dày trong vùng còn lại.

Dùng được cho cả Mũi nhọn 1 khi truy vấn KIS mô tả bối cảnh trước/sau
("người đàn ông bước vào SAU KHI đám đông giải tán").

Dùng:
    from thoi_gian import xep_lai
    kq = xep_lai(kenh.tim(q_hien_tai), kenh.tim(q_truoc), kenh.tim(q_sau))
"""

from collections import defaultdict

try:
    from .schema import Candidate
except ImportError:
    from schema import Candidate


def diem_tot_nhat_moi_video(ds: list[Candidate]) -> dict[str, float]:
    """Điểm cao nhất mà mỗi video đạt được trong một danh sách."""
    tot: dict[str, float] = defaultdict(float)
    for c in ds:
        if c.score > tot[c.video_id]:
            tot[c.video_id] = c.score
    return tot


def xep_lai(hien_tai: list[Candidate],
            truoc: list[Candidate] | None = None,
            sau: list[Candidate] | None = None,
            trong_so: tuple[float, float] = (1.0, 1.0)) -> list[Candidate]:
    """Cộng thưởng cho ứng viên nằm trong video có đủ cả cảnh trước và sau.

    Chỉ danh sách `hien_tai` sinh ra kết quả; `truoc`/`sau` chỉ dùng để cộng
    điểm. Video thiếu cảnh trước hoặc sau vẫn được giữ, chỉ là không được
    thưởng — KHÔNG lọc cứng, cùng nguyên tắc với `objects.py`: bóc tách truy
    vấn sai thì lọc cứng sẽ xóa mất đáp án đúng.

    ⚠️ Điểm ở đây là TỔNG các score gốc, chỉ so sánh được trong nội bộ lần gọi
    này. Muốn hợp nhất với kênh khác thì đưa qua `rrf.hop_nhat`, đừng cộng
    thẳng.
    """
    tot_truoc = diem_tot_nhat_moi_video(truoc or [])
    tot_sau = diem_tot_nhat_moi_video(sau or [])
    w_t, w_s = trong_so

    ra = []
    for c in hien_tai:
        them_t = w_t * tot_truoc.get(c.video_id, 0.0)
        them_s = w_s * tot_sau.get(c.video_id, 0.0)
        ra.append(Candidate(
            row_id=c.row_id, video_id=c.video_id, frame_idx=c.frame_idx,
            score=c.score + them_t + them_s, source="thoi_gian",
            meta={**c.meta, "goc": c.score, "them_truoc": round(them_t, 4),
                  "them_sau": round(them_s, 4),
                  "du_chuoi": bool(them_t and them_s)}))
    return sorted(ra, key=lambda c: -c.score)


def video_du_chuoi(cac_danh_sach: list[list[Candidate]]) -> list[str]:
    """`video_id` xuất hiện ở MỌI truy vấn con, xếp theo tổng điểm giảm dần.

    Dùng cho Bước 2 của TRAKE — chốt 1–3 video trước khi trích frame dày.
    Chuỗi N sự kiện thì truyền N danh sách.

    Trả về rỗng nghĩa là không video nào chứa đủ chuỗi: hoặc bóc tách truy vấn
    sai, hoặc phải nới `k` của từng truy vấn con. ĐỪNG coi đó là "không có đáp
    án" — PHẦN C mục 3: TRAKE chấm từng phần, vẫn phải nộp đủ N vị trí.

    ⚠️ Giao CỨNG: một sự kiện không tìm ra đúng video (rất có thể xảy ra — bóc
    tách/truy hồi một truy vấn con bị lệch) loại LUÔN video đó khỏi kết quả,
    dù N-1 sự kiện còn lại đúng. `xep_video_theo_chuoi()` bên dưới là bản MỀM,
    không đòi giao cứng — dùng bản đó làm nguồn xếp hạng chính (Bước 2b), giữ
    hàm này lại vì vẫn hữu ích khi cần một danh sách "chắc chắn đủ chuỗi".
    """
    if not cac_danh_sach:
        return []
    chung = set.intersection(*({c.video_id for c in ds} for ds in cac_danh_sach))
    tong: dict[str, float] = defaultdict(float)
    for ds in cac_danh_sach:
        for v, d in diem_tot_nhat_moi_video(ds).items():
            if v in chung:
                tong[v] += d
    return [v for v, _ in sorted(tong.items(), key=lambda x: -x[1])]


def xep_video_theo_chuoi(cac_danh_sach: list[list[Candidate]]) -> list[str]:
    """Xếp hạng MỀM mọi video ứng viên của chuỗi TRAKE — Bước 2b (A8.7 #3).

    Với mỗi sự kiện thứ `i`, thưởng điểm bằng `xep_lai()` nếu video đó cũng có
    mặt ở sự kiện liền trước (`i-1`) và/hoặc liền sau (`i+1`) — video chứa
    TRỌN chuỗi lân cận đáng tin hơn video chỉ khớp một cảnh rời rạc. Sau đó
    CỘNG DỒN điểm đã thưởng của MỌI sự kiện cho từng video.

    Khác `video_du_chuoi()`: không đòi một video phải xuất hiện ở ĐỦ N sự
    kiện — video chỉ khớp 2/3, 3/4 sự kiện (rất thường gặp khi một truy vấn
    con bị bóc tách hoặc truy hồi hơi lệch) vẫn được xếp hạng theo điểm thật
    thay vì rớt xuống một khối không thứ tự. Trả về MỌI video xuất hiện ở BẤT
    KỲ sự kiện nào — không cần bước "nới ra" riêng như khi dùng
    `video_du_chuoi()`.
    """
    n = len(cac_danh_sach)
    if n == 0:
        return []
    tong: dict[str, float] = defaultdict(float)
    for i, hien_tai in enumerate(cac_danh_sach):
        truoc = cac_danh_sach[i - 1] if i > 0 else None
        sau = cac_danh_sach[i + 1] if i < n - 1 else None
        thuong = xep_lai(hien_tai, truoc, sau)
        for v, d in diem_tot_nhat_moi_video(thuong).items():
            tong[v] += d
    return [v for v, _ in sorted(tong.items(), key=lambda x: -x[1])]
