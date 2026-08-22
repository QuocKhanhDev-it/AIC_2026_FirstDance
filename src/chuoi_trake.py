"""
chuoi_trake.py — Dóng hàng chuỗi sự kiện TRAKE theo THỜI GIAN THẬT.

VÌ SAO — A39, đo trên 42 câu TRAKE của tập dev
==============================================

"Mô phỏng luồng suy nghĩ con người" cho TRAKE, khi quy ra số, chính là điều
này: người soát **không** tìm sự kiện 2 trên cả kho. Họ tìm sự kiện 1, rồi
**đi tiếp về phía trước từ đó** một quãng hợp lý. Bản cũ (`run.dong_hang_dp`)
chỉ ràng buộc `khung(i) < khung(i+1)` — biết THỨ TỰ, nhưng **không biết
KHOẢNG CÁCH**. Mà khoảng cách mới là thứ đo được:

    độ trải cả chuỗi   trung vị  56,6 s   max 101,0 s   (n=42)
    khoảng cách 2 sự kiện liền kề
                       trung vị  18,7 s   p25 12,4   p75 28,9
                       min        5,1 s   max  55,7 s  (n=106 cặp)
    độ trải / độ dài video
                       trung vị  0,1      max 0,2

Ba hệ quả, và cả ba đều chỉ ra lỗi có thật trong bản cũ:

**1. Chuỗi nằm gọn trong ~2 phút, không rải khắp video.** Độ trải chiếm trung
vị 10% và **không bao giờ quá 20%** độ dài video. Nhưng chốt chống dồn cục của
bản cũ, khi nổ, rải N sự kiện **đều khắp TOÀN BỘ video** — tức xa gấp 5–10 lần
sự thật. Nó nổ trên **47/100 dòng** của `query-p1-18-trake` trong bài nộp thật.

**2. `DON_NHAU = 100` khung nằm DƯỚI mức sàn đo được.** 100 khung là 3,3 s
(30 fps) đến 4,0 s (25 fps), còn khoảng cách nhỏ nhất từng thấy là **5,1 s**.
Ngưỡng đặt thấp hơn cả sàn thì gần như không bắt được gì.

**3. Đếm bằng KHUNG là dính đúng cái bẫy fps.** Kho có 4 giá trị fps
(25 / 26,44 / 29,97 / 30) nên `DON_NHAU = 100` mang ý nghĩa khác nhau ở từng
video. Module này làm việc hoàn toàn trên `pts_time`.

⚠️ **Cảnh báo về chính phép đo trên.** 41/42 câu TRAKE của tập dev là câu TỰ
SOẠN, chỉ 1 câu do BTC viết (`trake-DE1-16`). Phân bố *câu hỏi* vì thế không
đáng tin (A19/A20/A31/A34/A37 — tập dev tự soạn đã mù 5 lần). Nhưng phân bố
*thời gian đáp án* thì đỡ hơn nhiều: nó là tính chất của **video và của cách
người ta chọn một chuỗi sự kiện**, không phải của cách viết câu hỏi. Và câu đề
thật duy nhất có độ trải **101,0 s** — nằm ở mép trên, vẫn trong khoảng. Vì
n=1 nên trần ở đây đặt **rộng gấp 1,8 lần** con số đó, không bám sát nó.
"""

import math

# Đo được, không phải chọn. Xem A39 ở trên.
GAN_NHAT_GIAY = 5.0       # min khoảng cách 2 sự kiện liền kề từng thấy: 5,1 s
XA_NHAT_GIAY = 56.0       # max: 55,7 s
TRUNG_VI_KHOANG = 18.7    # trung vị khoảng cách cặp
TRUNG_VI_TRAI = 56.6      # trung vị độ trải cả chuỗi
TRAI_TOI_DA_GIAY = 180.0  # max đo được 101,0 s; nới ×1,8 vì chỉ có 1 câu đề thật


def phat_khoang(dt: float,
                gan: float = GAN_NHAT_GIAY,
                xa: float = XA_NHAT_GIAY) -> float:
    """Phạt cho khoảng cách `dt` giây giữa hai sự kiện liền kề. Càng lớn càng tệ.

    Trả `inf` khi `dt <= 0` — sự kiện phải TĂNG THẬT theo thời gian; bằng nhau
    cũng không được (`nop_bai.soat` chặn, và BTC đòi đúng thứ tự thời gian).

    Trong khoảng `[gan, xa]` thì phạt 0: đây là vùng đã quan sát được, không có
    cơ sở nào để nói 12 s tốt hơn 28 s. Ra ngoài mới phạt, và phạt TUYẾN TÍNH
    chứ không chặn cứng — bóc tách truy vấn có thể sai, chặn cứng thì xoá luôn
    đáp án đúng (cùng nguyên tắc mềm của `objects.py` và `thoi_gian.py`).
    """
    if dt <= 0:
        return math.inf
    if dt < gan:
        return (gan - dt) / gan          # 0 -> 1 khi dt tiến về 0
    if dt > xa:
        return (dt - xa) / xa            # không chặn trên; trần độ trải lo phần còn lại
    return 0.0


def dong_hang_theo_thoi_gian(cac_su_kien: list,
                             he_so_phat: float = 0.0,
                             trai_toi_da: float = math.inf,
                             k_moi_su_kien: int = 20) -> list:
    """Chọn cho mỗi sự kiện một khung, tăng dần theo thời gian, CÓ prior khoảng cách.

    `cac_su_kien[i]` là `list[(frame_idx, pts_time, score)]` — ứng viên của sự
    kiện thứ `i` **trong cùng một video**. Sự kiện không có ứng viên thì `[]`.

    Tối đa hoá   Σ điểm(f_i)  −  he_so_phat · Σ phat_khoang(t_{i+1} − t_i)
    với ràng buộc  t tăng thật  và  t_cuối − t_đầu ≤ `trai_toi_da`.

    ⚠️ Mặc định `he_so_phat=0.0` và `trai_toi_da=inf` cho lại **đúng hành vi
    cũ** của `run.dong_hang_dp`: chỉ ràng buộc tăng dần. Cố ý — kỷ luật đo của
    dự án là chưa thắng trên tập dev thì chưa được bật. Bật bằng
    `scripts/35_do_chuoi_trake.py` để đo, không bật thẳng trong `run.py`.

    Cách cài phản chiếu đúng luồng suy nghĩ nó mô phỏng: **chọn một mốc bắt
    đầu, rồi đi tiếp về phía trước**. Vòng ngoài duyệt từng ứng viên của sự
    kiện hợp lệ đầu tiên làm NEO; vòng trong là quy hoạch động chỉ nhìn các
    khung nằm trong `[t_neo, t_neo + trai_toi_da]`. Chi phí O(K²·N·K) với K≈20,
    N≤5 — cỡ 4·10⁴ phép, không đo được thời gian.

    Trả `list[int | None]` cùng độ dài; `None` ở vị trí không chọn được. Người
    gọi tự nội suy (xem `rai_deu_hep`).
    """
    n = len(cac_su_kien)
    if n == 0:
        return []

    # Cắt còn top-K theo điểm, bỏ trùng frame_idx (giữ điểm cao nhất).
    lop = []
    for ds in cac_su_kien:
        tot: dict = {}
        for fidx, t, sc in ds:
            if fidx not in tot or sc > tot[fidx][1]:
                tot[fidx] = (float(t), float(sc))
        lop.append(sorted(((f, t, s) for f, (t, s) in tot.items()),
                          key=lambda x: -x[2])[:k_moi_su_kien])

    dau = next((i for i, x in enumerate(lop) if x), None)
    if dau is None:
        return [None] * n

    tot_nhat, tot_diem = [None] * n, -math.inf
    for neo in lop[dau]:
        chon, diem = _di_tiep(lop, dau, neo, he_so_phat, trai_toi_da)
        if diem > tot_diem:
            tot_nhat, tot_diem = chon, diem
    return tot_nhat


def _di_tiep(lop: list, dau: int, neo: tuple,
             he_so_phat: float, trai_toi_da: float):
    """Quy hoạch động xuôi từ một NEO đã chốt cho sự kiện `dau`.

    Tách riêng khỏi `dong_hang_theo_thoi_gian` để chỗ "đi tiếp về phía trước"
    đọc được thành một hàm — đó là toàn bộ ý tưởng của A39.
    """
    n = len(lop)
    t_neo = neo[1]
    han = t_neo + trai_toi_da

    # trang[i] = list (frame_idx, t, điểm tích luỹ, sự kiện trước, ứng viên trước)
    trang: list = [None] * n
    trang[dau] = [(neo[0], t_neo, neo[2], -1, -1)]
    truoc_i = dau

    for i in range(dau + 1, n):
        if not lop[i]:
            continue
        hang = []
        for fidx, t, sc in lop[i]:
            if t > han:
                continue
            tot_j, tot_d = -1, -math.inf
            for j, (_, tj, dj, _, _) in enumerate(trang[truoc_i]):
                p = phat_khoang(t - tj)
                if p == math.inf:
                    continue
                d = dj + sc - he_so_phat * p
                if d > tot_d:
                    tot_j, tot_d = j, d
            if tot_j >= 0:
                hang.append((fidx, t, tot_d, truoc_i, tot_j))
        if hang:
            trang[i] = hang
            truoc_i = i

    # Truy vết từ lớp hợp lệ cuối cùng.
    ket = trang[truoc_i]
    j = max(range(len(ket)), key=lambda x: ket[x][2])
    diem = ket[j][2]
    ra: list = [None] * n
    i = truoc_i
    while i >= 0 and j >= 0:
        fidx, _, _, pi, pj = trang[i][j]
        ra[i] = fidx
        if pi < 0:
            break
        i, j = pi, pj
    return ra, diem


def rai_deu_hep(neo_frame: int, neo_vi_tri: int, n: int,
                fps: float, lo: int, hi: int,
                trai_giay: float = TRUNG_VI_TRAI) -> list:
    """Rải N sự kiện quanh một NEO trong cửa sổ `trai_giay`, KHÔNG rải khắp video.

    Thay cho nhánh dồn cục của `run.dung_trake`, chỗ rải đều trên **toàn bộ**
    khoảng frame của video: đo được độ trải thật chỉ chiếm trung vị **10%**
    (max 20%) độ dài video, nên rải khắp video là chắc chắn sai — nó đẩy các sự
    kiện ra xa gấp 5–10 lần.

    Vẫn cần `fps` **của chính video đó** (kho có 4 giá trị fps — cấm hardcode),
    và vẫn kẹp trong `[lo, hi]` là khoảng frame thật của video.
    """
    buoc = trai_giay * fps / max(n - 1, 1)
    kh = [int(round(neo_frame + (i - neo_vi_tri) * buoc)) for i in range(n)]
    # Kẹp vào [lo, hi] rồi ép tăng thật — kẹp có thể làm hai giá trị bằng nhau.
    kh = [max(lo, min(hi, x)) for x in kh]
    for i in range(1, n):
        if kh[i] <= kh[i - 1]:
            kh[i] = kh[i - 1] + 1
    return kh
