"""
kbest_trake.py — Lắp bài nộp TRAKE: 100 dòng = 100 giả thuyết, không phải 100 video.

VÌ SAO MODULE NÀY THAY CÁCH CŨ (A63, A66, A74, A78, A79)

Cách cũ trong `run.dung_trake()` xếp **một dòng cho mỗi video**, rồi bù cho đủ
100 bằng 99 video xếp sau. Nhưng BTC chấm TRAKE **theo vị trí**: chuỗi của
video tốt nhất lệch một sự kiện ra ngoài cửa sổ là cả dòng đó 0 điểm — và 99
dòng còn lại dành cho những video gần như chắc chắn sai.

Với TRAKE, đáp án nằm trong MỘT video. Nên 100 dòng nên là **100 giả thuyết
khác nhau về video đó**, không phải 100 video khác nhau.

A63 đo được khâu lắp ráp cũ mất **37% ở ±2s** số điểm kênh đã tìm ra.

BA THAM SỐ, MỖI CÁI MỘT PHÉP ĐO RIÊNG

* `cach_nhau = 3,0s` — hai chuỗi coi là khác nhau nếu lệch quá ngần này (A74:
  0,5 / 1,5 / 3,0 đều dương, 3,0 tốt nhất ở cả hai mức dung sai).
* `ty_le = (0,40, 0,25, 0,15, 0,12, 0,08)` — chia ngân sách dòng cho top-5
  video. A78 dò 7 cách: **dồn hết cho hạng 1 là TỆ NHẤT** (✅ ổn định, 0 thắng
  / 5 thua), mà trải đều hẳn thì ❌ đảo dấu. Tối ưu nằm sát ngay cạnh cách cũ,
  phẳng hơn một chút.
* `n_duoi = 20` — số dòng cuối dành cho **1 chuỗi tốt nhất mỗi video từ hạng 6
  trở đi**.

VÌ SAO CÓ `n_duoi` — K-best đổi BỀ RỘNG lấy CHIỀU SÂU

Trên 20 câu TRAKE, video đúng nằm ở: **14 câu hạng 1, 4 câu hạng 2–5, 2 câu
NGOÀI top-5** (hạng 8 và hạng 23). Với hai câu đó K-best thuần sinh không một
giả thuyết nào -> đúng 0 điểm, trong khi cách cũ rải 1 dòng/video vẫn chạm tới
hạng 23 và được 0,4800 (`trake-L25-004`).

`n_duoi` gắn lại lưới an toàn đó. A79 đo bốn mức, và **cơ chế tự xác nhận**:

    n_duoi = 10  ->  −0,0050   phủ tới hạng 15, CHƯA chạm hạng 23
    n_duoi = 20  ->  +0,0135   phủ tới hạng 25, CHẠM
    n_duoi = 30  ->  +0,0135   không thêm gì
    n_duoi = 50  ->  +0,0140 nhưng 2-4-14, cắt quá nhiều chiều sâu top-5

Hiệu ứng xuất hiện đúng chỗ cơ chế nói nó phải xuất hiện — đáng tin hơn con số.

⚠️ ĐÂY LÀ THỨ ĐẦU TIÊN QUA NGƯỠNG KỂ TỪ A52. Trên 20 câu, cách cũ thua cấu
hình này **−0,0990 ở ±2s, 2 thắng / 11 thua, vượt ngưỡng nhiễu 0,0932 -> ✅ ỔN
ĐỊNH**. Hai điều phải nhớ khi đọc lại:

  * ✅ dựa trên ±2s; ở ±15s cùng dấu nhưng chưa vượt nhiễu.
  * 14/20 câu TRAKE là câu tự soạn. Dùng được ở đây vì thứ tự sự kiện và số
    Frame ID là ràng buộc **hình thức**, không phụ thuộc câu dễ hay khó (A63) —
    nhưng đó là lập luận, không phải phép đo.
"""

import math

try:
    from .schema import AnswerTRAKE
except ImportError:
    from schema import AnswerTRAKE

BEAM = 64
CACH_NHAU = 3.0
TY_LE = (0.40, 0.25, 0.15, 0.12, 0.08)
N_DUOI = 20
SO_VIDEO = 5
TOI_DA_UV = 20          # ứng viên xét cho mỗi sự kiện trong một video


def gom_theo_video(cac_su_kien: list) -> dict:
    """`{video_id: [[(row_id, điểm)] cho từng sự kiện]}`."""
    n = len(cac_su_kien)
    ra = {}
    for i, ds in enumerate(cac_su_kien):
        for c in ds:
            ra.setdefault(c.video_id, [[] for _ in range(n)])[i].append(
                (c.row_id, c.score))
    return ra


def cham_video(theo_video: dict) -> dict:
    """`{video_id: điểm}` = Σ log(điểm cao nhất cho từng sự kiện).

    Vì mọi video có cùng số sự kiện, tổng-log **tương đương đơn điệu với trung
    bình NHÂN**. A78 dò cả bốn cách hợp (cộng / nhân / điều hoà / min) và thấy
    chúng chọn ra gần như CÙNG một video — 10/17 đúng ở hạng 1, ba cách đầu
    giống hệt nhau tới từng câu. Nút này **trơ**, nên giữ nguyên bản đang có.

    Video thiếu ứng viên cho bất kỳ sự kiện nào thì bị loại.
    """
    ra = {}
    for v, uv in theo_video.items():
        if any(not x for x in uv):
            continue
        ra[v] = sum(math.log(max(s for _, s in x) + 1e-9) for x in uv)
    return ra


def phat_bac(gan: float = 1.0, nang: float = 1.0,
             xa: float = 60.0, beta: float = 0.0005):
    """Hàm phạt dạng BẬC — khác hẳn phạt tỷ lệ thuận đã bị bác ở A80.

        Δt < `gan`        -> phạt `nang`   (hai sự kiện khác nhau không thể
                                            rơi vào cùng một tích tắc)
        `gan` ≤ Δt ≤ `xa` -> KHÔNG phạt
        Δt > `xa`         -> phạt `beta · (Δt − xa)`

    A80 bác phạt **tỷ lệ thuận với khoảng cách ở mọi Δt**, và lý do đo được là:
    khoảng cách thật giữa hai sự kiện có trung vị 12,0s nhưng trải từ 1,5s tới
    259,3s, nên phạt đều tay trừng phạt cả những chuỗi ĐÚNG có khoảng cách dài
    thật.

    Hàm bậc này **né đúng chỗ đó**: vùng [1s, 60s] chứa phần lớn khoảng cách
    thật thì không bị đụng tới. A80 KHÔNG bác được nó — phải đo riêng.
    """
    def f(dt: float) -> float:
        if dt < gan:
            return nang
        if dt > xa:
            return beta * (dt - xa)
        return 0.0
    return f


def beam_video(uv_theo_su_kien: list, pts, k_chuoi: int,
               cach_nhau: float = CACH_NHAU, phat_giay: float = 0.0,
               phat=None) -> list:
    """Sinh tối đa `k_chuoi` chuỗi TĂNG DẦN NGẶT, khác nhau về thời gian.

    `uv_theo_su_kien[i]` = `[(row_id, điểm)]` của sự kiện i TRONG một video.

    Tăng dần ngặt vì BTC đòi *"thứ tự phải tuân theo thứ tự thời gian của các
    events"*. Lọc đa dạng ở cuối để 100 dòng không phải 100 biến thể lệch nhau
    vài phần trăm giây — chúng sẽ cùng đúng hoặc cùng sai, tức phí 99 dòng.

    `phat_giay` trừ `λ × (khoảng cách giây tới sự kiện trước)` mỗi lần nối —
    ý tưởng "phạt mềm": chuỗi có các sự kiện cách nhau đều và gần thì đáng tin
    hơn chuỗi nhảy cóc qua nửa video.

    **Mặc định 0,0 (TẮT) — A80 đo và BÁC.** Năm mức λ, hại đơn điệu theo cường
    độ; λ=0,001 và 0,005 đều ✅ ỔN ĐỊNH theo hướng xấu. Lý do nằm ở phân bố
    thật: khoảng cách giữa hai sự kiện liền kề có trung vị **12,0s** nhưng trải
    từ **1,5s tới 259,3s** — rộng gấp 170 lần. Phạt theo khoảng cách trừng phạt
    đúng những chuỗi ĐÚNG mà có khoảng cách dài thật, vì nó không phân biệt
    được "nhảy cóc vì đoán bừa" với "hai sự kiện cách nhau 4 phút".

    Giữ tham số lại để lần sau ai đó nghĩ ra ý này thì thấy nó đã được đo.
    """
    beam = [([], 0.0, -1e9)]                       # (chuỗi, điểm, pts cuối)
    for uv in uv_theo_su_kien:
        moi = []
        for chuoi, d, t_cuoi in beam:
            for rid, s in uv:
                if pts[rid] > t_cuoi:
                    if not chuoi:                  # sự kiện đầu, chưa có Δt
                        p = 0.0
                    elif phat is not None:
                        p = phat(pts[rid] - t_cuoi)
                    else:
                        p = phat_giay * (pts[rid] - t_cuoi)
                    moi.append((chuoi + [rid], d + s - p, pts[rid]))
        if not moi:                                # không nối tiếp được nữa
            return [c for c, _, _ in beam if len(c) == len(uv_theo_su_kien)]
        moi.sort(key=lambda x: -x[1])
        beam = moi[:BEAM]

    ra = []
    for chuoi, _, _ in beam:
        if all(any(abs(pts[a] - pts[b]) > cach_nhau
                   for a, b in zip(chuoi, cu)) for cu in ra):
            ra.append(chuoi)
        if len(ra) >= k_chuoi:
            break
    return ra


def lap_dong(cac_su_kien: list, master, so_dong: int = 100,
             ty_le=TY_LE, n_duoi: int = N_DUOI,
             cach_nhau: float = CACH_NHAU,
             phat_giay: float = 0.0, phat=None) -> list:
    """N danh sách ứng viên (mỗi sự kiện một danh sách) -> `list[list[row_id]]`.

    Trả `row_id` chứ không phải `frame_idx`: `lap_trake()` mới đổi sang
    `AnswerTRAKE`. Tách ra để `scripts/9x_*` đo thẳng ở tầng `row_id` — A5.7 đo
    được 614 keyframe dùng chung `frame_idx`, nên đi qua `frame_idx` là mất
    thông tin ngay giữa phép đo.
    """
    if not cac_su_kien:
        return []
    pts = master.pts_time.values
    theo_video = gom_theo_video(cac_su_kien)
    diem_v = cham_video(theo_video)
    if not diem_v:
        return []
    xep = sorted(diem_v, key=lambda v: -diem_v[v])

    def chuoi(v, k):
        uv = theo_video[v]
        for x in uv:
            x.sort(key=lambda t: -t[1])
        return beam_video([x[:TOI_DA_UV] for x in uv], pts, k,
                          cach_nhau, phat_giay, phat)

    n_tren = max(1, so_dong - n_duoi)
    ra = []
    for v, w in zip(xep[:SO_VIDEO], ty_le):
        ra += chuoi(v, max(1, round(n_tren * w)))
        if len(ra) >= n_tren:
            break
    ra = ra[:n_tren]
    for v in xep[SO_VIDEO:SO_VIDEO + n_duoi]:      # lưới an toàn
        ra += chuoi(v, 1)
    return ra[:so_dong]


def lap_trake(cac_su_kien: list, master, so_dong: int = 100, **k) -> list:
    """-> `list[AnswerTRAKE]`, dùng thẳng được cho bài nộp.

    ⚠️ `frame_idx` lấy từ cột `frame_idx` của bảng cái, **không** tính lại từ
    `pts_time` — làm tròn lệch 1 frame (12,9% dòng khác nhau nếu tính lại).
    """
    vid = master.video_id.values
    fx = master.frame_idx.values
    ra = []
    for dong in lap_dong(cac_su_kien, master, so_dong, **k):
        if not dong:
            continue
        # ⚠️ `beam_video` ép tăng dần theo `pts_time`, KHÔNG theo `frame_idx` —
        # và hai thứ đó không cùng nhịp: A5.7 đo được **614 cặp** cùng video có
        # `pts_time` tăng nhưng `frame_idx` BẰNG NHAU (không cặp nào giảm).
        # Nộp hai sự kiện khác nhau ở cùng một Frame ID là chắc chắn phí một
        # trong hai — hai sự kiện là hai khoảnh khắc khác nhau. `run.dung_trake`
        # đã chốt chỗ này ("phải TĂNG THẬT, không bằng nhau"); K-best thì chưa,
        # và `nop_bai.soat` không bắt được vì nó so với `sorted()`, mà
        # `[0, 0, 519]` thì đã sorted rồi.
        khung = [int(fx[r]) for r in dong]
        for i in range(1, len(khung)):
            if khung[i] <= khung[i - 1]:
                khung[i] = khung[i - 1] + 1
        ra.append(AnswerTRAKE(str(vid[dong[0]]), khung))
    return ra
