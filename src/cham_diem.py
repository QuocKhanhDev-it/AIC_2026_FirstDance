"""
cham_diem.py — Thước đo: đúng công thức chấm của BTC, cộng phép so theo cặp.

Đọc kèm: docs/07_lam_tap_dev.md

    Final Score = trung bình R@{1, 5, 20, 50, 100}
    R@k = max(R-Score của các kết quả xét tới hạng k)

Với KIS/QA thì R-Score là 0 hoặc 1 nên rút gọn thành hàm bậc thang theo thứ
hạng của câu đúng đầu tiên (PHẦN C):

    hạng 1 -> 1,00 | 2-5 -> 0,80 | 6-20 -> 0,60 | 21-50 -> 0,40
    51-100 -> 0,20 | >100 -> 0,00

TRAKE chấm TỪNG PHẦN theo tỷ lệ sự kiện khớp (A8.1) — nên bỏ trống một vị trí
là mất phần điểm đó, còn đoán sai thì cũng chỉ mất đúng phần đó. Luôn điền đủ.

VÌ SAO CÓ `so_sanh_cap`: với 40 câu, độ lệch chuẩn điểm ~0,35 nên sai số chuẩn
của trung bình là ~0,055 — chênh lệch dưới ~0,11 là nhiễu. Ta đã bị đúng cái
bẫy này hai lần (bench VLM 20 mẫu, bench OCR 13 mẫu). So THEO CẶP trên cùng bộ
câu hỏi thì phần lớn câu cho điểm y hệt và triệt tiêu nhau, phát hiện được
chênh lệch nhỏ hơn nhiều. Luôn báo cáo kèm thắng–thua–hòa.

    from cham_diem import cham, so_sanh_cap
    a = cham(tap_dev, lambda c: kenh.tim(c.cau_hoi, k=100))
    b = cham(tap_dev, lambda c: rrf.hop_nhat([...]))
    print(so_sanh_cap(a, b, "CLIP", "RRF"))
"""

from statistics import mean

import pandas as pd

MOC = (1, 5, 20, 50, 100)


def r_at_k(hang: int | None, k: int) -> float:
    """R@k cho câu nhị phân: 1 nếu câu đúng nằm trong top-k, 0 nếu không."""
    return 1.0 if hang is not None and hang <= k else 0.0


def diem_cau(hang: int | None) -> float:
    """Điểm một câu KIS/QA từ thứ hạng của kết quả đúng đầu tiên (1-based)."""
    return mean(r_at_k(hang, k) for k in MOC)


def diem_trake(hang_moi_su_kien: list) -> float:
    """Điểm TRAKE: trung bình điểm của từng sự kiện.

    Sự kiện không tìm thấy tính 0 nhưng KHÔNG loại cả câu — đó là ý nghĩa của
    'điểm từng phần' ở A8.1.
    """
    return mean(diem_cau(h) for h in hang_moi_su_kien) if hang_moi_su_kien else 0.0


def _hang(ket_qua, dung: set, gioi_han: int = 100) -> int | None:
    """Thứ hạng (1-based) của kết quả đúng đầu tiên trong top-`gioi_han`."""
    for i, c in enumerate(ket_qua[:gioi_han], start=1):
        if c.row_id in dung:
            return i
    return None


def no_cua_so(row_ids, master, dung_sai_giay: float) -> set:
    """Nở đáp án ra MỌI keyframe cùng video nằm trong ±`dung_sai_giay`.

    ⚠️ **Không có hàm này thì ta chấm CHẶT HƠN BTC** và mọi so sánh đều lệch.

    BTC xác nhận (A9) đáp án là một **cửa sổ `[s,e]` rộng 4 giây đến 5 phút**,
    không phải một frame. Keyframe cách đáp án 2 giây vẫn được BTC tính ĐÚNG,
    nhưng `_hang()` so `row_id` chính xác nên tính SAI. Chấm chặt hơn thật thì
    mọi cấu hình đều bị hạ điểm, và hạ **không đều** — cấu hình nào hay trả về
    keyframe lân cận bị phạt nặng hơn, tức thứ hạng giữa các cấu hình có thể
    ĐẢO NGƯỢC so với điểm thi thật.

    Dùng `pts_time` chứ không dùng `frame_idx`: fps khác nhau giữa các video
    (25 / 26,44 / 29,97 / 30) nên cùng một số frame là những khoảng thời gian
    khác nhau.
    """
    ra = set(row_ids)
    for r in row_ids:
        g = master.iloc[r]
        cung = master[(master.video_id == g.video_id)
                      & ((master.pts_time - g.pts_time).abs() <= dung_sai_giay)]
        ra |= set(int(x) for x in cung.row_id)
    return ra


def cham(tap_dev, chay, gioi_han: int = 100,
         master=None, dung_sai_giay: float | None = None) -> pd.DataFrame:
    """Chấm một cấu hình trên cả tập dev.

    `chay(cau_hoi) -> list[Candidate]` là cấu hình đang đo — có thể là một kênh
    đơn lẻ, hoặc kết quả RRF nhiều kênh. Hàm này không cần biết bên trong.

    `dung_sai_giay` mô phỏng cửa sổ `[s,e]` của BTC: kết quả cách đáp án không
    quá chừng đó giây (cùng video) được tính ĐÚNG. Cần `master`.
    **Để None là chấm chặt hơn BTC** — xem `no_cua_so()`.

    BTC nói cửa sổ rộng "4 giây đến 5 phút, tuỳ trường hợp" nên **không có một
    con số đúng duy nhất**. Hãy chấm ở vài mức và xem kết luận có đổi không —
    nếu đổi thì kết luận đó phụ thuộc vào một ẩn số, phải ghi rõ.

    Trả về một dòng mỗi câu, để so theo cặp về sau.
    """
    if dung_sai_giay is not None and master is None:
        raise ValueError("dung_sai_giay cần `master` để tra pts_time")

    def _dung(row_ids):
        return (no_cua_so(row_ids, master, dung_sai_giay)
                if dung_sai_giay is not None else set(row_ids))

    dong = []
    for c in tap_dev:
        if c.loai == "TRAKE":
            kq = chay(c)
            hang = [_hang(kq, _dung(b), gioi_han) for b in c.row_id_dung]
            d, h = diem_trake(hang), None
        else:
            kq = chay(c)
            h = _hang(kq, _dung(c.row_id_dung), gioi_han)
            d = diem_cau(h)
            # PHẦN C mục 4: `answer` sai -> 0 điểm bất kể frame đúng
            if c.loai == "QA" and c.dap_an:
                tra = (kq[0].meta.get("answer") if kq else None)
                if tra is not None and str(tra).strip().lower() != c.dap_an.strip().lower():
                    d = 0.0
        dong.append({"id": c.id, "loai": c.loai, "hang": h, "diem": d})
    return pd.DataFrame(dong)


def tom_tat(bang: pd.DataFrame) -> pd.DataFrame:
    """Điểm trung bình toàn bộ và theo từng loại câu."""
    theo = bang.groupby("loai").diem.agg(["size", "mean"])
    theo.columns = ["so_cau", "diem_tb"]
    tong = pd.DataFrame({"so_cau": [len(bang)], "diem_tb": [bang.diem.mean()]},
                        index=["TỔNG"])
    return pd.concat([theo, tong]).round(4)


def so_sanh_cap(a: pd.DataFrame, b: pd.DataFrame,
                ten_a="A", ten_b="B") -> str:
    """So hai cấu hình TRÊN CÙNG bộ câu hỏi.

    Báo cáo hiệu trung bình VÀ thắng–thua–hòa. Dòng thắng–thua–hòa nói nhiều
    hơn hiệu trung bình: 11 thắng / 4 thua trên 40 câu là tín hiệu thật, còn
    hiệu +0,04 kèm 6 thắng / 5 thua thì chỉ là nhiễu.
    """
    g = a.merge(b, on=["id", "loai"], suffixes=("_a", "_b"))
    hieu = g.diem_b - g.diem_a
    thang = int((hieu > 0).sum())
    thua = int((hieu < 0).sum())
    hoa = int((hieu == 0).sum())

    d = hieu.std(ddof=1) if len(g) > 1 else 0.0
    se = d / len(g) ** 0.5 if len(g) else 0.0

    ra = [
        f"{ten_a} = {g.diem_a.mean():.4f}    {ten_b} = {g.diem_b.mean():.4f}"
        f"    hiệu = {hieu.mean():+.4f}",
        f"{ten_b} thắng {thang} câu / thua {thua} / hòa {hoa}   (n = {len(g)})",
    ]
    if se:
        ra.append(f"sai số chuẩn của hiệu = {se:.4f}"
                  f"  ->  |hiệu| cần > {2 * se:.4f} mới đáng tin")
        if abs(hieu.mean()) < 2 * se:
            ra.append(f"⚠️  Chênh lệch NHỎ HƠN nhiễu. Chưa kết luận được "
                      f"{ten_b} hơn {ten_a}.")
    doi = g[hieu != 0].sort_values("diem_b", ascending=False)
    if len(doi):
        ra += ["", f"{len(doi)} câu đổi kết quả:",
               doi[["id", "loai", "hang_a", "hang_b", "diem_a", "diem_b"]]
               .to_string(index=False)]
    return "\n".join(ra)
