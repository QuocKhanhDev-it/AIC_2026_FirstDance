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


def diem_trake_bai_nop(cac_dong, dung_moi_su_kien: list,
                       gioi_han: int = 100) -> float:
    """Điểm TRAKE **theo đúng cách BTC chấm BÀI NỘP**.

    ⚠️ Khác hẳn `diem_trake()` ở trên, và phải hiểu rõ khác chỗ nào:

    * `diem_trake()` chấm **KÊNH**: một danh sách ứng viên, tìm xem mỗi sự kiện
      có nằm trong đó không. Trả lời câu hỏi *"kênh có tìm ra các sự kiện
      không"*.
    * Hàm này chấm **BÀI NỘP**: mỗi dòng là một BỘ N khung, và **vị trí i chỉ
      được so với sự kiện i**. Trả lời câu hỏi *"nộp cái này thì được mấy
      điểm"*.

    Hai con số có thể lệch rất xa: kênh tìm ra đủ ba sự kiện nhưng khâu lắp ráp
    xếp sai vị trí thì `diem_trake()` cho điểm cao còn BTC cho 0. Đó chính là
    tầng mà `run.dung_trake()` làm, và là tầng chưa ai đo.

    `cac_dong`: danh sách các dòng, mỗi dòng là list `row_id` theo thứ tự sự kiện.
    `dung_moi_su_kien`: list các `set` row_id đúng, một set cho mỗi sự kiện.

    R-Score một dòng = tỷ lệ vị trí khớp (A8.1: *"partial credit proportional to
    the number of correctly matched frames"*). Điểm cuối vẫn là trung bình
    `max R-Score trong top-k` trên các mốc R@{1,5,20,50,100}.
    """
    n = len(dung_moi_su_kien)
    if not n:
        return 0.0
    r = []
    for dong in cac_dong[:gioi_han]:
        khop = sum(1 for i, f in enumerate(dong[:n]) if f in dung_moi_su_kien[i])
        r.append(khop / n)
    if not r:
        return 0.0
    # max R-Score trong top-k, trung bình trên các mốc
    return mean(max(r[:k], default=0.0) for k in MOC)


def _hang(ket_qua, dung: set, gioi_han: int = 100, hop_le=None) -> int | None:
    """Thứ hạng (1-based) của kết quả đúng đầu tiên trong top-`gioi_han`.

    `hop_le(c)` là điều kiện PHỤ THÊM cho từng ứng viên — dùng cho câu Q&A,
    nơi một dòng chỉ ăn điểm khi đúng CẢ khung LẪN chuỗi `answer`.
    """
    for i, c in enumerate(ket_qua[:gioi_han], start=1):
        if c.row_id in dung and (hop_le is None or hop_le(c)):
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


def _dung_dap_an(c):
    """Điều kiện `answer` cho câu Q&A. Trả None nếu câu này không cần xét.

    ⚠️ **Xét theo TỪNG ỨNG VIÊN, không xét mỗi hạng 1.** Bài nộp Q&A là danh
    sách xếp hạng các bộ `(video_id, frame_idx, answer)` — mỗi dòng mang
    `answer` riêng, và một dòng ăn điểm khi đúng CẢ khung LẪN chuỗi answer
    (PHẦN C mục 4). Nên thứ hạng phải là dòng đầu tiên thỏa cả hai.

    Bản đầu chấm thứ hạng theo khung rồi mới xóa điểm nếu `kq[0].answer` sai.
    Sai ở CẢ HAI CHIỀU, đã dựng test cho từng chiều:

      * hạng 1 sai đáp án, hạng 4 đúng cả hai  -> chấm 0, đúng ra là 0,8
      * hạng 1 đúng đáp án nhưng SAI khung     -> chấm 0,8, đúng ra là 0

    Lệch kiểu này không đều giữa các cấu hình, tức nó **đảo được thứ hạng** —
    đúng loại hỏng mà `no_cua_so()` sinh ra để chặn.

    Ứng viên **không có** khóa `answer` được coi là hợp lệ: mọi kênh hiện tại
    chưa biết trả lời, và lúc này ta đang đo TRUY HỒI. Siết chỗ này sẽ làm mọi
    con số cũ hết so được với nhau.
    """
    if c.loai != "QA" or not c.dap_an:
        return None
    mong = c.dap_an.strip().lower()

    def hop_le(x):
        tra = x.meta.get("answer")
        return tra is None or str(tra).strip().lower() == mong
    return hop_le


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
            h = _hang(kq, _dung(c.row_id_dung), gioi_han, _dung_dap_an(c))
            d = diem_cau(h)
        dong.append({"id": c.id, "loai": c.loai, "hang": h, "diem": d})
    return pd.DataFrame(dong)


def tom_tat(bang: pd.DataFrame) -> pd.DataFrame:
    """Điểm trung bình toàn bộ và theo từng loại câu."""
    theo = bang.groupby("loai").diem.agg(["size", "mean"])
    theo.columns = ["so_cau", "diem_tb"]
    tong = pd.DataFrame({"so_cau": [len(bang)], "diem_tb": [bang.diem.mean()]},
                        index=["TỔNG"])
    return pd.concat([theo, tong]).round(4)


# Hai mốc dung sai để báo cáo. BTC nói cửa sổ rộng "4 giây đến 5 phút, TUỲ
# TRƯỜNG HỢP" (A9) — không có một con số đúng duy nhất, nên phải báo hai mức:
#   CHINH   ±2s  = cửa sổ 4s, mức HẸP NHẤT BTC nêu -> con số bảo thủ để báo cáo
#   KIEM    ±15s = cửa sổ 30s, mức rộng vừa phải   -> để kiểm kết luận có đổi không
# Kết luận đổi giữa hai mức = kết luận phụ thuộc vào ẩn số BTC chưa chốt.
DUNG_SAI_CHINH = 2.0
DUNG_SAI_KIEM = 15.0
# ⚠️ TÊN PHẢI KHÁC `MOC`. `MOC = (1, 5, 20, 50, 100)` ở đầu file là các mốc
# R@k của công thức chấm BTC. Đặt trùng tên là ĐÈ MẤT nó, và `diem_cau()` sẽ
# lặng lẽ tính trung bình R@2 và R@15 thay vì R@{1,5,20,50,100} — điểm sai mà
# không có gì báo. Đã vấp thật: bảng đo ra 0,3810 thay vì 0,5238, và tôi suýt
# sửa nhầm tài liệu theo con số hỏng đó. `test_bac_thang_dung_cong_thuc_btc`
# là thứ bắt được.
MOC_DUNG_SAI = (DUNG_SAI_CHINH, DUNG_SAI_KIEM)


def cham_nhieu_muc(tap_dev, chay, master, moc=MOC_DUNG_SAI,
                   gioi_han: int = 100) -> dict:
    """Chấm cùng một cấu hình ở nhiều mức dung sai. Trả `{dung_sai: DataFrame}`."""
    return {ds: cham(tap_dev, chay, gioi_han, master, ds) for ds in moc}


def _hieu(a: pd.DataFrame, b: pd.DataFrame):
    g = a.merge(b, on=["id", "loai"], suffixes=("_a", "_b"))
    h = g.diem_b - g.diem_a
    se = (h.std(ddof=1) / len(g) ** 0.5) if len(g) > 1 else 0.0
    return g, h, se


def bao_cao_do_nhay(tap_dev, cau_hinh: dict, master, moc=MOC_DUNG_SAI,
                    gioi_han: int = 100) -> str:
    """So nhiều cấu hình ở nhiều mức dung sai, và **báo kết luận có ỔN ĐỊNH không**.

    `cau_hinh` là `{"tên": hàm_chạy}`. Cấu hình ĐẦU TIÊN làm mốc nền, mọi cấu
    hình sau so theo cặp với nó.

    Vì sao cần: BTC chưa chốt độ rộng cửa sổ, mà độ rộng đó đổi điểm tới 0,10 —
    lớn hơn khác biệt giữa các cấu hình đang cần đo. Một kết luận chỉ đáng tin
    khi nó **giữ nguyên dấu ở cả hai mức** và vượt nhiễu ở ít nhất một mức.

    Ba kết luận có thể ra:

    | Ký hiệu | Nghĩa |
    | --- | --- |
    | `ON DINH` | cùng dấu ở cả hai mức, và vượt 2 sai số chuẩn ở ít nhất một mức |
    | `YEU`     | cùng dấu nhưng chưa vượt nhiễu ở mức nào — cần thêm câu hỏi |
    | `DAO DAU` | **đổi dấu giữa hai mức** — kết luận phụ thuộc vào ẩn số, KHÔNG được dùng để quyết |
    """
    bang = {ten: cham_nhieu_muc(tap_dev, f, master, moc, gioi_han)
            for ten, f in cau_hinh.items()}
    ten_moc = next(iter(cau_hinh))

    ra = [f"{len(tap_dev)} câu | mốc nền: {ten_moc}", ""]
    ra.append("ĐIỂM TRUNG BÌNH")
    ra.append(f"  {'cấu hình':<22}" + "".join(f"{'±' + str(ds) + 's':>12}" for ds in moc))
    ra.append("  " + "-" * (22 + 12 * len(moc)))
    for ten, d in bang.items():
        ra.append(f"  {ten:<22}" + "".join(f"{d[ds].diem.mean():>12.4f}" for ds in moc))

    ra += ["", f"SO THEO CẶP với `{ten_moc}`  (hiệu / thắng-thua-hòa / ngưỡng nhiễu)"]
    for ten in list(cau_hinh)[1:]:
        ra.append(f"\n  {ten}")
        dau, manh = [], False
        for ds in moc:
            g, h, se = _hieu(bang[ten_moc][ds], bang[ten][ds])
            tb = h.mean()
            dau.append(0 if tb == 0 else (1 if tb > 0 else -1))
            vuot = abs(tb) > 2 * se > 0
            manh |= vuot
            ra.append(f"    ±{ds:<5g}s  {tb:>+8.4f}   "
                      f"{int((h > 0).sum())}-{int((h < 0).sum())}-{int((h == 0).sum())}"
                      f"   ngưỡng {2 * se:.4f}{'  <= vượt nhiễu' if vuot else ''}")
        if len(set(d for d in dau if d)) > 1:
            ket = "❌ DAO DAU — phụ thuộc ẩn số độ rộng cửa sổ, KHÔNG dùng để quyết"
        elif not any(dau):
            ket = "⚪ KHÔNG ĐỔI GÌ — cấu hình này không tác động trên tập dev này"
        elif manh:
            ket = "✅ ON DINH"
        else:
            ket = "🟡 YEU — cùng dấu nhưng chưa vượt nhiễu, cần thêm câu hỏi"
        ra.append(f"    -> {ket}")
    return "\n".join(ra)


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
