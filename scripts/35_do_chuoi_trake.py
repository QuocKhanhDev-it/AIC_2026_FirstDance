"""
35_do_chuoi_trake.py — Đo prior KHOẢNG CÁCH thời gian cho TRAKE (A39).

CÂU HỎI ĐANG ĐO
===============

Bản cũ (`run.dong_hang_dp`) chỉ ràng buộc `khung(i) < khung(i+1)`: nó biết THỨ
TỰ nhưng **không biết KHOẢNG CÁCH**. Đo trên 42 câu TRAKE của tập dev thì
khoảng cách là thứ có phân bố rất chặt:

    độ trải cả chuỗi   trung vị 56,6 s   max 101,0 s
    khoảng cách cặp    trung vị 18,7 s   min 5,1 s   max 55,7 s
    độ trải / dài video trung vị 0,1     max 0,2

Script này hỏi: đưa phân bố đó vào khâu lắp ráp thì bài nộp được thêm mấy điểm.

BỐN BIẾN THỂ — MỖI LẦN ĐỔI ĐÚNG MỘT THỨ
========================================

    cu          bản đang chạy: DP chỉ ép tăng dần            <- MỐC NỀN
    tran        + trần độ trải (chuỗi phải gọn trong T giây)
    phat        + prior khoảng cách (phạt dồn cục / giãn quá)
    rai_hep     + rải hẹp thay cho rải khắp video khi dồn cục

`tran`/`phat`/`rai_hep` mỗi cái so RIÊNG với `cu`, không cộng dồn — cộng dồn
rồi quy công cho nhầm cái là lỗi đã vấp nhiều lần trong repo này. Biến thể
`tat_ca` in thêm ở cuối chỉ để biết trần trên, đừng dùng nó để kết luận.

    python scripts/35_do_chuoi_trake.py                       # kênh 3 (yếu)
    python scripts/35_do_chuoi_trake.py --cache index/truy_van.npz   # kênh 1

⚠️ HAI CẢNH BÁO PHẢI ĐỌC TRƯỚC KHI TIN CON SỐ
==============================================

**1. Không có `--cache` thì phép đo gần như vô nghĩa.** Kênh 3 (OCR+ASR) một
mình cho **0,0000** trên câu TRAKE (đã đo ở `26_do_don_cuc_trake.py`) — cả bốn
biến thể sẽ bằng nhau vì không có gì để lắp ráp. Máy 7,7 GB không nạp nổi
SigLIP2, nên phép đo THẬT phải chạy ở máy mạnh, hoặc ở đây với `truy_van.npz`
mã hoá sẵn từ máy mạnh.

**2. 41/42 câu TRAKE của tập dev là câu TỰ SOẠN**, chỉ `trake-DE1-16` do BTC
viết. Tập dev tự soạn đã mù 5 lần (A19/A20/A31/A34/A37). Cột `chỉ câu đề thật`
in riêng ra để nhìn — n=1, không kết luận được gì, nhưng thấy được nó có ĐẢO
DẤU so với phần còn lại hay không.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                                   # noqa: E402
import tap_dev                                                    # noqa: E402
from bm25 import KenhVanBan                                       # noqa: E402
from cham_diem import MOC_DUNG_SAI, diem_trake_bai_nop, no_cua_so  # noqa: E402
from chuoi_trake import TRAI_TOI_DA_GIAY                          # noqa: E402
from dense import KenhAnhCache                                    # noqa: E402

# Mốc nền LUÔN là `cu` — cấu hình đang thật sự chạy, không phải cái tiện tay.
BIEN_THE = {
    "cu":      dict(),
    "tran":    dict(dong_hang="thoi_gian", he_so_phat=0.0),
    "phat":    dict(dong_hang="thoi_gian", he_so_phat=1.0, trai_toi_da=float("inf")),
    "rai_hep": dict(rai_hep=True),
    "tat_ca":  dict(dong_hang="thoi_gian", he_so_phat=1.0, rai_hep=True),
}
NHAN = {
    "cu":      "MỐC NỀN — DP chỉ ép tăng dần",
    "tran":    f"+ trần độ trải {TRAI_TOI_DA_GIAY:.0f}s",
    "phat":    "+ prior khoảng cách (không trần)",
    "rai_hep": "+ rải hẹp 56,6s thay vì khắp video",
    "tat_ca":  "cả ba (chỉ để biết trần trên)",
}


def bang_tra_nguoc(master) -> dict:
    """`(video_id, frame_idx) -> [row_id, ...]` — A5.7: KHÔNG phải song ánh.

    614 keyframe dùng chung `frame_idx` với dòng liền trước, nên tra ngược có
    thể ra nhiều `row_id`. Lấy cái đầu tiên; chấm bằng `no_cua_so` nên chênh
    một dòng không đổi kết quả.
    """
    d = defaultdict(list)
    for r in master.itertuples():
        d[(r.video_id, int(r.frame_idx))].append(int(r.row_id))
    return d


def cham_mot_cau(kho_cau, c, master, tra, dung_sai, k, **cau_hinh) -> float:
    dong = R.dung_trake(kho_cau, master, so_dong=k, **cau_hinh)
    cac_dong = []
    for d in dong:
        r = [tra.get((d.video_id, int(f)), [None])[0] for f in d.frame_idxs]
        if all(x is not None for x in r):
            cac_dong.append(r)
    dung = [no_cua_so(rs, master, dung_sai) for rs in c.row_id_dung]
    return diem_trake_bai_nop(cac_dong, dung, k)


def sai_so_cap(diem_moc: dict, diem: dict) -> tuple[float, float]:
    """`(hiệu trung bình, 2·sai số chuẩn)` của hiệu THEO CẶP.

    ⚠️ Dùng đúng công thức `cham_diem._hieu` — hai script phải hiểu "vượt
    nhiễu" giống nhau, nếu không thì cùng một con số được gọi là ✅ ở đây và
    🟡 ở kia.

    Bản đầu của script này lấy ngưỡng là `1/(số sự kiện × số câu)` — đó là
    **lượng tử nhỏ nhất** mà điểm có thể đổi, KHÔNG phải ngưỡng thống kê. Nó
    trả lời "có đổi gì không", không trả lời "đổi có thật không". Và dòng kết
    luận thì chỉ xét DẤU, nên `+0,0016` (kém xa nhiễu) vẫn được gắn ✅ — nói
    quá về một thay đổi chỉ nhúc nhích 2 câu trong 42.
    """
    h = [diem[c] - diem_moc[c] for c in diem_moc]
    n = len(h)
    tb = sum(h) / n
    if n < 2:
        return tb, 0.0
    var = sum((x - tb) ** 2 for x in h) / (n - 1)
    return tb, 2 * (var / n) ** 0.5


def in_so_sanh(ten, diem_moc: dict, diem: dict, nguong: float):
    """Thắng–thua–hoà theo CẶP, kèm 2·SE — không in mỗi trung bình."""
    t = h = b = 0
    for cid in diem_moc:
        d = diem[cid] - diem_moc[cid]
        if abs(d) < nguong:
            h += 1
        elif d > 0:
            t += 1
        else:
            b += 1
    tb_moc = sum(diem_moc.values()) / len(diem_moc)
    tb = sum(diem.values()) / len(diem)
    hieu, se2 = sai_so_cap(diem_moc, diem)
    vuot = " <= vượt nhiễu" if abs(hieu) > se2 > 0 else ""
    print(f"  {NHAN[ten]:38} {tb:.4f}  ({tb - tb_moc:+.4f})  "
          f"thắng {t:2d} / thua {b:2d} / hoà {h:2d}   2·SE {se2:.4f}{vuot}")


def main():
    ap = argparse.ArgumentParser(description="do prior khoang cach cua TRAKE")
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--cache", type=Path, default=None,
                    help="index/truy_van.npz — dùng ứng viên KÊNH 1 (SigLIP2) "
                         "mà KHÔNG nạp model. Thiếu nó thì phép đo gần như vô "
                         "nghĩa: kênh 3 một mình cho 0,0000 trên TRAKE")
    ap.add_argument("--matrix", default="clip_siglip2.npy")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    tra = bang_tra_nguoc(master)
    cau = [c for c in tap_dev.doc() if c.loai == "TRAKE"]
    if not cau:
        raise SystemExit("Tập dev chưa có câu TRAKE nào.")

    if a.cache:
        print(f"Ứng viên: kênh 1 (SigLIP2) qua cache {a.cache} — KHÔNG nạp model")
        k = KenhAnhCache(str(a.index), a.cache, matrix=a.matrix)
        thieu = k.co_du([sk for c in cau for sk in R.tach_su_kien(c.cau_hoi)])
        if thieu:
            raise SystemExit(
                f"{len(thieu)} mệnh đề sự kiện chưa có trong cache, ví dụ:\n"
                f"  {thieu[0][:100]!r}\n"
                f"Mã hoá thêm: python scripts/25_ma_hoa_truy_van.py --tap-dev --gop")
    else:
        p = a.index / "ocr_asr.parquet"
        if not p.exists():
            raise SystemExit(f"Chưa có {p}")
        k = KenhVanBan.tu_bang_khung(master, pd.read_parquet(p),
                                     cot="text", ten="ocr_asr")
        print(f"Ứng viên: kênh 3 ({len(k):,} khung có chữ)")
        print("⚠️  KÊNH 3 MỘT MÌNH CHO 0,0000 TRÊN TRAKE (26_do_don_cuc_trake.py).")
        print("    Bốn biến thể nhiều khả năng bằng nhau — chạy lại với --cache.")

    de_that = [c.id for c in cau if c.id.startswith("trake-DE1")]
    print(f"\n{len(cau)} câu TRAKE  ({len(de_that)} câu do BTC viết, "
          f"{len(cau) - len(de_that)} câu tự soạn)")
    print("⚠️  đọc THẮNG-THUA-HOÀ, đừng đọc mỗi điểm trung bình\n")

    # Ứng viên dựng MỘT LẦN, dùng chung cho mọi biến thể: dựng lại theo từng
    # biến thể thì khác biệt đo được có thể đến từ nhiễu của kênh chứ không
    # phải từ khâu lắp ráp — đúng thứ đang cần cô lập.
    kho = {}
    for c in cau:
        sk = R.tach_su_kien(c.cau_hoi)
        kho[c.id] = ([k.tim(s, k=a.k) for s in sk] if a.cache
                     else [k.tim(R.tach_truy_van(s), k=a.k) for s in sk])

    ket: dict = {}
    for dung_sai in MOC_DUNG_SAI:
        print(f"=== dung sai ±{dung_sai}s " + "=" * 46)
        diem: dict = {}
        for ten, cau_hinh in BIEN_THE.items():
            diem[ten] = {c.id: cham_mot_cau(kho[c.id], c, master, tra,
                                            dung_sai, a.k, **cau_hinh)
                         for c in cau}
        # Ngưỡng nhiễu: một câu TRAKE đổi đúng MỘT vị trí trong N thì điểm đổi
        # cỡ 1/(N·số câu). Dưới mức đó là vô nghĩa, in ra để khỏi tự lừa mình.
        nguong = 1.0 / (min(len(c.row_id_dung) for c in cau) * len(cau))
        print(f"  (ngưỡng nhiễu ~{nguong:.4f}/câu)")
        for ten in BIEN_THE:
            in_so_sanh(ten, diem["cu"], diem[ten], nguong)
        if de_that:
            print(f"  --- chỉ {len(de_that)} câu ĐỀ THẬT (n nhỏ, không kết luận) ---")
            for ten in BIEN_THE:
                v = sum(diem[ten][i] for i in de_that) / len(de_that)
                d = v - sum(diem["cu"][i] for i in de_that) / len(de_that)
                print(f"  {NHAN[ten]:38} {v:.4f}  ({d:+.4f})")
        ket[dung_sai] = diem
        print()

    # Kết luận theo đúng kỷ luật `bao_cao_do_nhay`: ĐẢO DẤU giữa hai mức dung
    # sai nghĩa là KHÔNG KẾT LUẬN ĐƯỢC, không phải "hơi hơn".
    print("=== ket luan " + "=" * 52)
    a1, a2 = MOC_DUNG_SAI[0], MOC_DUNG_SAI[-1]
    for ten in BIEN_THE:
        if ten == "cu":
            continue
        d = [sum(ket[m][ten].values()) / len(cau)
             - sum(ket[m]["cu"].values()) / len(cau) for m in (a1, a2)]
        # ⚠️ Phải xét DẤU CỦA CÁC MỨC KHÁC 0, không xét `d[0] > 0 and d[1] > 0`.
        # Bản đầu viết thế và gán nhãn `🟡 TỆ HƠN` cho `±2s +0,0000 |
        # ±15s +0,0079` — một kết quả DƯƠNG. Một mức bằng 0 nghĩa là "ở mức đó
        # không đổi gì", không phải "xấu đi".
        # Cùng dấu là ĐIỀU KIỆN CẦN, chưa đủ. Phải vượt 2·SE ở ít nhất một
        # mức thì mới được gọi là ✅ — giống hệt `cham_diem.bao_cao_do_nhay`.
        manh = any(abs(h) > s > 0 for h, s in
                   (sai_so_cap(ket[m]["cu"], ket[m][ten]) for m in (a1, a2)))
        khac_khong = [x for x in d if abs(x) > 1e-9]
        if not khac_khong:
            kl = "⚪ KHÔNG ĐỔI GÌ"
        elif min(khac_khong) < 0 < max(khac_khong):
            kl = "❌ ĐẢO DẤU — không kết luận được"
        elif khac_khong[0] > 0:
            kl = ("✅ ON DINH — tốt hơn, vượt nhiễu" if manh
                  else "🟡 YEU — cùng dấu dương nhưng CHƯA vượt nhiễu")
        else:
            kl = ("🔻 TỆ HƠN, vượt nhiễu" if manh
                  else "🟠 tệ hơn nhưng chưa vượt nhiễu")
        print(f"  {NHAN[ten]:38} ±{a1}s {d[0]:+.4f} | ±{a2}s {d[1]:+.4f}  {kl}")


if __name__ == "__main__":
    main()
