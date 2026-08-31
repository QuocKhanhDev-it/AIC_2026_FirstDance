"""
47_chia_viec_encode.py — Chia việc encode ảnh cho nhiều người, mỗi người một
danh sách `--chi-video` RỜI NHAU.

Quota GPU Kaggle tính theo **tài khoản**: 6 người là 6 lần 30 giờ/tuần. Đây là
đòn bẩy lớn nhất còn lại, và nó không cần thêm phần cứng nào.

    python scripts/47_chia_viec_encode.py --nguoi 6
    python scripts/47_chia_viec_encode.py --nguoi 6 --ghi --ra chia_viec

Sinh ra `phan_1.txt` … `phan_N.txt`, mỗi dòng một `video_id`. Đưa thẳng vào
`08_encode.py --chi-video`.

VÌ SAO CHIA THEO VIDEO CHỨ KHÔNG THEO NHÓM L
=============================================

Nhóm L lệch nặng (A2): L25 có 37.445 ảnh, L23 chỉ 2.326. Chia theo nhóm thì
người ôm L25 làm gấp mười sáu lần người ôm L23 — cả nhóm vẫn phải chờ đúng
người chậm nhất, tức là không chia được gì cả.

Chia ở mức video, xếp video to trước rồi luôn bỏ vào phần đang nhẹ nhất
(thuật toán LPT). Vẫn giữ TRỌN VẸN từng video trong một phần — cắt đôi một
video giữa hai người là tự chuốc lấy chuyện hai bên encode lệch cấu hình trên
cùng một cảnh.

BA ĐIỀU BẮT BUỘC KHI GHÉP LẠI
==============================

  1. **Mọi người dùng ĐÚNG một `--model` và một `--pretrained`.** Không có
     ngoại lệ. Ma trận ghép từ hai model khác nhau là file hợp lệ hoàn toàn
     mà mọi kết quả đều sai — `18_ghep_ma_tran.py::kiem_cung_model` chặn
     chuyện đó, nhưng chỉ khi có sidecar.
  2. **Nộp `.npy` KÈM `.json` cùng tên.** Thiếu sidecar là mất luôn phép kiểm
     ở điều 1.
  3. **Một người giữ ma trận chính** và ghép lần lượt. Ghép xong chạy
     `08_encode.py --kiem-lech-hang` một lần cuối.
"""

import argparse
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent


def chia(master: pd.DataFrame, so_phan: int, chi_co_anh: bool = True):
    """Chia video thành `so_phan` phần cân nhau theo SỐ ẢNH.

    LPT (longest processing time first): xếp video giảm dần theo số ảnh, mỗi
    lần bỏ video kế tiếp vào phần đang nhẹ nhất. Đơn giản, tất định, và cho
    kết quả rất sát tối ưu ở bài toán này.
    """
    d = master[master.kf_path.notna()] if chi_co_anh else master
    kich_co = (d.groupby("video_id").size()
               .sort_values(ascending=False, kind="mergesort"))

    phan = [[] for _ in range(so_phan)]
    tai = [0] * so_phan
    for vid, n in kich_co.items():
        i = min(range(so_phan), key=lambda k: (tai[k], k))
        phan[i].append(vid)
        tai[i] += int(n)
    return [sorted(p) for p in phan], tai


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--nguoi", type=int, default=6, help="số người chia việc")
    ap.add_argument("--ca-kho", action="store_true",
                    help="chia cả 177.321 dòng, kể cả video CHƯA có ảnh ở máy "
                         "này (dùng khi lập kế hoạch cho lúc L26 đã lên)")
    ap.add_argument("--nhom", default=None, metavar="L26[,L25]",
                    help="chỉ chia trong những nhóm L này. Dùng khi một nhóm "
                         "lên Kaggle SAU khi cả nhóm đã chia xong phần còn "
                         "lại — chia lại từ đầu sẽ xáo trộn phần đã giao, "
                         "còn ai đã encode xong lại phải làm lại từ đầu.")
    ap.add_argument("--ra", type=Path, default=GOC / "chia_viec")
    ap.add_argument("--ghi", action="store_true",
                    help="ghi thật (mặc định chỉ xem trước)")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    if a.nhom:
        nh = {x.strip().upper() for x in a.nhom.split(",")}
        master = master[master.video_id.str[:3].isin(nh)]
        if master.empty:
            raise SystemExit(f"❌ Không có video nào thuộc {sorted(nh)}")
    phan, tai = chia(master, a.nguoi, chi_co_anh=not a.ca_kho)

    nguon = "cả kho" if a.ca_kho else "chỉ video CÓ ẢNH ở máy này"
    if a.nhom:
        nguon = f"nhóm {a.nhom.upper()}, " + nguon
    print(f"Chia {sum(tai):,} ảnh ({nguon}) cho {a.nguoi} người\n")
    print(f"{'phần':>6}  {'video':>6}  {'ảnh':>9}  nhóm L")
    print("-" * 62)
    for i, (p, t) in enumerate(zip(phan, tai), 1):
        nhom = sorted({v[:3] for v in p})
        print(f"{i:>6}  {len(p):>6}  {t:>9,}  {', '.join(nhom)}")
    print("-" * 62)
    print(f"{'lệch':>6}  {'':>6}  {max(tai) - min(tai):>9,}  "
          f"(max {max(tai):,} / min {min(tai):,})")

    # Không có video nào nằm ở hai phần — điều kiện để ghép lại không đè nhau.
    tat_ca = [v for p in phan for v in p]
    assert len(tat_ca) == len(set(tat_ca)), "CÓ VIDEO TRÙNG GIỮA HAI PHẦN"
    print(f"\n{len(tat_ca):,} video, không phần nào trùng nhau — đã kiểm.")

    if not a.ghi:
        print("\n(xem trước — thêm `--ghi` để ghi ra file)")
        return

    a.ra.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(phan, 1):
        f = a.ra / f"phan_{i}.txt"
        f.write_text("\n".join(p) + "\n", encoding="utf-8")
        print(f"  ✅ {f}  ({len(p)} video)")

    print(f"\nMỗi người chạy ĐÚNG một lệnh này, chỉ đổi số phần:\n"
          f"    python scripts/08_encode.py --model ViT-gopt-16-SigLIP2-384 \\\n"
          f"        --pretrained webli --chi-video phan_<N>.txt \\\n"
          f"        --workers 4 --batch 32 --out clip_gopt_phan<N>.npy\n\n"
          f"Nộp về CẢ `.npy` LẪN `.json` cùng tên. Thiếu sidecar là mất phép\n"
          f"kiểm cùng-model lúc ghép.")


if __name__ == "__main__":
    main()
