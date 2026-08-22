"""
36_do_cua_so.py — Đo cách chấm theo CỬA SỔ keyframe (A38) trên tập dev.

CÂU HỎI ĐANG ĐO
===============

`dense.KenhAnh.tim` lấy `max` **qua các mệnh đề trên cùng một khung** — nó
thưởng cho khung khớp MỘT mệnh đề thật mạnh. A38 nói cách đó lệch với cách đề
được viết: người ra đề **xem video** rồi tả một quãng thời gian, còn ta thì
khớp từng ảnh tĩnh rời rạc. Ca `query-p1-4-kis` là bằng chứng: kf183 phủ mệnh
đề 1, kf186/187 phủ mệnh đề 2, **không khung nào phủ cả hai**.

`cua_so.diem_cua_so` đảo hai phép toán lại: **cộng** qua mệnh đề, lấy `max`
qua các khung lân cận — thưởng cho vùng phủ được NHIỀU mệnh đề.

    python scripts/36_do_cua_so.py --cache index/truy_van.npz

Không có `--cache` thì phải nạp SigLIP2 (~3,5 GB) — máy 7,7 GB đã treo hai lần
vì việc này, `dense.kiem_ram` sẽ chặn trước.

TẠI SAO MỐC NỀN Ở ĐÂY PHẢI DỰNG LẠI BẰNG TAY
=============================================

Không gọi `kenh.tim()` làm mốc nền mà tự dựng bằng `diem_khung_roi(sim)`, dù
hai thứ cho cùng kết quả. Lý do: cả hai nhánh phải ăn **cùng một ma trận
`sim`**, mã hoá **cùng một lần**. Gọi `tim()` là mã hoá lại lượt hai, và nếu
danh sách mệnh đề bị dựng khác đi một chỗ nào đó thì khác biệt đo được đến từ
đấy chứ không từ cách chấm — đúng loại lỗi quy công nhầm mà repo này đã vấp.

Có một chốt tự kiểm ngay trong script: `--kiem-moc` so mốc nền tự dựng với
`kenh.tim()` thật, phải trùng khít.

ĐỌC KẾT QUẢ THẾ NÀO
===================

Cột `chỉ 23 câu ĐỀ THẬT` là cột đáng tin nhất — đó là câu do BTC viết, đúng
phân bố đang đi thi (63 từ / 2,4 mệnh đề). Tập dev tự soạn (~15-22 từ, 1,1
mệnh đề) **về nguyên tắc không lộ ra được A38**: câu một mệnh đề thì cộng qua
mệnh đề với lấy max qua mệnh đề là một. Nếu cột tự soạn ⚪ mà cột đề thật
dương thì đó chính là điều A38 dự đoán, không phải mâu thuẫn.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                            # noqa: E402
import tap_dev                                             # noqa: E402
from cham_diem import MOC_DUNG_SAI, bao_cao_do_nhay        # noqa: E402
from cua_so import bien_video, diem_cua_so, diem_khung_roi  # noqa: E402
from dense import KenhAnh, KenhAnhCache                    # noqa: E402
from schema import Candidate                               # noqa: E402

BAN_KINH = (1, 2, 3, 5)


def ma_tran_sim(kenh, cau_hoi: str) -> np.ndarray:
    """(số_mệnh_đề, số_dòng) — mã hoá ĐÚNG những chuỗi `run.py` sẽ tra.

    Dùng lại `R.tach_truy_van` chứ không tự cắt câu: chép lại logic cắt là mở
    đường cho hai bản lệch nhau, và cache vector cũng được sinh bằng chính hàm
    này (`25_ma_hoa_truy_van.py`).
    """
    return np.stack([kenh._nhan(kenh.encode_text(m))
                     for m in R.tach_truy_van(cau_hoi)])


def thanh_ung_vien(diem: np.ndarray, master, k: int) -> list:
    """Điểm mỗi dòng -> top-`k` `Candidate`, giống hệt phần đuôi của `tim()`."""
    lay = min(len(diem), k + 200)
    top = np.argpartition(-diem, lay - 1)[:lay]
    top = top[np.argsort(-diem[top])][:k]
    return [Candidate(row_id=int(i), video_id=r.video_id,
                      frame_idx=int(r.frame_idx), score=float(diem[i]),
                      source="cua_so",
                      meta={"pts_time": float(r.pts_time), "fps": float(r.fps),
                            "kf_n": int(r.kf_n), "title": r.title})
            for i, r in zip(top, master.iloc[top].itertuples())]


def main():
    ap = argparse.ArgumentParser(description="do cach cham theo cua so (A38)")
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--cache", type=Path, default=None,
                    help="index/truy_van.npz — chạy kênh 1 mà KHÔNG nạp model")
    ap.add_argument("--matrix", default="clip_siglip2.npy")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--ban-kinh", type=int, nargs="*", default=list(BAN_KINH))
    ap.add_argument("--kiem-moc", action="store_true",
                    help="chốt tự kiểm: mốc nền tự dựng phải trùng kenh.tim()")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    kenh = (KenhAnhCache(str(a.index), a.cache, matrix=a.matrix) if a.cache
            else KenhAnh(str(a.index), matrix=a.matrix))

    cau = [c for c in tap_dev.doc() if c.loai in ("KIS", "QA")]
    if a.cache:
        thieu = kenh.co_du([m for c in cau for m in R.tach_truy_van(c.cau_hoi)])
        if thieu:
            raise SystemExit(
                f"{len(thieu)} mệnh đề chưa có trong cache, ví dụ:\n  {thieu[0][:100]!r}\n"
                f"Mã hoá thêm: python scripts/25_ma_hoa_truy_van.py --tap-dev --gop")

    nhom = bien_video(master)

    # Mã hoá MỘT LẦN cho mọi biến thể — xem docstring: dùng chung `sim` là điều
    # kiện để khác biệt đo được thuộc về cách CHẤM, không thuộc về khâu mã hoá.
    print(f"Mã hoá {len(cau)} câu KIS/QA...", flush=True)
    sim_cua = {c.id: ma_tran_sim(kenh, c.cau_hoi) for c in cau}
    so_md = [len(sim_cua[c.id]) for c in cau]
    print(f"  trung bình {np.mean(so_md):.2f} mệnh đề/câu; "
          f"{sum(1 for x in so_md if x == 1)}/{len(cau)} câu chỉ có MỘT mệnh đề "
          f"(A38 không thể lộ ra ở những câu này)\n")

    if a.kiem_moc:
        c0 = cau[0]
        tu_dung = thanh_ung_vien(diem_khung_roi(sim_cua[c0.id]), master, a.k)
        that = kenh.tim(R.tach_truy_van(c0.cau_hoi), k=a.k)
        khop = [x.row_id for x in tu_dung] == [x.row_id for x in that]
        print(f"[kiểm mốc] {c0.id}: mốc tự dựng {'TRÙNG' if khop else '❌ LỆCH'} "
              f"kenh.tim()\n")
        if not khop:
            raise SystemExit("Mốc nền lệch — mọi so sánh sau đó vô nghĩa.")

    cau_hinh = {"khung rời (đang chạy)":
                lambda c: thanh_ung_vien(diem_khung_roi(sim_cua[c.id]), master, a.k)}
    for r in a.ban_kinh:
        cau_hinh[f"cửa sổ ±{r}"] = (
            lambda c, r=r: thanh_ung_vien(
                diem_cua_so(sim_cua[c.id], nhom, ban_kinh=r), master, a.k))

    de_that = [c for c in cau if "-DE1-" in c.id]
    tu_soan = [c for c in cau if "-DE1-" not in c.id]

    for ten, bo in (("TOÀN BỘ", cau),
                    (f"chỉ {len(de_that)} câu ĐỀ THẬT (do BTC viết)", de_that),
                    (f"chỉ {len(tu_soan)} câu TỰ SOẠN", tu_soan)):
        if not bo:
            continue
        print("=" * 70)
        print(f"### {ten}")
        print("=" * 70)
        print(bao_cao_do_nhay(bo, cau_hinh, master, MOC_DUNG_SAI, a.k))
        print()


if __name__ == "__main__":
    main()
