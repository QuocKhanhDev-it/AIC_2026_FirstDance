"""
13_do_dedup.py — Đo `dedup.gom_ban_sao` có đáng đưa vào đường ống không.

`src/dedup.py` mở đầu bằng một lời cảnh báo tự dán lên mình:

    ⚠️ Việc này chưa được chứng minh là tăng điểm. Theo kỷ luật GIAI ĐOẠN 3,
       đo trên tập dev: không tăng thì bỏ.

Script này trả lời dòng đó. Kết quả tóm tắt ở A11 của docs/Ke_hoach_AIC2026_v4.md.

BA CÁI BẪY ĐÃ VẤP KHI ĐO, ĐỀU DỰNG SẴN CHỐT Ở ĐÂY

1. TRUY VẤN HỎNG LÀM DEDUP TRÔNG HIỆU QUẢ GẤP TRĂM LẦN.
   Chấm 97 câu tiếng Việt: dedup bỏ 58,4/100 ứng viên. Nghe như một phát hiện
   lớn. Nhưng CLIP mù tiếng Việt (A10), nên vector truy vấn gần như ngẫu nhiên
   và top-100 đổ dồn vào một cảnh tĩnh duy nhất. Cùng script, truy vấn TIẾNG
   ANH mà CLIP đọc được: bỏ 0,5/100. Con số 58,4 đo cái hỏng của kênh 1, không
   đo cái lợi của dedup. `--doi-chung` là phần dựng để không tự lừa mình lần
   nữa.

2. ĐỔI HAI THỨ CÙNG LÚC.
   Lần đầu tôi so `k=100 không dedup` với `k=300 -> dedup -> cắt 100`, rồi định
   quy 30,5 vị trí lệch cho dedup. Hai nhánh khác nhau cả độ sâu. Ở đây cả hai
   nhánh dùng CHUNG một danh sách top-`--sau`, khác đúng một bước dedup.

3. BỂ ỨNG VIÊN LỆCH.
   Ma trận chạy thử chỉ encode một phần kho. Script tự khóa bể về phần đã
   encode và bỏ câu có đáp án nằm ngoài — giữ lại chỉ là cộng số 0 vào cả hai
   nhánh và pha loãng chênh lệch cần đo.

    python scripts/13_do_dedup.py                       # chẩn đoán + điểm
    python scripts/13_do_dedup.py --doi-chung           # thêm đối chứng tiếng Anh
    python scripts/13_do_dedup.py --matrix clip_siglip2.npy --moi-video 3
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import tap_dev                                       # noqa: E402
from cham_diem import bao_cao_do_nhay, MOC_DUNG_SAI  # noqa: E402
from dedup import gom_ban_sao, NGUONG                # noqa: E402
from dense import KenhAnh                            # noqa: E402

# Đối chứng: truy vấn TIẾNG ANH, thứ CLIP thật sự đọc được. Không có nhóm này
# thì mọi con số dedup đo trên tập dev tiếng Việt đều là ảo giác — xem bẫy 1.
DOI_CHUNG = [
    "a person riding a motorbike on the street",
    "a news anchor sitting at a desk in a studio",
    "a crowd of people at an outdoor festival",
    "firefighters spraying water on a burning building",
    "a woman cooking food in a kitchen",
    "an aerial view of a city with tall buildings",
    "students in a classroom raising their hands",
    "a football match in a stadium",
    "heavy flooding on a residential road",
    "two men shaking hands in front of flags",
    "a close-up of hands typing on a laptop",
    "traffic jam at an intersection with many motorbikes",
    "a doctor in a white coat talking to a patient",
    "workers harvesting rice in a green field",
    "a boat sailing on a river at sunset",
]


def nho(f):
    """Nhớ kết quả theo `id` câu hỏi.

    `bao_cao_do_nhay` chấm mỗi cấu hình một lần cho MỖI mức dung sai, mà truy
    hồi không phụ thuộc dung sai chút nào — không nhớ là chạy lại y hệt.
    """
    kho = {}

    def g(c):
        if c.id not in kho:
            kho[c.id] = f(c)
        return kho[c.id]
    return g


def han_moi_video(ds, mv, k=100):
    """Ràng buộc đa dạng của PHẦN C mục 2, áp SAU dedup."""
    if not mv:
        return ds[:k]
    dem, ra = {}, []
    for x in ds:
        if dem.get(x.video_id, 0) < mv:
            dem[x.video_id] = dem.get(x.video_id, 0) + 1
            ra.append(x)
    return ra[:k]


def loc_cau_tra_loi_duoc(cau, be):
    """Bỏ câu có đáp án nằm ngoài bể ứng viên (bẫy 3)."""
    if be is None:
        return cau, []
    duoc, bo = [], []
    for c in cau:
        phang = ([r for b in c.row_id_dung for r in b] if c.loai == "TRAKE"
                 else c.row_id_dung)
        (duoc if any(be[r] for r in phang) else bo).append(c)
    return duoc, bo


def do_mot_truy_van(kenh, cau_chu, sau, be, mv, nguong):
    """Dedup động tới đâu trên MỘT truy vấn. Hai nhánh dùng chung `goc` (bẫy 2)."""
    goc = kenh.tim(cau_chu, k=sau, be=be)
    sach = gom_ban_sao(goc, kenh.mat, nguong)
    a = han_moi_video(goc, mv)
    b = han_moi_video(sach, mv)
    lech = sum(1 for i in range(min(len(a), len(b)))
               if a[i].row_id != b[i].row_id) + abs(len(a) - len(b))
    return {
        "bo_trong_100": len(goc[:100]) - len(gom_ban_sao(goc[:100], kenh.mat, nguong)),
        "vi_tri_lech": lech,
        "hang_1_doi": bool(a and b and a[0].row_id != b[0].row_id),
        "so_video": len({x.video_id for x in a}),
    }


def bang_chan_doan(kenh, cac_cau, sau, be, mv, nguong, khoa) -> pd.DataFrame:
    return pd.DataFrame([
        {khoa: k, **do_mot_truy_van(kenh, q, sau, be, mv, nguong)}
        for k, q in cac_cau
    ])


def in_tom_tat(cd: str, d: pd.DataFrame):
    print(f"\n  {cd}: bỏ {d.bo_trong_100.mean():5.1f}/100 · "
          f"lệch {d.vi_tri_lech.mean():5.1f}/100 vị trí · "
          f"hạng 1 đổi {int(d.hang_1_doi.sum())}/{len(d)} câu · "
          f"top-100 trải {d.so_video.mean():.0f} video")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--matrix", default="clip.npy")
    ap.add_argument("--file", default=tap_dev.MAC_DINH, type=Path)
    ap.add_argument("--sau", type=int, default=300,
                    help="lấy sâu bao nhiêu trước khi dedup rồi cắt 100")
    ap.add_argument("--moi-video", type=int, default=0,
                    help="ràng buộc đa dạng, áp SAU dedup. 0 = tắt")
    ap.add_argument("--nguong", type=float, default=NGUONG)
    ap.add_argument("--doi-chung", action="store_true",
                    help="chạy thêm nhóm truy vấn tiếng Anh (bẫy 1)")
    ap.add_argument("--bo-diem", action="store_true",
                    help="chỉ chẩn đoán, không chấm điểm")
    a = ap.parse_args()

    cau = tap_dev.doc(a.file)
    if not cau:
        raise SystemExit(f"Chưa có câu nào trong {a.file}")

    kenh = KenhAnh(a.index, matrix=a.matrix)
    print(f"{a.matrix}  {kenh.mat.shape}  |  {kenh.model_tag} / {kenh.pretrained}")

    mat_na = kenh.dong_da_encode()
    day_du = bool(mat_na.all())
    be = None if day_du else mat_na
    print(f"bể ứng viên: {int(mat_na.sum()):,} / {len(mat_na):,} keyframe"
          f"{'' if day_du else '  (ma trận chạy thử — đã khóa bể)'}")

    cau, bo = loc_cau_tra_loi_duoc(cau, be)
    if bo:
        print(f"bỏ {len(bo)} câu có đáp án ngoài bể: "
              f"{', '.join(sorted({c.nhom(kenh.master) for c in bo}))}")
    if len(cau) < 10:
        raise SystemExit(f"Chỉ còn {len(cau)} câu trả lời được — quá ít để đo.")
    print(f"{len(cau)} câu đo được   |   moi_video = {a.moi_video or 'tắt'}"
          f"   |   lấy sâu {a.sau}   |   ngưỡng {a.nguong}\n")

    print("=" * 76)
    print("CHẨN ĐOÁN — dedup động tới đâu")
    print("=" * 76)

    cd = bang_chan_doan(kenh, [(c.nhom(kenh.master), c.cau_hoi) for c in cau],
                        a.sau, be, a.moi_video, a.nguong, "nhom")
    print(cd.groupby("nhom").agg(so_cau=("bo_trong_100", "size"),
                                 bo_100=("bo_trong_100", "mean"),
                                 lech=("vi_tri_lech", "mean"),
                                 video=("so_video", "mean")).round(1).to_string())
    in_tom_tat(f"tập dev ({len(cd)} câu tiếng Việt)", cd)

    if a.doi_chung:
        dc = bang_chan_doan(kenh, [("en", q) for q in DOI_CHUNG],
                            a.sau, be, a.moi_video, a.nguong, "nhom")
        in_tom_tat(f"đối chứng ({len(dc)} câu tiếng Anh)", dc)
        if cd.bo_trong_100.mean() > 5 * max(dc.bo_trong_100.mean(), 0.1):
            print("\n  ⚠️  Tập dev bỏ nhiều gấp bội đối chứng. Đó là dấu hiệu kênh\n"
                  "      KHÔNG ĐỌC ĐƯỢC truy vấn, không phải dấu hiệu dedup có ích.\n"
                  "      Đừng đọc con số tập dev là lợi ích của dedup — xem bẫy 1.")

    if a.bo_diem:
        return
    if cd.vi_tri_lech.max() == 0:
        print("\n⚪ Dedup không đổi vị trí nào trong top-100 của bất kỳ câu nào.\n"
              "   Không cần chấm điểm.")
        return

    print("\n" + "=" * 76)
    print("ĐIỂM")
    print("=" * 76)

    def nhanh(dedup: bool):
        def f(c):
            ds = kenh.tim(c.cau_hoi, k=a.sau, be=be)
            if dedup:
                ds = gom_ban_sao(ds, kenh.mat, a.nguong)
            return han_moi_video(ds, a.moi_video)
        return nho(f)

    print(bao_cao_do_nhay(cau, {"không dedup (mốc nền)": nhanh(False),
                                "có dedup": nhanh(True)},
                          kenh.master, MOC_DUNG_SAI))


if __name__ == "__main__":
    main()
