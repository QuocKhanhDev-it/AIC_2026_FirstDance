"""
19_tim_cau_trake.py — Tìm video ĐÁNG soạn câu TRAKE, và sinh sẵn khung JSONL.

Tập dev đang có **0 câu TRAKE / 117 câu**. Nghĩa là `run.dung_trake()` — thứ
quyết định điểm của cả một trong ba dạng truy vấn — **chưa từng được đo một
lần nào**. Không có câu TRAKE thì mọi lựa chọn trong đó (nội suy, rải đều khi
dồn cục, ép tăng dần) đều là phỏng đoán.

TRAKE KHÓ SOẠN HƠN KIS/QA, VÀ ĐÂY LÀ CHỖ SCRIPT GIÚP ĐƯỢC
==========================================================

Câu KIS chỉ cần **một** khung đẹp. Câu TRAKE cần **N khoảnh khắc phân biệt
được, đúng thứ tự thời gian, trong CÙNG một video** — duyệt tay cả nghìn
keyframe để tìm là việc nản. Script này thu hẹp lại còn vài chục video.

CHUỖI SỰ KIỆN ≠ NHIỀU CẢNH KHÁC NHAU

Bản đầu của script chấm sai chỗ này, nên ghi lại: nó thưởng cho việc **trải
rộng theo thời gian**, và trên bản tin *"60 Giây Sáng"* nó chọn ba mẩu tin cách
nhau 5 phút — ba chủ đề chẳng liên quan gì nhau. Câu TRAKE soạn từ đó là câu vô
nghĩa, mà lại **trông hợp lệ**, nên còn tệ hơn câu sai hẳn.

Chuỗi sự kiện là **các bước của MỘT hành động liên tục**: người bước vào ->
ngồi xuống -> cầm ly lên. Nên script tìm **cửa sổ ngắn** (~1 phút) trong đó
cảnh **đang biến đổi mà chưa đổi hẳn** — cosine giữa hai keyframe liền nhau nằm
trong "dải hành động" 0,72–0,92. Trên 0,92 là cảnh đứng yên (không có gì để tả
thành nhiều bước); dưới 0,72 là cắt cảnh sang chủ đề khác.

LOAI NOI DUNG NAO SOAN DUOC CAU TRAKE — DA KIEM TAN MAT

    L27 (Viet Nam Di La Ghien)   DUOC   phong su, hanh dong lien tuc
    L21 / L22 (60 Giay)          KHONG  ban tin: chuoi mau tin ROI RAC

Da mo anh xem: ung vien manh nhat cua L21_V019 la khung 1 studio hai bien tap
vien, khung 2 la B-roll cua MOT TIN HOAN TOAN KHAC ('Ha Giang: Xe o to lao
xuong vuc'). Diem cosine cao vi canh doi that, nhung do khong phai chuoi su
kien. **Ban tin khong dung duoc cho TRAKE ve mat cau truc** — dung phi thoi
gian duyet L21/L22, chuyen sang nhom co phong su.

    python scripts/19_tim_cau_trake.py --nhom L27 --so-su-kien 3
    python scripts/19_tim_cau_trake.py --video L27_V013 --so-su-kien 4 --mo

⚠️ **Script KHÔNG viết được câu hỏi.** Nó chỉ chỉ chỗ. Phải mở ảnh ra nhìn rồi
tự tả — đúng kỷ luật §4 của docs/07_lam_tap_dev.md. Một câu TRAKE bịa từ ranh
giới cosine mà không nhìn ảnh là câu vô nghĩa, và tệ hơn câu sai vì nó trông
hợp lệ.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

# DẢI HÀNH ĐỘNG. Cosine giữa hai keyframe liền nhau:
#   > 0,92   cảnh gần như đứng yên — không có gì để tả thành nhiều sự kiện
#   0,72–0,92  CÙNG bối cảnh nhưng ĐANG DIỄN RA — đây là thứ TRAKE cần
#   < 0,72   cắt cảnh, đổi hẳn chủ đề — hai mẩu tin khác nhau, không phải chuỗi
THAP, CAO = 0.72, 0.92

# Cửa sổ thời gian một chuỗi sự kiện thường nằm gọn trong đó.
CUA_SO_GIAY = 60.0


def dai_dien(doan: np.ndarray) -> int:
    """Khung giữa đoạn — ổn định hơn khung đầu/cuối (hay dính lúc chuyển cảnh)."""
    return int(doan[len(doan) // 2])


def tim_cua_so(mat, rid: np.ndarray, t: np.ndarray, n: int,
               cua_so: float) -> tuple[float, np.ndarray]:
    """Tìm đoạn LIÊN TỤC tốt nhất trong một video để soạn câu TRAKE.

    ⚠️ **Bản đầu của script này chấm SAI BẢN CHẤT.** Nó thưởng cho việc trải
    rộng theo thời gian, nên trên bản tin *"60 Giây Sáng"* nó chọn ba mẩu tin
    cách nhau 5 phút — ba chủ đề chẳng liên quan gì nhau. Đó không phải một
    chuỗi sự kiện, và một câu TRAKE soạn từ đó là câu vô nghĩa.

    Chuỗi sự kiện là **các bước của MỘT hành động liên tục**: người bước vào ->
    ngồi xuống -> cầm ly lên. Nên phải tìm cửa sổ ngắn (~1 phút) trong đó cảnh
    **đang biến đổi mà chưa đổi hẳn** — cosine nằm trong dải hành động.

    Trả `(điểm, row_id trong cửa sổ)`.
    """
    if len(rid) < n + 1:
        return 0.0, rid
    v = np.asarray(mat[rid], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    ke = (v[:-1] * v[1:]).sum(1)
    trong_dai = (ke >= THAP) & (ke <= CAO)

    tot, tot_lat = 0.0, rid
    i = 0
    while i < len(rid):
        j = i
        while j + 1 < len(rid) and t[j + 1] - t[i] <= cua_so:
            j += 1
        if j - i >= n - 1:
            cap = trong_dai[i:j]                       # các cặp trong cửa sổ
            if len(cap):
                # tỷ lệ cặp "đang diễn ra" × số khung, chuộng cửa sổ vừa đủ dày
                d = float(cap.mean()) * min(len(cap) / (n * 2), 1.0)
                if d > tot:
                    tot, tot_lat = d, rid[i:j + 1]
        i += 1
    return tot, tot_lat


def khung_jsonl(vid: str, rid_moi_su_kien: list, nhom: str, so: int) -> dict:
    """Khung JSONL để người điền — `row_id_dung` của TRAKE là list[list[int]]."""
    return {
        "id": f"trake-{nhom}-{so:03d}",
        "loai": "TRAKE",
        "cau_hoi": "\n".join(f"{i+1}. <TẢ SỰ KIỆN {i+1}>"
                             for i in range(len(rid_moi_su_kien))),
        "row_id_dung": [[int(r)] for r in rid_moi_su_kien],
        "dap_an": "",
        "nguon": f"19_tim_cau_trake {vid}",
        "ghi_chu": "CHƯA KIỂM BẰNG MẮT",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--nhom", help="chỉ xét một nhóm L, vd L21")
    ap.add_argument("--video", help="soi kỹ một video cụ thể")
    ap.add_argument("--so-su-kien", type=int, default=3)
    ap.add_argument("--cua-so", type=float, default=CUA_SO_GIAY,
                    help="do rong cua so thoi gian (giay) de tim chuoi")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--mo", action="store_true",
                    help="mở ảnh đại diện của từng đoạn (cần có ảnh trên máy)")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    mat = np.load(a.index / "clip.npy", mmap_mode="r")

    # Chỉ xét video CÓ ẢNH trên máy này — không mở được ảnh thì không kiểm
    # bằng mắt được, mà câu TRAKE chưa kiểm thì không dùng được.
    co_anh = master[master.kf_path.notna()]
    if co_anh.empty:
        raise SystemExit("Máy này chưa có ảnh keyframe nào. Cần gói Keyframes_*.")
    if a.nhom:
        co_anh = co_anh[co_anh.video_id.str[:3] == a.nhom.upper()]
    if a.video:
        co_anh = co_anh[co_anh.video_id == a.video]
    if co_anh.empty:
        raise SystemExit("Không còn video nào sau khi lọc.")

    print(f"Xét {co_anh.video_id.nunique()} video có ảnh "
          f"| cửa sổ {a.cua_so:.0f}s | dải hành động {THAP}–{CAO} "
          f"| tìm {a.so_su_kien} sự kiện\n")

    ket = []
    for vid, sub in co_anh.groupby("video_id", sort=True):
        rid = sub.index.to_numpy()
        t = sub.pts_time.to_numpy(dtype=float)
        d, lat = tim_cua_so(mat, rid, t, a.so_su_kien, a.cua_so)
        if d > 0:
            ket.append((d, vid, lat))
    ket.sort(key=lambda x: -x[0])

    if not ket:
        raise SystemExit(
            f"Không video nào có cửa sổ {a.cua_so:.0f}s đủ {a.so_su_kien} khung "
            f"đang diễn ra. Thử --cua-so lớn hơn, hoặc --so-su-kien nhỏ hơn.")

    print(f"{'điểm':>6}  {'video':<11} {'khung':>6}  {'dài (s)':>8}  tiêu đề")
    print("-" * 88)
    for d, vid, lat in ket[:a.top]:
        t = master.pts_time.iloc[lat]
        print(f"{d:6.3f}  {vid:<11} {len(lat):>6}  {t.max()-t.min():>8.0f}  "
              f"{str(master.title.iloc[lat[0]])[:44]}")

    # Video mạnh nhất: in chi tiết + khung JSONL sẵn để điền
    d, vid, lat = ket[0]
    print(f"\n{'=' * 88}\nỨNG VIÊN MẠNH NHẤT — {vid}\n{'=' * 88}")
    # rải đều TRONG cửa sổ — các bước của cùng một hành động
    buoc = max(1, (len(lat) - 1) // max(a.so_su_kien - 1, 1))
    rid = [int(lat[min(i * buoc, len(lat) - 1)]) for i in range(a.so_su_kien)]

    print(f"{'sự kiện':>8}  {'row_id':>8}  {'giây':>8}  {'kf_name':>10}")
    for i, r in enumerate(rid, 1):
        g = master.iloc[r]
        print(f"{i:>8}  {r:>8}  {g.pts_time:>8.1f}  {str(g.kf_name):>10}")

    print("\nDán vào dev/tap_dev_thanh_vien/tap_dev_<nhóm>.jsonl rồi ĐIỀN TAY:")
    print(json.dumps(khung_jsonl(vid, rid, vid[:3], 1), ensure_ascii=False))

    print("\nMở ảnh ra nhìn rồi mới tả — ĐỪNG tả từ ranh giới cosine:")
    print(f"    python scripts/10_contact_sheet.py --tra {' '.join(map(str, rid))} --mo")

    if a.mo:
        import subprocess
        for r in rid:
            p = master.kf_path.iloc[r]
            if isinstance(p, str) and Path(p).exists():
                subprocess.Popen(["cmd", "/c", "start", "", p], shell=False)


if __name__ == "__main__":
    main()
