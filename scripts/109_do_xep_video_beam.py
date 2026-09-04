"""
109_do_xep_video_beam.py — Xếp hạng video TRAKE bằng CHUỖI HỢP LỆ TỐT NHẤT, không bằng tổng-log-max.

    python scripts/109_do_xep_video_beam.py

LỖ HỔNG NHẮM TỚI, VÀ CON SỐ ĐÚNG ĐỂ TRÍCH DẪN

`cham_video()` chấm mỗi video bằng `Σ log(max_e)` — điểm cao nhất cho TỪNG sự
kiện, cộng lại. Nó **không kiểm tra có tồn tại một chuỗi tăng dần theo thời
gian hay không**. Một video mà cả ba sự kiện đều khớp mạnh, nhưng chỉ khớp ở
những mốc thời gian không xếp được thành dãy tăng, vẫn đứng hạng 1 — rồi
`beam_video` trả về ít chuỗi hơn hạn ngạch, và số dòng đó **mất trắng**.

⚠️ Con số **KHÔNG** nên trích cho lỗ hổng này là "37% ở ±2s" (A63). Đó là
khoảng cách giữa chấm ở tầng KÊNH và tầng NỘP — một hiện vật của cách đo, và
chính A79 đã cho thấy nó **sụp từ 48% xuống 4,3% ở ±15s**, tức phần lớn nằm ở
độ chính xác THỜI GIAN chứ không ở chọn video.

Con số đúng là **khoảng cách tới oracle** trên 17 câu (A79):

    ±2s    K-best 0,3547   oracle 0,4576   -> còn 0,1029
    ±15s   K-best 0,5488   oracle 0,6718   -> còn 0,1230

và A79 kết luận thẳng: *"Vẫn còn ngần ấy nằm ở khâu CHỌN VIDEO."* Đó mới là
bằng chứng cho phép đo này — và nó mạnh hơn con số 37%, vì nó KHÔNG sụp ở ±15s.

VÌ SAO ĐÂY KHÔNG PHẢI THỨ A78 ĐÃ THỬ

A78 dò bốn cách hợp điểm (tổng / tổng-log / điều hoà / min) và thấy nút này
**trơ** — cả bốn chọn ra gần như cùng một video. Nhưng cả bốn đều là hàm hợp
của `max_e`, tức đều **mù với ràng buộc thời gian**. Điểm chuỗi hợp lệ không
nằm trong họ đó: nó đòi tồn tại `t_1 < t_2 < … < t_N`, thứ mà không hàm hợp nào
của các giá trị max nhìn thấy được.

BỐN CẤU HÌNH — TÁCH "LỌC KHẢ THI" KHỎI "XẾP LẠI HẠNG"

Nếu chỉ so cấu hình cuối với mốc nền thì thắng cũng không biết thắng nhờ đâu.

    1. MỐC          tổng-log-max (đang chạy)
    2. LỌC          vẫn tổng-log-max, nhưng BỎ video không có chuỗi hợp lệ
    3. XẾP LẠI      xếp theo điểm chuỗi hợp lệ tốt nhất, bể = top-P tổng-log-max
    4. XẾP LẠI TOÀN xếp theo điểm chuỗi hợp lệ, bể = MỌI video có ứng viên

Dòng 2 là thứ rẻ nhất có thể làm. Nếu nó lấy hết phần thắng thì dòng 3 và 4 chỉ
là phức tạp thừa.

ĐIỂM CHUỖI TỐT NHẤT TÍNH BẰNG QUY HOẠCH ĐỘNG, KHÔNG BẰNG BEAM

Beam là xấp xỉ; ở đây cần con số ĐÚNG để xếp hạng, và bài toán đủ nhỏ để giải
chính xác. Với `m_i[j]` là điểm ứng viên `j` của sự kiện `i` tại thời điểm
`t_j`:

    D_0[j] = m_0[j]
    D_i[j] = m_i[j] + max{ D_{i-1}[k] : t_k < t_j }      (−∞ nếu không có k nào)
    điểm video = max_j D_{N-1}[j]

`N ≤ 5` sự kiện × `TOI_DA_UV = 20` ứng viên nên O(N·C²) = 2.000 phép — rẻ hơn
beam, và không có sai số xấp xỉ.

⚠️ Chấm ở tầng NỘP (`cham_trake_nhieu_muc`), không phải tầng KÊNH.
"""

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import bao_cao_tu_bang                 # noqa: E402
from cham_diem import cham_trake_nhieu_muc            # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

import kbest_trake as KB                              # noqa: E402

W3 = 0.5
AM_VO_CUC = float("-inf")


def diem_chuoi_tot_nhat(uv_theo_su_kien, pts) -> float:
    """Điểm chuỗi tăng dần ngặt TỐT NHẤT của một video, hoặc −∞ nếu không có.

    Quy hoạch động chính xác (xem docstring đầu file). Trả −∞ nghĩa là video
    này **không thể** tạo ra một dòng nộp hợp lệ nào — nó đang chiếm hạn ngạch
    dòng mà không sinh nổi dòng nào.
    """
    if not uv_theo_su_kien or any(not x for x in uv_theo_su_kien):
        return AM_VO_CUC
    truoc = [(pts[r], s) for r, s in uv_theo_su_kien[0]]
    for uv in uv_theo_su_kien[1:]:
        moi = []
        for r, s in uv:
            t = pts[r]
            tot = max((d for t0, d in truoc if t0 < t), default=AM_VO_CUC)
            moi.append((t, s + tot if tot != AM_VO_CUC else AM_VO_CUC))
        if all(d == AM_VO_CUC for _, d in moi):
            return AM_VO_CUC
        truoc = moi
    return max(d for _, d in truoc)


def xep_hang_video(theo_video, pts, cach: str, bể_P: int = 150):
    """`[video_id]` đã xếp hạng, theo một trong bốn cách."""
    diem_cu = KB.cham_video(theo_video)          # tổng-log-max, đã loại video thiếu
    xep_cu = sorted(diem_cu, key=lambda v: -diem_cu[v])
    if cach == "moc":
        return xep_cu

    def uv_cua(v):
        uv = [list(x) for x in theo_video[v]]
        for x in uv:
            x.sort(key=lambda t: -t[1])
        return [x[:KB.TOI_DA_UV] for x in uv]

    if cach == "loc":
        return [v for v in xep_cu
                if diem_chuoi_tot_nhat(uv_cua(v), pts) != AM_VO_CUC]

    be = xep_cu[:bể_P] if cach == "xep_lai" else xep_cu
    d = {v: diem_chuoi_tot_nhat(uv_cua(v), pts) for v in be}
    d = {v: x for v, x in d.items() if x != AM_VO_CUC}
    return sorted(d, key=lambda v: -d[v])


def lap_dong_theo_xep(cac_su_kien, master, xep, theo_video,
                      so_dong: int = 100):
    """Bản sao `kbest_trake.lap_dong` nhưng NHẬN SẴN thứ tự video.

    Chỉ đổi ĐÚNG MỘT THỨ so với bản đang chạy: nguồn của `xep`. Mọi thứ khác —
    hạn ngạch 40/25/15/12/8, 20 dòng đuôi, beam, giãn cách — giữ nguyên, nếu
    không thì thắng thua chẳng quy được cho ai.
    """
    pts = master.pts_time.values

    def chuoi(v, k):
        uv = [list(x) for x in theo_video[v]]
        for x in uv:
            x.sort(key=lambda t: -t[1])
        return KB.beam_video([x[:KB.TOI_DA_UV] for x in uv], pts, k,
                             KB.CACH_NHAU)

    n_tren = max(1, so_dong - KB.N_DUOI)
    ra = []
    for v, w in zip(xep[:KB.SO_VIDEO], KB.TY_LE):
        ra += chuoi(v, max(1, round(n_tren * w)))
        if len(ra) >= n_tren:
            break
    ra = ra[:n_tren]
    for v in xep[KB.SO_VIDEO:KB.SO_VIDEO + KB.N_DUOI]:
        ra += chuoi(v, 1)
    return ra[:so_dong]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl",
        GOC / "dev" / "tap_de_thi_thu.jsonl",
        GOC / "dev" / "tap_dev_trake.jsonl"])
    ap.add_argument("--be", type=int, default=100)
    ap.add_argument("--be-p", type=int, default=150,
                    help="cỡ bể sơ tuyển cho cách 'xếp lại'")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    pts = master.pts_time.values
    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    cau = []
    for f in a.file:
        cau += [c for c in tap_dev.doc(f) if c.loai == "TRAKE"]
    giu = [c for c in cau
           if not any(k1.co_du(R.tach_truy_van(m))
                      for m in R.tach_su_kien(c.cau_hoi))]
    print(f"\n{len(giu)} câu TRAKE | kênh 3 w={W3:g} | bể sơ tuyển "
          f"{a.be_p} video\n")

    nho = {}

    def ung_vien(c):
        """`[list[Candidate]]` — một danh sách cho mỗi sự kiện."""
        if c.id not in nho:
            ds = []
            for sk in R.tach_su_kien(c.cau_hoi):
                anh = hop_nhat([k1.tim(m, k=a.be)
                                for m in R.tach_truy_van(sk)])
                ds.append(hop_nhat([anh, k3.tim(sk, k=a.be)],
                                   trong_so=[1.0, W3])[:a.be])
            nho[c.id] = ds
        return nho[c.id]

    # Chẩn đoán TRƯỚC khi chấm: hạn ngạch dòng đang bị phí bao nhiêu?
    phi, tong_v, khong_kha_thi = [], [], []
    for c in giu:
        tv = KB.gom_theo_video(ung_vien(c))
        xep_cu = xep_hang_video(tv, pts, "moc")
        tong_v.append(len(xep_cu))
        top = xep_cu[:KB.SO_VIDEO + KB.N_DUOI]
        xau = sum(1 for v in top
                  if diem_chuoi_tot_nhat(
                      [sorted(list(x), key=lambda t: -t[1])[:KB.TOI_DA_UV]
                       for x in tv[v]], pts) == AM_VO_CUC)
        khong_kha_thi.append(xau)
        phi.append(len(lap_dong_theo_xep(ung_vien(c), master, xep_cu, tv)))

    import statistics as st
    print("CHẨN ĐOÁN — hạn ngạch dòng của cấu hình ĐANG CHẠY")
    print(f"  video có đủ ứng viên mọi sự kiện : trung vị "
          f"{int(st.median(tong_v))}  (min {min(tong_v)}, max {max(tong_v)})")
    print(f"  trong 25 video được chia dòng, số video KHÔNG có chuỗi hợp lệ: "
          f"trung vị {int(st.median(khong_kha_thi))}  "
          f"(min {min(khong_kha_thi)}, max {max(khong_kha_thi)})")
    print(f"  số dòng thực sự nộp được         : trung vị "
          f"{int(st.median(phi))}/100  (min {min(phi)}, max {max(phi)})")
    print("\n  -> Nếu cột giữa BẰNG 0 ở gần hết các câu thì lỗ hổng này KHÔNG")
    print("     tồn tại, và ba cấu hình dưới sẽ không đổi gì. Đọc bảng dưới")
    print("     với kỳ vọng đó.\n")

    def lam(cach):
        def f(c):
            tv = KB.gom_theo_video(ung_vien(c))
            xep = xep_hang_video(tv, pts, cach, a.be_p)
            if not xep:
                return []
            return lap_dong_theo_xep(ung_vien(c), master, xep, tv)
        return f

    ten = {"moc": "1. MỐC: tổng-log-max",
           "loc": "2. LỌC video không có chuỗi hợp lệ",
           "xep_lai": f"3. XẾP LẠI theo chuỗi (bể {a.be_p})",
           "xep_lai_toan": "4. XẾP LẠI theo chuỗi (bể TOÀN BỘ)"}
    bang = {ten[k]: cham_trake_nhieu_muc(giu, lam(k), master)
            for k in ("moc", "loc", "xep_lai", "xep_lai_toan")}
    print(bao_cao_tu_bang(bang))


if __name__ == "__main__":
    main()
