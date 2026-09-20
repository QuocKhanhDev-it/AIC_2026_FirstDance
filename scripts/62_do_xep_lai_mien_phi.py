"""
62_do_xep_lai_mien_phi.py — Xếp lại bằng tín hiệu ĐÃ CÓ, không model nào.

    python scripts/62_do_xep_lai_mien_phi.py

BỐI CẢNH: A54 đo khoảng trống giữa điểm thật và trần "xếp lại hoàn hảo" là
**33 điểm phần trăm** ở bể 1000. Câu hỏi của script này: lấy được bao nhiêu
phần trong đó mà KHÔNG cần VLM, không cần GPU, không cần dữ liệu mới?

HAI TÍN HIỆU ĐANG BỊ VỨT ĐI

**1. Đồng thuận mệnh đề.** RRF cộng `1/(k+hạng)`, nên một ứng viên hạng 3 ở
MỘT mệnh đề (0,0159) vượt ứng viên hạng 12 ở BỐN mệnh đề (4×0,0139 = 0,0556)?
Không — nhưng hạng 3 ở một mệnh đề (0,0159) vẫn hơn hạng 40 ở hai mệnh đề
(0,0200)... ranh giới là do hằng số k quyết định, chứ không do "trúng mấy
mệnh đề". Trúng NHIỀU mệnh đề là bằng chứng khác loại: nó nói ứng viên khớp
nhiều PHẦN của cảnh được tả, không chỉ khớp mạnh một phần.

**2. Ủng hộ theo video.** Đáp án đúng hiếm khi đứng một mình: video đúng
thường có nhiều khung khác cùng lọt bể. Một khung lẻ loi trong một video không
có khung nào khác đáng ngờ hơn. (Nhóm khác gọi là `support_weight`; họ đo và
KHÔNG roll out được — nên đây là đo lại cho hệ của mình, không phải chép.)

⚠️ Cả hai đều nhân vào điểm RRF chứ không thay nó. Thay hẳn thứ tự bằng một
tín hiệu phụ là cách chắc chắn để mất những câu kênh 1 vốn đã làm đúng.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import bao_cao_do_nhay                 # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402
from schema import Candidate                          # noqa: E402

W3 = 0.5
TOP_M = 5                                             # khung/video tính ủng hộ


def _xep_lai(ds, he_so):
    """Nhân điểm RRF với hệ số rồi xếp lại. Giữ nguyên đối tượng Candidate."""
    moi = [Candidate(row_id=c.row_id, video_id=c.video_id,
                     frame_idx=c.frame_idx, score=c.score * he_so(c),
                     source=c.source, meta=c.meta) for c in ds]
    return sorted(moi, key=lambda c: -c.score)


def dong_thuan(ds, dem: dict, w: float):
    """Ứng viên trúng nhiều mệnh đề được cộng thêm w cho mỗi mệnh đề dư."""
    return _xep_lai(ds, lambda c: 1.0 + w * (dem.get(c.row_id, 1) - 1))


def ung_ho_video(ds, w: float, top_m: int = TOP_M):
    """Điểm của video = tổng `top_m` điểm cao nhất của nó; chuẩn hoá về [0,1]."""
    theo_video = defaultdict(list)
    for c in ds:
        theo_video[c.video_id].append(c.score)
    ho = {v: sum(sorted(s, reverse=True)[:top_m]) for v, s in theo_video.items()}
    cao = max(ho.values()) if ho else 1.0
    return _xep_lai(ds, lambda c: 1.0 + w * (ho[c.video_id] / cao))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    ap.add_argument("--be", type=int, default=300)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    print(f"{a.file.name}: đo {len(giu)}/{len(cau)} câu | bể {a.be}\n")

    tho, so_md = {}, {}

    def bo(c):
        """Bể ứng viên + số mệnh đề mà mỗi ứng viên trúng."""
        if c.id not in tho:
            md = R.tach_truy_van(c.cau_hoi)
            cac = [k1.tim(m, k=a.be) for m in md]
            anh = hop_nhat(cac)
            dem = defaultdict(int)
            for ds in cac:
                for x in ds:
                    dem[x.row_id] += 1
            so_md[c.id] = dict(dem)
            tho[c.id] = hop_nhat([anh, k3.tim(c.cau_hoi, k=a.be)],
                                 trong_so=[1.0, W3])
        return tho[c.id]

    cau_hinh = {"1. mốc: run.py": lambda c: bo(c)[:100]}
    for w in (0.1, 0.25, 0.5):
        cau_hinh[f"2. đồng thuận mệnh đề w={w:g}"] = (
            lambda w: lambda c: dong_thuan(bo(c), so_md[c.id], w)[:100])(w)
    for w in (0.25, 0.5, 1.0):
        cau_hinh[f"3. ủng hộ video w={w:g}"] = (
            lambda w: lambda c: ung_ho_video(bo(c), w)[:100])(w)
    cau_hinh["4. cả hai (0,25 + 0,5)"] = lambda c: ung_ho_video(
        dong_thuan(bo(c), so_md[c.id], 0.25), 0.5)[:100]

    print(bao_cao_do_nhay(giu, cau_hinh, master))


if __name__ == "__main__":
    main()
