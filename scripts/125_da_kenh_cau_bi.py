"""
125_da_kenh_cau_bi.py — 100 dòng CHIA HẠN NGẠCH cho nhiều kênh, dành cho câu KHÔNG ai soi ra.

    python scripts/125_da_kenh_cau_bi.py --de dev/SOTUYEN3-bo-de-thi \\
        --goi query-p2-9-kis query-p2-11-kis query-p2-32-kis --ra submission_dakenh

VÌ SAO KHÔNG PHẢI "BẬT HẾT KÊNH RỒI HỢP NHẤT"

Bật thêm kênh rồi RRF là thứ đã đo và **bị bác ba lần**: kênh 4 objects làm tệ
đi (A62), kênh 5 caption ✅ tệ hơn ở 100% độ phủ (A90), kênh 6 BGE-M3 🟡 (A59).
Hợp nhất là phép **lấy trung bình ý kiến** — thêm một kênh yếu vào thì nó kéo
tụt những ứng viên mà kênh mạnh đã xếp đúng.

Nhưng ở đây bài toán KHÁC, và khác ở một điểm quyết định: **ta đã biết cấu hình
mặc định TRƯỢT câu này** — cả nhóm đã soi top-100 của nó và không thấy gì. Nên
tối đa hoá "chất lượng xếp hạng trung bình" là tối đa hoá sai thứ. Thứ cần tối
đa hoá là **xác suất có ÍT NHẤT MỘT dòng trúng**, mà cách chấm R@k thưởng đúng
điều đó: `R@k = max` trên top-k, không phải trung bình.

Nên: **chia hạn ngạch**, không hợp nhất. Mỗi kênh giữ nguyên thứ hạng của mình
trong phần đất của nó, và một kênh thường yếu nhưng thỉnh thoảng đúng vẫn có
chỗ đứng.

    40 dòng  ảnh + kênh 3 (w=0,5)   cấu hình MẠNH NHẤT đã đo — vẫn ưu tiên
    30 dòng  caption (kênh 5)       ĐỘC LẬP NHẤT: A71 đo Spearman 0,043 với
                                    kênh 1, tức nó nhìn thấy thứ khác hẳn
    20 dòng  objects (kênh 4)       tín hiệu khác: nhãn vật thể, không phải
                                    hình tổng thể hay chữ
    10 dòng  ảnh một mình           bỏ pha loãng của kênh 3

Con số 30 cho caption không phải bốc: A71 đo **chồng@20 giữa kênh 1 và kênh 5
chỉ 4,3%**, thấp nhất trong mọi cặp kênh. Kênh nào ít chồng nhất thì mỗi dòng
của nó mang nhiều thông tin MỚI nhất — đúng thứ cần khi kênh chính đã trượt.

⚠️ CHỈ DÙNG CHO CÂU KHÔNG CÓ ĐÁP ÁN SOI TAY. Câu đã soi thì khung người soi
đứng hạng 1 và phần còn lại chỉ là lưới an toàn — lúc đó cấu hình mặc định vẫn
là lựa chọn đúng, vì nó xếp hạng tốt hơn.

⚠️ Đây KHÔNG phải một phép đo. Không có nhãn cho ba câu này nên không chấm được
gì; đây là một quyết định về **phân bổ rủi ro** dựa trên cơ chế đã đo (độ độc
lập của các kênh), không phải một cấu hình được chứng minh là hơn.
"""

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import nop_bai                                        # noqa: E402
import run as R                                       # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from rrf import hop_nhat                              # noqa: E402
from schema import AnswerKIS, AnswerQA                # noqa: E402

W3 = 0.5
HAN_NGACH = [("ảnh + kênh 3", 40), ("caption", 30),
             ("objects", 20), ("ảnh một mình", 10)]
TOI_DA = 100


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--de", type=Path, required=True)
    ap.add_argument("--goi", nargs="+", required=True)
    ap.add_argument("--ra", type=Path, required=True)
    ap.add_argument("--be", type=int, default=300)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    de = {}
    for g in a.goi:
        f = a.de / f"{g}.txt"
        if not f.exists():
            raise SystemExit(f"❌ không có {f}")
        de[g] = f.read_text("utf-8").strip()

    # Gom ứng viên theo TỪNG kênh, dựng kênh nào xong thì giải phóng ngay —
    # máy 7,7 GB không giữ nổi hai chỉ mục BM25 cùng lúc.
    kho = {g: {} for g in de}

    from dense import KenhAnhCache
    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    for g, q in de.items():
        me = R.tach_truy_van(q)
        thieu = k1.co_du(me)
        if thieu:
            raise SystemExit(f"❌ {g}: {len(thieu)} mệnh đề chưa mã hoá")
        kho[g]["ảnh một mình"] = (hop_nhat([k1.tim(m, k=a.be) for m in me])
                                  if len(me) > 1 else k1.tim(me[0], k=a.be))
    del k1
    gc.collect()

    p3 = a.index / "ocr_asr.parquet"
    k3 = KenhVanBan.tu_bang_khung(master, pd.read_parquet(p3),
                                  cot="text", ten="ocr_asr")
    for g, q in de.items():
        v = k3.tim(q, k=a.be)
        kho[g]["ảnh + kênh 3"] = hop_nhat([kho[g]["ảnh một mình"], v],
                                          trong_so=[1.0, W3])
    del k3
    gc.collect()

    pc = a.index / "caption.parquet"
    if pc.exists():
        k5 = KenhVanBan.tu_bang_khung(master, pd.read_parquet(pc),
                                      cot="caption", ten="caption")
        for g, q in de.items():
            kho[g]["caption"] = k5.tim(q, k=a.be)
        del k5
        gc.collect()

    try:
        from objects import KenhObjects
        k4 = KenhObjects(str(a.index), master)
        for g, q in de.items():
            kho[g]["objects"] = k4.tim(q, k=a.be)
        del k4
        gc.collect()
    except Exception as e:
        print(f"⚠️ kênh 4 (objects) không dựng được: {e}")

    goi = {}
    for g, q in de.items():
        loai = g.rsplit("-", 1)[-1]
        dong, da = [], set()
        bao = []
        for ten, han in HAN_NGACH:
            uv = kho[g].get(ten)
            if not uv:
                bao.append(f"{ten}=0")
                continue
            them = 0
            for c in uv:
                if len(dong) >= TOI_DA or them >= han:
                    break
                khoa = (c.video_id, int(c.frame_idx))
                if khoa in da:
                    continue
                da.add(khoa)
                dong.append(AnswerQA(c.video_id, int(c.frame_idx), "")
                            if loai == "qa" else
                            AnswerKIS(c.video_id, int(c.frame_idx)))
                them += 1
            bao.append(f"{ten}={them}")
        # Còn trống thì lấp tiếp bằng kênh mạnh nhất.
        for c in kho[g]["ảnh + kênh 3"]:
            if len(dong) >= TOI_DA:
                break
            khoa = (c.video_id, int(c.frame_idx))
            if khoa in da:
                continue
            da.add(khoa)
            dong.append(AnswerQA(c.video_id, int(c.frame_idx), "")
                        if loai == "qa" else AnswerKIS(c.video_id, int(c.frame_idx)))
        goi[g] = dong[:TOI_DA]
        n_vid = len({d.video_id for d in dong})
        print(f"{g:<22}{len(dong):>4} dòng · {n_vid:>3} video · " + " ".join(bao))

    d = nop_bai.ghi_goi(goi, thu_muc=str(a.ra))
    print(f"\n✅ {d}")


if __name__ == "__main__":
    main()
