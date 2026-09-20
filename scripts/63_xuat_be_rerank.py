"""
63_xuat_be_rerank.py — Xuất bể ứng viên để máy có GPU chấm lại bằng VLM.

    python scripts/63_xuat_be_rerank.py                 # -> dev/be_rerank.jsonl
    python scripts/63_xuat_be_rerank.py --dau 50

VÌ SAO TÁCH LÀM HAI MÁY

Chấm VLM cần GPU và cần ẢNH; dựng bể ứng viên cần `clip_gopt.npy` (545 MB) và
cache truy vấn. Nhồi cả hai lên Kaggle là phải đẩy thêm nửa GB index và chạy
lại đúng đoạn mã đã chạy ở đây — thừa, mà lại thêm một chỗ để hai bên lệch
nhau.

Cách này: máy có index xuất bể ra một file **vài trăm KB**, máy có GPU chỉ cần
file đó + ảnh keyframe. Bể được sinh bởi ĐÚNG cấu hình `run.py` sau A52, nên
điểm đo về sau so được thẳng với mốc.

⚠️ KHÔNG XUẤT ĐÁP ÁN. File này đi sang máy khác và có thể lọt vào log Kaggle
công khai; kèm `row_id_dung` là tự tay làm rò tập dev. Máy chấm VLM **không
cần biết** đáp án — nó chỉ chấm độ khớp giữa ảnh và câu hỏi.

ĐỊNH DẠNG (mỗi dòng một câu)

    {"id": ..., "cau_hoi": ..., "ung_vien": [
        {"row_id": .., "video_id": "L21_V001", "kf_name": "001.jpg",
         "hang": 1}, ...]}

`video_id` + `kf_name` là cách định vị ảnh KHÔNG phụ thuộc máy — `kf_path`
trong bảng cái là đường dẫn tuyệt đối của máy dựng index, vô nghĩa ở nơi khác.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5                                              # trọng số kênh 3 (A52)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=GOC / "dev" / "tap_de_that.jsonl", type=Path)
    ap.add_argument("--be", type=int, default=300, help="cỡ bể khi hợp nhất")
    ap.add_argument("--dau", type=int, default=30,
                    help="xuất bao nhiêu ứng viên ĐẦU để VLM chấm")
    ap.add_argument("--ra", default=GOC / "dev" / "be_rerank.jsonl", type=Path)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)

    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    if len(giu) < len(cau):
        print(f"⚠️ loại {len(cau) - len(giu)} câu thiếu chuỗi trong cache")

    ten_kf = master.set_index("row_id")["kf_name"]
    thieu_anh = 0
    with a.ra.open("w", encoding="utf-8", newline="\n") as f:
        for c in giu:
            anh = hop_nhat([k1.tim(m, k=a.be) for m in R.tach_truy_van(c.cau_hoi)])
            ds = hop_nhat([anh, k3.tim(c.cau_hoi, k=a.be)],
                          trong_so=[1.0, W3])[:a.dau]
            uv = []
            for hang, x in enumerate(ds, 1):
                kf = ten_kf.get(x.row_id)
                if pd.isna(kf):
                    # `kf_name` trống = ảnh chưa tải VỀ MÁY NÀY, không phải
                    # "không có keyframe". Máy Kaggle vẫn có ảnh, nên dựng lại
                    # tên theo `kf_n` thay vì bỏ ứng viên.
                    kf = f"{int(master.kf_n.iloc[x.row_id]):03d}.jpg"
                    thieu_anh += 1
                uv.append({"row_id": int(x.row_id), "video_id": x.video_id,
                           "kf_name": str(kf), "hang": hang})
            f.write(json.dumps({"id": c.id, "loai": c.loai,
                                "cau_hoi": c.cau_hoi, "ung_vien": uv},
                               ensure_ascii=False) + "\n")

    kb = a.ra.stat().st_size / 1024
    print(f"✅ {a.ra}  ({len(giu)} câu × {a.dau} ứng viên, {kb:.0f} KB)")
    if thieu_anh:
        print(f"   {thieu_anh} ứng viên không có `kf_name` ở máy này — "
              f"đã dựng tên từ `kf_n`, máy có ảnh sẽ tra được")
    print("   KHÔNG chứa đáp án — an toàn để đưa lên Kaggle")


if __name__ == "__main__":
    main()
