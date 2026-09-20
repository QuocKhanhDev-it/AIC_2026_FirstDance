r"""
40_xuat_goi_vqa.py — Đóng gói câu ĐẾM/MÀU của tập dev để chạy VLM ở máy khác.

    python scripts/40_xuat_goi_vqa.py                # xem trước
    python scripts/40_xuat_goi_vqa.py --ghi          # ghi ra vqa/goi/
    python scripts/40_xuat_goi_vqa.py --ghi --nen    # ghi + nén thành .zip

CÂU HỎI ĐANG ĐO LÀ GÌ
=====================

**Không** phải "VLM có cứu được kênh Q&A không". Là câu hẹp hơn nhiều, và phải
trả lời trước:

> **Cho sẵn ĐÚNG khung đáp án, VLM có đọc ra đúng đáp án không?**

Đây là **trần trên**. Trong bài thật còn phải tìm ra khung đã, mà tìm sai thì
VLM giỏi mấy cũng vô nghĩa. Nếu ngay cả khi đưa đúng khung mà VLM vẫn sai phần
lớn thì hướng VQA chết tại đây, khỏi tốn công dựng đường ống.

A26 và ca `p1-15-qa` đã cho thấy câu đếm là **trần của model**, không phải lỗi
truy hồi. Phép đo này kiểm lại điều đó bằng model khác.

HAI KỶ LUẬT ÉP VÀO ĐÂY
======================

**1. Gói xuất KHÔNG mang đáp án.** `dap_an` giữ lại ở máy này trong
`vqa/dap_an.json`. Máy chạy VLM không nhìn thấy đáp án, nên không có đường nào
để đáp án lọt vào lời nhắc — kể cả do sơ ý. Chấm bằng `41_cham_vqa.py`.

**2. Ảnh là dữ liệu của BTC.** Thư mục `vqa/` bị `.gitignore` chặn. Đẩy lên
Kaggle thì phải **Private Dataset**, không bao giờ để public.

CHỌN CÂU THẾ NÀO
================

Lọc theo mẫu chữ trong câu hỏi (`bao nhiêu`, `mấy`, `màu gì`...) rồi **bỏ câu
nào không có ảnh ở máy này**. Cột `kf_path` nghĩa là "ảnh đã tải Ở MÁY NÀY",
không phải "có keyframe" (A5.5) — máy thiếu nhóm L nào thì câu nhóm đó rụng, và
script in ra rụng những câu nào để khỏi tưởng là đủ.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

MAU_DEM = re.compile(r"bao nhiêu|mấy |số lượng|đếm", re.I)
MAU_MAU = re.compile(r"màu gì|màu nào|màu sắc", re.I)


def phan_loai(cau_hoi: str) -> str | None:
    if MAU_DEM.search(cau_hoi):
        return "DEM"
    if MAU_MAU.search(cau_hoi):
        return "MAU"
    return None


def main():
    ap = argparse.ArgumentParser(description="dong goi cau dem/mau de chay VLM")
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--tap-dev", default=GOC / "dev" / "tap_dev.jsonl", type=Path)
    ap.add_argument("--ra", default=GOC / "vqa", type=Path)
    ap.add_argument("--rong", type=int, default=768, help="cạnh dài tối đa của ảnh")
    ap.add_argument("--toi-da-anh", type=int, default=3,
                    help="số khung tối đa gửi kèm mỗi câu")
    ap.add_argument("--tat-ca-qa", action="store_true",
                    help="lấy MỌI câu QA chứ không chỉ đếm/màu")
    ap.add_argument("--ghi", action="store_true")
    ap.add_argument("--nen", action="store_true", help="nén vqa/goi thành .zip")
    a = ap.parse_args()

    from PIL import Image

    master = pd.read_parquet(a.index / "master.parquet")
    kfp, vid = master.kf_path.values, master.video_id.values

    cau = [json.loads(l) for l in a.tap_dev.read_text("utf-8").splitlines() if l.strip()]
    chon, rung = [], []
    for c in cau:
        if c["loai"] != "QA":
            continue
        loai = phan_loai(c["cau_hoi"])
        if loai is None and not a.tat_ca_qa:
            continue
        rs = [r[0] if isinstance(r, list) else r for r in c["row_id_dung"]]
        anh = [r for r in rs if isinstance(kfp[r], str)][:a.toi_da_anh]
        if not anh:
            rung.append((c["id"], str(vid[rs[0]])))
            continue
        chon.append((c, loai or "khac", anh))

    print(f"{len(chon)} câu có ảnh ở máy này")
    if rung:
        print(f"{len(rung)} câu RỤNG vì máy này chưa có ảnh nhóm L đó:")
        for i, v in rung:
            print(f"   {i:14} {v}")
    print()
    for c, loai, anh in chon:
        print(f"  {c['id']:14} {loai:4} {len(anh)} ảnh  {c['cau_hoi'][:62]}")

    if not a.ghi:
        print("\n(xem trước — thêm `--ghi` để ghi thật)")
        return

    goi = a.ra / "goi"
    if goi.exists():
        shutil.rmtree(goi)
    (goi / "anh").mkdir(parents=True)

    hoi, dap = [], {}
    for c, loai, anh in chon:
        ten_anh = []
        for r in anh:
            t = f"{c['id']}_{r}.jpg"
            im = Image.open(kfp[r]).convert("RGB")
            im.thumbnail((a.rong, a.rong))
            im.save(goi / "anh" / t, quality=82, optimize=True)
            ten_anh.append(t)
        # CO Y khong kem `dap_an` — xem phan ky luat o dau file
        hoi.append({"id": c["id"], "loai_hoi": loai,
                    "cau_hoi": c["cau_hoi"], "anh": ten_anh})
        dap[c["id"]] = c.get("dap_an", "")

    (goi / "cau_hoi.json").write_text(
        json.dumps(hoi, ensure_ascii=False, indent=1), "utf-8")
    (a.ra / "dap_an.json").write_text(
        json.dumps(dap, ensure_ascii=False, indent=1), "utf-8")

    mb = sum(p.stat().st_size for p in goi.rglob("*")) / 1024 ** 2
    print(f"\n✅ {goi}  ({mb:.2f} MB, {len(hoi)} câu)")
    print(f"   đáp án giữ RIÊNG ở {a.ra / 'dap_an.json'} — không nằm trong gói")
    if a.nen:
        p = shutil.make_archive(str(a.ra / "goi_vqa"), "zip", goi)
        print(f"   nén: {p}  ({Path(p).stat().st_size / 1024 ** 2:.2f} MB)")
    print("\n   Chạy VLM xong, đem file trả lời về rồi:")
    print("   python scripts/41_cham_vqa.py vqa/tra_loi.json")


if __name__ == "__main__":
    main()
