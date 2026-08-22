"""
34_de_thanh_tap_dev.py — Biến bộ đề ĐÃ THI + đáp án nhóm soát tay thành câu dev.

VÌ SAO ĐÂY LÀ THỨ TẬP DEV THIẾU SUỐT TỪ ĐẦU
============================================

Tập dev tự soạn đã tỏ ra **mù năm lần** (A19, A20, A31, A34, A37), và mỗi lần
truy nguyên đều ra cùng một nguyên nhân: **câu dev không giống đề thật**.

    câu dev tự soạn   ~15-22 từ, 1,1 mệnh đề, người viết BIẾT TRƯỚC đáp án
    đề thi thật        63 từ,    2,4 mệnh đề, viết bởi người KHÔNG biết hệ thống ta

Bộ đề sơ tuyển đợt 1 phá được cả hai chỗ lệch cùng lúc: câu hỏi do BTC viết, dài
đúng kiểu đề thật, và **nay đã có đáp án** do nhóm mở ảnh soát tay.

    python scripts/34_de_thanh_tap_dev.py --kiem      # xem trước, không ghi
    python scripts/34_de_thanh_tap_dev.py --ghi

⚠️ BA ĐIỀU PHẢI GIỮ, VÌ ĐÁP ÁN Ở ĐÂY LÀ NIỀM TIN CHỨ KHÔNG PHẢI SỰ THẬT
=======================================================================

**1. Câu chưa có đáp án thì BỎ, không điền bừa.** `query-p1-3-qa` nhóm chưa tìm
ra; đưa vào với đáp án `"không rõ"` là gieo một sự thật GIẢ vào thước đo. A21 đã
dạy đúng bài này: 5 câu dev soạn từ OCR của chính khung đáp án cho ra mức tăng
0,400 -> 0,840 hoàn toàn ảo.

**2. Ghi rõ ĐỘ CHẮC vào `ghi_chu`.** Nhóm tự đánh dấu `xong` / `chờ double check`
/ `ko chắc`. Bài nộp được 11/13 điểm nên phần lớn đúng, nhưng **không biết cái
nào sai** — và một đáp án sai trong tập dev thì mọi phép đo sau đó lệch mà không
có gì báo.

**3. `p1-8-kis` và `p1-14-kis` là HAI GÓI ĐỀ TRÙNG NHAU NGUYÊN VĂN** (BTC ra
trùng). Chỉ lấy MỘT — giữ cả hai là tự nhân đôi trọng số của một câu trong mọi
phép đo về sau.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

DE = GOC / "dev" / "SOTUYEN1-bo-de-thi"
RA = GOC / "dev" / "tap_dev_thanh_vien" / "tap_dev_DE1.jsonl"

# Độ chắc do chính nhóm đánh dấu khi soát. Không có tên ở đây = "xong".
DO_CHAC = {
    "query-p1-8-kis":  "chờ double check",
    "query-p1-13-kis": "cần check",
    "query-p1-15-qa":  "cần check",
    "query-p1-21-kis": "chưa chắc",
    "query-p1-22-kis": "cần check",
    "query-p1-18-kis": "lụm được, chưa soát kỹ",
    "query-p1-19-kis": "lụm được, chưa soát kỹ",
}

# Gói bỏ hẳn, kèm lý do — ghi ra để người sau không tưởng là sót.
BO = {
    "query-p1-3-qa": "nhóm CHƯA tìm ra đáp án (câu đọc số trên mặt cân)",
    "query-p1-14-kis": "trùng nguyên văn query-p1-8-kis — chỉ giữ một",
}


def nap_dap_an() -> dict:
    """Lấy bảng `DAP_AN` từ `32_dung_tu_dap_an_tay.py` — một nguồn sự thật."""
    s = importlib.util.spec_from_file_location(
        "m32", GOC / "scripts" / "32_dung_tu_dap_an_tay.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m.DAP_AN


def row_id_cua(master, video_id: str, kf: int):
    r = master[(master.video_id == video_id) & (master.kf_n == int(kf))]
    return None if r.empty else int(r.row_id.iloc[0])


def main():
    ap = argparse.ArgumentParser(description="de thi -> cau dev")
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--ra", default=RA, type=Path)
    ap.add_argument("--ghi", action="store_true", help="ghi thật, mặc định xem trước")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    dap_an = nap_dap_an()
    cau, bo_qua = [], []

    for ten, muc in sorted(dap_an.items()):
        if ten in BO:
            bo_qua.append((ten, BO[ten]))
            continue
        vid, kfs = muc[0], list(muc[1])
        dap = muc[2] if len(muc) > 2 else ""
        loai = ten.rsplit("-", 1)[-1]

        f = DE / f"{ten}.txt"
        if not f.exists():
            bo_qua.append((ten, "không có file đề"))
            continue
        cau_hoi = f.read_text("utf-8").strip()

        rid = [row_id_cua(master, vid, k) for k in kfs]
        rid = [r for r in rid if r is not None]
        if not rid:
            bo_qua.append((ten, "không tra được row_id"))
            continue

        so = ten.split("-")[2]
        cid = f"{loai}-DE1-{int(so):02d}"
        chac = DO_CHAC.get(ten, "xong")

        # TRAKE: mỗi sự kiện một danh sách row_id, đúng thứ tự thời gian.
        rd = [[r] for r in rid] if loai == "trake" else rid

        cau.append({
            "id": cid,
            "loai": loai.upper(),
            "cau_hoi": cau_hoi,
            "row_id_dung": rd,
            "dap_an": dap,
            "nguon": f"đề sơ tuyển đợt 1, {vid} kf {kfs}",
            "ghi_chu": (f"CÂU ĐỀ THẬT do BTC viết ({len(cau_hoi.split())} từ) — "
                        f"đúng phân bố đang đi thi, khác hẳn câu dev tự soạn. "
                        f"Đáp án do nhóm soát tay, độ chắc: {chac}. "
                        f"Bài nộp chứa đáp án này được 11/13 điểm, nhưng KHÔNG "
                        f"biết câu nào trong số đó sai."),
        })

    n_tu = sum(len(c["cau_hoi"].split()) for c in cau) / max(len(cau), 1)
    print(f"{len(cau)} câu dev mới | trung bình {n_tu:.0f} từ/câu")
    import collections
    print("  ", dict(collections.Counter(c["loai"] for c in cau)))
    print("  (tập dev tự soạn: ~15-22 từ/câu — xem A19/A20)\n")
    for ten, ly_do in bo_qua:
        print(f"  bỏ {ten}: {ly_do}")

    if not a.ghi:
        print(f"\n(xem trước — thêm `--ghi` để ghi ra {a.ra})")
        return

    a.ra.parent.mkdir(parents=True, exist_ok=True)
    with open(a.ra, "w", encoding="utf-8", newline="\n") as f:
        for c in cau:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\n✅ Đã ghi {a.ra}")
    print("   Gộp vào tập dev: python src/tap_dev.py --gop "
          "dev/tap_dev_thanh_vien/*.jsonl")
    print("   ⚠️ `--gop` DỰNG LẠI từ danh sách truyền vào — phải truyền TẤT CẢ "
          "file thành viên, không chỉ file mới.")


if __name__ == "__main__":
    main()
