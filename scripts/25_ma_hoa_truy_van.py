"""
25_ma_hoa_truy_van.py — Mã hoá sẵn vector truy vấn, để máy yếu chạy được kênh 1.

Máy 7,7 GB không nạp nổi `ViT-SO400M-14-SigLIP2-378` (~3,5 GB), nên mọi phép đo
dính kênh 1 đều tắc — kể cả hai thứ đáng giá nhất còn lại: đo TRAKE (Mũi nhọn 2)
và RRF(SigLIP2, OCR). Nhưng model chỉ làm đúng MỘT việc: biến câu chữ thành
vector. Ma trận ảnh 177.321 × 1152 thì đã nằm sẵn trên đĩa mọi máy.

Mà tập truy vấn là **hữu hạn và biết trước**: 24 câu đề + 115 câu tập dev. Mã
hoá chúng một lần ra file vài trăm KB, rồi máy yếu dùng `dense.KenhAnhCache` làm
toàn bộ truy hồi bằng numpy thuần — **không nạp model, không cần 6,5 GB RAM**.

    # chạy MỘT LẦN trên máy >= 16 GB (hoặc Colab/Kaggle)
    python scripts/25_ma_hoa_truy_van.py --de dev/THUNGHIEM-bo-de-thi --tap-dev

    # từ đó về sau, trên máy nào cũng được
    python scripts/22_do_trake.py --cache index/truy_van.npz

PHẢI MÃ HOÁ ĐÚNG NHỮNG CHUỖI SẼ ĐƯỢC TRA
=========================================

⚠️ Đây là chỗ dễ hỏng nhất, và nó hỏng ÂM THẦM. `run.py` không đưa nguyên câu
truy vấn vào encoder — nó gọi `tach_truy_van()` cắt câu dài thành nhiều mệnh đề
rồi encode TỪNG MỆNH ĐỀ (A19/A20: 100% truy vấn đề mẫu vượt trần token). Câu
TRAKE còn qua `tach_su_kien()` trước. Nên cache phải chứa **mọi biến thể sẽ
thật sự được tra**, không phải chỉ câu gốc.

Script này tự sinh đủ bộ đó bằng chính các hàm của `run.py` — không chép lại
logic cắt câu, vì chép là mở đường cho hai bản lệch nhau.

TIẾT KIỆM RAM KHI MÃ HOÁ
========================

`--fp16` nạp trọng số ở nửa độ chính xác rồi tính ở fp32 khi encode, và **giải
phóng tháp ảnh ngay sau khi nạp** — phần đó chiếm quá nửa model mà mã hoá văn
bản không dùng tới. Đủ để một máy ~8 GB làm được việc này khi đã đóng bớt ứng
dụng, dù vẫn không đủ để chạy `KenhAnh` bình thường.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from dense import MODEL_MAC_DINH, PRETRAINED_MAC_DINH  # noqa: E402

# Cờ ép chạy bất chấp RAM, đặt ở main(). Dùng list để `ma_hoa()` đọc được.
BO_QUA_RAM = [False]


def thu_thap(de_dir: Path | None, lay_tap_dev: bool, them: list) -> list[str]:
    """Mọi chuỗi sẽ thật sự được đưa vào `encode_text`, đã bỏ trùng."""
    ra = []

    def nap(cau: str):
        ra.append(cau)
        ra.extend(R.tach_truy_van(cau))     # đúng thứ run.py đưa vào encoder

    if de_dir:
        for ten, nd in R.doc_de(de_dir).items():
            if R.loai_cua(ten) == "trake":
                for sk in R.tach_su_kien(nd):
                    nap(sk)
            else:
                nap(nd)

    if lay_tap_dev:
        for c in tap_dev.doc():
            nap(c.cau_hoi)
            # câu TRAKE của tập dev cũng bị tách sự kiện khi đo
            if c.loai == "TRAKE":
                for sk in R.tach_su_kien(c.cau_hoi):
                    nap(sk)

    for x in them:
        nap(x)

    thay, sach = set(), []
    for c in ra:
        c = c.strip()
        if c and c not in thay:
            thay.add(c)
            sach.append(c)
    return sach


def ma_hoa(cac_cau: list, matrix: str, index: Path, fp16: bool) -> tuple:
    """Trả `(ma trận vector đã chuẩn hoá L2, ghi chú)`."""
    canh = index / (Path(matrix).stem + ".json")
    gc = json.loads(canh.read_text("utf-8")) if canh.exists() else {}
    model_tag = gc.get("model", MODEL_MAC_DINH)
    pretrained = gc.get("pretrained", PRETRAINED_MAC_DINH)

    import re

    import open_clip
    import torch

    # Cùng quy tắc ép kích cỡ ảnh như `dense.KenhAnh` — giữ khớp để cache và
    # kênh thật dùng chung một cấu hình model.
    ep = None
    m = re.search(r"SigLIP2-(\d+)$", model_tag)
    if m and not str(pretrained).lower().startswith(
            ("webli", "openai", "laion", "datacomp", "dfn")):
        ep = int(m.group(1))

    # ⚠️ CHỐT CHỐNG TREO MÁY. Script này gọi thẳng `open_clip`, KHÔNG đi qua
    # `dense.kiem_ram`, nên phải tự canh — nếu không thì đúng cái nó sinh ra để
    # né (nạp model trên máy 7,7 GB) lại xảy ra ngay trong chính nó. Đã làm
    # đứng máy hai lần, phải khởi động lại.
    #
    # Ngưỡng thấp hơn `dense.RAM_CAN` vì ở đây chỉ mã hoá VĂN BẢN: nạp fp16 rồi
    # bỏ tháp ảnh. Vẫn là ước lượng, nên `--bo-qua-ram` để ép.
    # Ngưỡng theo CỠ MODEL, không phải một hằng số: `dense.RAM_CAN` tra theo số
    # chiều (512 = ViT-B/32 nhẹ, 1152 = SO400M nặng). Mã hoá văn bản bỏ được
    # tháp ảnh nên chỉ cần một phần: ~45% nếu fp16, ~85% nếu fp32.
    from dense import RAM_CAN, RAM_CAN_MAC_DINH, ram_trong_gb
    chieu = int(np.load(index / matrix, mmap_mode="r").shape[1])
    day_du = RAM_CAN.get(chieu, RAM_CAN_MAC_DINH)
    can = round(day_du * (0.45 if fp16 else 0.85), 1)
    tro = ram_trong_gb()
    if tro is not None and tro < can and not BO_QUA_RAM[0]:
        raise SystemExit(
            f"\n❌ KHÔNG ĐỦ RAM để mã hoá — dừng trước khi nạp model.\n\n"
            f"   Model      : {model_tag}\n"
            f"   Cần trống  : ~{can:.1f} GB{' (đã tính --fp16)' if fp16 else ''}\n"
            f"   Đang trống : {tro:.1f} GB\n\n"
            f"   Cách đi tiếp:\n"
            f"     • Đóng bớt ứng dụng (trình duyệt là thủ phạm thường gặp)\n"
            f"     • Thêm --fp16 nếu chưa dùng (hạ ngưỡng còn ~3 GB)\n"
            f"     • Chạy trên Colab/Kaggle rồi chép file .npz về — file chỉ\n"
            f"       vài trăm KB, chép qua chat cũng được\n"
            f"     • Ép chạy bất chấp: --bo-qua-ram (RỦI RO TREO MÁY)\n")

    print(f"Nạp {model_tag} / {pretrained}"
          f"{' (fp16)' if fp16 else ''} — chỉ để mã hoá văn bản...")
    model, _, _ = open_clip.create_model_and_transforms(
        model_tag, pretrained=pretrained, force_image_size=ep,
        precision="fp16" if fp16 else "fp32")
    model.eval()

    # Tháp ảnh chiếm quá nửa trọng số mà mã hoá văn bản không dùng tới.
    if hasattr(model, "visual"):
        del model.visual
        import gc as _gc
        _gc.collect()
    if fp16:
        model.float()          # tính ở fp32 trên CPU; RAM đỉnh vẫn giảm nhờ nạp fp16

    tok = open_clip.get_tokenizer(model_tag)
    ra = []
    with torch.no_grad():
        for i, c in enumerate(cac_cau, 1):
            v = model.encode_text(tok([c]))[0].numpy().astype(np.float32)
            ra.append(v / (np.linalg.norm(v) + 1e-9))
            if i % 25 == 0 or i == len(cac_cau):
                print(f"  {i}/{len(cac_cau)}")
    return np.vstack(ra), {"model": model_tag, "pretrained": pretrained,
                           "matrix": matrix, "chieu": int(ra[0].shape[0])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--de", type=Path, help="thư mục chứa query-*.txt")
    ap.add_argument("--tap-dev", action="store_true", help="thêm cả tập dev")
    ap.add_argument("--them", action="append", default=[], metavar="CAU")
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--matrix", default="clip_siglip2.npy")
    ap.add_argument("--ra", default=None, type=Path,
                    help="mặc định index/truy_van.npz")
    ap.add_argument("--fp16", action="store_true",
                    help="nạp trọng số fp16 để giảm RAM đỉnh")
    ap.add_argument("--bo-qua-ram", action="store_true",
                    help="ép nạp model dù RAM thấp — RỦI RO TREO MÁY")
    ap.add_argument("--gop", action="store_true",
                    help="gộp thêm vào cache đã có thay vì ghi đè")
    a = ap.parse_args()

    BO_QUA_RAM[0] = a.bo_qua_ram
    if not (a.de or a.tap_dev or a.them):
        raise SystemExit("Chưa chọn nguồn nào: --de, --tap-dev hoặc --them")

    ra_file = a.ra or (a.index / "truy_van.npz")
    cac_cau = thu_thap(a.de, a.tap_dev, a.them)

    cu = {}
    if a.gop and ra_file.exists():
        z = np.load(ra_file, allow_pickle=False)
        cu = {str(c): np.asarray(z["vec"][i], dtype=np.float32)
              for i, c in enumerate(z["cau"])}
        print(f"Cache cũ: {len(cu)} câu")

    can = [c for c in cac_cau if c not in cu]
    print(f"{len(cac_cau)} chuỗi cần có | {len(can)} chuỗi phải mã hoá mới\n")
    if not can:
        print("Cache đã đủ — không phải nạp model.")
        return

    vec, ghi_chu = ma_hoa(can, a.matrix, a.index, a.fp16)
    for c, v in zip(can, vec):
        cu[c] = v

    cau = list(cu)
    ra_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ra_file, cau=np.array(cau, dtype=object).astype(str),
                        vec=np.vstack([cu[c] for c in cau]).astype(np.float32),
                        ghi_chu=json.dumps(ghi_chu, ensure_ascii=False))
    mb = ra_file.stat().st_size / 1024 ** 2
    print(f"\n✅ {len(cau)} câu -> {ra_file}  ({mb:.2f} MB)")
    print(f"   {ghi_chu['model']} / {ghi_chu['chieu']} chiều\n")
    print("   Máy yếu dùng: dense.KenhAnhCache('./index', "
          f"'{ra_file.as_posix()}')")


if __name__ == "__main__":
    main()
