"""
nop_bai.py — Sinh và soát file nộp cho BTC, đúng quy định vòng Sơ tuyển.

Nguồn: "Hướng dẫn nộp bài sơ tuyển", https://sotuyenaic.oj.io.vn (19/08/2026).

VÌ SAO MODULE NÀY PHẢI KHẮT KHE ĐẾN MỨC TỪ CHỐI GHI
====================================================

**Mỗi gói truy vấn chỉ được nộp 3 lần, và sai định dạng VẪN TÍNH LÀ MỘT LẦN.**
Nên một dấu phẩy lạc chỗ không phải là "sửa rồi nộp lại" — nó là 1/3 số lần
nộp của cả gói. Vì vậy `soat()` chạy TRƯỚC khi ghi, và bất kỳ lỗi nào cũng
làm dừng hẳn, không ghi file nào. Cùng nguyên tắc với `12_va_duong_dan.py`.

BA ĐỊNH DẠNG

    KIS     <tên video>,<frame_idx>
    QA      <tên video>,<frame_idx>,<answer>
    TRAKE   <tên video>,<frame_1>,<frame_2>,...,<frame_N>

BỐN CÁI BẪY ĐÃ DỰNG SẴN CHỐT

1. **BOM.** Ghi bằng `utf-8-sig` là chèn ba byte `EF BB BF` vào đầu file, và
   tên video dòng đầu thành `﻿L01_V028` — hỏng đúng một dòng, âm thầm.
   Repo này có chỗ đang dùng `utf-8-sig` (`05_bench_vlm.py`), nên đây không
   phải rủi ro lý thuyết. Ở đây **luôn `utf-8` trần**.

2. **Tên video kèm `.mp4`.** BTC ghi rõ `L01_V028` ✅ / `L01_V028.mp4` ❌.
   `master.video_id` vốn không có đuôi, nhưng ai đó ghép từ `video_path` thì
   dính ngay.

3. **`answer` quá 100 ký tự.** Cắt bớt là ĐỔI câu trả lời — thà báo lỗi để
   người sửa còn hơn lặng lẽ nộp một đáp án khác.

4. **Dòng trùng nhau.** Chỉ có 100 chỗ. Hai dòng y hệt là phí một chỗ mà không
   tăng cơ hội nào — `R@k = max R-Score trong top-k`.

Dùng:
    from nop_bai import ghi_goi, dong_goi
    ghi_goi({"query-1-kis": [AnswerKIS(...), ...]}, "submission")
    dong_goi("submission", "team_xyz_round1.zip")

Soát lại một thư mục đã có trước khi nộp:
    python src/nop_bai.py --soat submission
"""

import argparse
import csv
import io
import re
import zipfile
from pathlib import Path

try:
    from .schema import AnswerKIS, AnswerQA, AnswerTRAKE
except ImportError:                     # chạy trực tiếp: python src/nop_bai.py
    from schema import AnswerKIS, AnswerQA, AnswerTRAKE

TOI_DA_DONG = 100          # "Đội thi có thể nộp file tối đa 100 dòng"
TOI_DA_ANSWER = 100        # "Answer (Q&A) có độ dài tối đa 100 ký tự"
THU_MUC = "submission"     # "PHẢI có thư mục `submission` bên trong file zip"

# Tên file quyết định BTC chấm bằng bộ luật nào — hậu tố sai là chấm sai loại.
HAU_TO = {"kis": AnswerKIS, "qa": AnswerQA, "trake": AnswerTRAKE}
TEN_FILE = re.compile(r"^query-\d+-(kis|qa|trake)$")
TEN_VIDEO = re.compile(r"^[A-Za-z0-9_]+$")


def _loai(ten_goi: str) -> str:
    """Hậu tố của tên gói: `query-3-qa` -> `qa`."""
    m = TEN_FILE.match(ten_goi)
    if not m:
        raise ValueError(
            f"Tên gói {ten_goi!r} không đúng quy ước `query-<số>-<kis|qa|trake>`. "
            f"BTC chấm theo hậu tố này, đặt sai là chấm nhầm loại truy vấn.")
    return m.group(1)


def _dong(dap_an) -> list:
    """Một đáp án -> một dòng CSV, dạng list các ô."""
    if isinstance(dap_an, AnswerKIS):
        return [dap_an.video_id, int(dap_an.frame_idx)]
    if isinstance(dap_an, AnswerQA):
        return [dap_an.video_id, int(dap_an.frame_idx), dap_an.answer]
    if isinstance(dap_an, AnswerTRAKE):
        return [dap_an.video_id, *(int(x) for x in dap_an.frame_idxs)]
    raise TypeError(f"Không phải kiểu đáp án: {type(dap_an).__name__}")


def soat(ten_goi: str, cac_dap_an: list, so_su_kien: int | None = None) -> list[str]:
    """Trả danh sách lỗi. Rỗng nghĩa là nộp được.

    `so_su_kien` chỉ dùng cho TRAKE — *"Số lượng Frame ID phải khớp CHÍNH XÁC
    với số events yêu cầu"*. Không truyền thì chỉ kiểm các dòng dài bằng nhau,
    còn đúng hay không thì không biết được.
    """
    loi = []
    try:
        loai = _loai(ten_goi)
    except ValueError as e:
        return [str(e)]

    mong = HAU_TO[loai]
    if not cac_dap_an:
        return [f"{ten_goi}: không có dòng nào"]
    if len(cac_dap_an) > TOI_DA_DONG:
        loi.append(f"{ten_goi}: {len(cac_dap_an)} dòng, tối đa {TOI_DA_DONG}")

    thay = set()
    dai_trake = set()
    for i, d in enumerate(cac_dap_an, 1):
        if not isinstance(d, mong):
            loi.append(f"{ten_goi} dòng {i}: là {type(d).__name__} "
                       f"nhưng tên gói đòi {mong.__name__}")
            continue

        v = str(d.video_id)
        if v.lower().endswith(".mp4"):
            loi.append(f"{ten_goi} dòng {i}: tên video còn đuôi .mp4 ({v!r})")
        elif not TEN_VIDEO.match(v):
            loi.append(f"{ten_goi} dòng {i}: tên video lạ ({v!r})")

        khung = ([d.frame_idx] if not isinstance(d, AnswerTRAKE)
                 else list(d.frame_idxs))
        for f in khung:
            if isinstance(f, bool) or not isinstance(f, int) or f < 0:
                loi.append(f"{ten_goi} dòng {i}: frame_idx phải là số nguyên "
                           f"không âm, nhận {f!r}")

        if isinstance(d, AnswerQA):
            a = d.answer
            if not isinstance(a, str) or not a.strip():
                loi.append(f"{ten_goi} dòng {i}: `answer` rỗng -> chắc chắn 0 điểm")
            elif len(a) > TOI_DA_ANSWER:
                # KHÔNG tự cắt: cắt là đổi câu trả lời thành một câu khác.
                loi.append(f"{ten_goi} dòng {i}: `answer` {len(a)} ký tự, "
                           f"tối đa {TOI_DA_ANSWER}")

        if isinstance(d, AnswerTRAKE):
            dai_trake.add(len(d.frame_idxs))
            if so_su_kien and len(d.frame_idxs) != so_su_kien:
                loi.append(f"{ten_goi} dòng {i}: {len(d.frame_idxs)} frame "
                           f"nhưng truy vấn đòi {so_su_kien} sự kiện")
            if list(d.frame_idxs) != sorted(d.frame_idxs):
                loi.append(f"{ten_goi} dòng {i}: frame TRAKE không tăng dần — "
                           f"*'thứ tự phải tuân theo thứ tự thời gian'*")

        khoa = tuple(_dong(d))
        if khoa in thay:
            loi.append(f"{ten_goi} dòng {i}: trùng hệt một dòng trước — "
                       f"phí một trong {TOI_DA_DONG} chỗ")
        thay.add(khoa)

    if len(dai_trake) > 1:
        loi.append(f"{ten_goi}: các dòng TRAKE dài khác nhau {sorted(dai_trake)} "
                   f"— mọi dòng phải cùng số sự kiện")
    return loi


def tu_ung_vien(ung_vien: list, loai: str, dap_an: str | None = None,
                so_su_kien: int | None = None, gioi_han: int = TOI_DA_DONG) -> list:
    """`list[Candidate]` từ đường ống -> danh sách đáp án nộp được.

    ⚠️ Lấy thẳng `c.frame_idx`, **không tính lại từ `pts_time`** — làm tròn
    lệch 1 frame (xem `schema.py`). `Candidate` đã mang sẵn giá trị đúng từ
    bảng cái; việc duy nhất ở đây là đổi vỏ.

    `dap_an` dùng cho Q&A khi mọi dòng cùng một câu trả lời — đúng cách PHẦN C
    mục 4 khuyên: giữ nhiều `frame_idx` khác nhau nhưng CÙNG `answer` nếu VLM
    tự tin. Muốn mỗi dòng một `answer` riêng thì đặt vào `c.meta["answer"]`.

    TRAKE **không** dựng được từ danh sách phẳng: mỗi dòng là một CHUỖI N
    khung, phải do bước dóng hàng thời gian sinh ra. Gọi vào đây là báo lỗi
    chứ không đoán bừa.
    """
    if loai == "trake":
        raise ValueError(
            "TRAKE phải dựng từ các chuỗi đã dóng hàng, không từ danh sách "
            "ứng viên phẳng. Tự tạo list[AnswerTRAKE] rồi gọi ghi_goi().")

    ra = []
    for c in ung_vien[:gioi_han]:
        if loai == "kis":
            ra.append(AnswerKIS(c.video_id, int(c.frame_idx)))
        elif loai == "qa":
            tra = c.meta.get("answer") if getattr(c, "meta", None) else None
            ra.append(AnswerQA(c.video_id, int(c.frame_idx),
                               str(tra if tra is not None else (dap_an or ""))))
        else:
            raise ValueError(f"loai không hợp lệ: {loai!r}")
    return ra


def canh_bao(ten_goi: str, cac_dap_an: list) -> list[str]:
    """Những thứ KHÔNG sai luật nhưng gần như chắc chắn là mất điểm."""
    ra = []
    if 0 < len(cac_dap_an) < TOI_DA_DONG:
        ra.append(f"{ten_goi}: mới {len(cac_dap_an)}/{TOI_DA_DONG} dòng. "
                  f"Không có điểm phạt cho dòng sai — dòng thứ 100 vẫn đáng "
                  f"0,2 điểm nếu trúng. Điền cho đủ.")
    return ra


def _viet_csv(cac_dap_an: list) -> str:
    """Sinh nội dung CSV dưới dạng chuỗi.

    `QUOTE_MINIMAL` đúng luật BTC: chỉ bọc ngoặc kép khi `answer` có dấu phẩy,
    ngoặc kép hoặc xuống dòng, và escape `"` thành `""`. Answer đơn giản để
    trần — BTC chấp nhận cả hai.
    """
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
    for d in cac_dap_an:
        w.writerow(_dong(d))
    return buf.getvalue()


def ghi_goi(goi: dict, thu_muc="submission", so_su_kien: dict | None = None) -> Path:
    """Ghi cả gói ra `<thu_muc>/query-*.csv`. **Có lỗi thì KHÔNG ghi gì cả.**

    `goi` là `{"query-1-kis": [AnswerKIS, ...], ...}`.
    `so_su_kien` là `{"query-4-trake": 4}` — số sự kiện truy vấn đòi.
    """
    so_su_kien = so_su_kien or {}
    loi = []
    for ten, ds in goi.items():
        loi += soat(ten, ds, so_su_kien.get(ten))
    if loi:
        raise SystemExit(
            "❌ KHÔNG ghi file nào. Sai định dạng vẫn tính là một lần nộp "
            "(chỉ có 3 lần).\n\n" + "\n".join("   " + x for x in loi))

    d = Path(thu_muc)
    d.mkdir(parents=True, exist_ok=True)
    for ten, ds in goi.items():
        # utf-8 TRẦN, không phải utf-8-sig: BOM làm hỏng tên video dòng đầu.
        # newline="" để csv tự quản lý xuống dòng, không bị nhân đôi \r.
        (d / f"{ten}.csv").write_text(_viet_csv(ds), encoding="utf-8", newline="")
    return d


def dong_goi(thu_muc="submission", zip_ra="submission.zip") -> Path:
    """Nén thành `.zip` với thư mục `submission/` NẰM BÊN TRONG.

    *"KHÔNG nén trực tiếp các file CSV — phải nén thư mục `submission`"*. Đây
    là lỗi BTC xếp thứ hai trong danh sách năm lỗi thường gặp nhất.
    """
    d = Path(thu_muc)
    ds = sorted(d.glob("*.csv"))
    if not ds:
        raise SystemExit(f"{d} không có file .csv nào")
    z = Path(zip_ra)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        for x in ds:
            f.write(x, arcname=f"{THU_MUC}/{x.name}")   # tiền tố bắt buộc
    return z


# ---- đọc ngược lại để soát một thư mục đã có ------------------------------

def doc_csv(f: Path) -> list:
    """Đọc lại file CSV đã ghi thành danh sách đáp án, theo hậu tố tên file."""
    loai = _loai(f.stem)
    tho = f.read_bytes()
    if tho.startswith(b"\xef\xbb\xbf"):
        raise SystemExit(f"{f.name}: file có BOM (utf-8-sig). Ghi lại bằng "
                         f"`utf-8` trần — BOM dính vào tên video dòng đầu.")
    ra = []
    for hang in csv.reader(io.StringIO(tho.decode("utf-8"))):
        if not hang:
            continue
        if loai == "kis":
            ra.append(AnswerKIS(hang[0], int(hang[1])))
        elif loai == "qa":
            ra.append(AnswerQA(hang[0], int(hang[1]), hang[2] if len(hang) > 2 else ""))
        else:
            ra.append(AnswerTRAKE(hang[0], [int(x) for x in hang[1:]]))
    return ra


def main():
    ap = argparse.ArgumentParser(
        description="Soát thư mục nộp bài trước khi nén. Chỉ có 3 lần nộp.")
    ap.add_argument("--soat", default=THU_MUC, type=Path)
    ap.add_argument("--nen", metavar="FILE.zip",
                    help="soát xong thì nén luôn ra file này")
    a = ap.parse_args()

    d = a.soat
    if not d.is_dir():
        raise SystemExit(f"Không có thư mục {d}")
    ds = sorted(d.glob("*.csv"))
    if not ds:
        raise SystemExit(f"{d} không có file .csv nào")

    loi, cb = [], []
    print(f"{d}  —  {len(ds)} file\n")
    for f in ds:
        try:
            dap_an = doc_csv(f)
        except Exception as e:
            loi.append(f"{f.name}: đọc không được — {e}")
            continue
        l = soat(f.stem, dap_an)
        loi += l
        cb += canh_bao(f.stem, dap_an)
        print(f"  {'❌' if l else '✅'} {f.name:<26} {len(dap_an):>3} dòng")

    if cb:
        print("\n⚠️  CẢNH BÁO (không sai luật, nhưng đang mất điểm):")
        for x in cb:
            print("   ", x)
    if loi:
        print(f"\n❌ {len(loi)} LỖI — ĐỪNG NỘP:")
        for x in loi:
            print("   ", x)
        raise SystemExit(1)

    print("\n✅ Định dạng hợp lệ.")
    if a.nen:
        print(f"   Đã nén -> {dong_goi(d, a.nen)}")
    else:
        print(f"   Nén bằng: python src/nop_bai.py --soat {d} --nen bai_nop.zip")


if __name__ == "__main__":
    main()
