"""
run.py — Đường ống đầu-cuối: thư mục đề của BTC -> thư mục `submission/`.

    python src/run.py --de de_thi --ra submission
    python src/run.py --de de_thi --ra submission --nen bai_nop.zip

Đọc mọi `query-*.txt` trong `--de`, dispatch theo hậu tố tên file (`kis` / `qa`
/ `trake` — BTC chấm theo đúng hậu tố này), chạy truy hồi, rồi ghi qua
`nop_bai.ghi_goi` (tự soát định dạng, có lỗi thì KHÔNG ghi file nào).

THIẾT KẾ CHẠY ĐƯỢC TRÊN MÁY YẾU
================================

`ViT-SO400M-14-SigLIP2-378` chiếm ~3,5 GB. Máy 7,7 GB đã crash nhiều lần khi
giữ hai model cùng lúc. Nên ở đây: **nạp MỘT model, chạy HẾT mọi truy vấn, giải
phóng**, chỉ giữ lại `list[Candidate]` — thứ nhẹ hều. Kênh văn bản và objects
nạp sau, chúng không đáng kể.

KÊNH NÀO THIẾU THÌ BỎ QUA, KHÔNG CHẾT
Máy khác nhau giữ dữ liệu khác nhau (B4), và `ocr_asr.parquet` của kênh 3 có
thể chưa ai chạy. Thiếu file thì in một dòng rồi đi tiếp — **không bao giờ để
một kênh phụ làm hỏng cả bài nộp**.

HỢP NHẤT MẶC ĐỊNH TẮT — ĐÂY LÀ QUYẾT ĐỊNH CÓ SỐ ĐO
Đã đo ba lần (A14, A14.1, A17): cộng một kênh yếu vào kênh mạnh làm **TỆ ĐI**.
RRF cộng `1/(k+hạng)` từ mỗi kênh mà không nhìn kênh đó tốt hay tệ, nên ứng viên
hạng 1 của một kênh chết được cộng đúng bằng ứng viên hạng 1 của kênh tốt.
SigLIP2 đang 0,3258 còn objects 0,0412 — chênh 8 lần. Muốn thử thì `--hop-nhat`,
và **đo trên tập dev trước khi dùng thật**.
"""

import argparse
import gc
import re
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

from nop_bai import (TOI_DA_DONG, dong_goi, ghi_goi,          # noqa: E402
                     tu_ung_vien)
from schema import AnswerTRAKE                                # noqa: E402

TEN_DE = re.compile(r"^(query-\d+-(kis|qa|trake))$")

# Duoi nguong nay coi la N khung "don cuc" -> rai deu. ~100 frame la 3-4 giay
# o moi muc fps trong kho (25 / 26,44 / 29,97 / 30).
DON_NHAU = 100


def doc_de(thu_muc: Path) -> dict:
    """`{tên gói: nội dung truy vấn}` từ các file `query-*.txt`."""
    d = Path(thu_muc)
    if not d.is_dir():
        raise SystemExit(f"Không có thư mục đề: {d}")
    ra = {}
    for f in sorted(d.glob("*.txt")):
        m = TEN_DE.match(f.stem)
        if not m:
            print(f"  ⚠️  bỏ qua {f.name} — tên không theo `query-<số>-<loại>.txt`")
            continue
        ra[m.group(1)] = f.read_text("utf-8").strip()
    if not ra:
        raise SystemExit(f"{d} không có file đề nào hợp lệ")
    return ra


def loai_cua(ten: str) -> str:
    return ten.rsplit("-", 1)[-1]


def tach_su_kien(noi_dung: str) -> list[str]:
    """Tách truy vấn TRAKE thành các sự kiện con.

    Số dòng quyết định **số Frame ID phải nộp** — BTC: *"Số lượng Frame ID phải
    khớp với số events được yêu cầu"*. Nên nếu tách sai là sai định dạng, mất
    trắng cả câu. Vì vậy `--so-su-kien` ghi đè được, và script LUÔN in ra số nó
    tách được để người kiểm bằng mắt.
    """
    dong = [x.strip() for x in noi_dung.splitlines() if x.strip()]
    if len(dong) > 1:
        # bỏ tiền tố đánh số "1." / "1)" / "- " nếu có
        return [re.sub(r"^\s*(\d+\s*[.)]|[-*])\s*", "", x) for x in dong]
    # một dòng: thử tách theo dấu chấm phẩy, rồi theo "1. ... 2. ..."
    if ";" in noi_dung:
        return [x.strip() for x in noi_dung.split(";") if x.strip()]
    phan = re.split(r"\s*\d+\s*[.)]\s*", noi_dung)
    phan = [x.strip() for x in phan if x.strip()]
    return phan if len(phan) > 1 else [noi_dung.strip()]


# ------------------------------------------------------------------ các kênh

def quet_anh(index: Path, matrix: str, de: dict, k: int, mmap=True) -> dict:
    """Chạy kênh ảnh cho MỌI truy vấn rồi giải phóng model.

    TRAKE cần một danh sách riêng cho từng sự kiện con, nên giá trị trả về là
    `{tên gói: list[Candidate]}` với KIS/QA và `{tên gói: [list, list, ...]}`
    với TRAKE.
    """
    from dense import KenhAnh
    kenh = KenhAnh(index, matrix=matrix, mmap=mmap)
    print(f"  kênh 1: {matrix} {kenh.mat.shape} "
          f"{kenh.model_tag}/{kenh.pretrained}")

    ra = {}
    for ten, noi_dung in de.items():
        if loai_cua(ten) == "trake":
            ra[ten] = [kenh.tim(sk, k=k) for sk in tach_su_kien(noi_dung)]
        else:
            ra[ten] = kenh.tim(noi_dung, k=k)
    master = kenh.master
    del kenh
    gc.collect()
    return ra, master


def quet_van_ban(master, de: dict, k: int, index: Path) -> dict:
    """Kênh 2 (metadata) + kênh 3 (OCR/ASR nếu có file). Nhẹ, không cần model."""
    from bm25 import KenhVanBan
    ra = {}

    k2 = KenhVanBan.tu_metadata(master)
    print(f"  kênh 2: metadata, {len(k2)} video")
    for ten, nd in de.items():
        if loai_cua(ten) != "trake":
            ra.setdefault(ten, []).append(k2.tim(nd, k=k, moi_video=3))
    del k2
    gc.collect()

    # Kênh 3: TV4 sinh ra `ocr_asr.parquet` với cột row_id + text.
    for p in (index / "ocr_asr.parquet",
              GOC / "pipeline_OCR_ASR" / "output" / "ocr_asr.parquet"):
        if p.exists():
            b = pd.read_parquet(p)
            k3 = KenhVanBan.tu_bang_khung(master, b, cot="text", ten="ocr_asr")
            print(f"  kênh 3: OCR/ASR, {len(k3):,} khung có chữ ({p.name})")
            for ten, nd in de.items():
                if loai_cua(ten) != "trake":
                    ra.setdefault(ten, []).append(k3.tim(nd, k=k))
            del k3
            gc.collect()
            break
    else:
        print("  kênh 3: chưa có ocr_asr.parquet — bỏ qua")
    return ra


# --------------------------------------------------------------------- TRAKE

def dung_trake(cac_su_kien: list, master, so_dong: int = TOI_DA_DONG) -> list:
    """Từ N danh sách ứng viên (mỗi sự kiện một danh sách) -> các dòng TRAKE.

    Chọn video chứa TRỌN chuỗi trước (`thoi_gian.video_du_chuoi`), rồi trong mỗi
    video lấy khung tốt nhất cho từng sự kiện. Ép **tăng dần theo thời gian** —
    BTC đòi *"thứ tự phải tuân theo thứ tự thời gian của các events"*, và
    `nop_bai.soat` sẽ chặn nếu không.

    Video không có đủ N sự kiện vẫn được dùng: TRAKE chấm **từng phần** theo số
    sự kiện khớp (A8.1), nên điền bừa một vị trí còn hơn bỏ trống — bỏ trống
    chắc chắn 0, đoán sai cũng 0.
    """
    from thoi_gian import video_du_chuoi
    n = len(cac_su_kien)
    if n == 0:
        return []

    uu_tien = video_du_chuoi(cac_su_kien)
    # nới ra: thêm mọi video xuất hiện ở bất kỳ sự kiện nào, giữ thứ tự
    for ds in cac_su_kien:
        for c in ds:
            if c.video_id not in uu_tien:
                uu_tien.append(c.video_id)

    # Khoảng frame thật của từng video, để điền chỗ trống cho có nghĩa.
    bien = master.groupby("video_id").frame_idx.agg(["min", "max"])

    ra = []
    for vid in uu_tien[:so_dong]:
        # Bước 1: khung TỐT NHẤT của từng sự kiện trong video này. Chưa ép thứ
        # tự — ép ngay sẽ khiến sự kiện đầu ăn mất khung tốt của sự kiện sau.
        tot = []
        for ds in cac_su_kien:
            trong = [c for c in ds if c.video_id == vid]
            tot.append(int(trong[0].frame_idx) if trong else None)

        if all(x is None for x in tot):
            continue

        # Bước 2: chỗ nào không tìm ra thì NỘI SUY giữa hai neo gần nhất, cùng
        # lắm thì rải đều trên khoảng frame của video.
        #
        # ⚠️ Bản đầu điền `khung_trước + 1`, ra những chuỗi kiểu 564,565,566 —
        # ba sự kiện cách nhau 0,02 giây, vô nghĩa. TRAKE chấm TỪNG PHẦN nên
        # một vị trí bịa không làm mất câu, nhưng bịa cho có nghĩa thì còn cơ
        # hội trúng; bịa +1 thì chắc chắn trượt.
        lo, hi = int(bien.loc[vid, "min"]), int(bien.loc[vid, "max"])
        co = [(i, v) for i, v in enumerate(tot) if v is not None]
        for i, v in enumerate(tot):
            if v is not None:
                continue
            truoc = [(j, x) for j, x in co if j < i]
            sau = [(j, x) for j, x in co if j > i]
            if truoc and sau:
                (j0, x0), (j1, x1) = truoc[-1], sau[0]
                tot[i] = x0 + round((x1 - x0) * (i - j0) / (j1 - j0))
            elif truoc:
                tot[i] = min(hi, truoc[-1][1] + round((hi - truoc[-1][1])
                                                      * (i - truoc[-1][0]) / n))
            else:
                tot[i] = max(lo, sau[0][1] - round((sau[0][1] - lo)
                                                   * (sau[0][0] - i) / n))

        # Bước 3: BTC đòi thứ tự tăng dần theo thời gian. Sắp xếp giữ nguyên
        # toàn bộ khung tốt, chỉ sửa những đảo lộn nhỏ — tốt hơn hẳn việc vứt
        # khung tốt đi để lấy một khung "đúng thứ tự" nhưng sai cảnh.
        khung = sorted(int(x) for x in tot)

        # N sự kiện KHÔNG THỂ nằm gọn trong vài phần trăm giây. Nếu cả N khung
        # dồn vào một chỗ thì truy hồi đã không phân biệt được các sự kiện —
        # giữ nguyên là dồn hết cơ hội vào một điểm. Rải đều trên khoảng frame
        # của video cho mỗi sự kiện một cửa độc lập; với cửa sổ BTC rộng tới
        # vài phút (A9) thì rải đều có xác suất trúng thật.
        if n > 1 and khung[-1] - khung[0] < DON_NHAU and hi > lo:
            buoc = (hi - lo) / (n + 1)
            khung = [lo + round(buoc * (i + 1)) for i in range(n)]

        for i in range(1, n):                    # phải TĂNG THẬT, không bằng nhau
            if khung[i] <= khung[i - 1]:
                khung[i] = khung[i - 1] + 1
        ra.append(AnswerTRAKE(vid, khung))
        if len(ra) >= so_dong:
            break
    return ra


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--de", required=True, type=Path, help="thư mục chứa query-*.txt")
    ap.add_argument("--ra", default="submission", type=Path)
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--matrix", default="clip_siglip2.npy",
                    help="ma trận kênh 1. SigLIP2 đo được 0,3258; "
                         "clip.npy chỉ 0,0000 trên tiếng Việt (A10/A17)")
    ap.add_argument("--k", type=int, default=TOI_DA_DONG)
    ap.add_argument("--hop-nhat", action="store_true",
                    help="RRF kênh 1 với kênh văn bản. ĐÃ ĐO LÀ LÀM TỆ ĐI "
                         "(A14/A17) — chỉ bật khi đo lại thấy thắng")
    ap.add_argument("--trong-so-phu", type=float, default=0.3,
                    help="trọng số cho kênh phụ khi --hop-nhat (A14.2)")
    ap.add_argument("--tra-loi", default="",
                    help="chuỗi `answer` dùng chung cho mọi dòng Q&A khi chưa "
                         "có VLM. Sai đáp án = 0 điểm, nhưng vẫn phải nộp")
    ap.add_argument("--so-su-kien", type=int, default=0,
                    help="ép số sự kiện TRAKE, thay vì tự tách từ đề")
    ap.add_argument("--nen", metavar="FILE.zip", help="soát xong thì nén luôn")
    a = ap.parse_args()

    de = doc_de(a.de)
    print(f"{len(de)} gói truy vấn: {', '.join(sorted(de))}\n")
    for ten, nd in sorted(de.items()):
        if loai_cua(ten) == "trake":
            sk = tach_su_kien(nd)
            n = a.so_su_kien or len(sk)
            print(f"  {ten}: tách được {len(sk)} sự kiện -> nộp {n} Frame ID")
            if not a.so_su_kien:
                print("     ⚠️  KIỂM LẠI BẰNG MẮT. Sai số sự kiện là sai định "
                      "dạng, mất trắng cả câu — ép bằng --so-su-kien nếu lệch.")
    print()

    kq1, master = quet_anh(a.index, a.matrix, de, a.k)
    phu = quet_van_ban(master, de, a.k, a.index) if a.hop_nhat else {}

    goi, so_su_kien = {}, {}
    for ten in sorted(de):
        loai = loai_cua(ten)
        if loai == "trake":
            ds = kq1[ten]
            if a.so_su_kien and len(ds) != a.so_su_kien:
                ds = (ds + [ds[-1]] * a.so_su_kien)[:a.so_su_kien]
            goi[ten] = dung_trake(ds, master, a.k)
            so_su_kien[ten] = len(ds)
        else:
            uv = kq1[ten]
            if a.hop_nhat and phu.get(ten):
                from rrf import hop_nhat
                ds = [uv] + phu[ten]
                uv = hop_nhat(ds, trong_so=[1.0] + [a.trong_so_phu] * len(phu[ten]))
            goi[ten] = tu_ung_vien(uv, loai, dap_an=a.tra_loi, gioi_han=a.k)
        print(f"  {ten:<20} {len(goi[ten]):>3} dòng")

    if any(loai_cua(t) == "qa" for t in de) and not a.tra_loi.strip():
        raise SystemExit(
            "\n❌ Có gói Q&A nhưng chưa có `answer`.\n"
            "   BTC chấm: khung đúng NHƯNG answer sai -> 0 điểm. Bỏ trống là\n"
            "   chắc chắn 0, và `nop_bai.soat` sẽ chặn.\n\n"
            "   Tạm thời: --tra-loi \"không rõ\"  (vẫn 0 điểm, nhưng nộp được\n"
            "   để kiểm định dạng đầu-cuối).\n"
            "   Đúng cách: cần VLM sinh đáp án — việc 12 của PHẦN H.")

    d = ghi_goi(goi, a.ra, so_su_kien)
    print(f"\n✅ Đã ghi {d}")
    if a.nen:
        print(f"   Đã nén -> {dong_goi(d, a.nen)}")
    else:
        print(f"   Soát + nén: python src/nop_bai.py --soat {d} --nen bai_nop.zip")


if __name__ == "__main__":
    main()
