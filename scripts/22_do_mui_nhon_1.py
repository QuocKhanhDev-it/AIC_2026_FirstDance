"""
22_do_mui_nhon_1.py — Đo từng bước của Mũi nhọn 1, mỗi lần một bước.

Ba bước vừa được nối vào (`src/mui_nhon_1.py`) đều là **giả thuyết chưa có số**:

    Bước 1   thu hẹp cấp video bằng metadata   — dạng MỀM và dạng CỨNG
    Bước 2b  khử trùng lặp trước khi cắt top-K — A11 hẹn đo lại, nay đo được
    Bước 4   VLM sinh `answer` cho Q&A         — cần `--vlm`, xem cuối file

    python scripts/22_do_mui_nhon_1.py
    python scripts/22_do_mui_nhon_1.py --vlm --so-cau 20

MỐC NỀN: RRF(objects, OCR) — cấu hình mạnh nhất chạy được KHÔNG cần model
=========================================================================

Máy 7,7 GB không nạp nổi SigLIP2 (`dense.kiem_ram` chặn trước khi treo máy),
nên mốc nền ở đây là cấu hình model-free tốt nhất: RRF(kênh 4, kênh 3) = 0,0640.

⚠️ **Kết luận rút ra ở đây CHƯA CHẮC đúng khi kênh 1 sống.** A14.2 đo được RRF
chỉ có lãi khi các kênh cùng tầm chất lượng; SigLIP2 (0,3258) hơn objects
(0,0412) tám lần, nên thứ tự các cấu hình có thể đảo hẳn. Mọi con số dưới đây
phải đo lại trên máy ≥ 16 GB bằng `--matrix clip_siglip2.npy`. Ghi rõ điều này
trong báo cáo, đừng để người sau đọc nhầm thành kết luận toàn cục.

VÌ SAO IN ĐỘ PHỦ VIDEO TRƯỚC KHI IN ĐIỂM
=========================================

Bước 1 dạng CỨNG bỏ hẳn ứng viên ngoài top-N video. Nếu video đúng chỉ nằm
trong top-50 ở 60% số câu thì **40% số câu về 0 trước khi bước nào phía sau kịp
chạy** — và điểm trung bình sẽ nói "tệ đi" mà không nói vì sao. Độ phủ là trần
trên của dạng cứng, phải biết nó trước.

Cùng lối làm với `16_do_rrf.py`: đo cả CƠ CHẾ, không chỉ đo điểm.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import tap_dev                                        # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from cham_diem import (MOC_DUNG_SAI, bao_cao_do_nhay,  # noqa: E402
                       cham, so_sanh_cap)
from mui_nhon_1 import (gan_dap_an, thu_hep,           # noqa: E402
                        uu_tien, video_uu_tien)
from objects import KenhObjects                       # noqa: E402
from rrf import hop_nhat                              # noqa: E402


def do_phu_video(cau, k2, master, cac_moc=(10, 50, 100, 200)) -> pd.DataFrame:
    """Video đúng nằm trong top-N của BM25 metadata ở bao nhiêu phần trăm câu?

    Đây là TRẦN TRÊN của Bước 1 dạng cứng: câu nào video đúng rơi ngoài top-N
    thì cắt cứng là mất trắng, không bước nào sau cứu được.
    """
    hang = []
    for c in cau:
        r = c.row_id_dung[0]
        r = r[0] if isinstance(r, list) else r
        dung = master.video_id.iloc[r]
        xep = video_uu_tien(k2, c.cau_hoi, so_video=10 ** 9)
        hang.append(xep.index(dung) + 1 if dung in xep else None)

    ra = [{"top_N": n,
           "phu": sum(1 for h in hang if h is not None and h <= n),
           "ty_le_%": round(100 * sum(1 for h in hang if h is not None and h <= n)
                            / len(hang), 1)}
          for n in cac_moc]
    ra.append({"top_N": "có mặt", "phu": sum(1 for h in hang if h is not None),
               "ty_le_%": round(100 * sum(1 for h in hang if h is not None)
                                / len(hang), 1)})
    return pd.DataFrame(ra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", default=tap_dev.MAC_DINH, type=Path)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--matrix", default="clip.npy",
                    help="ma trận cho Bước 2b. Chỉ dùng để so ẢNH với ảnh nên "
                         "clip.npy dùng được dù CLIP mù tiếng Việt (A10)")
    ap.add_argument("--vlm", action="store_true",
                    help="đo Bước 4 (VLM sinh answer) — cần Ollama + model VL")
    ap.add_argument("--so-cau", type=int, default=0,
                    help="giới hạn số câu Q&A khi đo Bước 4 (VLM chậm)")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    cau = tap_dev.doc(a.file)

    k2 = KenhVanBan.tu_metadata(master)
    k4 = KenhObjects(a.index, master)
    p = a.index / "ocr_asr.parquet"
    if not p.exists():
        raise SystemExit(f"Chưa có {p} — cần kênh 3 để dựng mốc nền mạnh nhất.")
    k3 = KenhVanBan.tu_bang_khung(master, pd.read_parquet(p),
                                  cot="text", ten="ocr_asr")

    print(f"{len(cau)} câu | mốc nền RRF(objects, OCR) | k={a.k}")
    print(f"kênh 2: {len(k2)} video | kênh 3: {len(k3):,} khung có chữ\n")

    print("=" * 76)
    print("BƯỚC 1 — ĐỘ PHỦ VIDEO của BM25 metadata (trần trên của dạng CỨNG)")
    print("=" * 76)
    print(do_phu_video(cau, k2, master).to_string(index=False))

    kho = {}

    def moc(c):
        if c.id not in kho:
            kho[c.id] = hop_nhat([k4.tim(c.cau_hoi, k=a.k),
                                  k3.tim(c.cau_hoi, k=a.k)])
        return kho[c.id]

    nho_vid = {}

    def vids(c, n):
        if (c.id, n) not in nho_vid:
            nho_vid[(c.id, n)] = video_uu_tien(k2, c.cau_hoi, so_video=n)
        return nho_vid[(c.id, n)]

    cau_hinh = {"RRF(objects, OCR) — mốc": moc}

    # Bước 1 — MỘT thứ đổi mỗi lần: dạng mềm ở ba mức N, rồi dạng cứng.
    for n in (10, 50, 200):
        cau_hinh[f"B1 mềm: ưu tiên top-{n} video"] = (
            lambda n: lambda c: uu_tien(moc(c), vids(c, n)))(n)
    cau_hinh["B1 CỨNG: chỉ top-50 video"] = (
        lambda c: thu_hep(moc(c), vids(c, 50)))

    print("\n" + "=" * 76)
    print("BƯỚC 1 — ĐIỂM")
    print("=" * 76)
    print(bao_cao_do_nhay(cau, cau_hinh, master, MOC_DUNG_SAI, gioi_han=a.k))

    # Bước 2b — dedup. Ma trận nạp mmap: chỉ đọc vài trăm dòng mỗi truy vấn.
    mat = np.load(a.index / a.matrix, mmap_mode="r")
    print("\n" + "=" * 76)
    print(f"BƯỚC 2b — KHỬ TRÙNG LẶP trước khi cắt top-K ({a.matrix})")
    print("=" * 76)
    cau_hinh2 = {"RRF(objects, OCR) — mốc": moc}
    for ng in (0.99, 0.95):
        cau_hinh2[f"B2b dedup ≥ {ng}"] = (
            lambda ng: lambda c: _dedup(moc(c), mat, ng, a.k))(ng)
    # dedup + ràng buộc đa dạng: A11 dự đoán hai cái BỔ SUNG nhau, chưa ai kiểm
    cau_hinh2["B2b dedup ≥ 0.99 + mỗi video ≤ 3"] = (
        lambda c: _moi_video(_dedup(moc(c), mat, 0.99, a.k), 3, a.k))
    print(bao_cao_do_nhay(cau, cau_hinh2, master, MOC_DUNG_SAI, gioi_han=a.k))

    if not a.vlm:
        print("\n(Bước 4 — VLM sinh `answer` — bỏ qua. Thêm `--vlm` để đo.)")
        return

    do_buoc_4(cau, master, moc, a)


def _dedup(uv, mat, nguong, k):
    from dedup import gom_ban_sao
    return gom_ban_sao(uv, mat, nguong=nguong)[:k]


def _moi_video(uv, moi_video, k):
    from rrf import gioi_han_moi_video
    return gioi_han_moi_video(uv, moi_video, k)


def do_buoc_4(cau, master, moc, a):
    """Bước 4 — đo `answer` do VLM sinh, trên các câu Q&A CÓ ẢNH ở máy này.

    ⚠️ **Đây là phép đo tuyệt đối, không phải phép so theo cặp.** Mốc nền hiện
    tại nộp `answer` rỗng/hằng số nên **chắc chắn 0 điểm** ở mọi câu Q&A
    (PHẦN C mục 4) — bất kỳ con số nào khác 0 cũng là hơn. Câu hỏi cần trả lời
    không phải *"có hơn không"* mà *"hơn được bao nhiêu"*.

    `cham()` xét `meta['answer']` theo TỪNG ứng viên (`_dung_dap_an`), nên chỉ
    cần gắn đáp án vào rồi chấm như thường.
    """
    qa = [c for c in cau if c.loai == "QA"]
    co_anh = []
    for c in qa:
        r = c.row_id_dung[0]
        p = master.kf_path.iloc[r]
        if isinstance(p, str) and p and Path(p).exists():
            co_anh.append(c)
    if a.so_cau:
        co_anh = co_anh[:a.so_cau]

    print("\n" + "=" * 76)
    print(f"BƯỚC 4 — VLM SINH `answer` ({len(co_anh)}/{len(qa)} câu Q&A có ảnh)")
    print("=" * 76)
    if not co_anh:
        print("Không câu Q&A nào có ảnh trên máy này — không đo được ở đây.")
        return

    nho = {}

    def co_vlm(c):
        if c.id not in nho:
            nho[c.id] = gan_dap_an(moc(c), master, c.cau_hoi)
        return nho[c.id]

    truoc = cham(co_anh, moc, a.k, master, MOC_DUNG_SAI[0])
    sau = cham(co_anh, co_vlm, a.k, master, MOC_DUNG_SAI[0])
    print(so_sanh_cap(truoc, sau, "answer rỗng (mốc)", "answer do VLM"))
    print("\n⚠️  Mốc nền ở đây chấm như thể `answer` LUÔN ĐÚNG — `cham()` coi ứng "
          "viên\n    không có khóa `answer` là hợp lệ. Nên cột mốc là TRẦN TRÊN "
          "của truy hồi,\n    còn cột VLM mới là điểm thi thật. Chênh lệch âm là "
          "chuyện đương nhiên;\n    thứ cần đọc là cột VLM có KHÁC 0 không.")


if __name__ == "__main__":
    main()
