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

# Tên file đề. Tài liệu BTC ví dụ `query-1-kis`, nhưng bộ đề mẫu thật dùng
# `query-p1-1-kis` — có thêm mã đợt. Nhận cả hai; thứ BẮT BUỘC đúng là HẬU TỐ,
# vì BTC chấm theo đó.
TEN_DE = re.compile(r"^(query-.+-(kis|qa|trake))$")

# Sự kiện TRAKE trong đề mẫu đánh dấu bằng `E1:`, `E2:`...
_SU_KIEN = re.compile(r"^\s*E\s*\d+\s*[:.]\s*", re.I)

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

    # Đề mẫu thật đánh dấu sự kiện bằng `E1:`, `E2:`... và thường có một dòng
    # MỞ ĐẦU tả cả video trước đó.
    #
    # ⚠️ ĐẾM THEO DÒNG, KHÔNG THEO SỐ. Đề mẫu `query-p1-18-trake` đánh nhầm
    # `E1, E2, E2, E4` — không có E3. Bốn dòng là bốn sự kiện; tin vào con số
    # thì ra ba, mà sai số sự kiện là **sai định dạng, mất trắng cả câu**.
    ky = [x for x in dong if _SU_KIEN.match(x)]
    if ky:
        # dòng trước sự kiện đầu tiên = bối cảnh chung, ghép vào MỌI sự kiện
        dau = dong.index(ky[0])
        boi_canh = " ".join(dong[:dau]).rstrip(":. ")
        return [f"{boi_canh} {_SU_KIEN.sub('', x)}".strip() if boi_canh
                else _SU_KIEN.sub("", x).strip() for x in ky]

    if len(dong) > 1:
        # bỏ tiền tố đánh số "1." / "1)" / "- " nếu có
        return [re.sub(r"^\s*(\d+\s*[.)]|[-*])\s*", "", x) for x in dong]
    # một dòng: thử tách theo dấu chấm phẩy, rồi theo "1. ... 2. ..."
    if ";" in noi_dung:
        return [x.strip() for x in noi_dung.split(";") if x.strip()]
    phan = re.split(r"\s*\d+\s*[.)]\s*", noi_dung)
    phan = [x.strip() for x in phan if x.strip()]
    return phan if len(phan) > 1 else [noi_dung.strip()]


# Trần token của text encoder: CLIP 77, SigLIP2 **chỉ 64** (xem `model_configs`
# của open_clip). Vượt trần là encoder LẶNG LẼ CẮT phần đuôi, không báo gì.
#
# Đổi ra TỪ thì phụ thuộc tokenizer, và hai model chênh nhau rất xa trên tiếng
# Việt — đo trên chính tập dev + đề mẫu:
#
#     CLIP (BPE tiếng Anh)        3,21 token/từ  -> 23 từ đã chạm trần 77
#     SigLIP2 (Gemma đa ngôn ngữ) 1,18 token/từ  -> ~50 từ mới chạm trần 64
#
# Lấy 40 từ: an toàn cho SigLIP2 (~47 token). CLIP vẫn bị cắt ở mức này, nhưng
# CLIP đang được 0,0000 trên tiếng Việt (A10) nên không đáng tối ưu theo nó.
TRAN_TOKEN = 40


def tach_truy_van(cau: str, tran_tu: int = TRAN_TOKEN) -> list[str]:
    """Truy vấn dài -> nhiều mệnh đề ngắn, để encoder không cắt mất phần đuôi.

    ⚠️ **ĐO ĐƯỢC: 100% truy vấn của bộ đề mẫu bị cắt cụt.** Đề thi thật dài
    trung bình 63 từ / 281 ký tự (gấp 3 lần câu tập dev tự soạn), trong khi
    `ViT-SO400M-14-SigLIP2-378` chỉ nhận **64 token** và CLIP nhận 77. Encoder
    không báo lỗi — nó cắt rồi chạy tiếp, nên nửa sau mỗi truy vấn **biến mất
    mà không có gì cảnh báo**.

    Cách chữa rẻ nhất: cắt theo CÂU, encode từng câu, rồi lấy ĐIỂM CAO NHẤT
    trên từng keyframe. `dense.KenhAnh.tim` và `bm25.KenhVanBan.tim` **đã nhận
    sẵn danh sách chuỗi** và tự lấy max — chỉ thiếu thứ cắt câu ra.

    Lấy max chứ không lấy trung bình là có chủ ý: một mệnh đề trúng là đủ, không
    nên để những mệnh đề tả bối cảnh chung kéo điểm xuống.
    """
    cau = " ".join(cau.split())
    if len(cau.split()) <= tran_tu:
        return [cau]
    manh = [x.strip() for x in re.split(r"(?<=[.!?;])\s+", cau) if x.strip()]
    ra, dem = [], []
    for m in manh:
        if dem and len(" ".join(dem + [m]).split()) > tran_tu:
            ra.append(" ".join(dem))
            dem = []
        dem.append(m)
    if dem:
        ra.append(" ".join(dem))
    # mệnh đề đơn lẻ vẫn quá dài thì cắt cứng theo từ
    cuoi = []
    for x in ra:
        t = x.split()
        cuoi += [" ".join(t[i:i + tran_tu]) for i in range(0, len(t), tran_tu)]             if len(t) > tran_tu else [x]
    return cuoi or [cau]


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
            ra[ten] = [kenh.tim(tach_truy_van(sk), k=k)
                       for sk in tach_su_kien(noi_dung)]
        else:
            # Xin GAP DOI: hai row_id khac nhau co the ra cung mot dong nop
            # (A5.7 — 614 keyframe trung frame_idx), nen bo trung xong phai con
            # du de bu cho tron 100.
            ra[ten] = kenh.tim(tach_truy_van(noi_dung), k=k * 2)
    master = kenh.master
    del kenh
    gc.collect()
    return ra, master


def quet_objects(index: Path, de: dict, k: int):
    """Kenh 4 (objects) — KHONG can model lon.

    Duong thoat cho may thieu RAM: chot `dense.kiem_ram` chan model anh, nhung
    van phai nop duoc bai. Chat luong kem han han (0,0412 so voi 0,3258 cua
    SigLIP2 — A14/A17), nhung ra file DUNG DINH DANG, va dinh dang sai thi van
    tinh mot trong ba lan nop.
    """
    import importlib.util
    import pandas as pd
    master = pd.read_parquet(index / "master.parquet")
    s = importlib.util.spec_from_file_location(
        "r16", GOC / "scripts" / "16_do_rrf.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    k4 = m.KenhObjects(index, master)
    print("  kênh 4: objects + IDF (không cần model)")
    ra = {}
    for ten, nd in de.items():
        if loai_cua(ten) == "trake":
            ra[ten] = [k4.tim(x, k=k) for x in tach_su_kien(nd)]
        else:
            ra[ten] = k4.tim(nd, k=k * 2)     # xem ghi chu o quet_anh
    del k4
    gc.collect()
    return ra, master


# Token xuất hiện ở nhiều hơn ngần này tài liệu OCR thì KHÔNG phải token hiếm.
NGUONG_HIEM = 200

# Chèn tối đa ngần này ứng viên khớp cứng. Lọc cứng để **đẩy lên**, không phải
# để **thay thế**: cho nó chiếm cả 100 chỗ là vứt bỏ hoàn toàn kênh xếp hạng,
# mà token "hiếm" vẫn có thể là một âm tiết tiếng Việt tình cờ ít gặp trong kho
# OCR (đo được: `nghiên` lọt qua ngưỡng tần suất và kéo về 100 khung).
TOI_DA_LOC_CUNG = 20


def loc_cung(de: dict, bang, master, k: int) -> dict:
    """Chế độ LỌC CỨNG cho token hiếm (A8.5) — trả ứng viên để chèn lên đầu.

    3/5 ví dụ thực chiến của đội AIC'25 thắng bằng cách này chứ không bằng xếp
    hạng: `/filter all ocr{hidro}`. Với token hiếm, lọc dứt khoát hơn hẳn BM25
    hoà RRF — `hidro` có ở 3 khung thì lọc trả đúng 3, còn hoà RRF thì ba kênh
    kia dìm nó xuống dưới hạng 20.

    ⚠️ **`phat_hien_token_hiem` của TV4 quá tham trên văn bản thật.** Nó coi chữ
    hoa là tên riêng, mà tiếng Việt câu nào cũng viết hoa chữ đầu — đo được: nó
    bắt `Một`, `Đây`, `Trong`, `Sau` và nổ ở **24/24 gói đề mẫu**. Nhét những
    thứ đó lên đầu là đẩy ứng viên tốt khỏi hạng 1, mà R@1 chiếm 1/5 tổng điểm.

    Nên lọc lại bằng **chính kho OCR**: token chỉ được coi là hiếm khi nó xuất
    hiện ở `<= NGUONG_HIEM` tài liệu. Đo trên 47.064 tài liệu OCR:

        Một 2.397 · Trong 4.994 · Sau 2.688   -> không hiếm
        Debby 16 · Carolina 7                 -> hiếm

    Đây là đúng ý IDF, cùng nguyên tắc đã dùng ở `objects.py`, và không cần
    bảng từ dừng viết tay.
    """
    import collections
    import re as _re
    # `pipeline_OCR_ASR` nam o goc repo, ma sys.path chi co src/ (xem dau file).
    if str(GOC) not in sys.path:
        sys.path.insert(0, str(GOC))
    from pipeline_OCR_ASR.loc_cung_token_hiem import bo_dau, phat_hien_token_hiem
    from schema import Candidate

    co = bang[bang.text.fillna("").str.strip() != ""].copy()
    co["kd"] = co.text.map(bo_dau)
    df = collections.Counter()
    for x in co.kd:
        df.update(set(_re.findall(r"[^\W_]+", x)))

    ra = {}
    for ten, nd in de.items():
        if loai_cua(ten) == "trake":
            continue
        hiem = [t for t in phat_hien_token_hiem(nd)
                if 0 < df.get(bo_dau(t), 0) <= NGUONG_HIEM]
        if not hiem:
            continue
        # ĐÒI ĐỦ MỌI token hiếm, không phải bất kỳ cái nào. Nhiều token cùng
        # xuất hiện là tín hiệu mạnh hơn hẳn; `any` thì một âm tiết lạc cũng
        # kéo về cả trăm khung không liên quan.
        kd_hiem = [bo_dau(t) for t in hiem]
        mat = co.kd.apply(lambda x: all(t in x for t in kd_hiem))
        khop = co[mat].head(min(k, TOI_DA_LOC_CUNG))
        if khop.empty:
            continue
        ra[ten] = [Candidate(row_id=int(r), video_id=master.video_id.iloc[r],
                             frame_idx=int(master.frame_idx.iloc[r]),
                             score=1.0, source="loc_cung")
                   for r in khop.row_id]
        print(f"  lọc cứng {ten}: {hiem} -> {len(ra[ten])} khung")
    return ra


def quet_van_ban(master, de: dict, k: int, index: Path,
                 bo_metadata: bool = False) -> dict:
    """Kênh 2 (metadata) + kênh 3 (OCR/ASR nếu có file). Nhẹ, không cần model."""
    from bm25 import KenhVanBan
    ra = {}

    if bo_metadata:
        print("  kênh 2: BỎ (đo được 0,0000 ở ±2s — cộng vào là pha loãng)")
    else:
        k2 = KenhVanBan.tu_metadata(master)
        print(f"  kênh 2: metadata, {len(k2)} video")
        for ten, nd in de.items():
            if loai_cua(ten) != "trake":
                ra.setdefault(ten, []).append(k2.tim(tach_truy_van(nd), k=k,
                                                     moi_video=3))
        del k2
        gc.collect()

    # Kênh 3: TV4 sinh ra `ocr_asr.parquet` với cột row_id + text.
    for p in (index / "ocr_asr.parquet",
              GOC / "pipeline_OCR_ASR" / "output" / "ocr_asr.parquet"):
        if p.exists():
            b = pd.read_parquet(p)
            # ⚠️ File TON TAI khong co nghia la CO DU LIEU. `run_production_
            # pipeline.py` tu sinh mot ocr_asr.parquet 177.321 dong TOAN RONG
            # khi chua co ocr.parquet nguon — da gap that. Nhan nham file do la
            # kenh 3 im lang khong tra ve gi, khong bao loi nao.
            if b.get("text", pd.Series(dtype=str)).fillna("").str.strip().eq("").all():
                print(f"  kênh 3: {p} KHÔNG có dòng nào có chữ — bỏ qua")
                continue
            k3 = KenhVanBan.tu_bang_khung(master, b, cot="text", ten="ocr_asr")
            print(f"  kênh 3: OCR/ASR, {len(k3):,} khung có chữ ({p.name})")
            for ten, nd in de.items():
                if loai_cua(ten) != "trake":
                    ra.setdefault(ten, []).append(k3.tim(tach_truy_van(nd), k=k))
            del k3
            gc.collect()
            break
    else:
        print("  kênh 3: chưa có ocr_asr.parquet — bỏ qua")
    return ra


def bu_cho_du(uv: list, master, k: int, mam: int = 0) -> list:
    """Không bao giờ nộp file RỖNG. Thiếu thì bù bằng khung rải đều toàn kho.

    ⚠️ Kênh trả về rỗng là chuyện có thật, không phải giả định: trên bộ đề mẫu,
    kênh objects không rút được nhãn nào cho `query-p1-15-qa` và
    `query-p1-21-kis` -> 0 dòng -> `nop_bai.soat` chặn cả gói.

    Nộp bừa **luôn tốt hơn nộp rỗng**: PHẦN C mục 1 — *"không có điểm phạt, câu
    thứ 100 vẫn đáng 0,2"*. File rỗng thì chắc chắn 0 VÀ có thể làm hỏng cả lần
    nộp (mà chỉ có 3 lần).

    Rải đều theo `row_id` thay vì lấy 100 dòng đầu: 100 khung liên tiếp của một
    video là gần như một khoảnh khắc, còn rải đều thì phủ được nhiều video.
    """
    from schema import Candidate
    if len(uv) >= k:
        return uv
    co = {(c.video_id, int(c.frame_idx)) for c in uv}
    ra = list(uv)
    n = len(master)
    buoc = max(1, n // (k * 2))
    for i in range(mam, n, buoc):
        if len(ra) >= k:
            break
        g = master.iloc[i]
        khoa = (g.video_id, int(g.frame_idx))
        if khoa in co:
            continue
        co.add(khoa)
        ra.append(Candidate(row_id=int(i), video_id=g.video_id,
                            frame_idx=int(g.frame_idx), score=0.0, source="bu"))
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

    # Bu cho du 100 dong. Video khong co su kien nao van dang nop: TRAKE cham
    # TUNG PHAN, va khong co diem phat — dong thu 100 van dang 0,2 neu trung
    # mot vi tri. De trong 75 dong la vut khong 75 co hoi.
    if len(ra) < so_dong:
        da_co = {x.video_id for x in ra}
        con = [v for v in bien.index if v not in da_co]
        buoc = max(1, len(con) // max(so_dong - len(ra), 1))
        for v in con[::buoc]:
            if len(ra) >= so_dong:
                break
            lo, hi = int(bien.loc[v, "min"]), int(bien.loc[v, "max"])
            b = (hi - lo) / (n + 1)
            kh = [lo + round(b * (i + 1)) for i in range(n)]
            for i in range(1, n):
                if kh[i] <= kh[i - 1]:
                    kh[i] = kh[i - 1] + 1
            ra.append(AnswerTRAKE(v, kh))
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
    ap.add_argument("--loc-cung", action="store_true",
                    help="chen ung vien khop CUNG token hiem len dau (A8.5). "
                         "CHUA DO DUOC tren tap dev — xem docstring loc_cung()")
    ap.add_argument("--kenh", default="anh", choices=("anh", "objects"),
                    help="objects = KHONG can model lon, chay duoc tren may "
                         "thieu RAM. Chat luong kem han (0,0412 so voi 0,3258) "
                         "nhung ra file dung dinh dang")
    ap.add_argument("--hop-nhat", action="store_true",
                    help="RRF kênh 1 với kênh văn bản. ĐÃ ĐO LÀ LÀM TỆ ĐI "
                         "(A14/A17) — chỉ bật khi đo lại thấy thắng")
    ap.add_argument("--bo-metadata", action="store_true",
                    help="hop nhat MA KHONG lay kenh 2. Kenh 2 duoc 0,0000 o "
                         "±2s (A12) nen cong vao la PHA LOANG (A14.2). Cau hinh "
                         "do duoc tot nhat khong can model la RRF(objects, OCR)")
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

    if a.kenh == "objects":
        kq1, master = quet_objects(a.index, de, a.k)
    else:
        kq1, master = quet_anh(a.index, a.matrix, de, a.k)
    phu = (quet_van_ban(master, de, a.k, a.index, a.bo_metadata)
           if a.hop_nhat else {})

    lc = {}
    if a.loc_cung:
        for p in (a.index / "ocr_asr.parquet",
                  GOC / "pipeline_OCR_ASR" / "output" / "ocr_asr.parquet"):
            if p.exists():
                b = pd.read_parquet(p)
                if not b.get("text", pd.Series(dtype=str)).fillna("").str.strip().eq("").all():
                    lc = loc_cung(de, b, master, a.k)
                    break
        else:
            print("  lọc cứng: chưa có ocr_asr.parquet — bỏ qua")

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
            if lc.get(ten):
                # Khớp CỨNG lên đầu, phần còn lại giữ nguyên thứ tự của kênh.
                da = {x.row_id for x in lc[ten]}
                uv = lc[ten] + [x for x in uv if x.row_id not in da]
            if not uv:
                print(f"     ⚠️  {ten}: kênh không trả về gì — bù cho đủ "
                      f"{a.k} dòng (nộp bừa hơn nộp rỗng)")
            if a.hop_nhat and phu.get(ten):
                from rrf import hop_nhat
                ds = [uv] + phu[ten]
                uv = hop_nhat(ds, trong_so=[1.0] + [a.trong_so_phu] * len(phu[ten]))
            goi[ten] = tu_ung_vien(bu_cho_du(uv, master, a.k), loai,
                                   dap_an=a.tra_loi, gioi_han=a.k)
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
