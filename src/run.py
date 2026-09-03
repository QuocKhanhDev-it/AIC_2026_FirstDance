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

HỢP NHẤT MẶC ĐỊNH **BẬT** — VÀ ĐÂY LÀ MỘT QUYẾT ĐỊNH ĐÃ BỊ LẬT MỘT LẦN
Ba phép đo cũ (A14, A14.1, A17) nói cộng kênh yếu vào kênh mạnh làm **tệ đi**,
nên mặc định từng là TẮT. A45 đo lại trên **49 câu ĐỀ THẬT** thì ngược hẳn:

    kênh 1 SigLIP2   0,3258 (tập dev tự soạn)  ->  0,1429 (đề thật)
    kênh 3 OCR/ASR   0,1183                    ->  0,1633   <- mạnh hơn kênh 1
    RRF(1,3) 1:1     -0,0144                   ->  +0,0694  ✅ ỔN ĐỊNH

Ba phép đo cũ không sai về số học — chúng đo trên tập dev tự soạn, thứ đã bị
chứng minh là **thổi phồng kênh 1 gấp 2,3 lần**. Muốn dựng lại hành vi cũ:
`--khong-hop-nhat`.
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

# Mốc sự kiện TRAKE. Hai nhánh, và chúng CỐ Ý chặt lỏng khác nhau:
#
#   `E1:` `E1.` `E1 `      — đề sơ tuyển đợt 1 viết `E1 ` KHÔNG dấu hai chấm
#                            (A40). Nên nhánh này nhận cả khoảng trắng trần.
#   `Cảnh 1:` `Sự kiện 1.` — đề sơ tuyển đợt 2 đổi sang từ tiếng Việt (A44).
#                            Nhánh này BẮT BUỘC có `:` hoặc `.`, không nhận
#                            khoảng trắng trần — vì "Cảnh 2 người đàn ông đang
#                            khiêng thùng" là một câu tả bình thường, nhận nhầm
#                            nó làm mốc thì tách ra thừa sự kiện.
#
# Tách sai số sự kiện = nộp sai số Frame ID = MẤT TRẮNG CẢ GÓI. Đã cắn hai lần.
_SU_KIEN = re.compile(
    r"^\s*(?:"
    r"E\s*\d+(?:\s*[:.]\s*|\s+)"
    r"|(?:cảnh|sự\s*kiện|scene|event)\s*\d+\s*[:.]\s*"
    r")(?=\S)", re.I)

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

def quet_anh(index: Path, matrix: str, de: dict, k: int, mmap=True,
            giu_kenh: bool = False, cache=None, rrf_menh_de: bool = True):
    """Chạy kênh ảnh cho MỌI truy vấn rồi giải phóng model.

    TRAKE cần một danh sách riêng cho từng sự kiện con, nên giá trị trả về là
    `{tên gói: list[Candidate]}` với KIS/QA và `{tên gói: [list, list, ...]}`
    với TRAKE.

    `giu_kenh=True` KHÔNG giải phóng model — trả thêm `kenh` làm giá trị thứ
    ba. Chỉ bật khi cần Bước 4 (trích dày) cho TRAKE: nó cần encode ảnh bằng
    ĐÚNG model kênh 1, mà model đã giải phóng thì phải nạp lại (tốn thêm thời
    gian + RAM đỉnh). Mặc định `False`, giữ đúng hành vi cũ — "nạp MỘT model,
    chạy HẾT, giải phóng".
    """
    # `cache` = file .npz vector truy vấn mã hoá sẵn (25_ma_hoa_truy_van.py).
    # Có nó thì KHÔNG nạp model, nên máy thiếu RAM vẫn sinh được bài nộp bằng
    # kênh 1 — kênh mạnh nhất (SigLIP2 0,3258 / leaderboard 3,8, A24).
    if cache:
        from dense import KenhAnhCache
        kenh = KenhAnhCache(index, cache, matrix=matrix, mmap=mmap)
        print(f"  kênh 1: {matrix} {kenh.mat.shape} {kenh.model_tag} "
              f"— TỪ CACHE {Path(cache).name}, không nạp model")
        # Hỏng SỚM: thiếu một chuỗi là thiếu giữa chừng, mà lúc đó đã chạy nửa
        # đường ống. `co_du` hỏi trước khi tốn công.
        can = []
        for ten, nd in de.items():
            nguon = tach_su_kien(nd) if loai_cua(ten) == "trake" else [nd]
            for x in nguon:
                can += tach_truy_van(x)
        thieu = kenh.co_du(can)
        if thieu:
            raise SystemExit(
                f"\n❌ {len(thieu)}/{len(can)} chuỗi truy vấn chưa có trong "
                f"cache.\n   Ví dụ: {thieu[0][:90]!r}\n\n"
                f"   Mã hoá thêm (máy đủ RAM hoặc Colab):\n"
                f"     python scripts/25_ma_hoa_truy_van.py "
                f"--de <thư mục đề> --gop")
    else:
        from dense import KenhAnh
        kenh = KenhAnh(index, matrix=matrix, mmap=mmap)
        print(f"  kênh 1: {matrix} {kenh.mat.shape} "
              f"{kenh.model_tag}/{kenh.pretrained}")

    def hoi(noi_dung: str, sl: int):
        """Một truy vấn -> danh sách ứng viên, hợp nhất các mệnh đề bằng RRF.

        ⚠️ HỢP NHẤT THEO HẠNG, KHÔNG PHẢI MAX COSINE (A51).

        `KenhAnh.tim` nhận cả danh sách và lấy **điểm cosine cao nhất** trên
        từng keyframe qua các mệnh đề. Nghe hợp lý, nhưng cosine của hai mệnh
        đề KHÁC NHAU không so được với nhau: "một người phụ nữ" khớp mờ với
        hàng nghìn khung ở cos cao, còn "biển hiệu màu tím ghi BỆNH VIỆN" khớp
        đúng một khung ở cos thấp hơn. Max cosine để **mệnh đề dễ nuốt mệnh đề
        đặc trưng**.

        Xếp theo hạng thì mỗi mệnh đề có tiếng nói ngang nhau — cùng lý do
        repo này hợp nhất KÊNH bằng RRF chứ không cộng điểm (xem `schema.py`).

        Đo trên 52 câu đề thật, so với chính `tim()` max-cosine:
        cùng kênh 3 thì +0,0721 / +0,0971 ✅ ỔN ĐỊNH.
        """
        md = tach_truy_van(noi_dung)
        if len(md) == 1 or not rrf_menh_de:
            return kenh.tim(md, k=sl)
        return hop_nhat([kenh.tim(m, k=sl) for m in md])[:sl]

    ra = {}
    for ten, noi_dung in de.items():
        if loai_cua(ten) == "trake":
            ra[ten] = [hoi(sk, k) for sk in tach_su_kien(noi_dung)]
        else:
            # Xin GAP DOI: hai row_id khac nhau co the ra cung mot dong nop
            # (A5.7 — 614 keyframe trung frame_idx), nen bo trung xong phai con
            # du de bu cho tron 100.
            ra[ten] = hoi(noi_dung, k * 2)
    master = kenh.master
    if giu_kenh:
        return ra, master, kenh
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
    from objects import KenhObjects
    master = pd.read_parquet(index / "master.parquet")
    k4 = KenhObjects(index, master)
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

K_UNG_VIEN_MOI_SU_KIEN = 20   # candidate xét mỗi sự kiện trong dong_hang_dp()


def dong_hang_dp(cac_su_kien_trong_video: list) -> list:
    """Bước 5 — quy hoạch động dóng N sự kiện thành N khung TĂNG DẦN, **giữ
    đúng chỉ số sự kiện** ở từng vị trí.

    ⚠️ SỬA LỖI THẬT ở bản heuristic trước: nó lấy khung tốt nhất (rank-1) của
    từng sự kiện trong video rồi `sorted(tot)` để ép tăng dần theo THỜI GIAN —
    nhưng sort theo GIÁ TRỊ, không theo sự kiện. Mỗi sự kiện truy hồi độc lập,
    không biết gì về thứ tự của các sự kiện khác, nên khung rank-1 của sự kiện
    2 hoàn toàn có thể nhỏ hơn khung rank-1 của sự kiện 1 — `sorted()` khi đó
    HOÁN ĐỔI khung giữa hai sự kiện, nộp nhầm khung sự kiện 2 cho vị trí sự
    kiện 1. Hàm này không sort gì cả: DP chọn ứng viên cho VỊ TRÍ i luôn lấy
    từ danh sách ứng viên CỦA sự kiện i, chỉ ràng buộc khung(i) < khung(i+1).

    Còn xét ĐA ỨNG VIÊN mỗi sự kiện (không chỉ rank-1): nếu ứng viên tốt nhất
    của sự kiện i không tăng dần được so với sự kiện trước, DP thử ứng viên
    tốt nhì, tốt ba... tối đa `K_UNG_VIEN_MOI_SU_KIEN` — thay vì heuristic cũ
    chỉ có đúng MỘT lựa chọn mỗi sự kiện nên bó tay khi nó không tăng dần.

    `cac_su_kien_trong_video[i]` là `list[(frame_idx, score)]` đã bỏ trùng
    frame_idx (giữ điểm cao nhất), KHÔNG cần sắp sẵn theo điểm hay theo frame.
    Sự kiện không có ứng viên nào thì truyền `[]`.

    Trả về `list[int | None]` cùng độ dài — `None` ở vị trí không chọn được
    ứng viên tăng dần hợp lệ (kể cả khi rỗng hoàn toàn); người gọi tự nội suy
    như trước. Độ phức tạp O(N·K²), K nhỏ (~20) nên không đáng kể — plan gốc
    ghi O(N·K) bằng cách giữ prefix-max, bỏ qua tối ưu đó vì N·K² ở quy mô này
    (~1.600 phép) không đo được khác biệt thời gian nào.
    """
    n = len(cac_su_kien_trong_video)
    dp: list[list[tuple[int, float, int, int]] | None] = [None] * n
    truoc_i = None   # chỉ số sự kiện HỢP LỆ gần nhất (có ít nhất 1 ứng viên)

    for i, cands in enumerate(cac_su_kien_trong_video):
        if not cands:
            continue
        lop = []
        if truoc_i is None:
            for fidx, sc in cands:
                lop.append((fidx, sc, -1, -1))
        else:
            truoc_lop = dp[truoc_i]
            for fidx, sc in cands:
                hop_le = [(j, t) for j, t in enumerate(truoc_lop) if t[0] < fidx]
                if hop_le:
                    j, t = max(hop_le, key=lambda x: x[1][1])
                    lop.append((fidx, sc + t[1], truoc_i, j))
                else:
                    # không có tiền nhiệm hợp lệ -> bắt đầu lại chuỗi từ đây
                    lop.append((fidx, sc, -1, -1))
        dp[i] = lop
        truoc_i = i

    if truoc_i is None:
        return [None] * n

    ket_lop = dp[truoc_i]
    j = max(range(len(ket_lop)), key=lambda x: ket_lop[x][1])
    ra: list = [None] * n
    i = truoc_i
    while i is not None and i != -1 and j != -1:
        fidx, _, pi, pj = dp[i][j]
        ra[i] = fidx
        i, j = (pi, pj) if pi != -1 else (None, -1)
    return ra


def lam_day_bang_trich_day(vid: str, video_path: str, cac_su_kien_text: list,
                           cac_ung_vien_video: list, kenh1, master,
                           ban_kinh_giay=2.0, stride=2) -> None:
    """Bước 4 — trích thêm khung QUANH ứng viên tốt nhất của mỗi sự kiện,
    encode bằng chính model kênh 1, thêm làm ứng viên MỚI cho DP chọn.

    ⚠️ Chỉ gọi khi máy CÓ file `.mp4` thật (`video_path`) — thiếu thì nơi gọi
    bỏ qua lặng lẽ, không phải lỗi (đúng nguyên tắc "kênh nào thiếu thì bỏ
    qua" của `run.py`).

    Sửa TRỰC TIẾP `cac_ung_vien_video[i]` (list `(frame_idx, score)` của
    từng sự kiện) bằng cách nối thêm ứng viên mới — không trả về gì, gọi ngay
    trước khi đưa vào `dong_hang_dp()`.

    Cần một ỨNG VIÊN NEO có sẵn cho sự kiện đó (từ keyframe) để biết trích
    quanh đâu và lấy `pts_time`/`fps` THẬT (không suy từ `frame_idx/fps` —
    xem cảnh báo ở `trich_day.py`). Sự kiện hoàn toàn không có ứng viên
    trong video này thì bỏ qua, để `dong_hang_dp` + nội suy xử lý như cũ.
    """
    from trich_day import trich_day

    for i, ung_vien in enumerate(cac_ung_vien_video):
        if not ung_vien:
            continue
        neo_fidx, _ = max(ung_vien, key=lambda x: x[1])
        hang = master[(master.video_id == vid) & (master.frame_idx == neo_fidx)]
        if hang.empty:
            continue
        r = hang.iloc[0]
        khung_moi = trich_day(video_path, vid, int(neo_fidx), float(r.pts_time),
                              float(r.fps), radius_sec=ban_kinh_giay, stride=stride)
        if not khung_moi:
            continue
        vec = kenh1.encode_image([k.anh for k in khung_moi])
        q = kenh1.encode_text(cac_su_kien_text[i])
        diem = vec @ q
        co = {f for f, _ in ung_vien}
        for k, sc in zip(khung_moi, diem):
            if k.frame_idx not in co:
                ung_vien.append((k.frame_idx, float(sc)))


def _don_cuc(khung: list, cach: str) -> bool:
    """Chuỗi khung này có bị DỒN CỤC không, theo chính sách `cach`.

    `"cap"` — có BẤT KỲ cặp liền kề nào cách nhau dưới `DON_NHAU`.
    `"tong"` — chỉ xét tổng độ trải (bản đầu; bỏ sót dồn cục một phần).
    `"khong"` — không bao giờ coi là dồn cục, tức tắt hẳn việc rải đều.
    """
    if cach == "khong":
        return False
    if cach == "tong":
        return khung[-1] - khung[0] < DON_NHAU
    return any(b - a < DON_NHAU for a, b in zip(khung, khung[1:]))


def dung_trake(cac_su_kien: list, master, so_dong: int = TOI_DA_DONG,
               su_kien_text: list | None = None, kenh1=None,
               so_video_trich_day: int = 5, don_cuc: str = "tong",
               dong_hang: str = "cu", he_so_phat: float = 0.0,
               trai_toi_da: float | None = None, rai_hep: bool = False) -> list:
    """Từ N danh sách ứng viên (mỗi sự kiện một danh sách) -> các dòng TRAKE.

    Chọn video bằng điểm MỀM (`thoi_gian.xep_video_theo_chuoi` — Bước 2b),
    rồi trong mỗi video lấy khung tốt nhất cho từng sự kiện. Ép **tăng dần
    theo thời gian** — BTC đòi *"thứ tự phải tuân theo thứ tự thời gian của
    các events"*, và `nop_bai.soat` sẽ chặn nếu không.

    Video không có đủ N sự kiện vẫn được dùng: TRAKE chấm **từng phần** theo số
    sự kiện khớp (A8.1), nên điền bừa một vị trí còn hơn bỏ trống — bỏ trống
    chắc chắn 0, đoán sai cũng 0.

    `su_kien_text` + `kenh1` bật Bước 4 (trích dày — xem
    `lam_day_bang_trich_day`) cho `so_video_trich_day` video ưu tiên đầu
    tiên: cần văn bản gốc của từng sự kiện (để encode) và kênh 1 đang nạp
    (để encode ảnh CÙNG model). Để trống thì bỏ qua Bước 4, giữ đúng hành vi
    cũ — máy chưa có `.mp4` vẫn chạy được bình thường.

    BỐN THAM SỐ A39 (`dong_hang`, `he_so_phat`, `trai_toi_da`, `rai_hep`) —
    xem `chuoi_trake.py`. **Mặc định cả bốn giữ nguyên hành vi cũ từng đẻ ra
    bài nộp 3,8 điểm**, cố ý: kỷ luật đo của dự án là chưa thắng trên tập dev
    thì chưa được bật. Đo bằng `scripts/35_do_chuoi_trake.py`.
    """
    from thoi_gian import xep_video_theo_chuoi
    n = len(cac_su_kien)
    if n == 0:
        return []

    # Nhập vô điều kiện: `rai_hep` bật được ĐỘC LẬP với `dong_hang` — đó là
    # cả điểm của việc đo (chỉ đổi MỘT thứ mỗi lần).
    from chuoi_trake import (TRAI_TOI_DA_GIAY, dong_hang_theo_thoi_gian,
                             rai_deu_hep)
    han_trai = TRAI_TOI_DA_GIAY if trai_toi_da is None else trai_toi_da
    # `row_id` trùng vị trí dòng trong bảng cái (kiểm ở A39) -> tra O(1), khỏi dict.
    pts_theo_row = master.pts_time.values
    fps_theo_video = master.groupby("video_id").fps.first()

    # xep_video_theo_chuoi() đã trả về MỌI video xuất hiện ở bất kỳ sự kiện
    # nào, xếp theo điểm mềm (thưởng video có sự kiện lân cận cũng khớp) —
    # không cần bước "nới ra" thủ công như bản dùng video_du_chuoi() (giao
    # cứng) trước đây.
    uu_tien = xep_video_theo_chuoi(cac_su_kien)

    # Khoảng frame thật của từng video, để điền chỗ trống cho có nghĩa.
    bien = master.groupby("video_id").frame_idx.agg(["min", "max"])

    # video nào CÓ file .mp4 thật — tra một lần, dùng lại cho mọi vòng lặp
    video_path_that = {}
    if su_kien_text and kenh1 is not None:
        co_video = master[master.video_path.notna()]
        video_path_that = dict(zip(co_video.video_id, co_video.video_path))

    ra = []
    so_da_trich_day = 0
    for vid in uu_tien[:so_dong]:
        # Bước 1 + 5: DP chọn khung cho từng sự kiện trong video này, GIỮ ĐÚNG
        # chỉ số sự kiện và đã ép tăng dần ngay trong DP (xem dong_hang_dp).
        cac_ung_vien_trong_video = []
        thoi_diem: dict = {}          # frame_idx -> pts_time, cho prior A39
        for ds in cac_su_kien:
            trong: dict = {}
            for c in ds:
                if c.video_id == vid and (c.frame_idx not in trong
                                          or c.score > trong[c.frame_idx]):
                    trong[c.frame_idx] = c.score
                    thoi_diem[c.frame_idx] = float(pts_theo_row[c.row_id])
            top = sorted(trong.items(), key=lambda x: -x[1])[:K_UNG_VIEN_MOI_SU_KIEN]
            cac_ung_vien_trong_video.append(top)

        # Bước 4: trích dày CHỈ cho vài video ưu tiên đầu (chi phí ffmpeg +
        # encode không rẻ) và CHỈ khi máy có file .mp4 thật cho video này.
        if (so_da_trich_day < so_video_trich_day and vid in video_path_that):
            lam_day_bang_trich_day(vid, video_path_that[vid], su_kien_text,
                                   cac_ung_vien_trong_video, kenh1, master)
            so_da_trich_day += 1

        fps_v = float(fps_theo_video.get(vid, 25.0))
        if dong_hang == "thoi_gian":
            # Khung do Bước 4 trích thêm KHÔNG có trong bảng cái nên không có
            # `pts_time` -> suy từ `frame_idx / fps`. Được phép ở ĐÂY vì con số
            # này chỉ vào prior khoảng cách, KHÔNG BAO GIỜ được nộp; cái cấm
            # suy lại (CLAUDE.md) là `frame_idx` — chiều ngược lại.
            bo_ba = [[(f, thoi_diem.get(f, f / fps_v), s) for f, s in ds]
                     for ds in cac_ung_vien_trong_video]
            tot = dong_hang_theo_thoi_gian(
                bo_ba, he_so_phat=he_so_phat, trai_toi_da=han_trai,
                k_moi_su_kien=K_UNG_VIEN_MOI_SU_KIEN)
        else:
            tot = dong_hang_dp(cac_ung_vien_trong_video)

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

        # Bước 3: DP (bước trên) đã chọn khung tăng dần cho các sự kiện CÓ
        # ứng viên, giữ đúng vị trí — KHÔNG sort ở đây nữa (sort theo giá trị
        # sẽ hoán đổi nhầm khung giữa các sự kiện, xem docstring dong_hang_dp).
        # Khung nội suy ở bước 2 nằm giữa hai neo DP nên vẫn tăng dần tự nhiên.
        khung = [int(x) for x in tot]

        # N sự kiện KHÔNG THỂ nằm gọn trong vài phần trăm giây. Nếu các khung
        # dồn vào một chỗ thì truy hồi đã không phân biệt được các sự kiện —
        # giữ nguyên là dồn hết cơ hội vào một điểm. Rải đều trên khoảng frame
        # của video cho mỗi sự kiện một cửa độc lập; với cửa sổ BTC rộng tới
        # vài phút (A9) thì rải đều có xác suất trúng thật.
        #
        # ⚠️ `don_cuc="tong"` (bản đầu) xét TỔNG ĐỘ TRẢI `khung[-1]-khung[0]`,
        # nên **không bắt được dồn cục MỘT PHẦN**: ba sự kiện đầu rơi trùng một
        # khung còn sự kiện cuối ở xa thì tổng độ trải vẫn lớn, chốt không nổ,
        # và bước ép tăng dần bên dưới đẻ ra `0,1,2,2298` — ba sự kiện cách nhau
        # 0,03 giây. Đo trên bài nộp thật: **47/100 dòng** ở `query-p1-18-trake`
        # và 33/100 ở `query-p1-4-trake` có cặp liền kề dưới 100 frame.
        # `don_cuc="cap"` xét TỪNG CẶP liền kề, và xoá sạch 17/17 dòng dồn cục
        # trong phép đo. NHƯNG **mặc định vẫn là `"tong"`**: đo trên 13 câu
        # TRAKE với ứng viên kênh 3 thì cả ba chính sách đều **0,0000** — không
        # có bằng chứng nào để rời khỏi hành vi đã sinh ra bài nộp 3,8 điểm.
        # Đo lại khi có ứng viên kênh 1 (SigLIP2), nơi TRAKE mới có điểm để so.
        #
        # ⚠️ A39 — RẢI KHẮP VIDEO LÀ SAI, và đo được sai bao nhiêu. Độ trải
        # thật của một chuỗi TRAKE chiếm trung vị **10%** (max 20%) độ dài
        # video, nên rải đều trên `[lo, hi]` đẩy các sự kiện ra xa gấp 5–10
        # lần. `rai_hep=True` rải trong cửa sổ 56,6 s (trung vị đo được) quanh
        # neo mà DP thật sự chọn được. Mặc định vẫn False cho tới khi thắng
        # trên tập dev.
        if n > 1 and hi > lo and _don_cuc(khung, don_cuc):
            if rai_hep and co:
                neo_vt, neo_f = co[0]
                khung = rai_deu_hep(neo_f, neo_vt, n, fps_v, lo, hi)
            else:
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
    ap.add_argument("--matrix", default="clip_gopt.npy",
                    help="ma trận kênh 1. gopt 0,3160 trên đề thật, SigLIP2 "
                         "chỉ 0,1400 — hơn 2,3 lần (A47). clip.npy 0,0000 "
                         "trên tiếng Việt (A10). Đi kèm truy_van_gopt.npz")
    ap.add_argument("--k", type=int, default=TOI_DA_DONG)
    ap.add_argument("--cache", default=None, metavar="FILE.npz",
                    help="vector truy vấn đã mã hoá sẵn — chạy kênh 1 mà "
                         "KHÔNG nạp model, dùng được trên máy thiếu RAM. "
                         "Sinh bằng scripts/25_ma_hoa_truy_van.py")
    ap.add_argument("--loc-cung", action="store_true",
                    help="chen ung vien khop CUNG token hiem len dau (A8.5). "
                         "CHUA DO DUOC tren tap dev — xem docstring loc_cung()")
    ap.add_argument("--kenh", default="anh", choices=("anh", "objects"),
                    help="objects = KHONG can model lon, chay duoc tren may "
                         "thieu RAM. Chat luong kem han (0,0412 so voi 0,3258) "
                         "nhung ra file dung dinh dang")
    # A45: BẬT MẶC ĐỊNH. Trên tập đề THẬT, RRF(kênh 1, kênh 3) trọng số 1:1
    # được +0,0694/+0,0735 so với kênh 1 một mình, vượt nhiễu ở ±2s, thắng 8
    # thua 4 — và cùng dấu ở CẢ HAI nửa của nhóm đối chứng.
    # A14/A17 đo được điều NGƯỢC LẠI (−0,0144) nhưng đo trên tập dev tự soạn,
    # thứ đã bị chứng minh là thổi phồng kênh 1 gấp 2,3 lần.
    ap.add_argument("--khong-hop-nhat", dest="hop_nhat", action="store_false",
                    help="TẮT RRF, chỉ dùng kênh 1 một mình — hành vi trước A45")
    ap.set_defaults(hop_nhat=True)
    ap.add_argument("--co-metadata", dest="bo_metadata", action="store_false",
                    help="cộng thêm kênh 2 vào RRF. Kênh 2 được 0,0000 ở ±2s "
                         "(A12) nên mặc định BỎ — cộng vào là pha loãng (A14.2)")
    ap.set_defaults(bo_metadata=True)
    ap.add_argument("--khong-rrf-menh-de", dest="rrf_menh_de",
                    action="store_false",
                    help="quay lại gộp mệnh đề bằng MAX COSINE. A51 đo trên 52 "
                         "câu đề thật: RRF hạng + kênh 3 hơn max-cosine + kênh 3 "
                         "+0,0721/+0,0971 ✅ ỔN ĐỊNH. Chỉ tắt để tái lập số cũ")
    ap.add_argument("--trong-so-phu", type=float, default=0.5,
                    help="trọng số kênh phụ trong RRF. MẶC ĐỊNH 0,5 (A52). "
                         "0,5 hơn 0,75 ở CẢ BA nhóm câu, không nhóm nào phản "
                         "đối; và trên đề thật 0,5 hơn BỎ HẲN kênh 3 +0,0394 "
                         "✅ ỔN ĐỊNH (1-8-43). Từ 1,0 trở lên là có hại ✅ ở cả "
                         "ba nhóm — 0,75 (A50) chọn khi truy vấn còn bị cắt cụt")
    ap.add_argument("--hop-nhat-chi-cau-ngan", action="store_true",
                    help="CHỈ áp RRF cho câu <=1 mệnh đề. ⚠️ A45 BÁC cờ này: "
                         "đo trên 49 câu đề THẬT (trung vị 62 từ, 2,29 mệnh "
                         "đề) thì RRF thắng ✅ ỔN ĐỊNH, tức thắng trên chính "
                         "loại câu dài mà A34 tưởng là thua. Giữ lại để dựng "
                         "lại phép đo cũ, đừng bật khi đi thi")
    ap.add_argument("--tra-loi", default="",
                    help="chuỗi `answer` DỰ PHÒNG cho dòng Q&A không đào được "
                         "gì. Sai đáp án = 0 điểm, nhưng vẫn phải nộp")
    ap.add_argument("--rai-bien-the", type=int, default=0, metavar="K",
                    help="Q&A: phát K biến thể `answer` cho mỗi khung trong 10 "
                         "dòng đầu, thay vì 1. A83 đo 0,0462 -> 0,1077 "
                         "(gấp 2,3 lần, 0 câu thua) nhưng 🟡 vì toàn bộ hiệu "
                         "ứng là MỘT câu trên 13. Dự bị: BTC không phạt dòng "
                         "sai nên đây là cược một chiều. K=2 là đủ — A83 đo "
                         "K=3 và 30 khung đều cho kết quả Y HỆT")
    ap.add_argument("--khong-dao-dap-an", action="store_true",
                    help="TẮT việc đào `answer` riêng cho từng dòng Q&A, quay "
                         "lại dùng một chuỗi chung (--tra-loi). A67 đo được "
                         "đào riêng đúng 1/13 câu còn chuỗi chung 0/13, nên "
                         "chỉ tắt để dựng lại số cũ")
    ap.add_argument("--so-su-kien", type=int, default=0,
                    help="ép số sự kiện TRAKE, thay vì tự tách từ đề")
    ap.add_argument("--trake-cu", action="store_true",
                    help="quay lại cách lắp TRAKE CŨ (1 dòng mỗi video). Mặc "
                         "định dùng K-best beam search (A79): trên 20 câu, "
                         "cách cũ thua −0,0990 ở ±2s, 2 thắng / 11 thua, vượt "
                         "ngưỡng nhiễu -> ✅ ỔN ĐỊNH. Cờ này để dựng lại bài "
                         "nộp cũ khi cần đối chiếu")
    # --- Mũi nhọn 1 (Giai đoạn 2). Cả ba mặc định TẮT — xem A22 ------------
    ap.add_argument("--uu-tien-video", type=int, default=0, metavar="N",
                    help="BƯỚC 1: đưa ứng viên thuộc top-N video của BM25 "
                         "metadata lên trước. Đo được +0,0095 ở N=50 nhưng "
                         "DƯỚI ngưỡng nhiễu (A22) — chưa đủ căn cứ bật")
    ap.add_argument("--thu-hep-cung", action="store_true",
                    help="BƯỚC 1 dạng CỨNG: bỏ hẳn ứng viên ngoài top-N. ĐÃ ĐO "
                         "LÀ TỆ HƠN (−0,0286): metadata chỉ phủ 37%% số câu ở "
                         "top-50 nên cắt cứng là mất 63%% (A22)")
    ap.add_argument("--dedup", nargs="?", type=float, const=0.99, default=None,
                    metavar="NGUONG",
                    help="BƯỚC 2b: khử keyframe gần trùng trước khi cắt 100. "
                         "ĐÃ ĐO: ĐẢO DẤU giữa hai mức dung sai (A22) — không "
                         "dùng để quyết")
    ap.add_argument("--vlm", action="store_true",
                    help="BƯỚC 4: VLM nhìn 3 khung quanh ứng viên hạng 1 rồi "
                         "sinh `answer` cho gói Q&A. Cần Ollama + model NHÌN "
                         "ĐƯỢC ẢNH (`ollama pull qwen2.5vl:7b`)")
    ap.add_argument("--vlm-so-khung", type=int, default=3,
                    help="số khung đưa vào VLM (PHẦN D Bước 4: 3-5)")
    ap.add_argument("--vlm-so-ung-vien", type=int, default=1,
                    help="hỏi VLM trên N ứng viên đầu ở N video khác nhau rồi "
                         "lấy đa số. Đắt gấp N lần, chưa đo được cái nào hơn")
    ap.add_argument("--trich-day", action="store_true",
                    help="TRAKE Bước 4: trích thêm khung quanh ứng viên tốt "
                         "nhất mỗi sự kiện (cần file .mp4 thật + giữ model "
                         "kênh 1 lâu hơn, tốn thêm RAM/thời gian — xem "
                         "lam_day_bang_trich_day). Video nào thiếu .mp4 thì "
                         "tự bỏ qua, không lỗi")
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

    co_trake = any(loai_cua(t) == "trake" for t in de)
    giu_kenh = bool(a.trich_day and co_trake and a.kenh != "objects")
    kenh1_obj = None
    if a.kenh == "objects":
        kq1, master = quet_objects(a.index, de, a.k)
    elif giu_kenh:
        kq1, master, kenh1_obj = quet_anh(a.index, a.matrix, de, a.k,
                                          rrf_menh_de=a.rrf_menh_de,
                                          giu_kenh=True, cache=a.cache)
    else:
        kq1, master = quet_anh(a.index, a.matrix, de, a.k, cache=a.cache,
                               rrf_menh_de=a.rrf_menh_de)
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

    # Bước 1 cần kênh 2 (nhẹ: 873 tài liệu). Dựng MỘT lần, không dựng trong
    # vòng lặp — `tu_metadata` phải quét cả bảng cái mỗi lần gọi.
    k2_uu = None
    if a.uu_tien_video:
        from bm25 import KenhVanBan
        k2_uu = KenhVanBan.tu_metadata(master)
        print(f"  bước 1: ưu tiên top-{a.uu_tien_video} video theo metadata"
              f"{' (CỨNG — bỏ hẳn phần ngoài)' if a.thu_hep_cung else ''}")

    # Bước 2b chỉ so ẢNH với ảnh, nên ma trận nào cũng được — mmap để không nạp
    # cả 390 MB vào RAM chỉ để đọc vài trăm dòng.
    mat_dedup = None
    if a.dedup:
        import numpy as np
        p = a.index / a.matrix
        if not p.exists():
            p = a.index / "clip.npy"
        mat_dedup = np.load(p, mmap_mode="r")
        print(f"  bước 2b: dedup ngưỡng {a.dedup} ({p.name})")

    goi, so_su_kien = {}, {}
    for ten in sorted(de):
        loai = loai_cua(ten)
        if loai == "trake":
            ds = kq1[ten]
            sk = tach_su_kien(de[ten])
            if a.so_su_kien and len(ds) != a.so_su_kien:
                ds = (ds + [ds[-1]] * a.so_su_kien)[:a.so_su_kien]
                sk = (sk + [sk[-1]] * a.so_su_kien)[:a.so_su_kien]
            # A79: 100 dòng = 100 GIẢ THUYẾT về video tốt nhất, không phải 100
            # VIDEO khác nhau. Trên 20 câu TRAKE, cách cũ thua −0,0990 ở ±2s
            # (2 thắng / 11 thua, vượt ngưỡng nhiễu) -> ✅ ỔN ĐỊNH.
            #
            # ⚠️ `dung_trake()` KHÔNG bị đổi — nó là mốc nền "CŨ" mà `75_`,
            # `78_`, `92_` đang so với. Đổi nó là làm mốc nền trôi theo, và
            # mọi phép đo TRAKE cũ thành không so lại được.
            if a.trake_cu:
                goi[ten] = (dung_trake(ds, master, a.k, su_kien_text=sk,
                                       kenh1=kenh1_obj) if giu_kenh
                            else dung_trake(ds, master, a.k))
            else:
                from kbest_trake import lap_trake
                goi[ten] = lap_trake(ds, master, a.k)
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
            cau_ngan = len(tach_truy_van(de[ten])) <= 1
            if a.hop_nhat and phu.get(ten) and (
                    cau_ngan or not a.hop_nhat_chi_cau_ngan):
                from rrf import hop_nhat
                ds = [uv] + phu[ten]
                uv = hop_nhat(ds, trong_so=[1.0] + [a.trong_so_phu] * len(phu[ten]))
            elif a.hop_nhat and a.hop_nhat_chi_cau_ngan and phu.get(ten):
                print(f"     {ten}: câu dài (bị tách mệnh đề) — bỏ qua "
                      f"--hop-nhat, giữ kênh 1 một mình (A34)")

            # --- Mũi nhọn 1, Bước 1 và 2b. Cả hai MẶC ĐỊNH TẮT (A22) ---------
            if k2_uu is not None:
                from mui_nhon_1 import thu_hep, uu_tien, video_uu_tien
                vids = video_uu_tien(k2_uu, tach_truy_van(de[ten]),
                                     a.uu_tien_video)
                uv = (thu_hep(uv, vids) if a.thu_hep_cung
                      else uu_tien(uv, vids))
            if mat_dedup is not None:
                from dedup import gom_ban_sao
                uv = gom_ban_sao(uv, mat_dedup, nguong=a.dedup)

            uv = bu_cho_du(uv, master, a.k)

            # --- Bước 3b: đào `answer` cho TỪNG DÒNG từ OCR/ASR của chính
            # khung đó (A67). Chạy TRƯỚC VLM để VLM ghi đè được nếu bật.
            #
            # ⚠️ VÌ SAO KHÔNG DÙNG MỘT CHUỖI CHUNG. BTC chấm `answer` theo TỪNG
            # DÒNG. `--tra-loi` một chuỗi cho cả 100 dòng nghĩa là hoặc trúng
            # cả trăm hoặc trắng cả trăm — vứt đi đúng cơ chế ăn điểm từng dòng.
            #
            # ⚠️ ĐÂY LÀ BẢN VÁ, KHÔNG PHẢI LỜI GIẢI. Đo trên 13 câu Q&A đề
            # thật: đào bằng regex đúng 1/13, trần của mọi cách đào từ văn bản
            # là 7/13 (sáu câu còn lại phải NHÌN ẢNH). Nhưng BTC không phạt đáp
            # án sai, nên 1/13 vẫn hơn 0/13 — xem A67.
            if loai == "qa" and not a.khong_dao_dap_an:
                from dap_an import gan_cho_moi_dong
                if "van_qa" not in locals():
                    _b = pd.read_parquet(a.index / "ocr_asr.parquet")
                    van_qa = {int(r): f"{o} {s}".strip() for r, o, s in zip(
                        _b.row_id.values, _b.ocr_text.fillna("").values,
                        _b.asr_text.fillna("").values)}
                if a.rai_bien_the > 1:
                    from dap_an import rai_bien_the
                    uv, n_dao = rai_bien_the(
                        uv, de[ten], van_qa, k=a.rai_bien_the,
                        mac_dinh=a.tra_loi or "không rõ", gioi_han=a.k)
                else:
                    n_dao = gan_cho_moi_dong(uv[:a.k], de[ten], van_qa,
                                             mac_dinh=a.tra_loi or "không rõ")
                print(f"     đào `answer`: {n_dao}/{min(len(uv), a.k)} dòng "
                      f"có chuỗi từ chính khung đó")

            # --- Bước 4: VLM sinh `answer`, chỉ cho gói Q&A ------------------
            if loai == "qa" and a.vlm:
                from mui_nhon_1 import gan_dap_an
                uv = gan_dap_an(uv[:a.k], master, de[ten],
                                so_khung=a.vlm_so_khung,
                                mac_dinh=a.tra_loi or "không rõ",
                                so_ung_vien=a.vlm_so_ung_vien)
                print(f"     VLM trả lời {ten}: {uv[0].meta['answer']!r}")

            goi[ten] = tu_ung_vien(uv, loai, dap_an=a.tra_loi, gioi_han=a.k)
        print(f"  {ten:<20} {len(goi[ten]):>3} dòng")

    if kenh1_obj is not None:      # giu_kenh=True: giải phóng khi đã dùng xong
        del kenh1_obj
        gc.collect()

    if (any(loai_cua(t) == "qa" for t in de) and not a.tra_loi.strip()
            and not a.vlm and a.khong_dao_dap_an):
        raise SystemExit(
            "\n❌ Có gói Q&A nhưng chưa có `answer`.\n"
            "   BTC chấm: khung đúng NHƯNG answer sai -> 0 điểm. Bỏ trống là\n"
            "   chắc chắn 0, và `nop_bai.soat` sẽ chặn.\n\n"
            "   Bỏ --khong-dao-dap-an để đào `answer` từ OCR/ASR của từng khung,\n"
            "   hoặc --tra-loi \"không rõ\" để nộp được mà kiểm định dạng.")
    if any(loai_cua(t) == "qa" for t in de) and not a.vlm:
        print("\n⚠️ Q&A: `answer` đào bằng regex từ OCR/ASR — A67 đo được ĐÚNG\n"
              "   1/13 câu, và trần của mọi cách đào từ văn bản là 7/13 (sáu\n"
              "   câu còn lại phải NHÌN ẢNH). Đây là bản vá để không dòng nào\n"
              "   bỏ trống, KHÔNG phải lời giải. Có VLM thì bật --vlm.")

    d = ghi_goi(goi, a.ra, so_su_kien)

    # Ghi lại ĐÚNG lệnh đã chạy, cạnh các file CSV.
    #
    # ⚠️ A23: bản nộp ăn 2,6 điểm trên leaderboard KHÔNG dựng lại được bằng lệnh
    # ghi trong sổ tay — nó chạy với `--trong-so-phu 1.0` còn mặc định lúc đó là
    # 0,3, và khác biệt đó ra **19/24 file khác nhau**. Không ai biết cho tới khi
    # dựng lại và so từng byte. Điểm ngoài mà không truy nguyên được về một lệnh
    # thì nó không dạy ta điều gì cả.
    #
    # `dong_goi` chỉ nén `*.csv` nên file này KHÔNG lọt vào bài nộp.
    (d / "lenh_da_chay.txt").write_text(
        "# Lệnh đã sinh ra các file .csv trong thư mục này.\n"
        "# Không nằm trong zip nộp BTC (dong_goi chỉ nén *.csv).\n\n"
        # Bọc ngoặc kép cho tham số có khoảng trắng — `--tra-loi không rõ`
        # chép nguyên si sẽ chạy sai, mà chép nguyên si là cách nó sẽ được dùng.
        f"python {' '.join(f'\"{x}\"' if ' ' in x else x for x in sys.argv)}\n",
        encoding="utf-8")

    print(f"\n✅ Đã ghi {d}")
    if a.nen:
        print(f"   Đã nén -> {dong_goi(d, a.nen)}")
    else:
        print(f"   Soát + nén: python src/nop_bai.py --soat {d} --nen bai_nop.zip")


if __name__ == "__main__":
    main()
