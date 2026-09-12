"""
mui_nhon_1.py — Đường ống KIS & Q&A của GIAI ĐOẠN 2, ghép năm bước thành một.

Kế hoạch (PHẦN D, Mũi nhọn 1) đặt ra sáu bước. Trước file này, `run.py` mới
chạy đúng **một** trong số đó — Bước 2 — nên bảng trạng thái đọc ra toàn dấu ❌
trong khi từng module rời rạc thì đã viết xong từ lâu. Việc của module này là
**nối chúng lại**, mỗi bước một cờ, và để phép đo quyết định cờ nào bật.

    Bước 1   thu hẹp cấp video bằng BM25 metadata      `thu_hep` / `uu_tien`
    Bước 2   truy hồi đa kênh, hợp nhất bằng RRF       `rrf.hop_nhat` (đã có)
    Bước 2b  khử trùng lặp TRƯỚC khi cắt top-K         `dedup.gom_ban_sao`
    Bước 3   tinh chỉnh vị trí frame                   ĐÃ BỎ (A8.1, chấm theo khoảng)
    Bước 3b  khung lân cận                             KHÔNG vào truy hồi — xem dưới
    Bước 4   VLM sinh `answer` cho Q&A                 `gan_dap_an`

VÌ SAO BƯỚC 3b KHÔNG ĐƯỢC NỐI VÀO TRUY HỒI
===========================================

A18 đo rồi: chèn khung lân cận vào danh sách ứng viên **làm TỆ ĐI**, ổn định ở
cả hai mức dung sai. Lý do không phải cài sai mà là **cộng dồn hai luật của
BTC**: điểm chấm theo KHOẢNG rộng 4 giây–5 phút (A9), nên khung cách đáp án
±2 giây **đã được tính đúng rồi**. Chèn nó vào là tiêu một trong 100 chỗ để mua
lại thứ mình đã có, đồng thời đẩy mọi ứng viên phía dưới lùi một bậc.

Nhưng kỹ thuật đó **không vô dụng** — nó chỉ đứng nhầm chỗ. Chỗ đúng của nó là
**Bước 4**: PHẦN D ghi *"3–5 frame trong cửa sổ ±2s"*, mà cửa sổ ±2s quanh một
khung chính là `lan_can`. Ở đó nó không tiêu chỗ nộp nào, chỉ cho VLM nhìn được
nhiều hơn. Nên module này gọi `lan_can` **đúng một lần, trong `khung_ngu_canh`**.

BƯỚC 1 CÓ HAI DẠNG, VÀ DẠNG CỨNG LÀ DẠNG NGUY HIỂM
===================================================

Kế hoạch viết *"BM25 metadata → top-50 video"*, không nói cắt cứng hay xếp lại.
Khác biệt rất lớn:

    thu_hep()   CỨNG — kênh chỉ được nhìn vào top-N video. Video đúng rơi ngoài
                top-N là **mất trắng câu đó**, không cứu lại được.
    uu_tien()   MỀM — chạy kênh trên toàn kho rồi đưa ứng viên thuộc top-N video
                lên trước. Sai thì chỉ xáo thứ tự, **không mất ứng viên nào**.

Cùng nguyên tắc đã ghi ở `objects.py`: *cho điểm mềm, tuyệt đối không lọc cứng*.
Nhưng ở đây phải ĐO chứ không suy — nên có cả hai, và
`scripts/22_do_mui_nhon_1.py` in kèm **độ phủ video ở top-N** để biết dạng cứng
mất bao nhiêu trước khi nó kịp lợi được gì.

Dùng:
    from mui_nhon_1 import truy_hoi, gan_dap_an
    uv = truy_hoi(cau, [k4, k3], k2=k2, so_video=50, ma_tran=mat)
    uv = gan_dap_an(uv, master, cau)          # chỉ Q&A
"""

import numpy as np
import pandas as pd

try:
    from .dedup import gom_ban_sao
    from .lan_can import bien_video, lan_can
    from .rrf import hop_nhat
    from .schema import Candidate
except ImportError:                     # chạy trực tiếp / sys.path có src/
    from dedup import gom_ban_sao
    from lan_can import bien_video, lan_can
    from rrf import hop_nhat
    from schema import Candidate

# PHẦN D: "BM25 metadata -> top-50 video".
SO_VIDEO = 50

# PHẦN D Bước 4: "3-5 frame trong cửa sổ ±2s". Cửa sổ ±2s cũng đúng bằng
# `DUNG_SAI_CHINH` của `cham_diem` — mức HẸP NHẤT BTC nêu.
SO_KHUNG_VLM = 3
CUA_SO_GIAY = 2.0


# ------------------------------------------------------- Bước 1: thu hẹp video

def video_uu_tien(k2, cau, so_video: int = SO_VIDEO) -> list[str]:
    """Top-`so_video` video theo BM25 metadata, tốt xuống kém.

    `k2` là `bm25.KenhVanBan.tu_metadata(master)`. Đọc thẳng điểm TÀI LIỆU chứ
    không gọi `k2.tim()`: `tim()` nở tài liệu ra thành keyframe rồi cắt ở `k`,
    nên hỏi nó "50 video nào" là hỏi vòng — với `moi_video=1` thì `k=50` mới ra
    đủ 50 video, mà con số đó lại phụ thuộc tham số của lời gọi. Ở đây cần
    đúng một xếp hạng video, và `k2` vốn có sẵn nó.

    Video không khớp token nào (điểm 0) **không được vào danh sách**. Đệm cho
    đủ 50 bằng những video không liên quan là bịa ra ưu tiên không có căn cứ.
    """
    d = k2.diem_tai_lieu(cau) if isinstance(cau, str) else np.max(
        [k2.diem_tai_lieu(c) for c in cau], axis=0)
    vids = getattr(k2, "video_id", None)
    if vids is None:
        raise ValueError("kênh 2 phải dựng bằng `KenhVanBan.tu_metadata` "
                         "(cần thuộc tính `video_id`)")
    thu_tu = np.argsort(-d)[:so_video]
    return [vids[i] for i in thu_tu if d[i] > 0]


def be_video(master: pd.DataFrame, video_ids) -> np.ndarray:
    """Mặt nạ bool trên MỌI dòng bảng cái: dòng có thuộc các video này không.

    Truyền vào tham số `be` của `dense.KenhAnh.tim`, `bm25.KenhVanBan.tim` và
    `objects.KenhObjects.tim` — cả ba kênh nhận cùng một kiểu mặt nạ, nên
    "thu hẹp cấp video" cài đúng một lần cho cả ba.
    """
    return master.video_id.isin(set(video_ids)).values


def thu_hep(uv: list, video_ids) -> list:
    """Bước 1 dạng CỨNG, áp lên một danh sách đã có: bỏ mọi ứng viên ngoài top-N.

    ⚠️ Bỏ THẬT, không xếp lại. Video đúng nằm ngoài `video_ids` là câu đó về 0
    và không bước nào phía sau cứu được. Chỉ dùng khi phép đo cho thấy độ phủ
    video ở top-N đủ cao — xem `scripts/22_do_mui_nhon_1.py`.
    """
    cho = set(video_ids)
    return [c for c in uv if c.video_id in cho]


def uu_tien(uv: list, video_ids) -> list:
    """Bước 1 dạng MỀM: ứng viên thuộc top-N video lên trước, phần còn lại giữ
    nguyên phía sau.

    Ổn định trong từng nhóm — thứ tự do các kênh xếp ra vẫn được tôn trọng,
    module này chỉ chia danh sách làm hai rồi nối lại. Nhờ vậy nó **không thể
    làm mất câu nào**: kịch bản xấu nhất là ứng viên đúng bị đẩy xuống nửa sau,
    vẫn nằm trong 100 dòng nộp và vẫn đáng 0,2 điểm (PHẦN C mục 1).

    Trả về danh sách MỚI, không sửa tại chỗ; `meta['uu_tien']` đánh dấu ứng
    viên nào được nâng, để soi lại khi con số trông lạ.
    """
    cho = set(video_ids)
    trong, ngoai = [], []
    for c in uv:
        (trong if c.video_id in cho else ngoai).append(
            Candidate(row_id=c.row_id, video_id=c.video_id,
                      frame_idx=c.frame_idx, score=c.score, source=c.source,
                      meta={**c.meta, "uu_tien": c.video_id in cho}))
    return trong + ngoai


# ------------------------------------------------------------- ghép cả đường ống

def truy_hoi(cau, cac_kenh: list, k: int = 100, k2=None,
             so_video: int = 0, cung: bool = False,
             trong_so: list | None = None,
             ma_tran=None, nguong_dedup: float | None = None,
             moi_video: int | None = None) -> list[Candidate]:
    """Chạy Bước 1 → 2 → 2b cho một truy vấn KIS/Q&A.

    `cac_kenh` là các đối tượng có `.tim(cau, k=...)`. Thứ tự trong danh sách
    KHÔNG quan trọng với RRF, nhưng `trong_so` thì theo đúng thứ tự đó.

    Mọi bước phụ đều **mặc định TẮT** (`so_video=0`, `ma_tran=None`,
    `moi_video=None`): kỷ luật của repo là không bật gì trước khi thắng trên tập
    dev, và ba trong bốn kỹ thuật gần nhất lấy từ bài báo AIC'25 đã bị chính
    phép đo bác bỏ (A11, A14, A18).

    `ma_tran` bật Bước 2b. Nó chỉ dùng để so ẢNH với ảnh nên **dùng `clip.npy`
    được, kể cả khi kênh truy hồi là kênh khác**: CLIP mù tiếng Việt (A10)
    nhưng không mù trong việc nhận ra hai keyframe gần trùng nhau.
    """
    ds = [kenh.tim(cau, k=k) for kenh in cac_kenh]
    uv = ds[0] if len(ds) == 1 else hop_nhat(ds, trong_so=trong_so)

    if so_video and k2 is not None:
        vids = video_uu_tien(k2, cau, so_video)
        uv = thu_hep(uv, vids) if cung else uu_tien(uv, vids)

    # Bước 2b: SAU khi xếp hạng, TRƯỚC khi cắt — PHẦN C mục 6.
    if ma_tran is not None:
        uv = (gom_ban_sao(uv, ma_tran) if nguong_dedup is None
              else gom_ban_sao(uv, ma_tran, nguong=nguong_dedup))

    if moi_video:
        dem, loc = {}, []
        for c in uv:
            if dem.get(c.video_id, 0) < moi_video:
                dem[c.video_id] = dem.get(c.video_id, 0) + 1
                loc.append(c)
        uv = loc
    return uv[:k]


# ------------------------------------------------- Bước 4: VLM trả lời câu Q&A

def khung_ngu_canh(master: pd.DataFrame, row_id: int,
                   so_khung: int = SO_KHUNG_VLM,
                   cua_so_giay: float = CUA_SO_GIAY,
                   bien: dict | None = None) -> list[str]:
    """Đường dẫn ảnh của `so_khung` khung quanh `row_id`, trong cửa sổ ±giây.

    ⚠️ **Đây là chỗ đúng của `lan_can`** — xem đầu file. Ở truy hồi nó làm tệ
    đi (A18) vì tiêu mất chỗ nộp; ở đây nó không tiêu chỗ nào.

    Lọc theo `cach_giay` chứ không theo số bước: mật độ keyframe không đều
    (trung vị 55 frame, p90 150 — A1), nên "10 keyframe quanh đây" có thể là 6
    giây ở video này và 30 giây ở video kia. Câu hỏi đếm mà nhìn sang cảnh khác
    thì đếm ra một con số của cảnh khác.

    Chỉ trả về ảnh **có thật trên máy này**, và hỏi qua `anh.tim()` chứ không
    đọc thẳng `kf_path`. `kf_path` nghĩa là "ảnh GỐC có ở máy này" (A5.5) — nó
    rỗng ở **cả 79.590 dòng của L26**, 45% kho, vì không máy nào giữ 12,13 GB
    ảnh gốc đó. Đọc thẳng cột ấy thì `--vlm` **không bao giờ trả lời được câu
    Q&A rơi vào L26**, mà không có gì báo: `gan_dap_an` chỉ thấy danh sách ảnh
    rỗng rồi rơi về `mac_dinh`. Bản thu nhỏ 256px phủ **100%** kho và đủ để
    NHẬN RA CẢNH.

    ⚠️ Bản thu nhỏ KHÔNG đủ để ĐỌC CHỮ nhỏ, mà 10/14 câu Q&A đề thật là câu đọc
    chữ (docs/15). Nên với câu đọc chữ thì ảnh nhỏ chỉ là đường lùi, không phải
    câu trả lời — xem cảnh báo ở đầu `src/anh.py`.

    Trả về rỗng vẫn là chuyện bình thường, không phải lỗi — `gan_dap_an` xử lý.

    Khung chính (`buoc == 0`) LUÔN đứng đầu nếu có ảnh: VLM đọc ảnh đầu tiên
    kỹ nhất, mà đó mới là khung ta tin.
    """
    from anh import tim as tim_anh

    quanh = lan_can(master, row_id, so_buoc=max(so_khung, 5), bien=bien)
    trong = [c for c in quanh if abs(c.meta["cach_giay"]) <= cua_so_giay]
    trong.sort(key=lambda c: (abs(c.meta["buoc"]), c.meta["buoc"]))

    ra = []
    for c in trong:
        g = master.iloc[c.row_id]
        p, _ = tim_anh(g.kf_path, g.video_id, g.kf_n)
        if p is not None:
            ra.append(str(p))
        if len(ra) >= so_khung:
            break
    return ra


def gan_dap_an(uv: list, master: pd.DataFrame, cau_hoi: str,
               so_khung: int = SO_KHUNG_VLM, cua_so_giay: float = CUA_SO_GIAY,
               goi=None, mac_dinh: str = "không rõ", so_ung_vien: int = 1,
               bien: dict | None = None) -> list[Candidate]:
    """Bước 4 — sinh `answer` rồi gán vào `meta` của MỌI ứng viên.

    PHẦN D: *"giữ nhiều `frame_idx` khác nhau nhưng **cùng một `answer`** nếu
    VLM tự tin"*. Nên mặc định gọi VLM **một lần** trên ứng viên hạng 1 và dùng
    chung đáp án — không phải để tiết kiệm mà vì đó là cách nộp đúng: 100 dòng
    là 100 phỏng đoán về VỊ TRÍ, không phải 100 phỏng đoán về ĐÁP ÁN.

    `so_ung_vien > 1` thì hỏi VLM trên vài ứng viên đầu **thuộc các video khác
    nhau** rồi lấy đáp án chiếm đa số. Đắt gấp N lần, và **chưa đo được cái nào
    hơn** — để mặc định 1 cho tới khi có số.

    `goi(cau_hoi, duong_dan_anh) -> str` cho phép test bơm hàm giả vào; để None
    thì dùng `tra_loi.tra_loi_qa` thật (nạp muộn, vì nó cần Ollama sống).

    ⚠️ **Không ứng viên nào được để `answer` rỗng.** `nop_bai.soat` chặn dòng
    Q&A có `answer` rỗng, mà chặn là KHÔNG ghi file nào — một câu hỏi không có
    ảnh trên máy sẽ giết cả gói. Nên mọi đường thoát đều rơi về `mac_dinh`.
    """
    if not uv:
        return uv
    if goi is None:
        from tra_loi import tra_loi_qa
        goi = lambda ch, anh: tra_loi_qa(ch, anh, so_khung=so_khung)  # noqa: E731

    bien = bien or bien_video(master)
    phieu, da_hoi = [], set()
    for c in uv:
        if len(phieu) >= so_ung_vien:
            break
        if c.video_id in da_hoi:
            continue                      # hỏi lại cùng video là hỏi lại cùng cảnh
        anh = khung_ngu_canh(master, c.row_id, so_khung, cua_so_giay, bien)
        if not anh:
            continue
        da_hoi.add(c.video_id)
        tra = (goi(cau_hoi, anh) or "").strip()
        if tra:
            phieu.append(tra)

    if phieu:
        # đa số; hòa thì lấy phiếu của ứng viên xếp trên
        dap = max(sorted(set(phieu), key=phieu.index),
                  key=lambda x: phieu.count(x))
    else:
        dap = mac_dinh

    return [Candidate(row_id=c.row_id, video_id=c.video_id,
                      frame_idx=c.frame_idx, score=c.score, source=c.source,
                      meta={**c.meta, "answer": dap})
            for c in uv]
