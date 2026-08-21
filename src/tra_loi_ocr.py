"""
tra_loi_ocr.py — Trả lời câu Q&A bằng CHỮ ĐÃ ĐỌC ĐƯỢC trên khung hình.

VÌ SAO ĐƯỜNG NÀY, KHÔNG PHẢI VLM — A24/A25 ĐO ĐƯỢC
===================================================

Ba câu Q&A của bộ đề mẫu, khi mở ra đọc, đều hỏi **chữ hiện trên hình**:

    query-p1-15-qa   tên một XÃ ở Khánh Hoà        -> băng rôn trong khung
    query-p1-19-qa   HAI CÂU THƠ ca ngợi anh hùng  -> câu đối khắc trong đình
    query-p1-22-qa   TÊN MÓN ĂN trên tờ công thức  -> tiêu đề in trên giấy

Không câu nào hỏi số lượng hay màu sắc — tức không câu nào cần *nhìn*. Chúng cần
**đọc**. Mà chữ đó đã nằm sẵn trong `ocr_asr.parquet` (93,2% khung có OCR, 77,4%
có ASR — A25).

Ba hệ quả, và cả ba đều lớn:

1. **Không cần VLM.** Model chỉ phải đọc VĂN BẢN, nên `qwen2.5:3b` có sẵn trên
   máy là đủ — không phải `ollama pull` bản vision 3–6 GB, không đụng chốt RAM.
2. **Không cần ảnh trên đĩa.** `kf_path` chỉ có ở 36.506/177.321 dòng (A5.5), mà
   `ocr_asr.parquet` phủ toàn kho. Đường VLM chết ở mọi nhóm L chưa tải ảnh.
3. **Đo được ngay trên 42 câu Q&A của tập dev**, không cần GPU.

⚠️ Nhưng đây KHÔNG phải đường thay thế VLM cho mọi câu. Câu hỏi đếm hay hỏi màu
thì OCR không giúp gì. Hai đường bổ sung cho nhau; cái này rẻ hơn nên thử trước.

    from tra_loi_ocr import tra_loi_tu_ocr, van_ban_quanh
    vb = van_ban_quanh(master, bang, row_id=175339, so_khung=5)
    dap = tra_loi_tu_ocr("xã này tên là gì", vb)

⚠️ **ĐỪNG DÁN THẲNG OCR THÔ LÀM ĐÁP ÁN.** OCR ra chữ dính và mất dấu
(`Xa Giang Ly.huyen Khanh Vinh`), nên phải có một bước đọc-hiểu ở giữa. Đó là
việc của LLM ở đây, và cũng là lý do `don_dap_an` của `tra_loi.py` được dùng lại
nguyên vẹn: cùng một ràng buộc 100 ký tự, cùng một cách cắt chữ đưa đẩy.
"""

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

from tra_loi import TOI_DA, don_dap_an, nap_khoa      # noqa: E402

API_OLLAMA = "http://localhost:11434/api/generate"

# Model THUẦN VĂN BẢN là đủ — đầu vào đã là chữ. Đây chính là chỗ khác biệt lớn
# so với `tra_loi.py`, nơi model bắt buộc phải nhìn được ảnh.
MODEL_MAC_DINH = "qwen2.5:3b"

# Gemini: nhanh, mạnh hơn hẳn model 3B local trên việc gỡ OCR dính chữ mất dấu.
# ⚠️ Khoá đi qua HEADER `x-goog-api-key`, không phải `?key=` trên URL — đã thử
# và xác nhận trên khoá thật của nhóm (dạng `AQ.A...`, 53 ký tự, khác dạng
# `AIza...` 39 ký tự mà `14_sinh_caption.py` viết cho).
API_GEMINI = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "{model}:generateContent")
MODEL_GEMINI = "gemini-3.1-flash-lite"

NHAC = """Dưới đây là chữ đọc được (OCR) và lời nói (ASR) trong một đoạn video.

--- VĂN BẢN ---
{van_ban}
--- HẾT ---

Câu hỏi: {cau_hoi}

Yêu cầu bắt buộc:
- Trả lời NGẮN NHẤT có thể, chỉ phần cốt lõi được hỏi.
- Chỉ dùng thông tin có trong VĂN BẢN trên. KHÔNG suy đoán, KHÔNG bịa.
- OCR có thể mất dấu hoặc dính chữ — hãy khôi phục lại tiếng Việt có dấu cho
  đúng chính tả khi trả lời.
- Hỏi tên riêng (xã, huyện, người, món ăn) -> chỉ ghi đúng tên đó.
- KHÔNG giải thích, KHÔNG lặp lại câu hỏi.
- Nếu VĂN BẢN không chứa câu trả lời, ghi đúng hai chữ: không rõ

Trả lời:"""


def van_ban_quanh(master: pd.DataFrame, bang: pd.DataFrame, row_id: int,
                  so_khung: int = 5, cot: str = "text") -> str:
    """Gộp văn bản của `so_khung` khung quanh `row_id`, trong CÙNG video.

    Gộp nhiều khung chứ không lấy một: chữ trên màn hình xuất hiện dần (băng
    rôn chạy, tiêu đề hiện ra rồi mất), và OCR mỗi khung bắt được một mẩu khác
    nhau. Đã thấy trên dữ liệu thật — `L27_V010` kf145 đọc được
    `TAO GIANG OANH KHAP THIEN QUY DIA THAN`, còn kf146 mới đủ
    `HONG BAT NHAT KIEN ...`; ghép hai khung mới ra trọn câu đối.

    ⚠️ **TÁCH OCR RA KHỎI ASR, VÀ ĐỂ OCR LÊN TRƯỚC.** Bản đầu dùng thẳng cột
    `text` (đã gộp `ocr + " . " + asr`) và đo được ba lỗi thật ngay trên ba câu
    của đề mẫu:

        "xã này tên gì"      -> "không rõ"      (OCR chứa đáp án bị ASR chôn)
        "hai câu thơ"        -> chỉ ra vế đầu
        "tên món ăn"         -> "Bánh ít trơn"  (nghe theo ASR; OCR ghi TRAN)

    ASR dài trung bình 463 ký tự và **lặp y hệt trên nhiều khung liền nhau**
    (một đoạn lời nói phủ vài keyframe), còn OCR chỉ ~90 ký tự. Gộp phẳng thì
    phần OCR — thứ duy nhất chứa chữ TRÊN MÀN HÌNH, mà đó mới là thứ mấy câu
    này hỏi — bị đẩy xuống giữa một biển lời thoại.
    """
    r = master.iloc[row_id]
    lo = max(0, row_id - so_khung // 2)
    hi = min(len(master) - 1, row_id + so_khung // 2)
    lat = master.iloc[lo:hi + 1]
    lat = lat[lat.video_id == r.video_id]

    ocr, asr, thay = [], [], set()
    for rid in lat.row_id:
        x = bang.iloc[int(rid)]
        for nguon, gio in (("ocr_text", ocr), ("asr_text", asr)):
            phan = " ".join(str(x.get(nguon, "") or "").split())
            if phan and phan not in thay:
                thay.add(phan)
                gio.append(phan)
    if not ocr and not asr:                # bảng chỉ có cột `text` gộp sẵn
        for rid in lat.row_id:
            phan = " ".join(str(bang.iloc[int(rid)].get(cot, "") or "").split())
            if phan and phan not in thay:
                thay.add(phan)
                ocr.append(phan)

    khoi = []
    if ocr:
        khoi.append("CHỮ TRÊN MÀN HÌNH (OCR):\n" + "\n".join(ocr))
    if asr:
        khoi.append("LỜI NÓI (ASR):\n" + "\n".join(asr))
    return "\n\n".join(khoi)


def _goi_ollama(loi_nhac: str, model: str, timeout: int = 120) -> str:
    """Gọi Ollama ở chế độ VĂN BẢN. `temperature=0` — bắt buộc, xem D0.3."""
    body = {"model": model, "prompt": loi_nhac, "stream": False,
            "options": {"temperature": 0.0}}
    req = urllib.request.Request(
        API_OLLAMA, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("response", "").strip()
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError) as e:
        raise RuntimeError(
            f"Ollama không phản hồi ở {API_OLLAMA} ({e}).\n"
            f"   Chạy `ollama serve` và `ollama pull {model}` trước.")


def _goi_gemini(loi_nhac: str, model: str = MODEL_GEMINI, key: str = "",
                timeout: int = 60) -> str:
    """Gọi Gemini ở chế độ VĂN BẢN. Khoá lấy từ `.env` nếu không truyền vào."""
    key = key or nap_khoa()
    if not key:
        raise RuntimeError(
            "Chưa có khoá Gemini. Đặt GEMINI_API_KEY trong .env ở gốc repo "
            "(file đó đã được .gitignore bỏ qua) hoặc trong biến môi trường.")
    body = {"contents": [{"parts": [{"text": loi_nhac}]}],
            "generationConfig": {"temperature": 0.0}}   # D0.3: temp=0 bắt buộc
    req = urllib.request.Request(
        API_GEMINI.format(model=model), data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    try:
        return d["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        # Bị chặn bởi bộ lọc an toàn, hoặc hết quota -> trả rỗng, ĐỪNG ném lỗi
        # giữa một vòng chấm 42 câu.
        return ""


def tra_loi_tu_ocr(cau_hoi: str, van_ban: str, goi=None, backend: str = "ollama",
                   model: str | None = None, toi_da: int = TOI_DA) -> str:
    """Đáp án ngắn rút từ `van_ban`. Rỗng nếu không rút được gì.

    `goi(loi_nhac) -> str` cho phép test bơm hàm giả.
    `backend`: `"ollama"` (local, miễn phí) hoặc `"gemini"` (mạnh hơn hẳn trên
    OCR dính chữ — xem `doan_dia_danh`).
    """
    van_ban = (van_ban or "").strip()
    if not van_ban:
        return ""
    if goi is None:
        if backend == "gemini":
            goi = (lambda ln: _goi_gemini(ln, model or MODEL_GEMINI))
        else:
            goi = (lambda ln: _goi_ollama(ln, model or MODEL_MAC_DINH))
    tho = goi(NHAC.format(van_ban=van_ban[:4000], cau_hoi=cau_hoi))
    return don_dap_an(tho, toi_da)


# Cấp hành chính đứng trước một địa danh trong băng rôn/phụ đề.
#
# ⚠️ `\s*` chứ không phải `\s+`: OCR **dính chữ** rất thường xuyên. Chuỗi thật
# đọc được từ `L30_V072` là `...DEN TRUONG XaGiang Ly.huyen Khanh Vinh...` —
# `Xa` dính liền `Giang`. Đòi khoảng trắng là trượt đúng ca mà luật này sinh ra
# để bắt. Ranh giới kết thúc là dấu chấm/phẩy hoặc cấp hành chính kế tiếp.
_CAP = {"xa": "xã", "xã": "xã", "phuong": "phường", "phường": "phường",
        "thi tran": "thị trấn", "thị trấn": "thị trấn",
        "huyen": "huyện", "huyện": "huyện", "quan": "quận", "quận": "quận",
        "tinh": "tỉnh", "tỉnh": "tỉnh"}
_DIA_DANH = re.compile(
    r"(?:^|[\s.,])(xã|xa|phường|phuong|thị trấn|thi tran|huyện|huyen|"
    r"quận|quan|tỉnh|tinh)\s*"
    r"([A-ZĐÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZĐÀ-Ỹ][\wÀ-ỹ]*){0,3})", re.I)


def doan_dia_danh(van_ban: str, loai: str = "xã") -> str:
    """Rút địa danh theo LUẬT, không cần LLM.

    Vì sao đáng có dù đã có LLM: `qwen2.5:3b` đọc đúng chuỗi OCR chứa
    `XaGiang Ly.huyen Khanh Vinh.tinh Khanh Hod` mà vẫn trả "không rõ" — model
    nhỏ không gỡ nổi chữ dính mất dấu. Luật thì gỡ được, vì nó không cần hiểu.

    Trả rỗng khi không chắc: đoán bừa một địa danh sai cũng 0 điểm y như bỏ
    trống, mà lại che mất việc câu đó chưa được trả lời.
    """
    muon = _CAP.get(str(loai).strip().lower(), loai)
    for m in _DIA_DANH.finditer(van_ban or ""):
        if _CAP.get(m.group(1).lower()) == muon:
            return " ".join(m.group(2).split())
    return ""
