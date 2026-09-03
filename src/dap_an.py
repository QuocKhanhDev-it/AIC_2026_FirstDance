"""
dap_an.py — Điền chuỗi `answer` cho TỪNG DÒNG câu Q&A.

VÌ SAO MODULE NÀY TỒN TẠI (A67)

BTC chấm Q&A theo TỪNG DÒNG: mỗi dòng mang `answer` riêng và chỉ ăn điểm khi
đúng CẢ khung LẪN chuỗi. Nhưng `run.py` chỉ có `--tra-loi` — **một chuỗi dùng
chung cho cả 100 dòng**. Nên bài nộp thật hoặc trúng cả trăm dòng (nếu người
chạy gõ đúng) hoặc **trắng cả trăm dòng**, bất kể truy hồi tốt đến đâu.

Và thước đo không lộ ra chuyện đó: `cham_diem._dung_dap_an()` coi ứng viên
KHÔNG CÓ `answer` là hợp lệ, nên mọi điểm Q&A trong repo được chấm như thể đáp
án luôn đúng.

⚠️ ĐÂY LÀ BẢN VÁ TẠM, KHÔNG PHẢI LỜI GIẢI. Đo trên 13 câu Q&A đề thật:

    đáp án có mặt trong OCR/ASR của khung đúng   7/13   <- trần của mọi cách
                                                          đào từ văn bản
    đào bằng regex, chọn theo từ khoá câu hỏi    1/13

Sáu câu còn lại phải NHÌN ẢNH mới trả lời được. Và ngay cả 7 câu có chữ thì
chọn đúng con số nào vẫn là bài toán ĐỌC HIỂU: OCR bản tin đầy dấu thời gian
("06:30:11"), số hiệu kênh, ngày tháng.

Lời giải thật là VLM đọc ảnh, hoặc LLM đọc `câu hỏi + văn bản của chính khung
đó`. Module này chỉ bảo đảm **không dòng nào bỏ trống** — 1/13 vẫn hơn 0/13, và
BTC không phạt đáp án sai.
"""

import re
import unicodedata

# Số kèm đơn vị hay gặp trong đề: "200g", "2,15", "46", "1204"
SO = re.compile(r"\b\d{1,4}(?:[.,]\d+)?\s?(?:g|kg|ml|l|%)?\b", re.I)
# Tên riêng tiếng Việt: 1–3 từ viết hoa liên tiếp
HOA = r"A-ZĐÂÊÔƯĂÁÀÃẢẠÉÈẼẺẸÍÌĨỈỊÓÒÕỎỌÚÙŨỦỤÝỲỸỶỴ"
THUONG = (r"a-zàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớ"
          r"ờởỡợụủứừửữựỳỵỷỹ")
TEN = re.compile(rf"\b[{HOA}][{THUONG}]+(?:\s+[{HOA}][{THUONG}]+){{0,2}}\b")
HOI_SO = re.compile(r"\b(bao nhiêu|mấy|số|khối lượng|gam|kg|phần trăm|"
                    r"nhiệt độ|năm|giờ|độ|lít)\b", re.I)
# Từ quá phổ biến, không đáng làm neo khi dò vị trí
DUNG = set("la gi cua o trong va co nao bao nhieu may mot hai ba cho nguoi "
           "duoc nhung khi tren duoi sau truoc".split())


def bo_dau(s) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def co_dau(s) -> bool:
    """Chuỗi có mang dấu tiếng Việt không (dấu thanh hoặc dấu mũ)."""
    return any(unicodedata.category(c) == "Mn"
               for c in unicodedata.normalize("NFD", str(s)))


def uu_tien_co_dau(chon: str, moi_uv: list) -> str:
    """Cùng một chuỗi mà có cả bản CÓ DẤU lẫn bản KHÔNG DẤU thì lấy bản có dấu.

    VÌ SAO — A68 đo được `ocr_text` chỉ **31% có dấu** còn `asr_text` **100%**.
    `run.py` ghép `OCR + " " + ASR` nên cùng một thực thể thường xuất hiện HAI
    lần trong cùng chuỗi văn bản: `Ta Pua` (OCR) rồi `Tà Pứa` (ASR). Cách chọn
    theo khoảng cách không phân biệt hai bản, mà OCR đứng trước nên hay thắng.

    BTC khớp CHUỖI, nên `Ta Pua` là 0 điểm dù đúng khung và đúng thực thể. Hiện
    **0/13 câu khớp đúng dấu, 7/13 chỉ khớp khi bỏ dấu** — nghĩa là khoảng cách
    giữa hai con số đó phần lớn là chuyện DẤU, không phải chuyện nội dung.

    Đây KHÔNG phải phục hồi dấu: nó không đoán dấu cho chuỗi chưa có bao giờ.
    Nó chỉ chọn đúng bản khi bản có dấu **đã nằm sẵn trong cùng văn bản**.
    """
    if co_dau(chon):
        return chon
    goc = bo_dau(chon)
    for x in moi_uv:
        if x != chon and co_dau(x) and bo_dau(x) == goc:
            return x
    return chon


def dao(van: str, cau_hoi: str) -> str:
    """Chuỗi `answer` ứng viên từ văn bản CỦA CHÍNH keyframe đó. '' nếu chịu.

    Chọn ứng viên GẦN NHẤT với một từ khoá của câu hỏi, không lấy ứng viên đầu
    tiên: OCR bản tin mở đầu bằng đồng hồ và tên kênh, nên "số đầu tiên" gần
    như luôn là giờ phát sóng.
    """
    if not van:
        return ""
    hoi_so = bool(HOI_SO.search(cau_hoi))
    mau = SO if hoi_so else TEN
    # ⚠️ LOẠI ứng viên vốn là chữ CỦA CÂU HỎI. Câu "tên phóng viên là gì" đứng
    # cạnh chữ "Phóng viên" trong OCR, nên chọn-theo-khoảng-cách sẽ trả về
    # đúng cái từ khoá vừa dùng để dò. Đáp án không bao giờ là từ đã có sẵn
    # trong câu hỏi — test `test_hoi_ten_thi_lay_ten_rieng` chốt chỗ này.
    q = bo_dau(cau_hoi)
    uv = [(m.group().strip(), m.start()) for m in mau.finditer(van)
          if len(m.group().strip()) > (0 if hoi_so else 2)
          and bo_dau(m.group().strip()) not in q]
    if not uv:
        return ""

    v = bo_dau(van)
    khoa = [w for w in bo_dau(cau_hoi).split() if len(w) > 2 and w not in DUNG]
    vi_tri = [m.start() for w in khoa for m in re.finditer(re.escape(w), v)]
    moi_uv = [x[0] for x in uv]
    if not vi_tri:
        return uu_tien_co_dau(uv[0][0], moi_uv)
    chon = min(uv, key=lambda x: min(abs(x[1] - p) for p in vi_tri))[0]
    return uu_tien_co_dau(chon, moi_uv)


def gan_cho_moi_dong(ung_vien: list, cau_hoi: str, van_theo_row: dict,
                     mac_dinh: str = "") -> int:
    """Gắn `answer` vào `meta` của TỪNG ứng viên. Trả số dòng đào được.

    `mac_dinh` dùng cho dòng không đào ra gì — để trống là chắc chắn 0 điểm,
    mà BTC không phạt đáp án sai, nên điền bừa vẫn hơn.
    """
    n = 0
    for c in ung_vien:
        tra = dao(van_theo_row.get(c.row_id, ""), cau_hoi)
        if tra:
            n += 1
        c.meta["answer"] = tra or mac_dinh
    return n
