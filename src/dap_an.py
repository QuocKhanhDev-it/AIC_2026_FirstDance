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
# ⚠️ `TEN` đòi MỌI từ viết hoa, nên nó KHÔNG BAO GIỜ bắt được danh từ thường
# tiếng Việt kiểu "Cá lóc", "Cá sòng": chỉ khớp "Cá", rồi bị bộ lọc `len > 2`
# gạt nốt. A84 đo được 3/5 câu có đáp án CÓ SẴN trong văn bản mà bộ đào không
# bao giờ chọn — và bảng tra dấu vô dụng vì không bao giờ được đưa ứng viên
# đúng. `TEN_RONG` cho 1 từ hoa + tối đa 2 từ THƯỜNG theo sau.
TEN_RONG = re.compile(
    rf"\b[{HOA}][{THUONG}]+(?:\s+[{HOA}{THUONG}][{THUONG}]+){{0,2}}\b")
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


def bang_tra_ngram(van: str, n_toi_da: int = 4) -> dict:
    """Văn bản CÓ DẤU -> `{dạng bỏ dấu: dạng có dấu}` cho mọi n-gram 1..n.

    VÌ SAO — A68 đo `asr_text` **100% có dấu** còn `ocr_text` chỉ 31%. Nên ASR
    của chính kho này là **một cuốn từ điển có dấu của chính nó**: không cần
    model phục hồi dấu, chỉ cần tra.

    Và A83 đo được `uu_tien_co_dau()` bản đầu **chỉ bắt được khi hai bản có
    CÙNG SỐ TỪ** — mẫu tên riêng bắt tới 3 từ viết hoa liên tiếp nên
    `"Tại Tà Pứa"` không khớp `"Ta Pua"` sau khi bỏ dấu. Quét n-gram 1–4 từ
    không vướng chuyện đó: `"tà pứa"` là một 2-gram riêng.

    Chỉ ghi n-gram CÓ dấu — n-gram không dấu chẳng giúp gì, mà còn đè mất bản
    có dấu nếu vào trước.
    """
    ra = {}
    tu = re.findall(r"[^\W\d_]+", str(van), re.UNICODE)
    for n in range(1, n_toi_da + 1):
        for i in range(len(tu) - n + 1):
            cum = " ".join(tu[i:i + n])
            if co_dau(cum):
                ra.setdefault(bo_dau(cum), cum)
    return ra


def uu_tien_co_dau(chon: str, moi_uv: list, tra: dict | None = None) -> str:
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
    if tra and goc in tra:            # bảng tra n-gram từ ASR (A84)
        return tra[goc]
    return chon


def dao(van: str, cau_hoi: str, rong: bool = False) -> str:
    """Chuỗi `answer` ứng viên từ văn bản CỦA CHÍNH keyframe đó. '' nếu chịu.

    Chọn ứng viên GẦN NHẤT với một từ khoá của câu hỏi, không lấy ứng viên đầu
    tiên: OCR bản tin mở đầu bằng đồng hồ và tên kênh, nên "số đầu tiên" gần
    như luôn là giờ phát sóng.
    """
    if not van:
        return ""
    hoi_so = bool(HOI_SO.search(cau_hoi))
    mau = SO if hoi_so else (TEN_RONG if rong else TEN)
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


def dao_nhieu(van: str, cau_hoi: str, k: int = 3,
              rong: bool = False) -> list[str]:
    """Tối đa `k` chuỗi `answer` ứng viên, xếp theo độ gần từ khoá câu hỏi.

    VÌ SAO CẦN NHIỀU HƠN MỘT — BTC cho 100 dòng và **không phạt dòng sai**, mà
    `dao()` chỉ trả về một chuỗi. Nếu chuỗi đó sai thì cả 100 dòng cùng sai,
    dù ứng viên thứ hai có thể đúng. Rải các biến thể ra nhiều dòng là cách
    khai thác đúng luật chấm — đo ở `96_do_rai_bien_the.py`.

    Trả về theo cùng tiêu chí `dao()`: gần từ khoá câu hỏi nhất đứng trước, và
    ưu tiên bản CÓ DẤU khi có cả hai bản của cùng một chuỗi (A76).
    """
    if not van or k <= 0:
        return []
    hoi_so = bool(HOI_SO.search(cau_hoi))
    mau = SO if hoi_so else (TEN_RONG if rong else TEN)
    q = bo_dau(cau_hoi)
    uv = [(m.group().strip(), m.start()) for m in mau.finditer(van)
          if len(m.group().strip()) > (0 if hoi_so else 2)
          and bo_dau(m.group().strip()) not in q]
    if not uv:
        return []

    v = bo_dau(van)
    khoa = [w for w in bo_dau(cau_hoi).split() if len(w) > 2 and w not in DUNG]
    vi_tri = [m.start() for w in khoa for m in re.finditer(re.escape(w), v)]
    if vi_tri:
        uv.sort(key=lambda x: min(abs(x[1] - p) for p in vi_tri))

    # `rong` bắt được cụm dài hơn đáp án ("Mon Ca loc kho" khi đáp án là
    # "Cá lóc"), nên phát ra cả ĐOẠN CON. Rẻ vì A83 đã có cơ chế rải nhiều biến
    # thể ra nhiều dòng, và BTC không phạt dòng sai.
    if rong:
        them = []
        for s, vt in uv:
            w = s.split()
            for n in range(1, min(3, len(w)) + 1):
                for i in range(len(w) - n + 1):
                    con = " ".join(w[i:i + n])
                    if len(con) > 2 and bo_dau(con) not in q:
                        them.append((con, vt))
        uv = uv + them

    moi_uv = [x[0] for x in uv]
    ra, da = [], set()
    for s, _ in uv:
        s = uu_tien_co_dau(s, moi_uv)
        if bo_dau(s) not in da:              # trùng sau khi bỏ dấu -> bỏ qua
            da.add(bo_dau(s))
            ra.append(s)
        if len(ra) >= k:
            break
    return ra


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


def rai_bien_the(ung_vien: list, cau_hoi: str, van_theo_row: dict,
                 k: int = 2, n_dong: int = 10, mac_dinh: str = "",
                 gioi_han: int = 100):
    """`n_dong` khung đầu phát ra `k` dòng, mỗi dòng một biến thể `answer`.

    BTC cho 100 dòng và **không phạt dòng sai**, mà `dao()` chỉ trả về MỘT
    chuỗi — chuỗi đó sai thì cả 100 dòng cùng sai. A83 đo trên 13 câu Q&A:

        1 biến thể/dòng   0,0462
        2 biến thể, 10 khung đầu   0,1077   (+0,0615, thắng-thua-hoà 1-0-12)
        TRẦN mọi biến thể          0,1231

    Gấp 2,3 lần, **0 câu thua**, nhưng vẫn 🟡: `+0,0615 = 0,8/13`, tức toàn bộ
    hiệu ứng là **một câu trên mười ba**. Nên đây là **cờ dự bị**, không phải
    mặc định — đúng kỷ luật "chưa thắng trên dev thì chưa bật".

    Đánh đổi có thật: `k` biến thể của một khung chiếm chỗ của `k−1` khung
    khác. A83 đo `k=3` và `n_dong=30` cho kết quả **y hệt** `k=2, n_dong=10`,
    nên rải rộng hơn không mua thêm gì.

    Trả về `(danh sách ứng viên mới, số dòng đào được chuỗi thật)`.
    """
    import copy

    ra, n = [], 0
    for i, c in enumerate(ung_vien):
        van = van_theo_row.get(c.row_id, "")
        bien = (dao_nhieu(van, cau_hoi, k) if i < n_dong
                else [dao(van, cau_hoi)])
        for b in (bien or [None]):
            x = copy.copy(c)
            x.meta = {**c.meta, "answer": b or mac_dinh}
            n += bool(b)
            ra.append(x)
            if len(ra) >= gioi_han:
                return ra, n
    return ra, n


TU = re.compile(r"[^\W\d_]+|\d+(?:[.,]\d+)?\s?(?:g|kg|ml|l|%)?", re.UNICODE)


def dao_cum(van: str, cau_hoi: str, idf: dict, k: int = 5,
            n_toi_da: int = 4) -> list[str]:
    """Trích đáp án bằng CHẤM ĐIỂM CỤM, không đòi cụm phải viết hoa.

    VÌ SAO PHẢI BỎ ĐIỀU KIỆN CHỮ HOA (A84)

    `TEN`/`TEN_RONG` chỉ khớp cụm bắt đầu bằng chữ HOA. Nhưng đáp án Q&A của
    kho này phần lớn là **danh từ thường nằm giữa câu**:

        'NGUYEN LIEU Thit ca loc 300g Gao deo 100g …'
        '… Online Nguyên Liệu Thịt cá lóc 300g Gạo dẻo …'   <- VietOCR

    A84 đo được **3/5 câu có sẵn chuỗi đúng trong văn bản mà bộ đào không bao
    giờ chọn**. Nới regex không cứu được vì điều kiện chữ hoa ở từ ĐẦU vẫn còn.

    CÁCH CHẤM, và vì sao IDF là tiêu chí đúng

    Sinh MỌI cụm 1–`n_toi_da` từ rồi xếp hạng — bài toán thành **xếp hạng cụm**
    chứ không phải trích cụm. Điểm một cụm:

        điểm = max(IDF của các token)  −  phạt theo khoảng cách tới từ khoá

    IDF vì đáp án Q&A gần như luôn là **thực thể hiếm** (`Tà Pứa`, `cá lóc`,
    `1204`), còn cụm rác là từ phổ biến (`của`, `trong`, `Online`). Đây đúng
    tiêu chí `objects.py` đã dùng và đo được ở A62.

    ⚠️ `idf` phải là bảng của CHÍNH kho này (`bm25.BM25.idf`), không phải bảng
    tiếng Việt chung: `HTV Online` hiếm trong tiếng Việt nhưng xuất hiện ở
    19.656 khung L26 (A88), nên chỉ bảng của kho mới hạ được nó.

    ⚠️⚠️ **ĐO RỒI: 0/13, TỆ HƠN CẢ REGEX CŨ (3/13).** Giữ hàm này lại để lần
    sau ai nghĩ ra ý "xếp hạng cụm bằng IDF" thì thấy nó đã được thử.

    Cơ chế hỏng, và nó là một bài học chung: **OCR sinh ra rác duy nhất**. Mỗi
    lần đọc sai một ký tự là một token chỉ xuất hiện MỘT lần trong cả kho — tức
    IDF cực đại. Nên xếp theo IDF là xếp rác OCR lên đầu, còn đáp án thật
    (`cá lóc`, `1204`) là từ có thật nên phổ biến hơn rác.

    IDF là tiêu chí "hiếm thì đáng chú ý", đúng cho nhãn vật thể (A62) nơi từ
    vựng đóng và sạch. Trên văn bản OCR nó đo nhầm: **hiếm ở đây nghĩa là SAI**,
    không phải đặc trưng.
    """
    if not van or k <= 0:
        return []
    q = bo_dau(cau_hoi)
    tu = [(m.group(), m.start()) for m in TU.finditer(van)]
    if not tu:
        return []

    v = bo_dau(van)
    khoa = [w for w in q.split() if len(w) > 2 and w not in DUNG]
    vi_tri = [m.start() for w in khoa for m in re.finditer(re.escape(w), v)]

    diem = {}
    for n in range(1, n_toi_da + 1):
        for i in range(len(tu) - n + 1):
            cum = " ".join(x for x, _ in tu[i:i + n])
            if len(cum) < 2 or bo_dau(cum) in q:
                continue
            hiem = max((idf.get(bo_dau(x), 0.0) for x, _ in tu[i:i + n]),
                       default=0.0)
            if vi_tri:
                gan = min(abs(tu[i][1] - p) for p in vi_tri)
                hiem -= gan / 500.0          # xa từ khoá thì hạ nhẹ
            # Cùng một cụm xuất hiện nhiều chỗ -> giữ lần GẦN từ khoá nhất.
            if hiem > diem.get(cum, float("-inf")):
                diem[cum] = hiem

    ra, da = [], set()
    for cum in sorted(diem, key=lambda x: -diem[x]):
        g = bo_dau(cum)
        if g in da:
            continue
        da.add(g)
        ra.append(cum)
        if len(ra) >= k:
            break
    return ra
