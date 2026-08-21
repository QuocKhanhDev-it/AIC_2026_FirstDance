"""
tra_loi.py — Sinh `answer` cho câu Q&A bằng VLM. Mở khoá 43% số câu đang ăn 0.

Câu Q&A chiếm **42/99 câu tập dev**, và hiện **chắc chắn 0 điểm** — BTC chấm
`khung đúng VÀ answer đúng`, mà ta chưa có gì sinh answer. Đây là chỗ đắt nhất
trong toàn hệ thống tính theo điểm trên mỗi đơn vị công.

Dùng lại `goi_ollama` của `scripts/14_sinh_caption.py` (TV4 viết) — Qwen2.5-VL
chạy local, không trần quota, không tốn tiền.

    from tra_loi import tra_loi_qa
    dap = tra_loi_qa("trên bàn có mấy cái bát", ["duong/dan/anh.jpg"])

    python src/tra_loi.py --thu          # thử 3 câu Q&A của tập dev

ĐÁP ÁN PHẢI NGẮN VÀ CHUẨN TẮC — ĐÂY LÀ RÀNG BUỘC CỦA BTC, KHÔNG PHẢI THẨM MỸ
============================================================================

Quy định nộp bài ghi **hai điều mâu thuẫn nhau** về cách chấm `answer`:

    trang 2:  "được so sánh chính xác về mặt NGỮ NGHĨA với đáp án"
    trang 8:  "Answer (Q&A) sẽ được so sánh dưới dạng CHUỖI CHÍNH XÁC"

⚠️ **Chưa hỏi được BTC thì phải chọn phương án AN TOÀN VỚI CẢ HAI.** Dạng ngắn
nhất, chuẩn tắc nhất thoả cả hai: nếu chấm chuỗi chính xác thì `"5"` có cơ hội
khớp còn `"Có 5 cái bát trên bàn"` thì không; nếu chấm ngữ nghĩa thì `"5"` vẫn
đúng. Ngược lại thì không đúng.

Nên `don_dap_an()` ép về dạng tối giản, và **không đoán thêm gì** — nó chỉ cắt
chữ thừa, không viết lại nội dung.

> ⚠️ Đừng nghe lời khuyên "BTC thường chấm khớp từ khoá hoặc F1". Đó là **phỏng
> đoán về một BTC khác**; quy định của AIC26 nói *"chuỗi chính xác"* và *"ngữ
> nghĩa"*, không nói F1. Đặt cược vào một luật chấm chưa ai xác nhận là cách
> nhanh nhất để mất trọn 43% số câu.

BA ĐIỀU ĐÃ ĐO Ở D0.3, ÁP DỤNG THẲNG VÀO ĐÂY
===========================================

1. **`temperature = 0` là bắt buộc.** API mặc định 1,0 — đó là lý do đồng thuận
   thấp ở mọi đợt bench trước, không phải model kém. Hạ về 0: đồng thuận
   50% → 100%, lại nhanh hơn.
2. **Tuân thủ định dạng quan trọng hơn độ đúng.** `gemma-4-31b-it` bị loại vì
   tuân thủ format 0% — nhả nguyên chuỗi suy luận. Sai định dạng = 0 điểm.
3. **Trần độ đúng ~30–50% ở MỌI model.** Khoảng cách giữa các model nhỏ hơn
   khoảng cách tới mức dùng được — nên đầu tư vào **ngữ cảnh đưa vào**
   (nhiều khung, không phải một), đừng loay hoay chọn model.
"""

import argparse
import importlib.util
import re
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

TOI_DA = 100          # BTC: "Answer (Q&A) có độ dài tối đa 100 ký tự"

NHAC = """Nhìn ảnh và trả lời câu hỏi sau bằng tiếng Việt.

Câu hỏi: {cau_hoi}

Yêu cầu bắt buộc:
- Trả lời NGẮN NHẤT có thể: chỉ phần cốt lõi, tối đa 4 từ.
- Câu hỏi đếm -> trả lời bằng CHỮ SỐ, ví dụ: 5
- Câu hỏi màu sắc -> chỉ tên màu, ví dụ: màu đỏ
- KHÔNG viết câu đầy đủ, KHÔNG giải thích, KHÔNG lặp lại câu hỏi.
- Không nhìn thấy đủ để trả lời thì ghi: không rõ

Trả lời:"""

# Chữ đưa đẩy model hay thêm dù đã bảo đừng. Cắt ở đây chứ không sửa nội dung.
_MO_DAU = re.compile(
    r"^\s*(trả lời|đáp án|answer|kết quả|theo (ảnh|hình)|trong (ảnh|hình)|"
    r"dựa vào (ảnh|hình))\s*[:\-–]?\s*", re.I)
_CAU_DAY = re.compile(r"^(có|là|gồm|khoảng)\s+", re.I)


def nap_khoa(ten=("GEMINI_API_KEY", "GOOGLE_API_KEY"), f=None) -> str:
    """Đọc khoá API từ `.env` ở gốc repo, trả về khoá đầu tiên tìm thấy.

    Vì sao tự viết thay vì dùng `python-dotenv`: thêm một phụ thuộc chỉ để đọc
    một file `KEY=VALUE` là không đáng, và `requirements.txt` càng mỏng thì máy
    thành viên càng ít cớ dựng hỏng.

    ⚠️ **Cắt khoảng trắng quanh TÊN biến, không chỉ quanh giá trị.** File thật
    trên máy ghi `GEMINI_API_KEY =...` — có dấu cách trước `=`. Không cắt thì
    tên biến thành `"GEMINI_API_KEY "` và tra cứu trượt, mà triệu chứng lại là
    "không tìm thấy khoá" — rất dễ đổ nhầm cho việc chưa đặt khoá.

    ⚠️ **KHÔNG BAO GIỜ in khoá ra màn hình hay log.** Repo công khai, và log
    hay bị dán vào chat/issue. Hàm này trả chuỗi; chỗ gọi tự giữ kín.
    """
    import os
    for t in ten:                       # biến môi trường thật thắng file
        if os.environ.get(t):
            return os.environ[t]
    f = Path(f) if f else Path(__file__).resolve().parent.parent / ".env"
    if not f.exists():
        return ""
    for dong in f.read_text("utf-8").splitlines():
        dong = dong.strip()
        if not dong or dong.startswith("#") or "=" not in dong:
            continue
        k, v = dong.split("=", 1)
        k, v = k.strip(), v.strip().strip("'\"")
        if k in ten and v:
            os.environ.setdefault(k, v)
            return v
    return ""


def don_dap_an(text: str, toi_da: int = TOI_DA) -> str:
    """Ép câu trả lời của VLM về dạng ngắn, chuẩn tắc.

    ⚠️ **Chỉ CẮT, không VIẾT LẠI.** Sửa nội dung là tự bịa ra một đáp án khác
    với thứ model nhìn thấy — mà ta không có cách nào biết cái nào đúng hơn.

    Cắt quá 100 ký tự là **hỏng cả dòng** chứ không phải mất một phần: BTC chặn
    cứng ở 100. Nên cắt ở ranh giới từ, và nếu vẫn dài thì trả về phần đầu.
    """
    # ⚠️ TÁCH DÒNG TRƯỚC, gộp khoảng trắng SAU. Làm ngược lại thì `" ".join(
    # text.split())` đã nuốt mất `\n`, và cả chuỗi suy luận dài của model bị
    # dính vào câu trả lời. Test bắt được.
    t = (text or "").strip().split("\n")[0]
    t = " ".join(t.split())
    if not t:
        return ""
    # bỏ mở đầu thừa, có thể nhiều lớp ("Trả lời: Đáp án: 5")
    for _ in range(3):
        moi = _MO_DAU.sub("", t)
        if moi == t:
            break
        t = moi
    t = _CAU_DAY.sub("", t).strip(" .,:;\"'")

    # câu hỏi đếm: model hay viết "5 cái bát" -> giữ nguyên con số nếu nó đứng
    # đầu và phần sau chỉ là danh từ lặp lại câu hỏi
    m = re.match(r"^(\d+)\b", t)
    if m and len(t) > len(m.group(1)) + 12:
        t = m.group(1)

    if len(t) > toi_da:
        cat = t[:toi_da].rsplit(" ", 1)[0]
        t = (cat or t[:toi_da]).strip()
    return t


def _sinh_caption():
    """Nạp `scripts/14_sinh_caption.py` để dùng lại `goi()` của nó.

    Tên file bắt đầu bằng số nên không `import` thẳng được. Dùng lại thay vì
    chép: bộ gọi Ollama/Gemini ở đó đã có lùi thời gian và thông báo lỗi tử tế.
    """
    s = importlib.util.spec_from_file_location(
        "sinh_caption", GOC / "scripts" / "14_sinh_caption.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def kiem_model_nhin_duoc(model: str) -> None:
    """Dừng nếu model chưa cài, hoặc cài rồi nhưng KHÔNG NHÌN ĐƯỢC ẢNH.

    ⚠️ **Đây là bẫy D0.3, và nó không tự lộ ra.** Model thuần văn bản nhận
    `images` rồi lặng lẽ bỏ qua, đoán mò từ câu hỏi — ra một đáp án trông hợp
    lệ và một con số độ đúng trông hợp lý, nhưng **vô nghĩa**. Bench cũ đã mất
    một đợt vì `deepseek-chat`; đừng mất đợt nữa.

    Máy đang dùng có `qwen2.5:3b`, `gemma:2b`, `nomic-embed-text` — **cả ba đều
    thuần văn bản**. Phải `ollama pull` một bản VL.
    """
    import json
    import urllib.request
    sc = _sinh_caption()
    goc = sc.API_OLLAMA.rsplit("/api/", 1)[0]
    try:
        with urllib.request.urlopen(f"{goc}/api/tags", timeout=10) as r:
            co = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception as e:
        raise SystemExit(
            f"❌ Không hỏi được Ollama ở {goc} ({e}).\n"
            f"   Chạy `ollama serve` trước.")

    if model not in co and f"{model}:latest" not in co:
        raise SystemExit(
            f"❌ Chưa cài model {model!r}.\n\n"
            f"   Đang có: {', '.join(co) or '(trống)'}\n\n"
            f"   Cài:  ollama pull {model}\n"
            f"   Máy ít RAM thì dùng bản nhỏ: ollama pull qwen2.5vl:3b")

    if not any(k in model.lower() for k in
               ("vl", "vision", "llava", "minicpm-v", "moondream", "gemma3")):
        raise SystemExit(
            f"❌ {model!r} là model THUẦN VĂN BẢN — không nhìn được ảnh.\n\n"
            f"   Nó vẫn trả lời, vẫn ra con số độ đúng trông hợp lý, nhưng là\n"
            f"   ĐOÁN MÒ TỪ CÂU HỎI. Bench D0.3 đã mất một đợt vì đúng lỗi này.\n\n"
            f"   Dùng: ollama pull qwen2.5vl:7b   (hoặc qwen2.5vl:3b nếu ít RAM)")


def tra_loi_qa(cau_hoi: str, duong_dan_anh: list, backend="ollama",
               model=None, key=None, so_khung=3) -> str:
    """Đáp án ngắn cho một câu Q&A, nhìn `so_khung` khung đầu.

    ⚠️ **Đưa NHIỀU khung, không phải một.** PHẦN C mục 4 và D0.3 đều chỉ về đây:
    câu hỏi đếm không đếm đủ trên một khung — BTC nêu thẳng ca *"1 em bé được bế
    bởi 4 người liên tiếp, dựa trên keyframe thì chỉ có 3"* (A9.3). Chênh lệch
    giữa "1 khung" và "3–5 khung" nhiều khả năng lớn hơn chênh lệch giữa các
    model.
    """
    sc = _sinh_caption()
    model = model or sc.MODEL_MAC_DINH[backend]
    if backend == "ollama":
        kiem_model_nhin_duoc(model)
    anh = [Path(p).read_bytes() for p in duong_dan_anh[:so_khung]
           if p and Path(p).exists()]
    if not anh:
        return ""

    if backend == "ollama":
        tho = _goi_ollama_nhieu_anh(sc, model, anh,
                                    NHAC.format(cau_hoi=cau_hoi))
    else:
        # Gemini: bộ gọi sẵn chỉ nhận một ảnh -> dùng khung tốt nhất
        tho = sc.goi(backend, model, key, anh[0], NHAC.format(cau_hoi=cau_hoi))
    return don_dap_an(tho)


def _goi_ollama_nhieu_anh(sc, model, anh: list, loi_nhac: str) -> str:
    """Như `sc.goi_ollama` nhưng gửi NHIỀU ảnh trong một lượt.

    Ollama nhận `images` là một danh sách, nên nhiều khung chỉ là thêm phần tử.
    Bộ gọi của `14_sinh_caption.py` viết cho caption (một ảnh một caption) nên
    không lộ tham số này ra.
    """
    import base64
    import json
    import time
    import urllib.error
    import urllib.request

    body = {"model": model, "prompt": loi_nhac, "stream": False,
            "images": [base64.b64encode(x).decode() for x in anh],
            "options": {"temperature": 0.0}}     # D0.3: temp=0 là BẮT BUỘC
    data = json.dumps(body).encode()
    cho = 3
    for lan in range(3):
        req = urllib.request.Request(sc.API_OLLAMA, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read()).get("response", "").strip()
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError) as e:
            if lan < 2:
                time.sleep(cho)
                cho *= 2
                continue
            raise RuntimeError(
                f"Ollama không phản hồi ở {sc.API_OLLAMA} ({e}).\n"
                f"   Đã chạy `ollama serve` và `ollama pull {model}` chưa?")
    raise RuntimeError("hết lượt thử lại")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thu", type=int, default=3,
                    help="thử N câu Q&A đầu của tập dev (cần ảnh trên máy này)")
    ap.add_argument("--backend", default="ollama", choices=("ollama", "gemini"))
    ap.add_argument("--model")
    ap.add_argument("--so-khung", type=int, default=3)
    a = ap.parse_args()

    import pandas as pd
    import tap_dev
    master = pd.read_parquet(GOC / "index" / "master.parquet")

    cau = [c for c in tap_dev.doc() if c.loai == "QA"]
    # chỉ câu có ảnh trên máy này mới thử được
    duoc = []
    for c in cau:
        p = [master.kf_path.iloc[r] for r in c.row_id_dung]
        if any(isinstance(x, str) and Path(x).exists() for x in p):
            duoc.append((c, [x for x in p if isinstance(x, str)]))
    if not duoc:
        raise SystemExit("Không câu Q&A nào có ảnh trên máy này.")

    print(f"{len(duoc)}/{len(cau)} câu Q&A có ảnh — thử {min(a.thu, len(duoc))}\n")
    dung = 0
    for c, p in duoc[:a.thu]:
        tra = tra_loi_qa(c.cau_hoi, p, a.backend, a.model, so_khung=a.so_khung)
        ok = tra.strip().lower() == c.dap_an.strip().lower()
        dung += ok
        print(f"  {'✅' if ok else '❌'} {c.id}")
        print(f"     hỏi   : {c.cau_hoi[:70]}")
        print(f"     đáp án: {c.dap_an!r}")
        print(f"     VLM   : {tra!r}\n")
    print(f"Khớp CHUỖI CHÍNH XÁC: {dung}/{min(a.thu, len(duoc))}")
    print("⚠️  n nhỏ như vậy không kết luận được gì — đây là phép thử ĐƯỜNG ỐNG,\n"
          "    không phải phép đo độ đúng. Đo thật cần cả 42 câu Q&A.")


if __name__ == "__main__":
    main()
