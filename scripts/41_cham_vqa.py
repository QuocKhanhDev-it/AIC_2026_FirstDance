r"""
41_cham_vqa.py — Chấm file trả lời của VLM, và tách phần công của kênh OCR ra.

    python scripts/41_cham_vqa.py vqa/tra_loi.json

`tra_loi.json` là `{"<id câu>": "<đáp án VLM viết ra>", ...}`.

VÌ SAO KHÔNG CHỈ IN MỘT CON SỐ ĐÚNG/SAI
=======================================

Bộ lọc ở `40_xuat_goi_vqa.py` gom bằng mẫu chữ, nên nó **trộn hai loại câu rất
khác nhau** vào cùng một rổ:

* `Số hiển thị trên tấm bảng đen ... là bao nhiêu?` — đây là việc **ĐỌC CHỮ**.
  Kênh 3 (OCR) làm được, và đã làm.
* `Có bao nhiêu con mèo được chàng trai cho ăn?` — đây là việc **NHÌN**. Không
  dòng chữ nào trong khung trả lời hộ.

Nếu chỉ in một tỷ lệ đúng gộp chung, VLM sẽ **ăn công của OCR**: nó trả lời
đúng mấy câu đọc-bảng-điểm rồi trông như thắng, trong khi thứ ta cần biết là nó
có làm được phần OCR *không* làm được hay không.

Nên script in **hai cột**, chia theo CÂU HỎI ĐÒI GÌ:

* **ĐỌC CHỮ** — câu hỏi trỏ thẳng vào chữ trong khung (bảng, biển, màn hình,
  điểm số, slide). Kênh 3 vốn là chỗ giải quyết. VLM đúng ở đây gần như không
  nói gì mới.
* **NHÌN** — đếm vật, hỏi màu. Không dòng chữ nào trả lời hộ. **Đây mới là cột
  quyết định** hướng VQA sống hay chết.

**Một cách chia đã thử và ĐÃ BỎ:** lúc đầu tôi chia theo "đáp án có nằm nguyên
trong văn bản OCR của khung đáp án không". Nghe gọn, nhưng hỏng ngay khi chạy
thật: đáp án `'2'` hay `'1'` khớp bừa vào bất cứ đoạn OCR nào có chữ số đó, còn
câu số-điện-thoại — câu đọc-chữ rõ nhất — lại rơi vào nhóm "OCR mù" chỉ vì OCR
đọc sai. Phép thử đó đo **may rủi của OCR**, không đo **bản chất câu hỏi**. Cột
`ocr?` bên dưới vẫn in ra để tham khảo, nhưng không dùng để chia nhóm nữa.

SO KHỚP LỎNG, VÀ VÌ SAO
=======================

Đáp án chuẩn viết tay (`'ba'`, `'2,15'`, `'trắng, đen'`), VLM viết câu đầy đủ.
So khớp chặt sẽ đếm nhầm thành sai. Ở đây: bỏ dấu, hạ chữ thường, đổi số viết
chữ sang chữ số, rồi hỏi mọi phần của đáp án chuẩn có xuất hiện trong câu trả
lời không. Vẫn **in cả câu trả lời thô** để người đọc tự phủ quyết — con số
tổng không bao giờ được thay cho việc nhìn.
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

# Câu hỏi TRỎ THẲNG vào chữ trong khung -> việc của kênh 3, không phải của VQA.
DOI_DOC = re.compile(
    r"bảng|biển|màn hình|slide|điểm|số hiển thị|số điện thoại|chú giải|"
    r"áp phích|dòng chữ|ghi trên|in trên|thành lập", re.I)

SO_CHU = {
    "không": "0", "một": "1", "hai": "2", "ba": "3", "bốn": "4", "năm": "5",
    "sáu": "6", "bảy": "7", "tám": "8", "chín": "9", "mười": "10",
}


def bo_dau(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def chuan(s: str) -> str:
    s = bo_dau(str(s or "")).lower()
    for chu, so in SO_CHU.items():
        s = re.sub(rf"\b{bo_dau(chu)}\b", so, s)
    s = s.replace(",", ".")
    return re.sub(r"\s+", " ", s).strip()


def khop(chuan_dap: str, tra_loi: str) -> bool:
    """Đáp án chuẩn có thể gồm nhiều phần ('trắng, đen') — đòi đủ mọi phần."""
    tl = chuan(tra_loi)
    phan = [p for p in re.split(r"[.;/]| va ", chuan(chuan_dap)) if p.strip()]
    if not phan:
        return False
    return all(re.search(rf"(?<![\w.]){re.escape(p.strip())}(?![\w.])", tl)
               for p in phan)


def main():
    ap = argparse.ArgumentParser(description="cham file tra loi cua VLM")
    ap.add_argument("tra_loi", type=Path)
    ap.add_argument("--dap-an", default=GOC / "vqa" / "dap_an.json", type=Path)
    ap.add_argument("--goi", default=GOC / "vqa" / "goi" / "cau_hoi.json", type=Path)
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--tap-dev", default=GOC / "dev" / "tap_dev.jsonl", type=Path)
    a = ap.parse_args()

    dap = json.loads(a.dap_an.read_text("utf-8"))
    tl = json.loads(a.tra_loi.read_text("utf-8"))
    hoi = {c["id"]: c for c in json.loads(a.goi.read_text("utf-8"))}

    # --- phep thu: dap an co nam trong OCR cua chinh khung dap an khong ---
    master = pd.read_parquet(a.index / "master.parquet")
    bang = pd.read_parquet(a.index / "ocr_asr.parquet").set_index("row_id")
    cau_dev = {json.loads(l)["id"]: json.loads(l)
               for l in a.tap_dev.read_text("utf-8").splitlines() if l.strip()}

    def ocr_voi_toi(cid: str) -> bool:
        c = cau_dev.get(cid)
        if not c:
            return False
        rs = [r[0] if isinstance(r, list) else r for r in c["row_id_dung"]]
        chu = " ".join(str(bang.text.get(r, "") or "") for r in rs)
        return khop(dap.get(cid, ""), chu)

    nhom = {"ĐỌC CHỮ": [], "NHÌN": []}
    for cid in hoi:
        q = hoi[cid]["cau_hoi"]
        nhom["ĐỌC CHỮ" if DOI_DOC.search(q) else "NHÌN"].append(cid)

    print(f"{len(hoi)} câu trong gói | {len(tl)} câu có trả lời\n")
    for ten, ids in nhom.items():
        if not ids:
            continue
        dung = 0
        print(f"── {ten}  ({len(ids)} câu) " + "─" * 34)
        for cid in sorted(ids):
            g, p = dap.get(cid, ""), tl.get(cid, "")
            ok = khop(g, p) if cid in tl else False
            dung += ok
            dau = "✅" if ok else ("❌" if cid in tl else "··")
            print(f"  {dau} {cid:14} ocr?{'Y' if ocr_voi_toi(cid) else 'n'}  "
                  f"chuẩn={g!r:18} VLM={str(p)[:52]!r}")
        print(f"  → {dung}/{len(ids)} = {dung / len(ids):.0%}\n")

    tong = sum(khop(dap.get(c, ""), tl.get(c, "")) for c in hoi if c in tl)
    print(f"GỘP CHUNG: {tong}/{len(hoi)} = {tong / len(hoi):.0%}"
          "   <- ĐỪNG dùng con số này để kết luận")
    mu = nhom["NHÌN"]
    if mu:
        d = sum(khop(dap.get(c, ""), tl.get(c, "")) for c in mu if c in tl)
        print(f"\nCột quyết định là **NHÌN**: {d}/{len(mu)}.")
        print("Thấp thì hướng VQA chết ở TRẦN TRÊN — đã cho sẵn đúng khung mà vẫn")
        print("không đọc ra; trong bài thật còn phải tìm được khung nữa.")
        if len(mu) < 15:
            print(f"\n⚠️  Chỉ {len(mu)} câu NHÌN — quá ít để kết luận theo hướng")
            print("   DƯƠNG. Đủ để kết luận theo hướng ÂM (sai gần hết thì bỏ),")
            print("   không đủ để bật tính năng. Muốn kết luận dương thì phải")
            print("   soạn thêm câu Q&A THUẦN THỊ GIÁC — đếm vật, hỏi màu, hỏi")
            print("   vị trí — chứ không phải câu đọc bảng điểm.")
    print("\n⚠️  Đọc từng dòng ở trên rồi tự phủ quyết — đừng tin mỗi tỷ lệ.")


if __name__ == "__main__":
    main()
