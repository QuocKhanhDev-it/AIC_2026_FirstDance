"""
97_do_bang_tra_asr.py — Phục hồi dấu bằng **chính ASR của kho**, không cần model.

    python scripts/97_do_bang_tra_asr.py

A68: `ocr_text` chỉ **31% có dấu**, `asr_text` **100%**. Nên ASR của kho này là
**một cuốn từ điển có dấu của chính nó**. Khi OCR đọc ra `Ta Pua`, quét n-gram
1–4 từ trong ASR rồi thay bằng bản có dấu — không model, không GPU, chạy CPU.

BA MỨC PHẠM VI, và chúng trả lời ba câu khác nhau

  1. **cùng KHUNG**  — chỉ ASR của chính keyframe đó. Đây gần như là A76 hiện
     tại, nhưng dò n-gram nên bắt được cả cụm nhiều từ.
  2. **cùng VIDEO**  — ASR của toàn bộ video. Từ vựng trong một video có tương
     quan cực cao; đây là mức đề xuất nhắm tới.
  3. **cả kho**      — mọi ASR. Rộng nhất, nhưng cũng dễ thay nhầm nhất: hai
     thực thể khác nhau ở hai video có thể trùng nhau sau khi bỏ dấu.

Mức 3 có mặt để **đo cái giá của việc quét rộng**, không phải vì kỳ vọng nó
thắng. Nếu mức 3 tệ hơn mức 2 thì đó là bằng chứng cho "tương quan CỤC BỘ trong
video" chứ không chỉ là một con số.

⚠️ A83 đo được `uu_tien_co_dau()` bản đầu chỉ bắt khi hai bản **cùng số từ**.
Bảng tra n-gram không vướng chuyện đó — đó là điều đang được kiểm ở đây.
"""

import argparse
import sys
import unicodedata
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import tap_dev                                        # noqa: E402
from dap_an import bang_tra_ngram, co_dau, dao        # noqa: E402


def bo_dau(s) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path, default=[
        GOC / "dev" / "tap_de_that.jsonl", GOC / "dev" / "tap_de_thi_thu.jsonl"])
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet", columns=["video_id"])
    bang = pd.read_parquet(a.index / "ocr_asr.parquet")
    ocr = dict(zip(bang.row_id.astype(int), bang.ocr_text.fillna("")))
    asr = dict(zip(bang.row_id.astype(int), bang.asr_text.fillna("")))
    vid = master.video_id.values

    cau = []
    for f in a.file:
        cau += [c for c in tap_dev.doc(f) if c.loai == "QA" and c.dap_an]
    co = [c for c in cau if co_dau(c.dap_an)]
    print(f"{len(cau)} câu Q&A | {len(co)} câu có đáp án CHỨA DẤU "
          f"<- chỉ nhóm này nói được gì về dấu\n")

    # Bảng tra CẢ KHO — dựng một lần, cũng dùng làm nguồn cho hai mức kia.
    print("dựng bảng tra…", flush=True)
    tra_kho, tra_video = {}, {}
    for r, t in asr.items():
        if not t:
            continue
        g = bang_tra_ngram(t)
        tra_kho.update(g)
        tra_video.setdefault(str(vid[r]), {}).update(g)
    print(f"  cả kho : {len(tra_kho):,} n-gram có dấu")
    print(f"  video  : {len(tra_video)} video có ASR, "
          f"trung bình {sum(len(x) for x in tra_video.values()) // max(len(tra_video), 1):,}"
          f" n-gram/video\n")

    def van_cua(r):
        return f"{ocr.get(r, '')} {asr.get(r, '')}".strip()

    def thu(c, lay_tra):
        """Đào đáp án cho mọi khung đúng, xem có khung nào ra ĐÚNG CHUỖI không."""
        mong = c.dap_an.strip().lower()
        for r in c.row_id_dung:
            t = dao(van_cua(r), c.cau_hoi)
            if not t:
                continue
            g = lay_tra(r)
            if g and not co_dau(t) and bo_dau(t) in g:
                t = g[bo_dau(t)]
            if t.strip().lower() == mong:
                return True
        return False

    muc = {
        "1. không bảng tra (A76)": lambda r: None,
        "2. cùng KHUNG": lambda r: bang_tra_ngram(asr.get(r, "")),
        "3. cùng VIDEO": lambda r: tra_video.get(str(vid[r])),
        "4. cả KHO": lambda r: tra_kho,
    }

    print(f"{'câu':<16}{'đáp án':<18}" + "".join(f"{k[:14]:>16}" for k in muc))
    print("-" * (34 + 16 * len(muc)))
    dem = {k: 0 for k in muc}
    for c in co:
        o = []
        for k, f in muc.items():
            ok = thu(c, f)
            dem[k] += ok
            o.append("✅" if ok else "—")
        print(f"{c.id:<16}{c.dap_an[:16]:<18}" + "".join(f"{x:>16}" for x in o))
    print()
    for k, n in dem.items():
        print(f"  {k:<26}{n}/{len(co)}")
    print("\n⚠️ Mẫu số là số câu có đáp án CHỨA DẤU. 8/13 đáp án còn lại không")
    print("   có dấu nào, nên chúng không nói gì về phép phục hồi dấu (A76).")


if __name__ == "__main__":
    main()
