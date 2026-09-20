"""
115_do_char_ngram.py — BM25 trên n-gram KÝ TỰ: có trị được rác OCR không?

    python scripts/115_do_char_ngram.py --file dev/tap_de_that.jsonl

BÀI TOÁN

A88 đo được VietOCR đọc `HTV` thành `HIV` ở **8,3% khung L26** (3.971 dòng), và
mỗi lỗi đọc như vậy sinh ra một token **chỉ xuất hiện một lần trong cả kho** —
hapax. Với BM25 theo TỪ, một từ sai chính tả là một từ HOÀN TOÀN khác: `Bapnep`
và `Bắp nếp` không chia sẻ gì cả.

Lập luận thông thường là n-gram ký tự thì chia sẻ được phần chung. **Đo trước
khi tin** (Jaccard giữa hai tập n-gram, có đệm biên từ):

| ca | n=2 | n=3 | n=4 |
| --- | ---: | ---: | ---: |
| `HTV` vs `HIV` — sai 1 ký tự giữa từ NGẮN | 0,33 | **0,00** | **0,00** |
| `Bapnep` vs `bap nep` — dính chữ, từ DÀI | 0,75 | 0,50 | 0,29 |
| `Tà Pứa` vs `Ta Pua` — mất dấu | 1,00 | 1,00 | 1,00 |

**Ca đầu — chính là ca thúc đẩy ý tưởng này — không được cứu.** Ký tự sai nằm
GIỮA một từ ba chữ, nên với n ≥ 3 không n-gram nào sống sót. Chỉ n=2 chạm được,
mà 2-gram thì mọi từ tiếng Việt đều chạm nhau, tức nhiễu tối đa.

Ca ba thì n-gram thắng tuyệt đối — nhưng **nhánh bỏ dấu đã trị xong ca đó rồi**,
nên không có gì để lấy thêm.

Còn lại đúng **ca hai** (lỗi dính chữ) là chỗ n-gram có thể đóng góp thật. Nên
chạy `n = 3`: nó giữ được 0,50 ở ca hai trong khi n=4 chỉ còn 0,29.

BỐN CẤU HÌNH — VÀ VÌ SAO PHẢI CÓ ĐỦ BỐN

`bm25.py` hiện chạy **hai nhánh**: có dấu và bỏ dấu, trộn `α·có + (1−α)·không`
với α = 0,5. Nhánh bỏ dấu đã là một dạng "chuẩn hoá chịu lỗi" rồi. Nên câu hỏi
không phải *"n-gram có tốt không"* mà *"n-gram có tốt hơn cái đang có không"*:

    1. MỐC          từ, hai nhánh (có dấu / bỏ dấu), α = 0,5   <- đang chạy
    2. chỉ n-gram ký tự
    3. từ + n-gram, hợp nhất bằng RRF hạng
    4. chẩn đoán: chỉ nhánh CÓ DẤU (để thấy nhánh bỏ dấu đang gánh bao nhiêu)

Dòng 4 là đối chứng bắt buộc: nếu nhánh bỏ dấu đã lấy gần hết phần "chịu lỗi"
thì n-gram không còn gì để lấy, và dòng 2/3 sẽ không hơn được.

⚠️ ĐÂY LÀ KÊNH TRUY HỒI, KHÔNG PHẢI KHÂU ĐÀO ĐÁP ÁN. Bản rà soát đề nghị
n-gram cho khâu đào; nhưng khâu đào bị chặn bởi **trần 6/13** (A88) — đáp án
chỉ TỒN TẠI trong văn bản ở 46% số câu, và không phép so khớp nào vượt được
trần đó. Ở kênh truy hồi thì không có trần như vậy, nên đo ở đây có ý nghĩa
hơn.

⚠️ DỰ ĐOÁN GHI TRƯỚC (đã hạ xuống sau bảng Jaccard): dòng 2 thua rõ, dòng 3
hoà. Bảng trên cho thấy n-gram chỉ cứu được MỘT trong ba loại lỗi, và loại
nó cứu tốt nhất thì nhánh bỏ dấu đã cứu xong. Nếu dòng 4 gần bằng dòng 1 thì
nhánh bỏ dấu cũng không gánh gì, và cả hướng này lẫn `alpha` đều là chỗ trống.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
import tap_dev                                        # noqa: E402
from bm25 import BM25, KenhVanBan, bo_dau, tach       # noqa: E402
from cham_diem import bao_cao_do_nhay                 # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402
from schema import Candidate                          # noqa: E402

W3 = 0.5
N_GRAM = 3          # xem bảng Jaccard ở đầu file


def thanh_ngram(s: str, n: int = N_GRAM) -> str:
    """Chuỗi -> chuỗi các n-gram ký tự, ngăn bằng dấu cách.

    Bỏ dấu trước: n-gram trên chữ CÓ dấu thì `Tà` và `Ta` vẫn là hai ký tự
    khác nhau, tức mất đúng tính chịu lỗi đang cần. Thêm `_` ở hai đầu mỗi từ
    để n-gram biết đâu là biên từ — nếu không thì `HTV` nằm giữa một từ dài
    cũng khớp như `HTV` đứng riêng.
    """
    ra = []
    # bigram=False: `tach` mặc định sinh cả bigram nối bằng `_`, và cắt
    # n-gram ký tự từ token `ha_noi` là vô nghĩa — nó nhân đôi rác.
    for t in tach(bo_dau(s), bigram=False):
        t = f"_{t}_"
        ra += [t[i:i + n] for i in range(max(1, len(t) - n + 1))]
    return " ".join(ra)


class KenhNgram:
    """Kênh BM25 trên n-gram ký tự. Cùng giao diện `tim()` như `KenhVanBan`."""

    def __init__(self, master, bang, cot="text", n=N_GRAM):
        b = bang[bang[cot].fillna("").str.strip() != ""]
        self.rid = b.row_id.values.astype(int)
        # bigram=False: tài liệu ở đây ĐÃ là chuỗi n-gram; ghép bigram của
        # n-gram ra token dài vô nghĩa và làm phình từ điển gấp đôi.
        self.bm = BM25([thanh_ngram(str(t), n) for t in b[cot].fillna("")],
                       bigram=False)
        self.master = master
        self.n = n

    def tim(self, cau, k=100, be=None) -> list[Candidate]:
        cac = [cau] if isinstance(cau, str) else list(cau)
        import numpy as np
        d = np.max([self.bm.diem(thanh_ngram(c, self.n)) for c in cac], axis=0)
        lay = min(len(d), k)
        top = np.argpartition(-d, lay - 1)[:lay]
        top = top[np.argsort(-d[top])]
        top = [i for i in top if d[i] > 0]
        ra = []
        for i in top:
            r = int(self.rid[i])
            if be is not None and not be[r]:
                continue
            m = self.master.iloc[r]
            ra.append(Candidate(row_id=r, video_id=m.video_id,
                                frame_idx=int(m.frame_idx), score=float(d[i]),
                                source="ngram",
                                meta={"pts_time": float(m.pts_time)}))
        return ra[:k]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--file", nargs="*", type=Path,
                    default=[GOC / "dev" / "tap_de_that.jsonl"])
    ap.add_argument("--be", type=int, default=100)
    ap.add_argument("--n", type=int, default=N_GRAM)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    bang = pd.read_parquet(a.index / "ocr_asr.parquet")
    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(master, bang, cot="text", ten="ocr_asr")
    k3_dau = KenhVanBan.tu_bang_khung(master, bang, cot="text", ten="co_dau")
    k3_dau.alpha = 1.0                       # chỉ nhánh CÓ DẤU
    print(f"\ndựng chỉ mục {a.n}-gram ký tự…", flush=True)
    kg = KenhNgram(master, bang, cot="text", n=a.n)

    cau = []
    for f in a.file:
        cau += tap_dev.doc(f)
    giu = [c for c in cau if not k1.co_du(R.tach_truy_van(c.cau_hoi))]
    print(f"{len(giu)} câu | kênh 3 w={W3:g}\n")

    n1, nv = {}, {}

    def anh(c):
        if c.id not in n1:
            ds = [k1.tim(m, k=a.be) for m in R.tach_truy_van(c.cau_hoi)]
            n1[c.id] = hop_nhat(ds) if len(ds) > 1 else ds[0]
        return n1[c.id]

    def van(c, ten, kenh):
        if (c.id, ten) not in nv:
            nv[(c.id, ten)] = kenh.tim(c.cau_hoi, k=a.be)
        return nv[(c.id, ten)]

    def dung(ds_kenh):
        n = {}

        def g(c):
            if c.id not in n:
                phu = [van(c, t, k) for t, k in ds_kenh]
                n[c.id] = hop_nhat([anh(c)] + phu,
                                   trong_so=[1.0] + [W3] * len(phu))[:100]
            return n[c.id]
        return g

    cau_hinh = {
        "1. MỐC: từ, hai nhánh α=0,5": dung([("cu", k3)]),
        f"2. chỉ {a.n}-gram ký tự": dung([("ng", kg)]),
        f"3. từ + {a.n}-gram (RRF)": dung([("cu", k3), ("ng", kg)]),
        "4. chẩn đoán: chỉ nhánh CÓ DẤU": dung([("dau", k3_dau)]),
    }
    print(bao_cao_do_nhay(giu, cau_hinh, master))
    print("\nĐỌC BẢNG: nếu dòng 4 ≈ dòng 1 thì nhánh bỏ dấu KHÔNG gánh gì, và\n"
          "cả hướng n-gram lẫn tham số `alpha` đều là chỗ không có gì để lấy.")


if __name__ == "__main__":
    main()
