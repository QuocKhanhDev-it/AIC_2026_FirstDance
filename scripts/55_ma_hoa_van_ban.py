"""
55_ma_hoa_van_ban.py — KÊNH 6: nhúng OCR/ASR bằng chính tháp văn bản của gopt.

    python scripts/55_ma_hoa_van_ban.py --uoc-tinh
    python scripts/55_ma_hoa_van_ban.py --ghi          # trên Kaggle, cần GPU

Ý TƯỞNG

Kênh 3 hiện dùng BM25 — khớp **mặt chữ**. Truy vấn nói "xe cứu thương" mà bản
tin viết "xe cấp cứu" thì điểm bằng 0. Nhúng cùng văn bản đó vào không gian
vector 1536 chiều của gopt thì hai cách gọi nằm gần nhau.

Được thêm một thứ: ảnh và văn bản **chung một không gian**, nên hợp nhất không
còn là trộn hai thang điểm khác nhau.

⚠️ TRẦN 64 TOKEN — LÝ DO PHẢI CHIA ĐOẠN

Tháp văn bản SigLIP2 có `context_length = 64`. Đo trên `ocr_asr.parquet`:

    176.009 tài liệu có chữ, trung vị ~140 token
    75,7% VƯỢT trần 64 token
    ở trung vị chỉ giữ được 46% văn bản

Nhúng thẳng là vứt quá nửa nội dung ở ba phần tư tài liệu — và không có gì báo,
chỉ là kênh yếu đi một cách không giải thích được. Nên script này **chia mỗi tài
liệu thành nhiều đoạn ≤ 64 token**, nhúng từng đoạn, rồi điểm của một keyframe
là **max trên các đoạn của nó** (một đoạn khớp là đủ).

⚠️ CHƯA CHẮC KÊNH NÀY CHẠY ĐƯỢC — ĐO TRƯỚC KHI TIN

SigLIP2 huấn luyện để khớp **ảnh ↔ chữ**, không phải **chữ ↔ chữ**. Nhúng tài
liệu rồi so với vector truy vấn là dùng nó **ngoài phân bố huấn luyện**. Có thể
tốt, có thể không — nhưng rẻ để biết.

Vì vậy `--chon tap` chỉ mã hoá văn bản của các video tập dev đụng tới: đủ để đo,
mà rẻ hơn cả kho nhiều lần.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))


def chon_dong(bang: pd.DataFrame, master: pd.DataFrame, chon: str) -> pd.DataFrame:
    """Chọn những dòng cần mã hoá. Chỉ dòng CÓ CHỮ."""
    b = bang[bang["text"].fillna("").str.strip() != ""]
    if chon == "tat-ca":
        return b
    if chon.startswith("tap:"):
        import json
        rid = []
        for d in Path(chon.split(":", 1)[1]).read_text("utf-8").splitlines():
            if not d.strip():
                continue
            r = json.loads(d)["row_id_dung"]
            rid += [x for bb in r for x in bb] if isinstance(r[0], list) else r
        vid = set(master.video_id.iloc[rid])
        giu = set(master[master.video_id.isin(vid)].row_id)
        return b[b.row_id.isin(giu)]
    raise SystemExit(f"--chon không hiểu: {chon!r}. Dùng: tat-ca | tap:<f.jsonl>")


def dem_token(chuoi: list[str], tok, lo: int = 8192) -> list[int]:
    """Đếm token nội dung của từng chuỗi, theo LÔ.

    `tok` trả tensor [n, context_length] đã đệm 0. Đếm ô khác 0 rồi trừ phần
    chung của mọi chuỗi (token mở/đóng) để ra token NỘI DUNG — nhờ vậy cộng
    token của các từ lại mới có nghĩa.
    """
    rong = int((tok([""])[0] != 0).sum())
    ra = []
    for i in range(0, len(chuoi), lo):
        m = tok(chuoi[i:i + lo])
        ra += [max(1, int(x) - rong) for x in (m != 0).sum(1)]
    return ra


def chia_doan_hang_loat(cac_van: list[str], tok, tran: int,
                        moi_lan_in: int = 20000) -> list[list[str]]:
    """Cắt N tài liệu thành các đoạn ≤ `tran` token. Trả về N danh sách đoạn.

    ⚠️ VÌ SAO HÀM NÀY TỒN TẠI — MỘT SỰ CỐ THẬT

    Bản đầu gọi tokenizer MỘT LẦN CHO MỖI TỪ, mà mỗi lần lại token hoá cả đoạn
    đang dồn. Cả kho là ~17 triệu lượt gọi tokenizer, chạy tuần tự trên CPU,
    trong khi GPU nằm không. Một máy Kaggle chạy **4 tiếng vẫn chưa qua nổi
    bước chia đoạn** — và vì bước này không in gì cho tới lúc xong, nhìn log
    không biết nó đang làm gì hay đã treo.

    Cách ở đây: token hoá **mỗi TỪ DUY NHẤT một lần** theo lô, rồi cộng dồn.
    Kho ~176k tài liệu chỉ còn vài trăm nghìn từ duy nhất — nhanh hơn hàng
    trăm lần, và phần lớn thời gian trả về cho GPU, chỗ đáng tiêu.

    Cộng token của từng từ là ƯỚC THỪA: đứng riêng, một từ hay bị thêm token
    mở đầu mà khi ghép vào câu thì không có. Ước thừa là hướng AN TOÀN (đoạn
    ngắn hơn trần chứ không dài hơn), nhưng vẫn phải soát lại bằng tokenizer
    thật — `soat_tran()` làm việc đó.
    """
    tu_duy_nhat = sorted({t for v in cac_van for t in v.split()})
    print(f"  {len(tu_duy_nhat):,} từ duy nhất — token hoá một lần", flush=True)
    dem = dict(zip(tu_duy_nhat, dem_token(tu_duy_nhat, tok)))

    ra = []
    for i, van in enumerate(cac_van):
        tu = van.split()
        doan, hien, n = [], [], 0
        for t in tu:
            c = dem[t]
            if hien and n + c > tran:
                doan.append(" ".join(hien))
                hien, n = [t], c
            else:
                hien.append(t)
                n += c
        if hien:
            doan.append(" ".join(hien))
        ra.append(doan)
        if moi_lan_in and (i + 1) % moi_lan_in == 0:
            print(f"  chia đoạn {i + 1:,}/{len(cac_van):,}", flush=True)
    return ra


def soat_tran(doan: list[str], tok, tran_cung: int = 64,
              lo: int = 8192) -> list[int]:
    """Trả chỉ số các đoạn CHẠM trần cứng của tháp văn bản.

    Ước thừa ở `chia_doan_hang_loat` đáng ra không cho phép chuyện này, nhưng
    "đáng ra" không phải là phép đo. Đoạn bị cắt cụt không ném lỗi — nó chỉ
    lặng lẽ mất phần đuôi, đúng loại hỏng mà A51 đã cắn một lần.
    """
    xau = []
    for i in range(0, len(doan), lo):
        m = tok(doan[i:i + lo])
        for j, x in enumerate((m != 0).sum(1)):
            if int(x) >= tran_cung:
                xau.append(i + j)
    return xau


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--matrix", default="clip_gopt.npy",
                    help="đọc model/số chiều từ sidecar .json cạnh file này")
    ap.add_argument("--chon", default="tap:dev/tap_dev.jsonl",
                    help="tat-ca | tap:<file.jsonl>")
    ap.add_argument("--tran-token", type=int, default=60,
                    help="token mỗi đoạn. Để dưới 64 cho chắc")
    ap.add_argument("--lo", type=int, default=256)
    ap.add_argument("--ra", default=GOC / "index" / "van_ban_gopt.npz", type=Path)
    ap.add_argument("--uoc-tinh", action="store_true",
                    help="chỉ đếm đoạn và ước thời gian, KHÔNG nạp model")
    ap.add_argument("--ghi", action="store_true")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    bang = pd.read_parquet(a.index / "ocr_asr.parquet")
    d = chon_dong(bang, master, a.chon)
    print(f"chọn '{a.chon}': {len(d):,} tài liệu có chữ\n")

    if a.uoc_tinh:
        # Ước số đoạn mà KHÔNG nạp model: ~3,5 ký tự/token cho tiếng Việt với
        # SentencePiece 256k. Là ước lượng, ghi rõ vậy.
        kt = d["text"].str.len()
        so_doan = np.ceil(kt / 3.5 / a.tran_token).clip(lower=1)
        tong = int(so_doan.sum())
        print(f"{'đoạn ước tính':<28}{tong:>12,}")
        print(f"{'đoạn / tài liệu':<28}{tong / len(d):>12.2f}")
        for r in (69, 200, 400):
            print(f"{'  @' + str(r) + ' đoạn/giây':<28}{tong / r / 60:>10.0f} phút")
        print(f"\n{'ma trận fp16':<28}{tong * 1536 * 2 / 1024 ** 3:>11.2f} GB")
        print("\n⚠️ Đây là ƯỚC theo ký tự. Chạy thật đếm bằng tokenizer.")
        return

    import open_clip
    import torch
    import json
    canh = a.index / (Path(a.matrix).stem + ".json")
    gc = json.loads(canh.read_text("utf-8"))
    print(f"nạp {gc['model']} / {gc['pretrained']} — chỉ tháp văn bản…")
    model, _, _ = open_clip.create_model_and_transforms(
        gc["model"], pretrained=gc["pretrained"], precision="fp16")
    model.eval()
    if hasattr(model, "visual"):
        del model.visual
    model.float()
    thiet_bi = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(thiet_bi)
    tok = open_clip.get_tokenizer(gc["model"])
    print(f"  thiết bị: {thiet_bi}")

    print("chia đoạn…")
    t_chia = time.perf_counter()
    van = [str(x) for x in d["text"]]
    rid = [int(x) for x in d["row_id"]]
    nhom = chia_doan_hang_loat(van, tok, a.tran_token)

    doan, chu = [], []          # chu[i] = row_id của đoạn i
    for r, ds in zip(rid, nhom):
        doan += ds
        chu += [r] * len(ds)
    print(f"  {len(doan):,} đoạn từ {len(d):,} tài liệu "
          f"({len(doan) / len(d):.2f} đoạn/tài liệu) "
          f"— {time.perf_counter() - t_chia:.0f} giây")

    xau = soat_tran(doan, tok)
    if xau:
        # Không tự sửa: cắt cụt làm kênh yếu đi mà không báo, nên dừng để người
        # ta hạ `--tran-token` chứ đừng ghi ra một file trông hợp lệ.
        raise SystemExit(
            f"❌ {len(xau):,}/{len(doan):,} đoạn chạm trần 64 token, ví dụ "
            f"{doan[xau[0]][:80]!r}\n   Hạ --tran-token (đang {a.tran_token}).")
    print("  ✅ không đoạn nào chạm trần 64 token")

    ra, t0 = [], time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(doan), a.lo):
            v = model.encode_text(tok(doan[i:i + a.lo]).to(thiet_bi))
            v = v.float().cpu().numpy()
            ra.append(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9))
            xong = min(i + a.lo, len(doan))
            if xong % (a.lo * 20) == 0 or xong == len(doan):
                print(f"  {xong:,}/{len(doan):,}  "
                      f"{xong / (time.perf_counter() - t0):.0f} đoạn/giây", flush=True)

    vec = np.vstack(ra).astype(np.float16)
    print(f"\n{vec.shape} {vec.dtype}  ({vec.nbytes / 1024 ** 3:.2f} GB)")
    if not a.ghi:
        print("(xem trước — thêm `--ghi` để ghi file)")
        return
    np.savez(a.ra, vec=vec, row_id=np.array(chu, dtype=np.int64),
             ghi_chu=json.dumps({**gc, "so_doan": len(doan),
                                 "tran_token": a.tran_token}, ensure_ascii=False))
    print(f"✅ {a.ra}")


if __name__ == "__main__":
    main()
