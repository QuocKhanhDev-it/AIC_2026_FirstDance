# Cell Kaggle — nhúng OCR/ASR bằng **BGE-M3** (kênh 6 làm lại cho đúng)

## Vì sao làm lại thứ đã thất bại một lần

A53 đã bác kênh 6: nhúng OCR/ASR bằng chính **tháp văn bản của SigLIP2**. Chỉ
kênh 6 đứng một mình được **0,0490**, so với 0,4779 của kênh 1 — gần như không
truy hồi được gì.

Nguyên nhân đã ghi rõ: SigLIP2 huấn luyện để khớp **ảnh ↔ chữ**, không phải
**chữ ↔ chữ**. So vector truy vấn với vector tài liệu là dùng model ngoài phân
bố huấn luyện; hai loại vector nằm hai cụm khác nhau nên khoảng cách giữa chúng
gần như không mang thông tin.

**BGE-M3 thì được huấn luyện đúng cho việc này** — text-to-text retrieval, đa
ngôn ngữ, có tiếng Việt, 1024 chiều. Đây không phải thử lại cùng một thứ; đây
là thay đúng bộ phận đã hỏng.

Điều nó chữa: kênh 3 dùng BM25 nên khớp **mặt chữ**. Truy vấn "xe cứu thương"
mà bản tin nói "xe cấp cứu" thì điểm bằng 0.

## Chi phí

| | |
| --- | ---: |
| tài liệu có chữ | 176.009 |
| đoạn ≤ 512 token | ~180.000 (BGE-M3 nhận 8192 token, **không cần chia nhỏ như A53**) |
| thời gian trên T4 | ~40–60 phút |
| ma trận float16 lúc thi | 176.009 × 1024 × 2 = **344 MB** |

⚠️ Máy thi 7,7 GB RAM **đã treo cứng một lần** vì nạp thẳng ma trận 1,32 GB
cùng lúc với BM25. Ma trận này nhỏ hơn, nhưng vẫn phải `mmap` — `src/van_ban_dense.py`
đã sửa để làm đúng thế, dùng lại được nguyên vẹn.

**Add Input: `aic2026-index`** (cần `master.parquet` + `ocr_asr.parquet`).
Không cần ảnh. Settings → **GPU T4**.

```python
# ── Nhúng OCR/ASR bằng BGE-M3 ────────────────────────────────────────
import subprocess, json, pathlib, shutil, glob, os, time

def chay(lenh, im=False):
    p = subprocess.run(lenh, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not im or p.returncode != 0:
        print(p.stdout[-4000:])
    if p.returncode != 0:
        raise RuntimeError(f"ma thoat {p.returncode}: {lenh}")
    return p.stdout

# KHONG `pip -U` — no keo pillow/pandas moi vao va pha vo torchvision co san.
chay("pip -q install --no-deps FlagEmbedding")
chay("pip -q install --no-deps sentence-transformers")

import torch
assert torch.cuda.is_available(), "Settings > Accelerator > GPU T4"
print("GPU:", torch.cuda.get_device_name(0))

import pandas as pd, numpy as np
hit = glob.glob("/kaggle/input/**/ocr_asr.parquet", recursive=True)
assert hit, "khong thay ocr_asr.parquet — Add Input dataset aic2026-index chua?"
d = pd.read_parquet(hit[0])
d = d[d["text"].fillna("").str.strip() != ""]
print(f"{len(d):,} tai lieu co chu")

# BGE-M3 nhan toi 8192 token nen KHONG phai chia doan nhu A53 (SigLIP2 chi 64).
# Van cat cung o 2000 ky tu: doan dai hon the la ASR ca doan tin, nhung phan
# duoi hiem khi lien quan den mot khung hinh cu the.
van = d["text"].str.slice(0, 2000).tolist()
rid = d["row_id"].to_numpy(np.int64)

from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

t0 = time.perf_counter()
vec = model.encode(van, batch_size=64, max_length=512)["dense_vecs"]
vec = np.asarray(vec, dtype=np.float32)
vec /= (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9)
print(f"\n{vec.shape} sau {(time.perf_counter()-t0)/60:.1f} phut")

# Dinh dang GIONG HET kenh 6 cu -> src/van_ban_dense.py dung lai duoc nguyen ven
np.savez("/kaggle/working/van_ban_bge.npz",
         vec=vec.astype(np.float16), row_id=rid,
         ghi_chu=json.dumps({"model": "BAAI/bge-m3", "chieu": int(vec.shape[1]),
                             "so_doan": len(van)}, ensure_ascii=False))
print("van_ban_bge.npz:",
      os.path.getsize("/kaggle/working/van_ban_bge.npz")/1024**2, "MB")

# ── Truy van cung phai ma hoa bang CHINH model nay ───────────────────
# Khong dung chung cache gopt duoc: khac model, khac so chieu, khac khong gian.
chay("rm -rf /tmp/repo && git clone -q -b giai-doan-0 "
     "https://github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git /tmp/repo")
import sys; sys.path.insert(0, "/tmp/repo/src")
import run as R

cau = set()
for f in sorted(pathlib.Path("/tmp/repo/dev").glob("*.jsonl")) + \
         sorted(pathlib.Path("/tmp/repo/de_thi_thu").glob("*.txt")):
    if f.suffix == ".jsonl":
        noi = [json.loads(l)["cau_hoi"] for l in f.read_text("utf-8").splitlines() if l.strip()]
    else:
        noi = [f.read_text("utf-8").strip()]
    for x in noi:
        muc = R.tach_su_kien(x) if "trake" in f.name.lower() else [x]
        for v in muc:
            cau.add(v)
            cau.update(R.tach_truy_van(v))
cau = sorted(cau)
print(f"\n{len(cau):,} chuoi truy van")

qv = model.encode(cau, batch_size=64, max_length=512)["dense_vecs"]
qv = np.asarray(qv, dtype=np.float32)
qv /= (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
np.savez("/kaggle/working/truy_van_bge.npz", cau=np.array(cau), vec=qv,
         ghi_chu=json.dumps({"model": "BAAI/bge-m3", "chieu": int(qv.shape[1])},
                            ensure_ascii=False))
print("✅ xong — tai ve ca HAI file:")
print("   van_ban_bge.npz  (ma tran tai lieu)")
print("   truy_van_bge.npz (cache truy van CUNG model)")
```

## Sau khi chạy

Tải **cả hai file** về `index/`, rồi trên máy có `index/`:

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe scripts\58_do_kenh6.py `
    --van-ban index\van_ban_bge.npz --cache index\truy_van_bge.npz
```

`58_do_kenh6.py` đã có sẵn từ A53 và đo đúng ba câu hỏi cần: kênh mới **thêm**
vào có lãi không, **thay** được kênh 3 không, và **đứng một mình** được bao
nhiêu.

⚠️ Dòng "chỉ kênh mới (chẩn đoán)" là dòng quan trọng nhất. A53 ra **0,0490** ở
đó và nhờ vậy mới biết mọi con số hợp nhất phía trên chỉ là kênh 1 đội lốt.
BGE-M3 mà cũng dưới ~0,15 thì hướng này chết y như lần trước, khỏi dò trọng số.

## Hai chỗ dễ sai

**Phải nộp cả cache truy vấn.** Không dùng chung `truy_van_gopt.npz` được: khác
model, khác số chiều (1024 so với 1536), khác không gian vector.
`van_ban_dense.py` sẽ **dừng** nếu số chiều lệch — đó là chủ ý, ghép nhầm hai
không gian là hỏng câm.

**Đừng `pip -U`.** Đã cắn: `-U` kéo pillow 12.3 + pandas 3.0.5 vào và phá vỡ
torchvision có sẵn, lỗi nổ tận lúc import model chứ không phải lúc cài.
