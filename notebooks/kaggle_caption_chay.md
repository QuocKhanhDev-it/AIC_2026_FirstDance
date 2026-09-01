# Cell Kaggle — SINH caption (kênh 5)

Hai cell, dùng cho hai việc khác nhau. **Chạy cell A trước, đo, rồi mới quyết
định có chạy cell B hay không.**

Tốc độ đã đo trên T4 (`kaggle_caption_dothu.md`):

| cấu hình | s/ảnh | cả kho |
| --- | ---: | ---: |
| 512px / 180 chữ | 3,77 | 185,7 h |
| 256px / 180 chữ | 2,52 | 123,9 h |
| **256px / 120 chữ** ← dùng cái này | **2,10** | **103,4 h** |

---

## Cell A — chỉ 47 video của tập đề thật (**6,1 giờ, một phiên**)

52 câu đề thật chỉ đụng tới 47 video = **10.488 ảnh = 5,9% kho**. Sinh caption
cho riêng chúng là đủ để **đo kênh 5 có lãi hay không**.

Nếu kênh 5 vô dụng, ta vừa tiết kiệm 97 giờ GPU. Đó đúng là bài học A53: kênh 6
mã hoá xong cả kho rồi mới biết nó không chạy.

Add Input: `aic2026-index` + các dataset ảnh. Settings → **GPU T4 x2**.

```python
# ── CELL A: caption cho video tập đề thật (~6,1 giờ) ─────────────────
import subprocess, pathlib, shutil, glob, os, time

def chay(lenh, im=False):
    p = subprocess.run(lenh, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not im or p.returncode != 0:
        print(p.stdout[-4000:])
    if p.returncode != 0:
        raise RuntimeError(f"ma thoat {p.returncode}: {lenh}")
    return p.stdout

chay("rm -rf /tmp/repo && git clone -q -b giai-doan-0 "
     "https://github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git /tmp/repo")
os.chdir("/tmp/repo")
chay("pip -q install --no-deps qwen-vl-utils")

import transformers
print("transformers", transformers.__version__)
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception as e:
    print("thieu Qwen2_5_VL:", e)
    chay("pip -q install --no-deps -U 'transformers>=4.51'")
    import importlib; importlib.reload(transformers)
    from transformers import Qwen2_5_VLForConditionalGeneration
import qwen_vl_utils

import torch
assert torch.cuda.is_available(), "Settings > Accelerator > GPU T4"
print("GPU:", torch.cuda.get_device_name(0))

pathlib.Path("index").mkdir(exist_ok=True)
hit = glob.glob("/kaggle/input/**/master.parquet", recursive=True)
assert hit, "khong thay master.parquet — Add Input dataset aic2026-index chua?"
shutil.copy(hit[0], "index/master.parquet")
chay("python scripts/12_va_duong_dan.py --roots /kaggle/input --ghi")

t0 = time.perf_counter()
chay("python scripts/14_sinh_caption.py --backend hf "
     "--chon tap:dev/tap_de_that.jsonl --diem-anh 256 --so-chu 120 --batch 8")
print(f"\nxong sau {(time.perf_counter()-t0)/3600:.2f} gio")

chay("python scripts/14_sinh_caption.py --bien")     # jsonl -> parquet
for f in ("caption.jsonl", "caption.parquet"):
    shutil.copy(f"index/{f}", "/kaggle/working/")
    print(f"  {f}: {os.path.getsize(f'/kaggle/working/{f}')/1024**2:.1f} MB")
```

Tải cả hai file về `index/` rồi báo lại — phép đo kênh 5 chạy ở máy có tập dev.

⚠️ `soat_ro_dap_an()` sẽ cảnh báo vì cách chọn này dồn caption vào vùng có đáp
án. Ở đây **chấp nhận được**: nó sinh cho MỌI ảnh của 47 video, nên trong mỗi
video các khung vẫn cạnh tranh công bằng. Cái A21 cấm là sinh cho RIÊNG khung
đáp án.

---

## Cell B — cả kho, chia 12 phần (mỗi phần ~8,6 giờ)

Cell A đã chạy: kênh 5 đứng một mình đạt **0,3904** so với 0,6474 của kênh 1
trong cùng bể — một kênh truy hồi thật sự chạy được (A59). Lãi khi hợp nhất
+0,0106 🟡, dưới ngưỡng nhiễu của 52 câu, nên con số cuối cùng phải đợi tập đề
thật lên 76 câu. Nhóm đã quyết định cứ sinh cho cả kho.

`chia_caption/phan_1.txt` … `phan_12.txt`: **826 video, 166.833 ảnh** — đã TRỪ
47 video mà cell A caption rồi, khỏi ai phải chạy lại 6 giờ cho việc đã có kết
quả. Chia theo LPT, lệch **30 ảnh** giữa phần nặng nhất và nhẹ nhất, không phần
nào trùng nhau.

Mỗi phần ~13.900 ảnh × 2,10 s = **8,1 giờ** — an toàn dưới trần 12 giờ một
phiên. Tổng **97,3 giờ**.

Chia 12 chứ không 6 vì 6 phần là 17,2 giờ/phần — **vượt trần, mất trắng phiên
đó**.

**KHÔNG cần tài khoản mới.** Quota là 30 giờ/tuần mỗi tài khoản, 6 người là 180
giờ — thừa cho 97,3 giờ. Mỗi người nhận 2 phần, chạy hai lượt, tổng 16,2 giờ,
vẫn còn dư một nửa quota. (Kaggle cho mỗi người MỘT tài khoản; lập thêm là vi
phạm điều khoản và có thể mất cả tài khoản lẫn dữ liệu đã tải lên — mà ở đây
cũng không cần.)

**Mỗi người đổi đúng một dòng: `PHAN = <số của mình>`.**

```python
# ── CELL B: caption một phần của kho (~8,6 giờ) ──────────────────────
PHAN = 1          # ⚠️ ĐỔI SỐ NÀY THEO PHẦN ĐƯỢC GIAO (1..12)

import subprocess, pathlib, shutil, glob, os, time

def chay(lenh, im=False):
    p = subprocess.run(lenh, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not im or p.returncode != 0:
        print(p.stdout[-4000:])
    if p.returncode != 0:
        raise RuntimeError(f"ma thoat {p.returncode}: {lenh}")
    return p.stdout

chay("rm -rf /tmp/repo && git clone -q -b giai-doan-0 "
     "https://github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git /tmp/repo")
os.chdir("/tmp/repo")
chay("pip -q install --no-deps qwen-vl-utils")

import transformers
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:
    chay("pip -q install --no-deps -U 'transformers>=4.51'")
    import importlib; importlib.reload(transformers)
import qwen_vl_utils, torch
assert torch.cuda.is_available(), "Settings > Accelerator > GPU T4"
print("GPU:", torch.cuda.get_device_name(0))

pathlib.Path("index").mkdir(exist_ok=True)
hit = glob.glob("/kaggle/input/**/master.parquet", recursive=True)
assert hit, "khong thay master.parquet"
shutil.copy(hit[0], "index/master.parquet")
chay("python scripts/12_va_duong_dan.py --roots /kaggle/input --ghi")

# Kiem TRUOC khi chay 8 tieng: phan nay co du anh tren may nay khong?
import pandas as pd
vids = [x.strip() for x in open(f"chia_caption/phan_{PHAN}.txt") if x.strip()]
m = pd.read_parquet("index/master.parquet")
d = m[m.video_id.isin(vids)]
co = int(d.kf_path.notna().sum())
print(f"phan {PHAN}: {len(vids)} video, {len(d):,} anh, {co:,} co file")
assert co > len(d) * 0.98, (
    f"THIEU ANH: chi {co:,}/{len(d):,}. Add Input du cac dataset L2x chua? "
    f"Chay tiep la mat 8 tieng de sinh caption cho mot phan kho.")
print(f"uoc: {len(d)*2.10/3600:.1f} gio")

t0 = time.perf_counter()
chay(f"python scripts/14_sinh_caption.py --backend hf "
     f"--chon video:chia_caption/phan_{PHAN}.txt "
     f"--diem-anh 256 --so-chu 120 --batch 8")
print(f"\nxong sau {(time.perf_counter()-t0)/3600:.2f} gio")

chay("python scripts/14_sinh_caption.py --bien")
for f, t in (("caption.jsonl", "jsonl"), ("caption.parquet", "parquet")):
    ra = f"/kaggle/working/caption_phan{PHAN}.{t}"
    shutil.copy(f"index/{f}", ra)
    print(f"  {ra}: {os.path.getsize(ra)/1024**2:.1f} MB")
```

## Quy tắc bắt buộc khi nhiều người cùng chạy

**1. Cùng một cấu hình, không ngoại lệ: `--diem-anh 256 --so-chu 120`.** Caption
dài ngắn khác nhau giữa các phần thì BM25 chấm lệch theo phần, và không ai nhìn
ra được điều đó từ kết quả.

**2. Đổi tên file khi nộp** — cell đã tự thêm `_phan<N>`. Hai người nộp cùng
tên `caption.parquet` là một bản đè bản kia, im lặng.

**3. Đừng nhận phần chưa có ảnh.** Cell tự kiểm và dừng nếu thiếu quá 2% ảnh —
kiểm mất 5 giây, chạy nhầm mất 8 tiếng.

**4. Nộp cả `.jsonl` lẫn `.parquet`.** `.jsonl` ghi nối từng dòng nên còn dùng
được khi phiên chết giữa chừng; `.parquet` chỉ có nếu chạy trọn.
