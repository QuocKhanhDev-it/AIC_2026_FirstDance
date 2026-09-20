# Cell Kaggle — ĐO TỐC ĐỘ caption ở cấu hình rẻ (Nấc 2)

Mục đích: trả lời **một** câu hỏi trước khi ai đó tiêu 12 giờ phiên Kaggle —
ở `--diem-anh 256 --so-chu 120`, một ảnh mất bao nhiêu giây, và cả kho mất bao
lâu?

**ĐÃ ĐO XONG (31/08, T4):** 512px/180 chữ **3,77 s/ảnh**, 256px/180 chữ
**2,52**, 256px/120 chữ **2,10** — cả kho 103,4 giờ. Con số 2,85 s/ảnh ghi
trước đó là ƯỚC, và ước thấp hơn thực tế 32%. Cell này giữ lại để tái lập;
việc sinh caption thật thì xem `kaggle_caption_chay.md`.

⚠️ Đây là cell ĐO, không phải cell sinh dữ liệu. Nó chạy 3 cấu hình × 40 ảnh
rồi in bảng. Đừng để nó chạy cả kho.

Yêu cầu: Settings → Accelerator → **GPU T4 x2** (P100 KHÔNG dùng được: sm_60,
torch dựng cho sm_70+). Add Input → dataset `aic2026-index` + dataset ảnh của
phần được giao.

```python
# ── ĐO TỐC ĐỘ CAPTION — 3 cấu hình × 40 ảnh ──────────────────────────
import subprocess, json, pathlib, shutil, glob, os, time, re

def chay(lenh, im=False):
    p = subprocess.run(lenh, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # LUON in khi hong: `im=True` tung nuot mat traceback that va chi con
    # "ma thoat 1", khong doan duoc gi tu do.
    if not im or p.returncode != 0:
        print(p.stdout[-4000:])
    if p.returncode != 0:
        raise RuntimeError(f"ma thoat {p.returncode}: {lenh}")
    return p.stdout

chay("rm -rf /tmp/repo && git clone -q -b giai-doan-0 "
     "https://github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git /tmp/repo")
%cd /tmp/repo
# ⚠️ KHONG `pip -U` vao moi truong Kaggle. Ban truoc lam vay va pip keo theo
# pillow 12.3 + pandas 3.0.5, pha vo torchvision co san:
#     ImportError: cannot import name '_Ink' from 'PIL._typing'
# Kaggle da co torch/torchvision/pillow/pandas/pyarrow khop nhau san. Chi cai
# thu CHUA co, va luon `--no-deps` de pip khong dung toi nhung thu khac.
chay("pip -q install --no-deps qwen-vl-utils")

import transformers
print("transformers", transformers.__version__)
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception as e:
    print("ban transformers nay chua co Qwen2_5_VL:", e)
    chay("pip -q install --no-deps -U 'transformers>=4.51'")
    import importlib, transformers
    importlib.reload(transformers)
    from transformers import Qwen2_5_VLForConditionalGeneration
import qwen_vl_utils
print("✅ du thu vien")

import torch
assert torch.cuda.is_available(), "Settings > Accelerator > GPU T4"
print("GPU:", torch.cuda.get_device_name(0))

pathlib.Path("index").mkdir(exist_ok=True)
for ten in ("master.parquet",):
    hit = glob.glob(f"/kaggle/input/**/{ten}", recursive=True)
    assert hit, f"khong thay {ten} — Add Input dataset aic2026-index chua?"
    shutil.copy(hit[0], f"index/{ten}")
    print(f"  {ten:20} <- {hit[0]}")

# Ảnh: script đọc theo `kf_path` trong master, vốn là đường dẫn máy khác.
# `12_va_duong_dan.py` vá lại theo thư mục ảnh có trên máy NÀY.
#
# ⚠️ Cờ là `--roots` (nhận NHIỀU nơi), KHÔNG phải `--goc`. Và thiếu `--ghi`
# thì nó chỉ XEM TRƯỚC rồi thoát — bảng cái không đổi, caption sau đó không
# tìm thấy ảnh nào. Đưa thẳng `/kaggle/input`: nó tự quét đệ quy tìm mọi thư
# mục tên `L\d\d_V\d+` có ảnh, nên gắn bao nhiêu dataset ảnh cũng nhận hết.
anh = glob.glob("/kaggle/input/**/L*_V*/*.jpg", recursive=True)
assert anh, "khong thay anh keyframe trong /kaggle/input"
print(f"{len(anh):,} anh keyframe")
chay("python scripts/12_va_duong_dan.py --roots /kaggle/input --ghi")

# ── ba cấu hình, cùng 40 ảnh, chỉ đổi MỘT thứ mỗi lần so với dòng trên ──
CAU_HINH = [
    ("cũ    512px / 180 chữ", "--diem-anh 512 --so-chu 180"),
    ("rẻ    256px / 180 chữ", "--diem-anh 256 --so-chu 180"),
    ("rẻ+   256px / 120 chữ", "--diem-anh 256 --so-chu 120"),
]
N, KHO = 40, 177_321
bang = []
for ten, co in CAU_HINH:
    t0 = time.perf_counter()
    # `co-anh` = mọi dòng CÓ ẢNH ở máy này. Không có `tat-ca`.
    out = chay(f"python scripts/14_sinh_caption.py --backend hf --chon co-anh "
               f"--n {N} {co} --batch 8", im=True)
    giay = (time.perf_counter() - t0) / N
    bang.append((ten, giay))
    print(f"{ten}: {giay:.2f} s/anh")
    # in vài caption để soát CHẤT LƯỢNG — nhanh mà vô nghĩa thì vô dụng
    for d in out.splitlines()[-4:]:
        print("   ", d[:160])

print(f"\n{'cấu hình':<24}{'s/ảnh':>8}{'cả kho':>12}{'1 phần (6)':>14}")
print("-" * 58)
for ten, g in bang:
    kho_h, phan_h = g * KHO / 3600, g * KHO / 6 / 3600
    canh = "  ⚠️ VƯỢT 12h/phiên" if phan_h > 12 else ""
    print(f"{ten:<24}{g:>8.2f}{kho_h:>10.1f}h{phan_h:>12.1f}h{canh}")
```

## Đọc kết quả thế nào

Chỉ có một ngưỡng phải qua: **một phần (1/6 kho = 29.554 ảnh) phải dưới 12
giờ**, tức **dưới 1,46 s/ảnh**. Không qua thì chia nhỏ hơn 6 phần, hoặc bỏ
caption.

Và đừng chỉ nhìn tốc độ: cell in vài caption mẫu của từng cấu hình. Caption
120 chữ ở 256px có thể nhanh gấp ba mà mô tả rỗng tuếch ("a person in a
room") — lúc đó nhanh không cứu được gì, vì kênh 5 là BM25 trên chính những
chữ đó.

## Sau khi đo xong

Đừng sinh cả kho ngay. Sinh cho **các video tập dev đụng tới** trước
(`--chon tap:dev/tap_de_that.jsonl`), rồi đo kênh 5 có lãi không — đúng như
kênh 6 đã bị bác ở A53 sau khi mã hoá xong cả kho. Chi phí để biết một kênh vô
dụng nên là 40 phút, không phải 140 giờ.

⚠️ **Bẫy đã cắn ở A21 và ở tập TRAKE mới**: caption sinh cho ĐÚNG khung đáp án
làm mọi phép đo sau đó vô nghĩa. `14_sinh_caption.py` có `soat_ro_dap_an()` in
cảnh báo — đọc nó, đừng bỏ qua.
