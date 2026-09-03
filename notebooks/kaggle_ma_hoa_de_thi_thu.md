# Cell Kaggle — mã hoá 63 chuỗi truy vấn của `de_thi_thu/`

Bước chặn duy nhất trước khi soi 24 gói đề thi thử: **63 chuỗi truy vấn của
chúng chưa có trong `index/truy_van_gopt.npz`**. Không có vector thì
`KenhAnhCache` không chạy được, mà máy 7,7 GB thì không nạp nổi model.

Rẻ tới mức buồn cười: A49 đo **69 chuỗi/giây**, tức **~1 giây GPU**. Toàn bộ
thời gian còn lại là tải model.

## Cell này KHÔNG cần cache cũ

Nó sinh một file **riêng** (`truy_van_de_thi_thu.npz`, ~400 KB), rồi máy có
`index/` gộp vào bằng `scripts/67_gop_cache_truy_van.py`.

Cách kia — đưa cache cũ lên Kaggle rồi `--gop` rồi tải bản gộp về — phải truyền
file hai lượt cho 63 chuỗi, và mỗi lượt là một cơ hội ghi đè nhầm. **Mất cache
cũ là mọi script đo đều tắc**, vì máy này không sinh lại được. Cache cũ không
nên rời khỏi máy.

**Add Input: không cần gì cả.** Không cần ảnh, không cần `aic2026-index`.
Settings → Accelerator → **GPU T4** (P100 không dùng được — sm_60).

```python
# ── Mã hoá truy vấn de_thi_thu ───────────────────────────────────────
import subprocess, json, pathlib, shutil, os

def chay(lenh):
    p = subprocess.run(lenh, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-3000:])
    if p.returncode != 0:
        raise RuntimeError(f"ma thoat {p.returncode}: {lenh}")

chay("rm -rf /tmp/repo && git clone -q -b giai-doan-0 "
     "https://github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git /tmp/repo")
os.chdir("/tmp/repo")
chay("pip -q install open_clip_torch pandas pyarrow")

import torch
assert torch.cuda.is_available(), "Settings > Accelerator > GPU T4"
print("GPU:", torch.cuda.get_device_name(0))

# Sidecar: `25_` doc TEN MODEL va SO CHIEU o day, khong can ma tran 545 MB.
pathlib.Path("index").mkdir(exist_ok=True)
json.dump({"model": "ViT-gopt-16-SigLIP2-384", "pretrained": "webli",
           "chieu": 1536}, open("index/clip_gopt.json", "w"))

# KHONG co --gop: co tinh. File nay dung mot minh, may co index/ moi gop.
chay("python scripts/25_ma_hoa_truy_van.py --de de_thi_thu "
     "--matrix clip_gopt.npy --ra index/truy_van_de_thi_thu.npz")

shutil.copy("index/truy_van_de_thi_thu.npz", "/kaggle/working/")

import numpy as np
z = np.load("/kaggle/working/truy_van_de_thi_thu.npz", allow_pickle=False)
print(f"\n✅ {len(z['cau']):,} chuoi, {z['vec'].shape[1]} chieu")
print("   ghi chu:", json.loads(str(z["ghi_chu"])))
assert z["vec"].shape[1] == 1536, "sai so chieu — sai model?"
print(f"   kich thuoc: {os.path.getsize('/kaggle/working/truy_van_de_thi_thu.npz')/1024:.0f} KB")
```

## Sau khi chạy

Tải `truy_van_de_thi_thu.npz` từ tab **Output** về thư mục `index/`, rồi:

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe scripts\67_gop_cache_truy_van.py index\truy_van_de_thi_thu.npz
.venv\Scripts\python.exe scripts\67_gop_cache_truy_van.py index\truy_van_de_thi_thu.npz --ghi
```

Lần đầu chỉ **xem trước** — đọc kỹ rồi mới chạy lần hai với `--ghi`. Phải thấy
số chuỗi đi từ **1.158 → khoảng 1.221**. Script tự sao lưu bản cũ thành
`truy_van_gopt.npz.truoc_khi_gop` và tự dừng nếu model hoặc số chiều lệch.

Xong thì sang `docs/18_soi_de_thi_thu.md`.

## Nếu muốn đưa cache lên dataset cho cả nhóm

Không bắt buộc cho việc này, nhưng cần nếu người khác cũng phải chạy phép đo.
`index/truy_van_gopt.npz` chỉ **6,6 MB** nên đừng cập nhật dataset 339 MB —
tạo một dataset riêng, nhanh hơn nhiều:

```powershell
mkdir kaggle_cache
copy index\truy_van_gopt.npz kaggle_cache\
kaggle datasets init -p kaggle_cache
# sửa kaggle_cache\dataset-metadata.json: đặt title và id (<user>/aic2026-cache-truy-van)
kaggle datasets create -p kaggle_cache
# lần sau chỉ cần:
kaggle datasets version -p kaggle_cache -m "them 63 chuoi de_thi_thu"
```

Không có `kaggle` CLI thì làm bằng web: **Create → New Dataset**, kéo file vào.

⚠️ **Đặt Private.** Dataset công khai chứa dữ liệu BTC là chuyện khác hẳn một
repo mã nguồn công khai.
