# Cell Kaggle — mã hoá 63 chuỗi truy vấn của `de_thi_thu/`

Bước chặn duy nhất trước khi soi 24 gói đề thi thử: **63 chuỗi truy vấn của
chúng chưa có trong `index/truy_van_gopt.npz`**. Không có vector thì
`KenhAnhCache` không chạy được, mà máy 7,7 GB thì không nạp nổi model.

Rẻ tới mức buồn cười: A49 đo **69 chuỗi/giây**, tức **~1 giây GPU**. Toàn bộ
thời gian là tải model. Gộp chung vào một lượt chạy khác nếu tiện.

Yêu cầu: GPU T4 (P100 không dùng được — sm_60). Add Input: **`aic2026-index`**
(cần `truy_van_gopt.npz` để gộp vào). Không cần dataset ảnh.

```python
# ── Mã hoá truy vấn de_thi_thu, gộp vào cache gopt ───────────────────
import subprocess, json, pathlib, shutil, glob, os

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

pathlib.Path("index").mkdir(exist_ok=True)
hit = glob.glob("/kaggle/input/**/truy_van_gopt.npz", recursive=True)
assert hit, "khong thay truy_van_gopt.npz — Add Input dataset aic2026-index chua?"
shutil.copy(hit[0], "index/truy_van_gopt.npz")
print("cache cu <-", hit[0])

# Sidecar: `25_` đọc TÊN MODEL và SỐ CHIỀU ở đây, không cần ma trận 545 MB.
json.dump({"model": "ViT-gopt-16-SigLIP2-384", "pretrained": "webli",
           "chieu": 1536}, open("index/clip_gopt.json", "w"))

# --gop = thêm vào cache cũ, không ghi đè. Mất chuỗi cũ là mọi phép đo
# đang chạy tắc hết, nên đây là cờ bắt buộc.
chay("python scripts/25_ma_hoa_truy_van.py --de de_thi_thu "
     "--matrix clip_gopt.npy --gop --ra index/truy_van_gopt.npz")

shutil.copy("index/truy_van_gopt.npz", "/kaggle/working/")
import numpy as np
z = np.load("/kaggle/working/truy_van_gopt.npz", allow_pickle=False)
print(f"\n✅ cache moi: {len(z['cau']):,} chuoi, {z['vec'].shape[1]} chieu")
print("   (truoc do 1.158 chuoi — phai TANG, khong duoc giam)")
```

## Sau khi chạy

Tải `truy_van_gopt.npz` từ tab **Output**, ghi đè `index/truy_van_gopt.npz`,
rồi trên máy có index:

```powershell
.venv\Scripts\python.exe scripts\66_soat_de_thi_thu.py
```

Ra `dev/soat_de_thi_thu.html` — mở bằng trình duyệt, bấm chọn ô ảnh đúng, bấm
**Xuất JSONL**, chép kết quả vào `dev/tap_de_thi_thu.jsonl`.

⚠️ Trước khi ghi đè cache cũ, kiểm số chuỗi: phải **tăng** từ 1.158 lên
khoảng 1.221. Nếu giảm là `--gop` không ăn và bạn vừa mất cache của cả tập dev
— lúc đó mọi script đo đều tắc.
