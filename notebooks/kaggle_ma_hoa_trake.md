# Cell Kaggle — mã hoá **219 chuỗi TRAKE** của `dev/tap_dev_trake.jsonl`

Đây là bước chặn duy nhất giữa hiện tại và việc **dò được bốn tham số lắp ráp
TRAKE**. A63 đo được khâu lắp ráp mất **52%** số điểm kênh đã tìm ra; A66 cho
thấy K-best beam search lấy lại **80%** phần đã mất ở ±15s. Cả hai đo trên
**3 câu** — quá ít để đổi mặc định.

`dev/tap_dev_trake.jsonl` có 14 câu nữa, và **câu tự soạn dùng được ở đây**:
thứ tự sự kiện và số Frame ID là ràng buộc **hình thức**, không phụ thuộc câu
hỏi dễ hay khó — khác hẳn việc so kênh, nơi A50/A58 đã cấm dùng câu tự soạn.
3 + 14 = **17 câu**, đủ để thấy hướng có ổn định không.

Nhưng 14 câu đó **không đo được**: cả 219 chuỗi của chúng chưa có trong
`index/truy_van_gopt.npz`, mà máy 7,7 GB không nạp nổi model để tự sinh.

## Vì sao tới 219 chuỗi cho 14 câu

Câu TRAKE bị tách **hai tầng**: tách **sự kiện** trước, rồi mỗi sự kiện lại
tách **mệnh đề**. Cache phải có đủ cả ba mức (câu gốc, từng sự kiện, từng mệnh
đề) thì `KenhAnhCache` mới chạy. Thiếu tầng giữa thì cache trông đủ mà
`75_do_lap_rap_trake.py` vẫn báo "thiếu chuỗi" — đã cắn thật: 14/17 câu không
đo được.

Rẻ tới mức buồn cười: A49 đo **69 chuỗi/giây**, tức **~3 giây GPU**. Toàn bộ
thời gian còn lại là tải model.

## Cell này KHÔNG cần cache cũ

Nó sinh một file **riêng** (`truy_van_trake.npz`), rồi máy có `index/` gộp vào
bằng `scripts/67_gop_cache_truy_van.py`. Đưa cache cũ lên rồi tải bản gộp về là
truyền hai lượt, mỗi lượt một cơ hội ghi đè nhầm — **mất cache cũ là mọi script
đo đều tắc**, vì máy đó không sinh lại được.

**Add Input: không cần gì cả.** Không ảnh, không `aic2026-index`.
Settings → Accelerator → **GPU T4** (P100 không dùng được — sm_60).

```python
# ── Ma hoa 219 chuoi TRAKE ───────────────────────────────────────────
import subprocess, json, pathlib, shutil, os

def chay(lenh):
    p = subprocess.run(lenh, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-3000:])
    if p.returncode != 0:
        raise RuntimeError(f"ma thoat {p.returncode}: {lenh}")

# ⚠️ URL NOI CHUOI CO Y — copy tu cho da render Markdown thi URL lien khoi bi
# boc thanh `[nhan](dich)` va `sh` chet: Syntax error: "(" unexpected.
KHO_GIT = "https://" + "github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git"

chay(f"rm -rf /tmp/repo && git clone -q -b giai-doan-0 {KHO_GIT} /tmp/repo")
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
# Ma hoa CA HAI file co cau TRAKE. `tap_de_thi_thu.jsonl` vua them 3 cau
# TRAKE de THAT (10 chuoi moi); phan da co trong cache se bi `67_` bo qua khi
# gop, nen ma hoa thua vai chuoi khong ton gi — con thieu thi tac ca phep do.
chay("python scripts/25_ma_hoa_truy_van.py "
     "--tap dev/tap_dev_trake.jsonl --tap dev/tap_de_thi_thu.jsonl "
     "--matrix clip_gopt.npy --ra index/truy_van_trake.npz")

shutil.copy("index/truy_van_trake.npz", "/kaggle/working/")

import numpy as np
z = np.load("/kaggle/working/truy_van_trake.npz", allow_pickle=False)
print(f"\n✅ {len(z['cau']):,} chuoi, {z['vec'].shape[1]} chieu")
print("   ghi chu:", json.loads(str(z["ghi_chu"])))
assert z["vec"].shape[1] == 1536, "sai so chieu — sai model?"
assert len(z["cau"]) >= 219, f"cho it nhat 219 chuoi, nhan {len(z['cau'])}"
print(f"   kich thuoc: "
      f"{os.path.getsize('/kaggle/working/truy_van_trake.npz')/1024:.0f} KB")
```

## Sau khi chạy

Tải `truy_van_trake.npz` từ tab **Output** về thư mục `index/`, rồi:

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe scripts\67_gop_cache_truy_van.py index\truy_van_trake.npz
.venv\Scripts\python.exe scripts\67_gop_cache_truy_van.py index\truy_van_trake.npz --ghi
```

Lần đầu chỉ **xem trước** — đọc kỹ rồi mới chạy lần hai với `--ghi`. Phải thấy
số chuỗi đi từ **1.239 → 1.458**. Script tự sao lưu bản cũ thành
`truy_van_gopt.npz.truoc_khi_gop` và tự dừng nếu model hoặc số chiều lệch.

Rồi mở khoá được cả hai phép đo đang tắc:

```powershell
.venv\Scripts\python.exe scripts\78_do_kbest_trake.py `
    --file dev\tap_de_that.jsonl dev\tap_de_thi_thu.jsonl dev\tap_dev_trake.jsonl
.venv\Scripts\python.exe scripts\75_do_lap_rap_trake.py
```

## Ngưỡng đặt TRƯỚC khi xem số

A66 đo trên 3 câu: ±15s **0,2500 → 0,4500**, ±2s **không nhúc nhích**. Trên 17
câu, để đổi mặc định trong `run.py` thì cần:

* ±15s vẫn **dương và vượt ngưỡng nhiễu** in kèm, và
* ±2s **không âm** vượt nhiễu.

Đảo dấu giữa hai mức = **không kết luận được**, không phải "hơi hơn". Ghi ra
trước để khỏi tự thuyết phục mình sau khi thấy con số.
