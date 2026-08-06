# 01 — Cài đặt: từ clone tới chạy được

Làm đúng thứ tự. Mất khoảng 15 phút nếu chỉ lấy `index/`.

## Bước 1 — Clone

```powershell
cd C:\Code
git clone https://github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git aic2026
cd aic2026
git checkout dev
```

Thư mục dự án đặt đâu cũng được, nhưng đường dẫn **đừng có dấu tiếng Việt
hay khoảng trắng** — ffmpeg và một số thư viện hay vấp.

## Bước 2 — Python

Cần Python 3.11 trở lên (đã chạy thật trên 3.13.5). Tải từ python.org,
**nhớ tích "Add Python to PATH"** lúc cài.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Thấy `(.venv)` ở đầu dòng lệnh là đúng. Mỗi lần mở terminal mới phải chạy
lại `.\.venv\Scripts\Activate.ps1`.

Nếu PowerShell báo "cannot be loaded because running scripts is disabled":

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Thư viện tùy chọn — chỉ cài nếu cần

`requirements.txt` (~50 MB) đủ cho hầu hết mọi người: truy vấn bảng cái,
viết code xếp hạng, thử ý tưởng.

Riêng `scripts/03_verify_CLIP.py` cần thêm torch + open_clip, **~2,5 GB**:

```powershell
pip install -r requirements-clip.txt
```

| Bạn làm gì | Cài gì |
| --- | --- |
| Truy vấn bảng cái, code xếp hạng, phân tích | `requirements.txt` |
| Chạy `02_verify.py` (cần ảnh keyframe + video) | `requirements.txt` + ffmpeg |
| Chạy `03_verify_CLIP.py` (kiểm chứng vector CLIP) | thêm `requirements-clip.txt` |

Bản `torch` trên PyPI là **CPU-only** — chạy được, ~1–2 giây/ảnh. Có GPU
NVIDIA và muốn dùng thì cài theo hướng dẫn ở <https://pytorch.org/get-started/locally/>,
script tự nhận `cuda`.

## Bước 3 — ffmpeg

**Chỉ cần nếu bạn tải video** (để chạy `02_verify.py` hoặc cắt clip). Không
tải video thì bỏ qua bước này.

```powershell
winget install Gyan.FFmpeg
```

Xong phải **đóng terminal, mở cửa sổ mới**, rồi kiểm tra:

```powershell
ffmpeg -version
```

Vẫn "not found" thì PATH chưa cập nhật — xem [04_loi_hay_gap.md](04_loi_hay_gap.md).

## Bước 4 — Lấy `index/`

`index/` **không nằm trong git** (395 MB, quá nặng). Lấy từ Drive của nhóm.

Tải về rồi giải nén vào `C:\Code\aic2026\index\`, phải có đủ:

```text
index/
├── master.parquet      2,9 MB   BẢNG CÁI — một dòng = một keyframe
├── clip.npy            346 MB   (177321, 512) float32 chuẩn L2
├── objects.parquet     45 MB    1.122.384 detection
├── paths.parquet       36 KB    bản đồ video_id -> đường dẫn
├── problems.csv        34 KB    video có vấn đề
└── *_report.txt        nhỏ      báo cáo 3 bước
```

Kiểm tra nhanh:

```powershell
python -c "import pandas as pd; d=pd.read_parquet('index/master.parquet'); print(len(d),'dong,',d.video_id.nunique(),'video')"
```

Phải ra `177321 dong, 873 video`.

## Bước 5 — Chạy thử

```powershell
$env:PYTHONIOENCODING = "utf-8"
python -c "import duckdb; duckdb.sql(\"SELECT video_id, count(*) n FROM 'index/master.parquet' GROUP BY 1 ORDER BY n DESC LIMIT 5\").show()"
```

Ra bảng 5 dòng là xong. Sang [02_bang_cai.md](02_bang_cai.md).

---

## QUAN TRỌNG — nếu bạn có tải dữ liệu về

Các cột `kf_path`, `obj_path`, `video_path` trong `master.parquet` là
**đường dẫn tuyệt đối trên máy người dựng index**, tức `C:\Code\aic_data\...`.

Máy bạn để dữ liệu chỗ khác thì ba cột này **trỏ vào hư không**. Mọi cột
khác (`row_id`, `frame_idx`, `pts_time`, `fps`, `title`...) và `clip.npy`,
`objects.parquet` vẫn dùng bình thường — chỉ ba cột đường dẫn là hỏng.

Hai cách xử lý:

**Cách 1 — để dữ liệu đúng chỗ đó.** Giải nén vào `C:\Code\aic_data\` với
đúng tên gói tải về. Không phải làm gì thêm. Đây là cách đỡ phiền nhất.

**Cách 2 — sửa đường dẫn trong bảng.** Dữ liệu của bạn ở `D:\AIC` chẳng hạn:

```python
import pandas as pd
m = pd.read_parquet('index/master.parquet')
for c in ['kf_path', 'obj_path', 'video_path']:
    m[c] = m[c].str.replace(r'C:\Code\aic_data', r'D:\AIC', regex=False)
m.to_parquet('index/master.parquet', index=False)
```

**Cách 3 — tự dựng lại từ đầu.** Nếu bạn tải đủ dữ liệu thì chạy lại 3 bước,
đường dẫn sẽ là của máy bạn. Mất ~35 phút, xem [../README.md](../README.md).

Đừng commit `master.parquet` đã sửa đường dẫn lên git — `index/` bị
`.gitignore` chặn sẵn rồi, nhưng nhớ đừng cố gỡ.
