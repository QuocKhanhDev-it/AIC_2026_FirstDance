# AI Challenge 2026 — First Dance

## Giai đoạn 0: Bảng cái

**Trạng thái: ĐÃ XONG.** `02_verify.py` cho **282/282 KHỚP (100%)** trên
7 nhóm L — vượt ngưỡng 95%. Được phép sang Giai đoạn 1.
Độ phủ kiểm chứng hiện **308/873 video (35,3%)**, 7/10 nhóm L;
**L24 và L30 đã phủ 100%** —
xem `python scripts/07_gop_kiem_chung.py`.

**Kế hoạch hiện hành: [docs/Ke_hoach_AIC2026_v4.md](docs/Ke_hoach_AIC2026_v4.md)** (thay thế v3).
**Mới vào nhóm? Đọc [docs/01_cai_dat.md](docs/01_cai_dat.md) trước.**
Toàn bộ tài liệu ở [docs/](docs/).

Số liệu đầy đủ: [dev/so_lieu_giai_doan_0.md](dev/so_lieu_giai_doan_0.md)

| Chỉ số | Giá trị |
| --- | --- |
| video_id tìm thấy | **873** (L21–L30) |
| keyframe trong bảng cái | **177.321** |
| ma trận CLIP | `(177321, 512)` float32, chuẩn hóa L2 |
| detection objects (≥ 0,3) | **1.122.384** — 6,3 / keyframe |
| kiểm chứng bằng ffmpeg | **282/282 KHỚP** (278 KHOP + 4 KHOP_YEU) |
| kiểm chứng vector CLIP | **281/286 đạt**, 0 lệch chỉ số |

## Bốn con số phải nhớ

- **% có object JSON: 100%** — object là kênh dùng được, đừng chốt bỏ
- **Trung vị mật độ keyframe: 55 frame** (p90 = 150) → vẫn phải trích dày cho TRAKE
- **fps tồn tại: 25.0 / 26.44 / 29.97 / 30.0** → CẤM hardcode fps
- **Tổng keyframe: 177.321** trên 873 video

## Bố cục

```text
C:\Code\aic2026\          DỰ ÁN (code, output — cái này giữ gọn)
├── .venv/                môi trường Python 3.13
├── scripts/              00_discover.py, 01_build_index.py, 02_verify.py
├── index/                OUTPUT ~395 MB — không đẩy git, đồng bộ qua Drive
├── src/                  code dùng chung (io.py, schema Candidate/Answer)
├── dev/                  ghi chú, số liệu, queries_draft.md
└── cache/                file tạm

C:\Code\aic_data\         DỮ LIỆU — giải nén tại chỗ, KHÔNG gom, KHÔNG copy
├── Keyframes_L21/keyframes/L21_V001/*.jpg
├── Videos_L21_a/video/L21_V001.mp4
├── objects-aic25-b1/objects/L21_V001/*.json
├── map-keyframes-aic25-b1/map-keyframes/L21_V001.csv
├── media-info-aic25-b1/media-info/L21_V001.json
└── clip-features-32-aic25-b1/clip-features-32/L21_V001.npy
```

Script quét **đệ quy** và tự nhận diện theo tên, nên không cần ép cấu trúc
gói tải về vào khuôn nào — cứ giải nén nguyên trạng, lồng sâu bao nhiêu tầng
cũng nhận ra.

### Nội dung `index/`

| File | Cỡ | Là gì |
| --- | --- | --- |
| `clip.npy` | 346 MB | `(177321, 512)` float32, chuẩn L2, cùng thứ tự `row_id` |
| `objects.parquet` | 45 MB | 1.122.384 detection, nối với master bằng `row_id` |
| `master.parquet` | 2,9 MB | **BẢNG CÁI** — một dòng = một keyframe |
| `thieu_tai_san.csv` | 0,2 MB | 813 video chưa tải keyframe/video |
| `paths.parquet` | 36 KB | bản đồ video_id → đường dẫn tài sản |
| `problems.csv` | 34 KB | video có vấn đề khi ghép |
| `*_report.txt`, `verify_report.csv` | nhỏ | báo cáo 3 bước |

Chỉ `index/` cần đồng bộ giữa 6 người. Không ai phải copy video cho nhau.

## Độ phủ hiện tại

| Tài sản | Số video | Ghi chú |
| --- | --- | --- |
| csv / clip / objects / media-info | 873 / 873 (100%) | đủ cả L21–L30 |
| keyframes / video mp4 **trên máy này** | **60 / 873 (6,9%)** | L21 + L22 |
| **đã kiểm chứng — toàn nhóm** | **308 / 873 (35,3%)** | L24+L30 phủ 100%; thiếu L23, L26, L27 |

`problems.csv` có 813 dòng `lech_so_keyframe` — đó là **chưa tải**, không phải
lỗi ghép. Không có dòng `lech_so_vector` nào (đây mới là lỗi nghiêm trọng).

Tải thêm nhóm L nào thì chạy lại cả 3 bước rồi `02_verify.py` cho nhóm đó.

### Gửi kết quả kiểm chứng cho cả nhóm

Chỉ cần **hai file**, mỗi file vài KB:

```text
index/verify_report.csv      <- script 02   (ảnh keyframe ↔ dòng CSV)
index/verify_clip*.csv       <- script 03   (vector CLIP ↔ dòng CSV)
```

Bỏ vào `dev/verify/<nhóm_L>/`, đổi tên thành `verify_report.csv` và
`verify_clip.csv`, rồi:

```powershell
python scripts\07_gop_kiem_chung.py
```

**Đừng gửi `master.parquet` / `clip.npy` / `objects.parquet`** — mỗi bộ
395 MB và giống hệt nhau trên mọi máy trừ ba cột đường dẫn tuyệt đối.
`row_id` là như nhau ở mọi máy (đã đối chiếu thật 226/226 dòng của bốn lô),
nên gộp bằng `row_id` là an toàn.

## Setup từ đầu

```powershell
cd C:\Code\aic2026
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
winget install Gyan.FFmpeg
```

Sau khi cài ffmpeg phải **mở cửa sổ PowerShell mới** rồi kiểm tra
`ffmpeg -version`. Nếu PowerShell chặn script:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Terminal Windows mặc định cp1252 — in tiếng Việt hoặc bảng DuckDB sẽ crash
`UnicodeEncodeError`. Đặt `$env:PYTHONIOENCODING = "utf-8"` trước khi chạy.

## Ba bước

```powershell
python scripts\00_discover.py   --roots "C:\Code\aic_data" --out .\index
python scripts\01_build_index.py --out .\index --limit 50    # thử nhanh trước
python scripts\01_build_index.py --out .\index
python scripts\02_verify.py     --out .\index --n 60
```

Điều kiện qua Giai đoạn 0: `verify_report.csv` cho ≥ 95% kết luận `KHOP`.

Có thêm bước kiểm chứng tùy chọn cho nhóm L **chưa tải ảnh keyframe** —
`02_verify.py` bó tay ở đó vì cần ảnh, còn script này chỉ cần file `.mp4`:

```powershell
pip install -r requirements-clip.txt        # ~2,5 GB, chỉ cài nếu chạy bước này
python scripts\03_verify_CLIP.py --out .\index --n 40 --group L26
```

`02` kiểm **ảnh keyframe ↔ dòng CSV**, `03` kiểm **vector CLIP ↔ dòng CSV**.
Trên 7 nhóm L cả hai đều sạch: 282/282 KHỚP và 281/286 đạt, không mẫu nào
lệch chỉ số.

Thời gian thực tế trên máy Windows đã đo:

| Bước | Thời gian |
| --- | --- |
| `00_discover.py` | ~2 phút (duyệt 188.683 mục) |
| `01_build_index.py` | ~20 phút phần bảng cái + ~12 phút phần objects |
| `02_verify.py --n 60` | ~1 phút (29 mẫu) |

### Cờ hữu ích

```powershell
# Sửa ngưỡng object rồi dựng lại objects.parquet, KHÔNG dựng lại bảng cái
python scripts\01_build_index.py --out .\index --objects-only --min-obj-score 0.5

# Chỉnh số luồng đọc file JSON (mặc định 16)
python scripts\01_build_index.py --out .\index --objects-only --workers 24

# Nhiều nơi chứa dữ liệu
python scripts\00_discover.py --roots "C:\Code\aic_data" "D:\Downloads" "E:\AIC" --out .\index
```

## Truy vấn bảng cái bằng DuckDB

Không cần server, không cần Supabase:

```python
import duckdb
con = duckdb.connect()

# Keyframe có đủ cả Person và Car với độ tin cậy cao
con.sql("""
    SELECT m.video_id, m.frame_idx, m.kf_name
    FROM 'index/objects.parquet' o JOIN 'index/master.parquet' m USING (row_id)
    WHERE o.score >= 0.5
    GROUP BY 1,2,3
    HAVING count(*) FILTER (WHERE o.label='Person') > 0
       AND count(*) FILTER (WHERE o.label='Car')    > 0
    LIMIT 20
""").show()

# Tìm video theo metadata (tiền thân của kênh BM25 ở Giai đoạn 1)
con.sql("""
    SELECT DISTINCT video_id, title FROM 'index/master.parquet'
    WHERE lower(description) LIKE '%gốm%'
""").show()
```

Metadata là tiếng Việt **có dấu**, dùng được cho BM25.

## Hiệu năng: đọc file nhỏ trên Windows

Chỗ nghẽn duy nhất của Giai đoạn 0 là 177.321 file object JSON. Đọc tuần tự
chỉ **63 file/giây** (Defender quét mỗi lần mở file) → ~47 phút. `build_objects`
đã chuyển sang `ThreadPoolExecutor`: **253 file/giây** → ~12 phút. Nghẽn ở I/O
chứ không phải CPU nên thread ăn thua ngay.

Muốn nhanh hơn: loại trừ `C:\Code\aic_data` khỏi Windows Defender (cần admin).

Đây là chi phí **một lần**. Xong `objects.parquet` rồi thì không đụng lại
177k file JSON nữa — nén hoặc xóa thư mục đó cho Windows Explorer đỡ ì.
