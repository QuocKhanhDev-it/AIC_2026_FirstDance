# 04 — Lỗi hay gặp

Đây là những lỗi **đã gặp thật** khi dựng bảng cái, không phải danh sách
phòng xa.

## `UnicodeEncodeError: 'charmap' codec can't encode characters`

Terminal Windows mặc định cp1252, không in được tiếng Việt lẫn khung bảng
của DuckDB.

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

Đặt một lần mỗi khi mở terminal mới. Muốn khỏi phải nhớ thì đặt vĩnh viễn:

```powershell
[Environment]::SetEnvironmentVariable('PYTHONIOENCODING','utf-8','User')
```

Rồi mở cửa sổ mới.

## `ffmpeg: not found` dù đã cài bằng winget

winget có sửa PATH nhưng **cửa sổ terminal đang mở không thấy được**. Đóng
hẳn terminal, mở cái mới.

Vẫn không thấy thì ffmpeg nằm ở:

```text
%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_*\ffmpeg-9.0-full_build\bin
```

Thêm đường dẫn đó vào PATH thủ công, hoặc gọi thẳng bằng đường dẫn đầy đủ.

## `TypeError: argument should be a str ... not 'float'`

Xảy ra ở `Path(rec.keyframe_dir)`. Nguyên nhân: parquet trả ô trống thành
`NaN` (kiểu float), mà **`NaN` là truthy trong Python** — nên
`if rec.keyframe_dir` vẫn lọt qua rồi crash.

Đã sửa trong `01_build_index.py`. Nếu bạn viết code mới đọc parquet, nhớ:

```python
df = df.astype(object).where(df.notna(), None)
```

## `AttributeError: 'DataFrame' object has no attribute 'video_id'`

pandas 3.0 **loại cột nhóm ra khỏi** `groupby(...).apply(...)`. Code cũ viết
theo pandas 1.x/2.x sẽ vỡ.

```python
# hỏng trên pandas 3.0
df.groupby('video_id', group_keys=False).apply(lambda g: g.sample(1))

# đúng
df.groupby('video_id', as_index=False).sample(n=1)
```

## Script chạy hàng chục phút ở phần objects

Bình thường. Phải đọc **177.321 file JSON nhỏ**, mà Windows Defender quét
từng lần mở file — đọc tuần tự chỉ được ~63 file/giây.

`build_objects` đã chuyển sang `ThreadPoolExecutor` (mặc định 16 luồng,
~250 file/giây, ~12 phút). Máy khỏe thì tăng lên:

```powershell
python scripts\01_build_index.py --out .\index --objects-only --workers 24
```

Nhanh hơn nữa: loại trừ thư mục dữ liệu khỏi Windows Defender (cần quyền
admin). Windows Security → Virus & threat protection → Manage settings →
Exclusions → Add folder → chọn thư mục dữ liệu.

Đây là **chi phí một lần**. Xong `objects.parquet` rồi thì không đụng lại
177k file JSON nữa.

## `problems.csv` có 844 dòng — có phải hỏng không?

Không. Xem cột `loai_loi`:

| Loại | Nghĩa | Nghiêm trọng? |
| --- | --- | --- |
| `lech_so_keyframe` | số ảnh ≠ số dòng CSV | **không**, thường là chưa tải ảnh |
| `thieu_metadata` | không có file media-info | không, một số video vốn không có |
| `thieu_csv`, `thieu_npy` | thiếu tài sản | tải tiếp là hết |
| `lech_so_vector` | **số vector CLIP ≠ số dòng CSV** | **CÓ — nghiêm trọng** |
| `sai_chieu_vector` | npy không phải 512 chiều | **CÓ** |

Hiện tại 844 dòng đều là `lech_so_keyframe` vì mới tải keyframes của L21.
**Không có dòng `lech_so_vector` nào** — đó mới là thứ cần lo.

## `kf_path` trỏ vào file không tồn tại

Đường dẫn trong `master.parquet` là **tuyệt đối, của máy người dựng index**.
Xem mục cuối [01_cai_dat.md](01_cai_dat.md) để sửa.

## `02_verify.py` chỉ kiểm được 29 mẫu dù `--n 60`

Đúng. Script chỉ lấy dòng có **cả** `kf_path` lẫn `video_path`, mà hiện chỉ
29 video L21 có đủ hai thứ. Tải thêm nhóm L nào thì số mẫu tự tăng.

## `git push` báo `could not read Username`

Chưa đăng nhập GitHub. Chạy `git push` trong terminal **của bạn** (không
phải qua công cụ tự động) để Git Credential Manager mở cửa sổ đăng nhập.

## Cài `pip install` báo lỗi biên dịch

Đang dùng Python quá mới nên chưa có wheel dựng sẵn. Lùi về 3.12 hoặc 3.11.
Đã chạy thật trên 3.13.5 — mọi gói trong `requirements.txt` đều có wheel.

## Không chắc lỗi gì

Ba file báo cáo trong `index/` ghi lại đầy đủ lần chạy gần nhất:

```text
index/discover_report.txt   độ phủ từng loại tài sản, phân bố theo nhóm L
index/build_report.txt      thống kê bảng cái, mật độ keyframe, fps, objects
index/verify_report.csv     từng mẫu kiểm chứng và kết luận
```

Mở ba file đó trước khi hỏi. Chúng cũng nằm trong git nên xem được cả trên
GitHub.
