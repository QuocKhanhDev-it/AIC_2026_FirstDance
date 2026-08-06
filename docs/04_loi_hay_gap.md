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

Muốn kiểm chứng các nhóm chưa có ảnh keyframe thì dùng `03_verify_CLIP.py`.
Nó trích frame rồi encode lại bằng CLIP ViT-B/32 và so cosine với vector đã
lưu, nên chỉ cần file `.mp4`:

```powershell
pip install -r requirements-clip.txt    # ~2,5 GB, chỉ cài nếu cần
python scripts\03_verify_CLIP.py --out .\index --n 40 --group L26
```

Hai script kiểm hai liên kết khác nhau: `02` kiểm **ảnh keyframe ↔ dòng CSV**,
`03` kiểm **vector CLIP ↔ dòng CSV**. Chạy cả hai mới phủ hết bảng cái.

## `03_verify_CLIP.py` báo NGHI_NGO/LỆCH hàng loạt

Gần như chắc chắn là **sai tag model**, không phải sai dữ liệu. Phải dùng
`ViT-B-32-quickgelu`, không phải `ViT-B-32`.

OpenAI CLIP dùng hàm kích hoạt QuickGELU. Nạp bản thường thì open_clip chỉ
in một dòng cảnh báo lẫn trong đống log rồi vẫn chạy — nhưng embedding lệch
đủ để mọi mẫu trượt ngưỡng. Đo trên L21 (nhóm đã biết chắc đúng):

| Tag model | Cosine trung bình | Thấp nhất |
| --- | --- | --- |
| `ViT-B-32` | 0,9513 | 0,9417 |
| `ViT-B-32-quickgelu` | **0,9913** | 0,9293 |

Cảnh báo cần để ý trong log:

```text
UserWarning: QuickGELU mismatch between final model config (quick_gelu=False)
and pretrained tag 'openai' (quick_gelu=True)
```

**Đừng tin cosine tuyệt đối, hãy nhìn cột `hang`.** Cosine nhạy với tiền xử
lý (JPEG vs PNG, cách resize, phiên bản model). Cột `hang` mới là bằng chứng:
nó hỏi vector vừa encode có giống dòng `row_id` của nó hơn *mọi* keyframe
khác trong cùng video không. Trên L21: **29/29 đúng hạng 1**, cách ứng viên
nhì trung bình **+0,15** — dù có một mẫu cosine chỉ 0,9293.

Hạng 1 mà cosine thấp là chuyện tiền xử lý. Hạng khác 1 mới là lệch chỉ số
thật.

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
