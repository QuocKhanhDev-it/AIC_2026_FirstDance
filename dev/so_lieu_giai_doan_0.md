# Số liệu Giai đoạn 0 — đo thật trên dữ liệu, ngày 2026-08-05

Dữ liệu tại `C:\Code\aic_data`. **Mới tải L21** (keyframes + video);
L22–L30 hiện chỉ có csv/clip/objects/media-info.

## Bốn con số phải nhớ

| Con số | Giá trị | Ý nghĩa |
| --- | --- | --- |
| **% có object JSON** | **100,0%** | 177.321 file JSON = đúng 177.321 dòng CSV, khớp 1-1 |
| **Trung vị mật độ keyframe** | **55 frame** | p90 = 150; chỉ 12,62% cặp cách ≤ 10 frame |
| **Các fps tồn tại** | **25.0 / 26.44 / 29.97 / 30.0** | 4 giá trị → CẤM hardcode fps |
| **Tổng số keyframe** | **177.321** | trên 873 video |

## Kết luận rút ra

1. **Object là kênh dùng được — đừng bỏ.** 1.122.384 detection ở ngưỡng ≥ 0,3
   (6,3 detection/keyframe). Top nhãn: Person, Clothing, Human face, Food,
   Man, Tree, Woman, Table, Human hand, Flower...
   Lo ngại "tên file thưa (001, 005, 008...) nên object vô dụng" là **sai**:
   tên file thưa là ở *video_id* (L21_V004 không tồn tại), không phải ở
   keyframe. Trong mỗi video, keyframe đánh số liên tục 001, 002, 003...

2. **Vẫn phải xây module trích dày cho TRAKE.** Trung vị 55 frame ≈ 1,8 giây
   ở 30 fps. Xa mức đủ dày.

3. **fps 26.44 là cái bẫy.** Mọi hàm quy đổi giây ↔ frame phải nhận `fps`
   làm tham số, đọc từ cột `fps` của `master.parquet`.

4. **Bảng cái ĐÚNG.** `02_verify.py` chạy 29 mẫu trên 29 video L21:
   **29/29 KHỚP**, tương quan 0,990–1,000. Cách ghép dòng CSV thứ i ↔ ảnh
   keyframe thứ i ↔ `frame_idx` là chính xác.
   *Lưu ý:* chỉ kiểm được trên L21 vì các nhóm L khác chưa có video gốc.
   Tải thêm nhóm L nào thì chạy lại `02_verify.py` cho nhóm đó.

## Quy ước tên file (đã xác minh)

```
csv:  n=1, pts_time=0.0,    fps=30.0, frame_idx=0
      n=2, pts_time=3.0,    fps=30.0, frame_idx=90
      ...
ảnh:  001.jpg, 002.jpg, ...      (zero-pad 3 chữ số)
json: 001.json, 002.json, ...    (cùng số thứ tự)
```

Tức `n` ↔ `f"{n:03d}.jpg"` ↔ `f"{n:03d}.json"`.

## Độ phủ hiện tại

| | Số video | Ghi chú |
| --- | --- | --- |
| csv / clip / objects / media-info | 873 / 873 (100%) | đủ cả 10 nhóm L21–L30 |
| keyframes / video mp4 | 29 / 873 (3,3%) | chỉ L21, gói `Videos_L21_a` |

`index/problems.csv` có 844 dòng `lech_so_keyframe` — đó là **chưa tải**,
không phải lỗi ghép. Không có dòng `lech_so_vector` nào (đây mới là lỗi
nghiêm trọng).

Độ dài video: min 30s, trung vị 317s, max 2735s.

## Hai chỗ đã sửa trong script

1. `01_build_index.py` — parquet trả ô trống thành `NaN` (float), mà `NaN`
   là truthy nên `if rec.keyframe_dir` lọt qua và crash. Đổi hết về `None`.

2. `01_build_index.py` — object trước đây chỉ khớp qua tên file **ảnh**
   keyframe, nên 844 video chưa tải ảnh bị bỏ object oan (báo cáo 4,4%).
   Nay khớp theo cột `n`, ra đúng 100%.

3. `02_verify.py` — pandas 3.0 loại cột nhóm khỏi `groupby.apply`; đổi sang
   `groupby.sample`.

## Hiệu năng — đọc file nhỏ trên Windows

Đọc 177.321 file JSON tuần tự: **63 file/giây → ~47 phút**. Defender quét
mỗi lần mở file. Đã đổi sang `ThreadPoolExecutor` 24 luồng: **253 file/giây
→ 12 phút**. Muốn nhanh nữa thì loại trừ `C:\Code\aic_data` khỏi Defender.

Chi phí một lần: xong `objects.parquet` (2,7 MB) thì không đụng lại 177k
file JSON nữa.

## Vặt vãnh nhưng hay vấp

- Terminal Windows mặc định cp1252, in tiếng Việt / bảng DuckDB sẽ crash
  `UnicodeEncodeError`. Đặt `PYTHONIOENCODING=utf-8` trước khi chạy.
- ffmpeg cài bằng winget nằm ở
  `%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_*\ffmpeg-9.0-full_build\bin`,
  phải **mở cửa sổ terminal mới** thì PATH mới có.
