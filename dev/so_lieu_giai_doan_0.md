# Số liệu Giai đoạn 0 — đo thật trên dữ liệu

*Cập nhật 2026-08-08: thêm L22 (máy này) và L29 (do thành viên khác gửi).*

Dữ liệu tại `C:\Code\aic_data`. **Máy này đã tải L21 + L22** (keyframes +
video); L23–L30 hiện chỉ có csv/clip/objects/media-info.

Kiểm chứng thì gộp được từ nhiều máy: xem [verify/](verify/) và chạy
`python scripts/07_gop_kiem_chung.py`.

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

4. **Bảng cái ĐÚNG.** `02_verify.py` chạy trên 143 mẫu thuộc 5 nhóm L
   (L21+L22 máy này; L29; L24+L30 do thành viên khác gửi): **143/143 đạt**
   (142 KHỚP + 1 KHOP_YEU). Cách ghép dòng CSV thứ i ↔ ảnh keyframe thứ i ↔
   `frame_idx` là chính xác.
   *Lưu ý:* chỉ kiểm được nhóm nào đã có video gốc. Tải thêm nhóm L nào thì
   chạy lại `02_verify.py` cho nhóm đó.

5. **Liên kết CSV ↔ clip.npy cũng ĐÚNG.** `03_verify_CLIP.py` encode lại
   frame bằng CLIP ViT-B/32 rồi so với vector đã lưu: **122/123 đúng hạng 1**,
   cosine 0,9506–1,000. Mẫu duy nhất ở hạng 2 (`L22_V013` kf 116) là keyframe
   TRÙNG LẶP — hai frame của cùng một đoạn đồ họa chuyển cảnh, cosine giữa
   chúng 0,9787 — không phải lệch chỉ số.

6. **`row_id` giống nhau trên mọi máy.** `00_discover.py` duyệt theo
   `sorted(video_id)` và `01_build_index.py` đánh `row_id` tuần tự, nên
   `row_id ↔ (video_id, kf_n)` là như nhau ở bất kỳ ai có đủ 873 file CSV.
   Đã **đo thật**, không phải suy từ code: đối chiếu 23 dòng L29 trong
   `verify_clip.csv` của máy khác với `master.parquet` của máy này —
   **23/23 trùng khít** cả `kf_n` lẫn `frame_idx`.

   Hệ quả: kết quả kiểm chứng của 6 người gộp được bằng `row_id`, và
   **không cần gửi `master.parquet`/`clip.npy` cho nhau** (395 MB, giống hệt
   nhau trừ ba cột đường dẫn tuyệt đối).

   Đã đối chiếu hai lô, **63/63 dòng trùng khít** cả `video_id`, `kf_n`,
   `frame_idx` lẫn `fps`: 23 dòng L29 và 40 dòng L24+L30.

7. **fps 26.44 đã hết là bẫy — nhưng 29.97 thì chưa.** `L24_V044` là video
   duy nhất trong kho chạy 26,44 fps, và giờ đã qua cả hai bài kiểm
   (pixel corr 0,9997; CLIP hạng 1, cosine 0,9891). Kiểm được đến mức này là
   nhờ nó khắc nghiệt: keyframe `003.jpg` ứng với `frame_idx=5`, tức
   **0,19 giây**, mà bài kiểm pixel vẫn phân biệt được với hai dòng kề
   (biên độ 0,148). Nếu code có chỗ nào làm tròn 26,44 → 25 hoặc 30 thì mẫu
   này đã trượt.

   Còn lại **30 video L25 chạy 29,97 fps — chưa ai kiểm.** Đây là nhóm fps lạ
   duy nhất còn hở, và là chỗ nguy hiểm nhất trong kho.

8. **`L24_V044` là video dày keyframe nhất kho — 42 keyframe trong 35,18 giây.**
   Ba keyframe cuối cách nhau đúng **1 frame** (928, 929, 930). Trung vị toàn
   kho là 55 frame, nên video này lệch hẳn khỏi phân bố. Đáng chú ý cho TRAKE:
   ở đây không cần trích dày, dữ liệu gốc đã dày sẵn.

   Nghĩa là vector thứ i trong `clip.npy` thật sự là vector của keyframe
   thứ i, không lệch hàng.

   Bẫy khi chạy script này: phải dùng tag `ViT-B-32-quickgelu`. Nạp
   `ViT-B-32` thường làm cosine tụt từ 0,9913 xuống 0,9513 và mọi mẫu
   trượt ngưỡng dù dữ liệu hoàn toàn đúng. Ngưỡng gốc 0,98 cũng quá gắt,
   đã hiệu chuẩn lại còn 0,95.

## Quy ước tên file (đã xác minh)

```text
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
| keyframes / video mp4 **trên máy này** | **60 / 873 (6,9%)** | L21 + L22 |
| **đã kiểm chứng — toàn nhóm** | **162 / 873 (18,6%)** | L21, L22, L24, L29, L30 — 5/10 nhóm |

Còn thiếu người kiểm: **L23, L25, L26, L27, L28**. Riêng fps lạ — sai một chỗ
là hỏng mọi phép quy đổi giây ↔ frame:

| Nhóm | fps | Số video | Đã kiểm |
| --- | --- | --- | --- |
| L24 | 26.44 | 1 (`L24_V044`) | ✅ **1/1** |
| L25 | 29.97 | 30 | ❌ **0 — chưa ai** |

Máy giữ L24+L30 đã tải đủ **cả 139 video** của hai nhóm nhưng mới kiểm 79.
Chạy lại `--n 139` cho cả hai script là độ phủ lên **222/873 (25,4%)** mà
không cần tải thêm gì.

`index/problems.csv` có 813 dòng `lech_so_keyframe` — đó là **chưa tải**,
không phải lỗi ghép. Không có dòng `lech_so_vector` nào (đây mới là lỗi
nghiêm trọng). Máy giữ L29 cũng báo y hệt: 850 dòng, toàn `lech_so_keyframe`.

Độ dài video: min 30s, trung vị 317s, max 2735s.

## Bốn chỗ đã sửa trong script

1. `01_build_index.py` — parquet trả ô trống thành `NaN` (float), mà `NaN`
   là truthy nên `if rec.keyframe_dir` lọt qua và crash. Đổi hết về `None`.

2. `01_build_index.py` — object trước đây chỉ khớp qua tên file **ảnh**
   keyframe, nên 844 video chưa tải ảnh bị bỏ object oan (báo cáo 4,4%).
   Nay khớp theo cột `n`, ra đúng 100%.

3. `02_verify.py` — pandas 3.0 loại cột nhóm khỏi `groupby.apply`; đổi sang
   `groupby.sample`.

4. `02_verify.py` — thêm phán quyết `KHOP_YEU`: tương quan pixel dưới 0,95
   nhưng vượt dòng kề ≥ 0,30 thì vẫn là ghép đúng. Cần vì tương quan pixel
   sụt mạnh trên cảnh động. Bằng chứng: `L22_V013/116.jpg` chỉ đạt corr
   0,675 nhưng đồng hồ trên hình đọc cùng `18:36:43` với frame trích từ
   video — đúng giây, chỉ là đoạn đồ họa chuyển cảnh chạy nhanh.

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
