# Số liệu Giai đoạn 0 — đo thật trên dữ liệu

*Cập nhật 2026-08-09. Gộp kết quả kiểm chứng của 5 máy: L21+L22 (máy này),
L29, L24+L30, L25+L28, L23+L26+L27. Độ phủ **97,0%**.*

Dữ liệu tại `C:\Code\aic_data`. **Máy này đã tải L21 + L22** (keyframes +
video); L23–L30 hiện chỉ có csv/clip/objects/media-info.

Kiểm chứng gộp được từ nhiều máy: xem [verify/](verify/), chạy
`python scripts/07_gop_kiem_chung.py`.

## Bốn con số phải nhớ

| Con số | Giá trị | Ý nghĩa |
| --- | --- | --- |
| **% có object JSON** | **100,0%** | 177.321 file JSON = đúng 177.321 dòng CSV, khớp 1-1 |
| **Trung vị mật độ keyframe** | **55 frame** | p90 = 150; chỉ 12,62% cặp cách ≤ 10 frame |
| **Các fps tồn tại** | **25.0 / 26.44 / 29.97 / 30.0** | 4 giá trị → CẤM hardcode fps |
| **Tổng số keyframe** | **177.321** | trên 873 video |

## Độ phủ kiểm chứng

| | Số video | Ghi chú |
| --- | --- | --- |
| csv / clip / objects / media-info | 873 / 873 (100%) | đủ cả 10 nhóm L21–L30 |
| keyframes / video mp4 **trên máy này** | **60 / 873 (6,9%)** | L21 + L22 |
| **đã kiểm chứng — toàn nhóm** | **847 / 873 (97,0%)** | **đủ 10/10 nhóm L** |

**Năm nhóm phủ 100%:** L23, L24, L26, L27, L30. Riêng L26 — nhóm chiếm 57%
kho — nay kiểm đủ **498/498 video ở cả hai script**.

Còn thiếu:

| Nhóm | Video | 02 | 03 | Thiếu |
| --- | --- | --- | --- | --- |
| L25 | 88 | 47 | 40 | 26 video |
| L28 | 24 | 13 | **24** | script 02 thiếu 11 |

L28 đã đủ ở script 03 (phép kiểm quan trọng hơn cho `clip.npy`), chỉ script 02
còn thiếu. Máy giữ L25+L28 chạy lại với `--n 88 --group L25` và
`--n 24 --group L28` là xong 873/873.

`index/problems.csv` có 813 dòng `lech_so_keyframe` — đó là **chưa tải**,
không phải lỗi ghép. **Không máy nào có dòng `lech_so_vector`** (đây mới là
lỗi nghiêm trọng).

Độ dài video: min 30s, trung vị 317s, max 2735s.

## Kết luận rút ra

### 1. Bảng cái ĐÚNG — ảnh keyframe ↔ dòng CSV

`02_verify.py` chạy trên 821 mẫu thuộc cả 10 nhóm L: **821/821 đạt**
(817 `KHOP` + 4 `KHOP_YEU`). Cách ghép dòng CSV thứ *i* ↔ ảnh keyframe thứ
*i* ↔ `frame_idx` là chính xác.

`KHOP_YEU` là phán quyết cho cảnh động: tương quan pixel dưới 0,95 nhưng
vượt dòng kề ≥ 0,30. Bằng chứng hiệu chuẩn: `L22_V013/116.jpg` chỉ đạt corr
0,675 nhưng đồng hồ trên hình đọc cùng `18:36:43` với frame trích từ video —
đúng giây, chỉ là đoạn đồ họa chuyển cảnh chạy nhanh.

*Lưu ý:* chỉ kiểm được nhóm nào đã có video gốc.

### 2. Liên kết CSV ↔ clip.npy cũng ĐÚNG

`03_verify_CLIP.py` encode lại frame bằng CLIP ViT-B/32 rồi so với vector đã
lưu: **819/825 đạt, KHÔNG mẫu nào lệch chỉ số.**

- 6 mẫu `NGHI_NGO` — đúng hạng 1, cosine 0,919–0,947, cách biệt hạng 2 dương
  (+0,03…+0,16). Khác biệt tiền xử lý JPEG/resize, không phải lệch chỉ số.
- 13 mẫu `KHOP_TRUNG_LAP` — cosine ≥ 0,95 nhưng không đúng hạng 1 vì dòng
  thắng là **bản sao** của chính nó. Xem mục 6.

Nghĩa là vector thứ *i* trong `clip.npy` thật sự là vector của keyframe thứ
*i*, không lệch hàng.

**Bẫy khi chạy script này:** phải dùng tag `ViT-B-32-quickgelu`. Nạp
`ViT-B-32` thường làm cosine tụt từ 0,9913 xuống 0,9513 và mọi mẫu trượt
ngưỡng dù dữ liệu hoàn toàn đúng.

### 3. `row_id` giống nhau trên mọi máy

`00_discover.py` duyệt theo `sorted(video_id)` và `01_build_index.py` đánh
`row_id` tuần tự, nên `row_id ↔ (video_id, kf_n)` là như nhau ở bất kỳ ai có
đủ 873 file CSV.

Đã **đo thật**, không phải suy từ code: đối chiếu năm lô kết quả của máy khác
với `master.parquet` máy này — **765/765 dòng trùng khít** cả `video_id`,
`kf_n`, `frame_idx` lẫn `fps` (23 dòng L29, 139 dòng L24+L30, 64 dòng
L25+L28, 539 dòng L23+L26+L27).

**Mạnh hơn thế: cả pipeline tái lập được.** Máy giữ L23+L26+L27 chạy lại
**cả nhóm L21** — nhóm máy này đã chạy từ đầu. Kết quả trùng khít 29/29 ở
**mọi cột**, kể cả `cosine` tới 4 chữ số thập phân. Hai máy khác nhau, hai
ổ đĩa khác nhau, cùng chuỗi ffmpeg → CLIP, ra đúng cùng một con số. Nghĩa là
số liệu của mọi người so sánh trực tiếp được, không cần hiệu chuẩn chéo.

Hệ quả: kết quả của 6 người gộp được bằng `row_id`, và **không cần gửi
`master.parquet`/`clip.npy` cho nhau** (395 MB, giống hệt nhau trừ ba cột
đường dẫn tuyệt đối).

### 4. Rủi ro fps đã ĐÓNG — cả hai giá trị lạ

31 video có fps không phải 25/30. Cả hai loại nay đã kiểm và đạt:

| fps | Số video | Đã kiểm | Kết quả |
| --- | --- | --- | --- |
| **26.44** | 1 (`L24_V044`) | 1/1, **hai lần** | pixel corr 0,9997 và 0,9985; CLIP hạng 1, cosine 0,9891 và 0,9870 |
| **29.97** | 30 (L25) | 17/30 | pixel corr 0,9810–0,9999; CLIP cosine 0,9878–0,9979 |

`L24_V044` là mẫu khắc nghiệt nhất từng chạy: keyframe `003.jpg` ứng với
`frame_idx=5`, tức **0,19 giây**, mà bài kiểm pixel vẫn tách được khỏi hai
dòng kề (biên độ 0,148). Nghi vấn variable frame rate không thành hiện thực —
`frame_idx / pts_time` đo được là **26,4380**, khớp `26.44` trong CSV.

Lần chạy `--n 139` bốc trúng **một keyframe khác** của cùng video (`kf_n=10`,
`frame_idx=83`, 3,14s) và cho **26,4331** — xác nhận độc lập lần thứ hai.

Mọi hàm quy đổi giây ↔ frame vẫn **phải nhận `fps` làm tham số**, đọc từ cột
`fps` của `master.parquet`. Rủi ro đã được chứng minh là không hiện thực,
không có nghĩa là được phép hardcode.

### 5. Objects là kênh dùng được — đừng bỏ

1.122.384 detection ở ngưỡng ≥ 0,3 (6,3 detection/keyframe). Top nhãn:
Person, Clothing, Human face, Food, Man, Tree, Woman, Table, Human hand,
Flower...

Lo ngại "tên file thưa (001, 005, 008...) nên object vô dụng" là **sai**: tên
file thưa là ở *video_id* (L21_V004 không tồn tại), không phải ở keyframe.
Trong mỗi video, keyframe đánh số liên tục 001, 002, 003...

### 6. Keyframe trùng lặp — phát hiện mới, rủi ro lớn nhất còn lại

Đo cosine từng keyframe với mọi keyframe khác **trong cùng video**, cả
177.321 dòng:

| Ngưỡng | Keyframe có bản sao | Tỷ lệ |
| --- | --- | --- |
| ≥ 0,999 | 9.994 | 5,64% |
| **≥ 0,99** | **20.975** | **11,83%** |
| ≥ 0,98 | 28.859 | 16,28% |

**Lệch cực mạnh theo nhóm L** (ngưỡng 0,99): L25 **49,82%**; mọi nhóm khác
0,27%–2,16%. L25 lệch **23 lần** so với nhóm cao thứ nhì. Video tệ nhất
`L25_V085`: 408/599 keyframe có bản sao. Có cặp cosine đúng **1,0000**.

Bản sao **nằm liền nhau** (cảnh tĩnh kéo dài), không rải rác: trung vị cách
5 keyframe, 72,2% cách ≤ 10, chỉ 6,1% cách > 50. Cụm điển hình 5 phần tử,
p90 là 19, lớn nhất 125.

**Vì sao là rủi ro điểm số:** `Answer_KIS` nộp một `frame_idx` cụ thể. Frame
đáp án nằm trong cụm 20 bản sao thì hệ thống rất dễ trả về một thành viên
khác — đúng cảnh, sai `frame_idx`, **0 điểm**. Xác suất một keyframe bất kỳ
rơi vào cụm: **~12% toàn kho, ~50% nếu rơi vào L25** (L25 chiếm 21% số
keyframe).

Cách xử lý phụ thuộc câu trả lời của BTC cho câu 0.a — xem PHẦN A5.6 của
kế hoạch v4. Bảng cụm đã dựng sẵn tại `index/trung_lap.parquet`.

### 7. Vẫn phải xây module trích dày cho TRAKE

Trung vị 55 frame ≈ 1,8 giây ở 30 fps. Xa mức đủ dày.

Ngoại lệ đáng chú ý: **`L24_V044` là video dày keyframe nhất kho** — 42
keyframe trong 35,18 giây, ba cái cuối cách nhau đúng **1 frame** (928, 929,
930). Ở video này không cần trích dày, dữ liệu gốc đã dày sẵn.

## Quy ước tên file (đã xác minh)

```text
csv:  n=1, pts_time=0.0,    fps=30.0, frame_idx=0
      n=2, pts_time=3.0,    fps=30.0, frame_idx=90
      ...
ảnh:  001.jpg, 002.jpg, ...      (zero-pad 3 chữ số)
json: 001.json, 002.json, ...    (cùng số thứ tự)
```

Tức `n` ↔ `f"{n:03d}.jpg"` ↔ `f"{n:03d}.json"`.

## Năm chỗ đã sửa trong script

1. `01_build_index.py` — parquet trả ô trống thành `NaN` (float), mà `NaN`
   là truthy nên `if rec.keyframe_dir` lọt qua và crash. Đổi hết về `None`.

2. `01_build_index.py` — object trước đây chỉ khớp qua tên file **ảnh**
   keyframe, nên 844 video chưa tải ảnh bị bỏ object oan (báo cáo 4,4%).
   Nay khớp theo cột `n`, ra đúng 100%.

3. `02_verify.py` — pandas 3.0 loại cột nhóm khỏi `groupby.apply`; đổi sang
   `groupby.sample`.

4. `02_verify.py` — thêm phán quyết `KHOP_YEU` (xem mục 1).

5. `03_verify_CLIP.py` — **đổi thứ tự phán quyết: cosine tuyệt đối trước,
   thứ hạng sau.** Bản cũ xét thứ hạng trước nên báo 8 dòng L25 là
   `LECH_INDEX` trong khi chúng chỉ là keyframe trùng lặp. Lý do bản mới
   đúng: lệch chỉ số thật nghĩa là `clip[row_id]` là vector của một **cảnh
   khác**, khi đó cosine tụt xuống 0,3–0,7 chứ không thể ≥ 0,95. Thêm phán
   quyết `KHOP_TRUNG_LAP` cho trường hợp cosine ≥ 0,95 nhưng hạng > 1.

## Hiệu năng — đọc file nhỏ trên Windows

Đọc 177.321 file JSON tuần tự: **63 file/giây → ~47 phút**. Defender quét
mỗi lần mở file. Đã đổi sang `ThreadPoolExecutor` 24 luồng: **253 file/giây
→ 12 phút**. Muốn nhanh nữa thì loại trừ `C:\Code\aic_data` khỏi Defender.

Chi phí một lần: xong `objects.parquet` (2,7 MB) thì không đụng lại 177k
file JSON nữa.

## Hai điểm còn treo ở lô L23+L26+L27

**1. Máy đó thiếu ảnh keyframe L21.** `02_verify.py` báo 8 dòng
`loi_doc_anh: No such file or directory` cho `Keyframes_L21`, và
`03_verify_CLIP.py` báo 5 dòng `khong_trich_duoc`. Đây là **sự cố tải dữ liệu
trên máy đó**, không phải lỗi bảng cái — L21 đã được máy khác kiểm 29/29 đạt.
Người giữ máy nên tải lại gói `Keyframes_L21` và kiểm lại `Videos_L21`.

**2. Cột `kf_name` rỗng trong `verify_report.csv` của họ.** Theo
`01_build_index.py` dòng 104 và 108, `kf_name` và `kf_path` cùng lấy từ một
biến `kf` nên phải cùng rỗng hoặc cùng có. Mà `02_verify.py` lọc
`kf_path.notna()`, tức những dòng đó phải có `kf_path`. Tôi **chưa giải thích
được** mâu thuẫn này từ code hiện tại.

Điều đó **không làm hỏng kết quả**: `corr` được tính từ ảnh đọc ở `kf_path`
chứ không phải `kf_name`, và ba phép kiểm độc lập đều sạch —

- `row_id` khớp `master.parquet` 35/35 (cả `kf_n`, `frame_idx`, `pts_time`, `fps`);
- `(video_id, frame_idx)` của script 02 tồn tại trong bảng cái 52/52;
- công thức `bien_do = corr − max(dòng kề)` khớp **52/52**, đúng logic
  `02_verify.py`.

Vẫn nên hỏi người gửi xem họ chạy đúng bản script trong repo không.

## Vặt vãnh nhưng hay vấp

- Terminal Windows mặc định cp1252, in tiếng Việt / bảng DuckDB sẽ crash
  `UnicodeEncodeError`. Đặt `PYTHONIOENCODING=utf-8` trước khi chạy.
- ffmpeg cài bằng winget nằm ở
  `%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_*\ffmpeg-9.0-full_build\bin`,
  phải **mở cửa sổ terminal mới** thì PATH mới có.
