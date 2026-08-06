# 02 — Bảng cái: có gì và dùng thế nào

Ba file, nối với nhau bằng **`row_id`**. Nhớ đúng một điều này là dùng được
tất cả:

```text
master.parquet   dòng thứ i  ─┐
clip.npy         dòng thứ i  ─┼─ CÙNG một keyframe
objects.parquet  row_id = i  ─┘
```

`row_id` chính là **chỉ số dòng** trong `master.parquet` và **chỉ số hàng**
trong `clip.npy`. Không cần tra cứu gì, `clip[row_id]` là vector của dòng đó.

## `master.parquet` — 177.321 dòng, một dòng = một keyframe

| Cột | Kiểu | Ví dụ | Ghi chú |
| --- | --- | --- | --- |
| `row_id` | int64 | `0` | khóa nối sang `clip.npy` và `objects.parquet` |
| `video_id` | str | `L21_V001` | |
| `kf_n` | int64 | `1` | số thứ tự keyframe trong video, bắt đầu từ 1 |
| `kf_name` | str | `001.jpg` | rỗng nếu chưa tải keyframes |
| `frame_idx` | int64 | `0` | **GIÁ TRỊ NỘP CHO BTC** |
| `pts_time` | float64 | `0.0` | giây, dùng để `ffmpeg -ss` |
| `fps` | float64 | `30.0` | **25.0 / 26.44 / 29.97 / 30.0 — đừng hardcode** |
| `kf_path` | str | `C:\...\001.jpg` | đường dẫn tuyệt đối, xem cảnh báo ở 01 |
| `obj_path` | str | `C:\...\001.json` | |
| `video_path` | str | `C:\...\L21_V001.mp4` | |
| `has_clip` | bool | `True` | |
| `title` | str | `60 Giây Sáng - Ngày 01082024...` | tiếng Việt **có dấu** |
| `description` | str | | tiếng Việt có dấu |
| `keywords` | str | `HTV Tin tức HTV News...` | đã nối thành một chuỗi |
| `video_length` | int64 | `1262` | giây |
| `publish_date` | str | `01/08/2024` | **chuỗi**, không phải date |

### Ba cột hay bị dùng nhầm

- **`frame_idx` mới là cái nộp cho BTC**, không phải `kf_n` và không phải
  `row_id`. `kf_n` chỉ là số thứ tự ảnh trong thư mục.
- **`pts_time` dùng cho ffmpeg**, `frame_idx` dùng để nộp. Đừng đổi
  `frame_idx / fps` ra giây rồi tưởng bằng `pts_time` — sai số làm lệch frame.
- **`fps` phải đọc từ bảng.** Có 4 giá trị khác nhau, khác nhau ngay trong
  cùng nhóm L21 (V001 là 30.0, V003 là 25.0). Mọi hàm quy đổi giây↔frame
  phải nhận `fps` làm tham số.

## `clip.npy` — (177321, 512) float32

Đã chuẩn hóa L2 sẵn (norm mỗi dòng = 1,0), nên **tích vô hướng chính là
cosine similarity**. Không cần chuẩn hóa lại.

```python
import numpy as np, pandas as pd

clip = np.load('index/clip.npy', mmap_mode='r')   # mmap: không nuốt 346 MB RAM
m = pd.read_parquet('index/master.parquet')

# tìm 20 keyframe giống keyframe số 5000 nhất
q = clip[5000]
sim = clip @ q                    # cosine, vì đã chuẩn L2
top = np.argpartition(-sim, 20)[:20]
top = top[np.argsort(-sim[top])]
print(m.iloc[top][['video_id', 'kf_name', 'frame_idx']])
```

Video chưa có vector CLIP thì dòng đó là vector 0 — lọc bằng `has_clip`.

## `objects.parquet` — 1.122.384 detection

| Cột | Kiểu | Ví dụ |
| --- | --- | --- |
| `row_id` | int64 | `0` |
| `label` | str | `Lantern` |
| `score` | float64 | `0.7967` |
| `box` | array | `[y_min, x_min, y_max, x_max]` chuẩn hóa 0–1 |

Ngưỡng khi dựng là `score >= 0.3`. Muốn ngưỡng khác thì dựng lại, **không
cần dựng lại bảng cái**:

```powershell
python scripts\01_build_index.py --out .\index --objects-only --min-obj-score 0.5
```

### Hai điều phải biết trước khi dùng objects

**Nhãn là tiếng Anh** (bộ Open Images), truy vấn thi đấu là tiếng Việt. Cần
bảng ánh xạ. Và thứ bậc nhãn của Open Images khá lạ: `Car`, `Land vehicle`,
`Vehicle` là ba nhãn *riêng biệt*, không phải cha-con tự động gộp.

**Nhãn người vô dụng khi lọc.** `Person` + `Clothing` + `Human face` +
`Man` + `Woman` chiếm **~50%** toàn bộ detection — lọc theo chúng gần như
không loại được gì. Giá trị phân biệt nằm ở nhãn hiếm:

| Nhãn | Số detection |
| --- | --- |
| Person | 161.352 (14,4%) |
| Clothing | 160.208 (14,3%) |
| Human face | 107.090 (9,5%) |
| Food | 75.635 (6,7%) |
| ... | |
| Boat | 9.255 |
| Bicycle | 4.434 |
| Car | 4.252 |
| Wok | 3.161 |
| Chopsticks | 2.909 |

Lưu ý `Tomato` (10.909) nhiều gấp 2,5 lần `Car` (4.252) — bộ dữ liệu này
nặng nội dung ẩm thực hơn giao thông rất nhiều. Đừng đoán nội dung khi soạn
truy vấn nháp.

Bảng đầy đủ 60 nhãn: `index/label_top60.csv`.

## Truy vấn bằng DuckDB

Không cần server, không cần Supabase. Cùng cú pháp SQL, chạy thẳng trên file.

```python
import duckdb
con = duckdb.connect()

# Video nào nhiều keyframe nhất
con.sql("""
    SELECT video_id, count(*) AS n_kf, max(pts_time) AS dai_giay
    FROM 'index/master.parquet'
    GROUP BY video_id ORDER BY n_kf DESC LIMIT 10
""").show()

# Tìm theo metadata — tiền thân của kênh BM25
con.sql("""
    SELECT DISTINCT video_id, title FROM 'index/master.parquet'
    WHERE lower(description) LIKE '%gốm%'
""").show()

# Keyframe có đủ cả Person và Car, độ tin cậy cao
con.sql("""
    SELECT m.video_id, m.frame_idx, m.kf_path
    FROM 'index/objects.parquet' o JOIN 'index/master.parquet' m USING (row_id)
    WHERE o.score >= 0.5
    GROUP BY 1,2,3
    HAVING count(*) FILTER (WHERE o.label='Person') > 0
       AND count(*) FILTER (WHERE o.label='Car')    > 0
    LIMIT 20
""").show()

# Keyframe có nhãn hiếm — cái này mới lọc được thật
con.sql("""
    SELECT m.video_id, m.frame_idx, o.label, o.score
    FROM 'index/objects.parquet' o JOIN 'index/master.parquet' m USING (row_id)
    WHERE o.label IN ('Wok','Chopsticks','Boat') AND o.score >= 0.5
    ORDER BY o.score DESC LIMIT 20
""").show()
```

Nhớ `$env:PYTHONIOENCODING = "utf-8"` trước khi chạy, không thì bảng DuckDB
làm crash terminal Windows.

## Xem tận mắt một keyframe

```powershell
# lấy pts_time từ bảng rồi trích frame từ video gốc
ffmpeg -ss 11.7 -i "C:\Code\aic_data\Videos_L21_a\video\L21_V030.mp4" -frames:v 1 xem.png -y
start xem.png
```

## File mẫu để xem nhanh

- `index/master_sample.csv` — 200 dòng đầu, mở bằng Excel được
- `index/label_top60.csv` — 60 nhãn phổ biến nhất

Hai file này cũng bị `.gitignore` chặn (nằm trong `index/`), lấy từ Drive.
