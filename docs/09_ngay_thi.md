# Sổ tay NGÀY THI — Sơ tuyển đợt 3

> Đọc mục **0** và **1** trước. Phần còn lại là tra cứu khi cần.

---

## 0. Thành viên cần làm gì ngoài `git pull`?

**Chỉ pull là đủ để có MÃ NGUỒN. Nhưng mã nguồn không chạy được nếu thiếu
`index/`.**

| việc | ai | bắt buộc? |
| --- | --- | --- |
| `git pull` nhánh `giai-doan-0` | tất cả | ✅ |
| Đồng bộ `index/` từ Google Drive | tất cả | ✅ **nếu muốn chạy** |
| Chạy `scripts/12_va_duong_dan.py` | ai vừa tải `index/` mới | ✅ |
| Cài lại thư viện | **không ai** | ❌ đợt này không thêm phụ thuộc nào |
| Encode lại gì đó | **không ai** | ❌ |

### `index/` phải có những file nào

```powershell
.venv\Scripts\python.exe -c "
from pathlib import Path
can = ['master.parquet','clip_gopt.npy','clip_gopt.json',
       'ocr_asr.parquet','truy_van_gopt.npz']
tuy = ['ocr_vietocr.parquet','caption.parquet','objects.parquet',
       'hubness_clip_gopt.npy']
for f in can: print(('OK  ' if Path('index',f).exists() else 'THIEU'), f)
for f in tuy: print(('OK  ' if Path('index',f).exists() else '(tuy chon)'), f)
"
```

Thiếu một trong năm file đầu thì **không chạy được**. `ocr_vietocr.parquet` chỉ
cần nếu dùng `--van-ban-gop`.

### Đường dẫn trong `master.parquet` là TUYỆT ĐỐI của máy dựng index

Máy nào vừa tải `index/` từ Drive về đều phải chạy **một lần**:

```powershell
.venv\Scripts\python.exe scripts\12_va_duong_dan.py
```

Không chạy thì `kf_path` trỏ vào ổ đĩa của máy khác — UI không hiện được ảnh,
và nó **không báo lỗi**, chỉ hiện ô trống.

---

## 1. Trình tự ngày thi — sáu bước

### Bước 1. Có đề → mở Kaggle

Theo `notebooks/kaggle_ma_hoa_dot3.md`. **Cell 1 chạy TRƯỚC khi có đề** (nạp sẵn
trọng số, ~2 phút). Cell 2 chạy khi đã dán đề vào.

Kết quả: `dot3.zip` chứa `index/truy_van_dot3.npz` + thư mục đề.

### Bước 2. Giải nén về máy

```powershell
Expand-Archive dot3.zip -DestinationPath C:\Code\aic2026 -Force
```

### Bước 3. Gộp cache — KHÔNG cần model

```powershell
.venv\Scripts\python.exe scripts\120_gop_cache.py `
    index\truy_van_gopt.npz index\truy_van_dot3.npz
```

### Bước 4. 🔴 TIỀN KIỂM — bước KHÔNG ĐƯỢC BỎ

```powershell
.venv\Scripts\python.exe scripts\119_kiem_truy_van.py --de dev\SOTUYEN3-bo-de-thi
```

Phải thấy `✅ ĐỦ`. Nếu thấy `❌ THIẾU` thì nó in sẵn lệnh vá — chạy rồi kiểm lại.

> **Vì sao bước này tồn tại.** Ở Sơ tuyển 2, `p2-22-kis` **mất trắng** vì câu
> đó chưa nằm trong cache vector: kênh 1 bị bỏ, chỉ còn kênh văn bản. Mất
> 1/30 câu = **3,3% bài thi**, và mất vì **vận hành**, không phải vì mô hình.
> Chốt này chạy thử trên đề Sơ tuyển 2 và **bắt đúng gói đó**.

### Bước 5. Sinh bài nộp

```powershell
.venv\Scripts\python.exe src\run.py `
    --de dev\SOTUYEN3-bo-de-thi `
    --cache index\truy_van_gopt.npz `
    --ra submission `
    --van-ban-gop --trong-so-hoi 0.25
```

Bỏ hai cờ cuối để chạy cấu hình bảo thủ — xem mục 2.

### Bước 6. Mở UI để soi bằng mắt trước khi nộp

```powershell
.venv\Scripts\python.exe web\server.py
```

UI đã bật sẵn đúng những kênh đang dùng. Nếu terminal in một khối `!!!!!!` thì
có gói chưa mã hoá — **đừng nộp gói đó**, quay lại bước 4.

---

## 2. Chọn cấu hình cho lần nộp 1/2 hay lần cuối

BTC tính **lần nộp CUỐI CÙNG**, không phải lần tốt nhất (C7). Nên hai cấu hình:

### A. BẢO THỦ — dùng cho lần nộp CUỐI

```powershell
.venv\Scripts\python.exe src\run.py --de dev\SOTUYEN3-bo-de-thi `
    --cache index\truy_van_gopt.npz --ra submission
```

Chỉ gồm những thứ đã ✅ vượt ngưỡng nhiễu. **0,5173 / 0,6096** trên 52 câu đề
thật nhãn sạch (KIS/Q&A), và TRAKE đã có hai bản vá ✅ ở mục 3.

### B. MẠNH NHẤT — dùng cho lần nộp 1 hoặc 2, để lấy số THẬT từ BTC

```powershell
.venv\Scripts\python.exe src\run.py --de dev\SOTUYEN3-bo-de-thi `
    --cache index\truy_van_gopt.npz --ra submission `
    --van-ban-gop --trong-so-hoi 0.25
```

Thêm hai thứ, mỗi thứ riêng lẻ là 🟡 nhưng **gộp lại thì ✅ vượt ngưỡng**:
**0,5433 / 0,6173** (+0,0260, thắng 7 thua 2 hoà 43) — A100.

**Vì sao không để B làm mặc định:** riêng đợt đo vừa rồi đã chạy ~20 cấu hình,
mà ngưỡng 2×SE là khoảng tin cậy cho **một** phép so, và B chỉ vượt **5%**.
Không thành phần nào tự thắng. Nộp B ở lần 1-2 để BTC chấm hộ; nếu điểm thật
cao hơn thì lần cuối dùng B, không thì dùng A.

---

## 3. Đợt này có gì đổi trong đường nộp

Hai bản vá TRAKE, **cả hai đã ✅ ỔN ĐỊNH**, mặc định BẬT:

| | trước | sau |
| --- | ---: | ---: |
| TRAKE ±2s | 0,2994 | **0,4317** |
| TRAKE ±15s | 0,5139 | **0,6483** |

1. **TRAKE nay có kênh 3.** Trước đây `quet_van_ban` bỏ qua câu TRAKE, nên bài
   nộp chạy TRAKE bằng kênh 1 một mình — trong khi **mọi script đo TRAKE** đều
   giả định có kênh 3. Toàn bộ kết luận TRAKE của repo đo trên một cấu hình bài
   nộp không chạy.
2. **`--be-trake` mặc định 300** (trước là 100). Với TRAKE, `--k` không phải
   "số dòng nộp" mà là **bể để GIAO**: một video chỉ vào danh sách khi có ứng
   viên cho *mọi* sự kiện. Ở 100 chỉ còn trung vị **11 video**, ở 300 là **25**
   — đúng con số mà hạn ngạch dòng `40/25/15/12/8 + 20` được thiết kế cho.

Quay lại hành vi cũ: `--be-trake 100` và `--khong-hop-nhat`.

⚠️ **Cảnh báo cỡ mẫu:** 15/18 câu TRAKE dùng để đo là **tự soạn**, chỉ 3 câu là
đề thật. Hai bản vá này đúng hướng ở cả hai mức dung sai và khớp với kiểu trượt
soi tay được ở cả hai đợt sơ tuyển, nhưng đừng coi con số là chắc.

---

## 4. Bảng kênh — cái nào bật, vì sao

| kênh | trạng thái | căn cứ |
| --- | --- | --- |
| 1 — ảnh (SigLIP2 gopt 1536) | **BẬT** | A87: gấp 2,4 lần SigLIP2-1152 |
| 3 — OCR/ASR, w = 0,5 | **BẬT** (nay cả TRAKE) | A45, A52, A102 |
| 2 — metadata | TẮT | A12/A14.2: 0,0000 ở ±2s |
| 4 — objects | TẮT | A62: làm tệ đi khi hợp nhất |
| 5 — caption | TẮT | A73/A90: ✅ tệ hơn ở 100% độ phủ |
| 6 — BGE-M3 | TẮT | A59: 🟡, tốn 360 MB RAM |

Bật tay được bằng `--co-objects`, `--co-caption`, `--co-bge` (UI) — nhưng cả ba
đều đã đo là không giúp. **Đừng bật ở bài nộp thật.**

---

## 5. Hỏng giữa chừng thì làm gì

| triệu chứng | nguyên nhân | xử lý |
| --- | --- | --- |
| `❌ N/M chuỗi truy vấn chưa có trong cache` | thiếu mã hoá | bước 4, chạy lệnh vá nó in ra |
| UI không hiện ảnh, ô trống | `kf_path` của máy khác | `scripts\12_va_duong_dan.py` |
| `UnicodeEncodeError` khi in | terminal cp1252 | `$env:PYTHONIOENCODING = "utf-8"` |
| `ModuleNotFoundError: pyarrow` | gọi `python` trần | luôn dùng `.venv\Scripts\python.exe` |
| TRAKE nộp thiếu dòng | bể ứng viên hẹp | đã vá (mục 3); ép thêm bằng `--be-trake 500` |
| Số sự kiện TRAKE tách sai | câu viết lạ | `--so-su-kien N` ép đúng số |

**Sai định dạng vẫn tính là một lần nộp** (C7) — nên `nop_bai.soat()` từ chối
ghi file khi phát hiện sai. Đừng lách nó.

---

## 6. Hai điều tuyệt đối không làm

1. **Không chạy `src/tap_dev.py --tach-test`.** Nó tự từ chối, nhưng đừng thử.
   Rò tập test là loại hỏng không tự lộ ra.
2. **Không commit `index/`, `*.npy`, `*.parquet`, ảnh, khoá API.** Repo công
   khai. `.gitignore` đã chặn, nhưng đã từng lọt 15 ảnh keyframe.
