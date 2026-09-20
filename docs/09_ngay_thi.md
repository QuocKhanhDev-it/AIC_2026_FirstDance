# Sổ tay NGÀY THI — Sơ tuyển đợt 3

> Đọc mục **0** và **1** trước. Phần còn lại là tra cứu khi cần.

---

## 0. Thành viên cần làm gì ngoài `git pull`?

**Chỉ pull là đủ để có MÃ NGUỒN. Nhưng mã nguồn không chạy được nếu thiếu
`index/`.**

| việc | ai | bắt buộc? |
| --- | --- | --- |
| `git pull` nhánh `giai-doan-0` | tất cả | ✅ |
| **Giải nén `CHO_NHOM_dot3.zip`** (9,2 MB) | tất cả | ✅ **mở khoá kênh 1 cho đề đợt 3** |
| Đồng bộ `index/` từ Google Drive | ai chưa có | ✅ **nếu muốn chạy** |
| Chạy `scripts/12_va_duong_dan.py` | ai vừa tải `index/` mới | ✅ |
| Cài lại thư viện | **không ai** | ❌ đợt này không thêm phụ thuộc nào |
| Encode lại gì đó | **không ai** | ❌ **đã mã hoá xong, dùng chung** |

### 🔴 Mở khoá kênh 1 cho đề đợt 3 — hai file, một lệnh kiểm

Kênh 1 (ảnh, SigLIP2 gopt) là **kênh mạnh nhất** và nó chạy từ **vector truy
vấn mã hoá sẵn**. Truy vấn nào không có trong cache thì kênh 1 **im lặng bị
bỏ** cho gói đó — `p2-22` ở Sơ tuyển 2 mất trắng đúng vì vậy.

`git pull` **KHÔNG** mang hai thứ này về, vì `.gitignore` chặn cả `index/` lẫn
`*-bo-de-thi/`. Phải nhận qua Drive/Zalo:

```powershell
Expand-Archive CHO_NHOM_dot3.zip -DestinationPath C:\Code\aic2026 -Force
```

Gói gồm đúng hai thứ:

| | |
| --- | --- |
| `index/truy_van_gopt.npz` | cache **đã gộp** 1.591 chuỗi (1.468 cũ + 123 của đợt 3) |
| `dev/SOTUYEN3-bo-de-thi/` | 36 file `.txt` |

⚠️ **Phải lấy đúng 36 file `.txt` này, đừng gõ lại hay chép từ chỗ khác.** Cache
tra theo **chuỗi chính xác** — lệch một dấu cách là một chuỗi khác, là kênh 1
tắt cho gói đó.

⚠️ **Ai đã tự mã hoá thêm truy vấn riêng** thì đừng ghi đè; gộp thay vì thay:

```powershell
.venv\Scripts\python.exe scripts\120_gop_cache.py `
    index\truy_van_gopt.npz <cache vừa nhận>.npz
```

**Rồi bắt buộc kiểm — một lệnh:**

```powershell
.venv\Scripts\python.exe scripts\119_kiem_truy_van.py --de dev\SOTUYEN3-bo-de-thi
```

Phải thấy `thiếu : 0` và `✅ ĐỦ — kênh 1 sẽ chạy cho MỌI truy vấn`. Thấy `❌`
thì **đừng nộp**, nó in sẵn lệnh vá.

### `index/` chỉ cần **6 file**, không cần cả thư mục

`index/` trên máy dựng index là **13,24 GB** và Drive báo ~5 giờ. Phần lớn là
**đầu vào của các bước đã xong** và **bản sao lưu**. Chạy:

```powershell
.venv\Scripts\python.exe scripts\121_goi_index_toi_thieu.py
```

Nó liệt kê từng file cần / không cần **kèm lý do**, và không để file nào không
có lý do.

| | MB | |
| --- | ---: | --- |
| `master.parquet` | 3,5 | bảng cái |
| `clip_gopt.npy` | 544,7 | kênh 1 |
| `clip_gopt.json` | ~0 | **tên model** — thiếu là dùng SAI model |
| `ocr_asr.parquet` | 26,7 | kênh 3 |
| `truy_van_gopt.npz` | 13,6 | vector truy vấn — thiếu là kênh 1 TẮT |
| `ocr_vietocr.parquet` | 5,7 | chỉ cho `--van-ban-gop` |
| **CỘNG** | **594** | **đủ chạy `run.py`** = 4,5% |
| `anh_nho/` | 541 | **chỉ** để xem ảnh trong UI |
| **CỘNG + ảnh** | **1.135** | = **8,6%** |

**Đã đo, không phải đoán:** dựng một thư mục chỉ chứa 6 file đó rồi chạy
`run.py` — **24/24 file bài nộp giống hệt từng dòng** so với khi chạy trên
`index/` đầy đủ.

Dựng sẵn bộ tải lên:

```powershell
.venv\Scripts\python.exe scripts\121_goi_index_toi_thieu.py --ra D:\len_drive --co-anh
```

`--co-anh` nén `anh_nho` thành **một** file zip — 81.916 file nhỏ thì Drive tải
chậm hơn hẳn một file lớn cùng dung lượng.

Ai chỉ chạy `run.py` mà không mở UI thì **bỏ `--co-anh`**, còn 594 MB.

⚠️ **Đừng xoá phần còn lại trên máy dựng index.** 15 file `clip_gopt_*.npy` là
các phần đã gộp; giữ chúng thì gộp lại được nếu `clip_gopt.npy` hỏng.

### Đường dẫn trong `master.parquet` là TUYỆT ĐỐI của máy dựng index

Máy nào vừa tải `index/` từ Drive về đều phải chạy **một lần**:

```powershell
.venv\Scripts\python.exe scripts\12_va_duong_dan.py
```

Không chạy thì `kf_path` trỏ vào ổ đĩa của máy khác — UI không hiện được ảnh,
và nó **không báo lỗi**, chỉ hiện ô trống.

---

## 1. Trình tự ngày thi — bảy bước

### Bước 0. Có đề → giải nén vào thư mục RIÊNG

```powershell
Expand-Archive SOTUYEN3-bo-de-thi.zip -DestinationPath dev\SOTUYEN3-bo-de-thi
```

⚠️ **Thư mục riêng, không đè lên đợt trước.** Đề đợt 3 dùng lại tiền tố `p2-`:
**16 tên file trùng đợt 2 nhưng nội dung khác hẳn**. Giải nén đè là mất đề cũ,
và tệ hơn là dễ nộp nhầm bộ.

⚠️ **Giữ nguyên tên file BTC đặt.** `run.py` đọc loại câu từ tên
(`-kis`/`-qa`/`-trake`), và tên gói cũng là tên file nộp.

### Bước 1. Sinh cell Kaggle — KHÔNG dán tay câu nào

```powershell
.venv\Scripts\python.exe scripts\122_sinh_cell_kaggle.py --de dev\SOTUYEN3-bo-de-thi
```

Ra `dev\SOTUYEN3-bo-de-thi\_cell_kaggle.py`. Mở Kaggle, chạy **cell 1** trong
`notebooks/kaggle_ma_hoa_dot3.md` (nạp sẵn trọng số, ~2 phút — làm được từ
trước khi có đề), rồi **copy toàn bộ file vừa sinh, dán vào cell 2, chạy**.

Cell tự mã hoá, tự tiền kiểm, tự đóng gói. Nó `assert` ở hai chỗ nên **hỏng là
dừng**, không im lặng chạy tiếp.

> Vì sao không dán tay: 36 gói tiếng Việt có dấu, mỗi lần dán là một lần có thể
> lệch một ký tự — mà lệch một ký tự là **một chuỗi khác**, là trượt cache, là
> đúng lỗi làm mất trắng `p2-22` ở Sơ tuyển 2. Script nhúng nội dung dưới dạng
> JSON và đã kiểm: **36/36 gói đi qua nguyên vẹn**.

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
