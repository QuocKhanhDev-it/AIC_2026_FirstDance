# Giao diện truy vấn cục bộ

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe web\server.py
```

Rồi mở **<http://127.0.0.1:8000>**. Không cần cài thêm gì: `http.server` của thư
viện chuẩn, HTML thuần, không build.

**Chỉ một lệnh.** Không còn phải truyền `--cache`: máy chủ tự tìm
`index/truy_van_gopt.npz`. Việc duy nhất phải làm trước ngày thi là **mã hoá đề
mới**:

```powershell
.venv\Scripts\python.exe scripts\25_ma_hoa_truy_van.py --de <thư_mục_đề> `
    --matrix clip_gopt.npy --ra index\truy_van_moi.npz
.venv\Scripts\python.exe scripts\67_gop_cache_truy_van.py index\truy_van_moi.npz
.venv\Scripts\python.exe scripts\67_gop_cache_truy_van.py index\truy_van_moi.npz --ghi
```

Chạy `67_` hai lần là cố ý — lần đầu chỉ xem trước. Việc mã hoá cần GPU, xem
`notebooks/kaggle_ma_hoa_trake.md` để chạy trên Kaggle (~3 phút).

## Mặc định = ĐÚNG cấu hình `run.py` nộp thật

| kênh | mặc định | vì sao |
| --- | :---: | --- |
| 1 — ảnh SigLIP2 `gopt` | **BẬT** | tự tìm cache, không nạp model (A47) |
| 3 — OCR+ASR, trọng số 0,5 | **BẬT** | A52; đóng góp +0,0285 ✅ ổn định (A82) |
| 4 — objects | tắt | A62: sửa 2 lỗi công thức, mạnh gấp 2,5 lần, **vẫn** làm tệ đi |
| 5 — caption | tắt | A73: ở độ phủ 76% thì ❌ đảo dấu |
| 6 — BGE-M3 | tắt | A59: +0,0140 nhưng 🟡; tốn ~360 MB RAM |

Ba cờ `--co-objects`, `--co-caption`, `--co-bge` để soi từng kênh khi cần, kèm
cảnh báo hiện thẳng lên màn hình.

> ⚠️ **"Bật càng nhiều càng tốt" là sai ở đây.** Bật thêm kênh làm người soát
> nhìn một bể ứng viên rồi gửi đi một bể khác. Ba kênh trên đều đã đo và đều
> không vào được bài nộp; giao diện phải phản ánh **đúng thứ sẽ nộp**.

## Hai file

| | |
| --- | --- |
| `server.py` | gọi thẳng `run`, `bm25`, `kbest_trake`, `dense`, `nop_bai` — không chép lại logic nào |
| `index.html` | toàn bộ phần nhìn, một file, không phụ thuộc |

`tests/test_web_server.py` chốt ba chỗ từng lệch giữa giao diện và bài nộp:
trọng số hợp nhất, hằng số `k` của RRF, và **hợp nhất mệnh đề bằng RRF hạng chứ
không phải max cosine**.

## Vì sao màn hình xếp như vậy

**Cột phải là 100 dòng bài nộp, không phải danh sách rút gọn.** BTC chấm
`trung bình R@{1,5,20,50,100}`, cho nộp 100 dòng và **không phạt dòng sai** —
nên xoá bớt ứng viên là vứt vé số mà không nhận lại gì. Việc của người soát chỉ
là **đẩy dòng đã soi bằng mắt lên hạng 1**; thứ tự đổi, số dòng không đổi.

Ô điểm ở cột phải quy đúng công thức đó ra số: hạng 1 → 1,00 · hạng 2–5 → 0,80 ·
6–20 → 0,60 · 21–50 → 0,40 · 51–100 → 0,20.

## Ba chỗ trang này cố tình nói thật thay vì nói cho đẹp

1. **Kênh 1 tắt thì có băng cảnh báo.** Thiếu cache thì bể ứng viên yếu hơn hẳn
   bài nộp thật. Đọc kết quả mà không biết điều đó là tự lừa mình.
2. **Ảnh thiếu thì vẽ khác hẳn.** Ô "máy này không có ảnh" khác hẳn ô đang tải.
   `/api/trang_thai` báo số dòng thật sự soi được ảnh.
3. **`nop_bai.soat()` từ chối thì hiện nguyên văn lý do.** Không file nào được
   ghi. Mỗi gói chỉ nộp được 3 lần, sai định dạng vẫn tính một lần.

## TRAKE

Chuỗi do **`kbest_trake.lap_trake`** lắp — beam search sinh 100 giả thuyết khác
nhau về video tốt nhất, thay vì 1 dòng cho mỗi trong 100 video. Trên 20 câu
TRAKE, cách cũ thua **−0,0990 ở ±2s, 2 thắng / 11 thua, ✅ ổn định** (A79).

Hệ quả dễ chịu cho người soát: K-best **không nội suy**. Mọi vị trí trong chuỗi
đều là khung THẬT, không còn ô viền vàng "DP bịa ra" — trước đây có câu chỉ 36%
vị trí là khung thật.

`--trake-cu` quay lại cách cũ để dựng lại bài nộp cũ khi cần đối chiếu; lúc đó
ô viền vàng xuất hiện trở lại và cờ `that` cho biết vị trí nào là nội suy.
