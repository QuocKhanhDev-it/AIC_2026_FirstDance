# Giao diện truy vấn cục bộ

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe web\server.py                          # kênh 3 + 4
.venv\Scripts\python.exe web\server.py --cache index\truy_van.npz   # + kênh 1
```

Rồi mở **<http://127.0.0.1:8000>**. Không cần cài thêm gì: `http.server` của thư
viện chuẩn, HTML thuần, không build.

## Hai file

| | |
| --- | --- |
| `server.py` | gọi thẳng `run`, `bm25`, `objects`, `dense`, `nop_bai` — không chép lại logic nào |
| `index.html` | toàn bộ phần nhìn, một file, không phụ thuộc |

## Vì sao màn hình xếp như vậy

**Cột phải là 100 dòng bài nộp, không phải danh sách rút gọn.** BTC chấm
`trung bình R@{1,5,20,50,100}`, cho nộp 100 dòng và **không phạt dòng sai** —
nên xoá bớt ứng viên là vứt vé số mà không nhận lại gì. Việc của người soát chỉ
là **đẩy dòng đã soi bằng mắt lên hạng 1**; thứ tự đổi, số dòng không đổi.

Ô điểm ở cột phải quy đúng công thức đó ra số: hạng 1 → 1,00 · hạng 2–5 → 0,80 ·
6–20 → 0,60 · 21–50 → 0,40 · 51–100 → 0,20.

## Ba chỗ trang này cố tình nói thật thay vì nói cho đẹp

1. **Kênh 1 tắt thì có băng cảnh báo.** Không có `--cache`, bể ứng viên yếu hơn
   ~3 lần bài nộp thật (0,3258 so với 0,1183 + 0,0417). Đọc kết quả mà không
   biết điều đó là tự lừa mình.
2. **Ảnh thiếu thì vẽ khác hẳn.** Máy này chỉ 36.506/177.321 dòng có `kf_path`
   (20,6% — A5.5). Ô "máy này không có ảnh" khác hẳn ô đang tải.
3. **`nop_bai.soat()` từ chối thì hiện nguyên văn lý do.** Không file nào được
   ghi. Mỗi gói chỉ nộp được 3 lần, sai định dạng vẫn tính một lần.

## TRAKE

Chuỗi do `run.dung_trake` lắp (quy hoạch động, giữ đúng chỉ số sự kiện, ép
tăng dần theo thời gian) — trang này chỉ vẽ lại và cho đẩy chuỗi đúng lên hạng 1.

Mỗi vị trí trong chuỗi được đánh dấu **thật** hay **nội suy**. Khung nội suy là
thứ DP *bịa ra* khi truy hồi không tìm được ứng viên cho sự kiện đó — nó có ô
viền vàng ghi rõ, và không có ảnh để soi. Trên câu `query-p1-16-trake` chạy
bằng kênh 3+4, **chỉ 36% vị trí là khung thật**; con số đó là thước đo bể ứng
viên yếu tới đâu, không phải lỗi hiển thị.

Hai nút **A39** (`rải hẹp 56,6s`, `prior khoảng cách`) lắp lại chuỗi ngay để
nhìn khác biệt bằng mắt. **Mặc định TẮT** — chưa thắng trên tập dev, xem A39.
