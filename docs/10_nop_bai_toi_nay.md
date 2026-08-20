# 10 — Quy trình nộp bài (làm theo đúng thứ tự)

*Viết cho đợt nộp thử đầu tiên. Đọc hết trước khi chạy — chỉ có **3 lần nộp**
mỗi gói, và **sai định dạng vẫn tính là một lần**.*

---

## ⛔ Ai chạy trên máy nào — quyết định TRƯỚC KHI làm gì khác

| Máy | Chạy được gì | Điểm ước tính |
| --- | --- | --- |
| **Máy ≥ 16 GB RAM** *(máy Khánh)* | SigLIP2 — `--matrix clip_siglip2.npy` | **0,3258** (A17) |
| Máy 7,7 GB | **chỉ** `--matrix clip.npy` | **0,0000** trên tiếng Việt (A10) |

**Máy 7,7 GB không nạp nổi SigLIP2.** Đã thử hai lần, cả hai lần **treo cứng
máy** — không phải báo lỗi, mà đứng hẳn phải khởi động lại. `dense.kiem_ram()`
nay chặn trước khi điều đó xảy ra.

> **Nghĩa là: bài nộp có điểm PHẢI chạy trên máy khỏe.** Máy yếu chạy được, ra
> file hợp lệ, nhưng gần như chắc chắn 0 điểm vì CLIP mù tiếng Việt. Vẫn đáng
> chạy để **kiểm định dạng đầu-cuối** trước khi máy khỏe chạy thật.

---

## 1. Chuẩn bị thư mục đề

Tải gói truy vấn của BTC, đặt mỗi câu một file `.txt`, **giữ nguyên tên**:

```text
de_thi/
├── query-1-kis.txt
├── query-2-kis.txt
├── query-3-qa.txt
└── query-4-trake.txt
```

Hậu tố `kis` / `qa` / `trake` **quyết định BTC chấm bằng bộ luật nào**. Đặt sai
là bị chấm nhầm loại — `run.py` bỏ qua file không đúng quy ước và in cảnh báo.

---

## 2. Chạy đường ống

**Trên máy khỏe** (mặc định đã là SigLIP2):

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe src\run.py --de de_thi --ra submission --tra-loi "không rõ"
```

**Trên máy yếu**, chỉ để kiểm định dạng:

```powershell
.venv\Scripts\python.exe src\run.py --de de_thi --ra submission --matrix clip.npy --tra-loi "không rõ"
```

### Đọc kỹ hai thứ script in ra

**Số sự kiện TRAKE.** Script tự tách từ đề và in ra. **Kiểm bằng mắt** — sai số
sự kiện là sai định dạng, mất trắng cả câu:

```text
query-4-trake: tách được 4 sự kiện -> nộp 4 Frame ID
```

Lệch thì ép: `--so-su-kien 4`.

**Kênh nào chạy.** Nếu thấy `kênh 3: chưa có ocr_asr.parquet — bỏ qua` mà TV4
bảo đã chạy xong, thì file chưa được chép về — đặt vào
`pipeline_OCR_ASR/output/ocr_asr.parquet` hoặc `index/ocr_asr.parquet`.

---

## 3. Soát rồi nén

```powershell
.venv\Scripts\python.exe src\nop_bai.py --soat submission --nen bai_nop.zip
```

Phải thấy `✅ Định dạng hợp lệ.` Bộ soát chặn sẵn:

| Chặn | Vì sao |
| --- | --- |
| BOM đầu file | tên video dòng đầu hỏng, **âm thầm** |
| Tên video còn `.mp4` | BTC ghi rõ `L01_V028` ✅ / `L01_V028.mp4` ❌ |
| `answer` > 100 ký tự | **không tự cắt** — cắt là đổi câu trả lời |
| Quá 100 dòng | BTC chỉ nhận 100 |
| Dòng trùng hệt nhau | phí một trong 100 chỗ |
| TRAKE sai số sự kiện / không tăng dần | sai định dạng |
| Thiếu thư mục `submission/` trong zip | lỗi BTC xếp thứ hai trong 5 lỗi thường gặp |

Cảnh báo `mới 30/100 dòng` **không phải lỗi** — nhưng đang mất điểm. Không có
điểm phạt cho dòng sai, dòng thứ 100 vẫn đáng 0,2 nếu trúng.

---

## 4. Trước khi bấm nộp

- [ ] Mở `submission/query-1-kis.csv` bằng Notepad — thấy text thuần, không ký tự lạ
- [ ] Tên file khớp tên truy vấn BTC cấp
- [ ] Trong zip có thư mục `submission/` (đã kiểm ở bước 3)
- [ ] Còn mấy lần nộp? **Tối đa 3, lấy lần CUỐI để xếp hạng**

---

## Điều phải biết về câu Q&A

**Ta chưa có VLM sinh đáp án**, nên `--tra-loi "không rõ"` chỉ là chỗ giữ chỗ.
BTC chấm: khung đúng **nhưng** `answer` sai → **0 điểm**. Nên mọi câu Q&A tối
nay chắc chắn 0.

Vẫn nộp, vì: không có điểm phạt, và nó kiểm được đường ống đầu-cuối. Muốn có
điểm Q&A thì phải chốt **việc 12** (trả phí API hay chạy Qwen local).

> ⚠️ Còn một chỗ **phải hỏi BTC**: tài liệu tự mâu thuẫn — trang 2 nói `answer`
> *"so sánh chính xác về mặt **ngữ nghĩa**"*, trang 8 nói *"so sánh dưới dạng
> **chuỗi chính xác**"*. Khác nhau hoàn toàn: `"5"` / `"Năm người"` / `"5 người"`
> là ba đáp án khác nhau nếu so chuỗi. Điều này đổi hẳn cách nhắc VLM.

---

## Nếu hỏng giữa chừng

| Triệu chứng | Nguyên nhân | Xử lý |
| --- | --- | --- |
| `❌ KHÔNG ĐỦ RAM` | máy yếu, model lớn | đổi máy, hoặc `--matrix clip.npy` |
| Máy đứng hẳn | đã bỏ qua chốt RAM | khởi động lại; **đừng dùng `bo_qua_ram=True`** |
| `❌ KHÔNG ghi file nào` | sai định dạng | đọc danh sách lỗi in kèm, sửa rồi chạy lại |
| `Có gói Q&A nhưng chưa có answer` | quên `--tra-loi` | thêm `--tra-loi "không rõ"` |
| `UnicodeEncodeError` | terminal cp1252 | `$env:PYTHONIOENCODING = "utf-8"` |
