# 10 — Quy trình nộp bài (làm theo đúng thứ tự)

*Viết cho đợt nộp thử đầu tiên. Đọc hết trước khi chạy — chỉ có **3 lần nộp**
mỗi gói, và **sai định dạng vẫn tính là một lần**.*

---

## ⛔ Ai chạy trên máy nào — quyết định TRƯỚC KHI làm gì khác

| Máy | Chạy được gì | Điểm dev | Điểm leaderboard |
| --- | --- | --- | --- |
| **Máy ≥ 16 GB RAM** *(máy Khánh)* | SigLIP2 — `--matrix clip_siglip2.npy` | **0,3258** (A17) | **3,8** — lượt #3 (A24) |
| Máy 7,7 GB | objects + OCR, không cần model | 0,0781 (A23) | 2,6 — lượt #2 và #4 |
| Máy 7,7 GB | `--matrix clip.npy` (CLIP) | **0,0000** trên tiếng Việt (A10) | — |

> **Vòng thật phải chạy trên máy khỏe.** Điều này nay có bằng chứng NGOÀI, không
> chỉ tập dev: lượt #3 (SigLIP2) được **3,8**, lượt #2 và #4 (objects+OCR) được
> **2,6**. Cấu hình model-free chỉ là đường lui khi không mượn được máy.

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

**Trên máy yếu** — đây là cấu hình **đã ăn 2,6 điểm** trên leaderboard, không
phải bản chạy cho có:

```powershell
.venv\Scripts\python.exe src\run.py --de de_thi --ra submission `
    --kenh objects --hop-nhat --bo-metadata --tra-loi "không rõ"
```

> **Đừng thêm `--trong-so-phu 0.3`.** Mặc định nay là **1,0** (ngang nhau) và đó
> là cấu hình đã ghi điểm. Dìm OCR xuống 0,3 đo được là **tệ đi ổn định**
> (0,0781 → 0,0476, gần bằng objects đứng một mình) — A23. Con số 0,3 chỉ đúng
> cho cảnh một kênh mạnh cộng một kênh yếu (A14.2), không phải cảnh này.

Chỉ để **kiểm định dạng** bằng kênh ảnh trên máy yếu (gần như chắc chắn 0 điểm,
vì CLIP mù tiếng Việt):

```powershell
.venv\Scripts\python.exe src\run.py --de de_thi --ra submission --matrix clip.npy --tra-loi "không rõ"
```

Chạy xong, `submission\lenh_da_chay.txt` giữ **nguyên văn lệnh đã sinh ra các
file CSV**. Giữ file đó lại: một điểm số trên leaderboard mà không truy được về
một lệnh cụ thể thì không dạy được gì (A23 — bản nộp 2,6 suýt mất dấu vì đúng
chuyện này). File này **không lọt vào zip** — `dong_goi()` chỉ nén `*.csv`.

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
.venv\Scripts\python.exe src\nop_bai.py --soat submission --nen firstdance1.zip
```

Lệnh này nay chạy **hai cổng**: soát thư mục, nén, rồi **soát lại chính file
zip** theo checklist BTC. Soát một file zip đã có:

```powershell
.venv\Scripts\python.exe src\nop_bai.py --soat-zip firstdance1.zip
```

> **Vì sao phải soát chính file zip.** Soát thư mục rồi nén là soát thứ sắp nộp
> *gián tiếp* — giữa hai bước còn một thao tác nén mà người ta làm tay được, và
> BTC xếp *"nén trực tiếp file CSV thay vì thư mục"* là lỗi phổ biến **thứ hai**.
> Cổng này đọc đúng thứ sẽ upload.

**Tên file zip: chỉ chữ và số.** BTC khuyến cáo vậy (`firstdance1.zip`, không
phải `firstdance_p1_v2.zip`). Bản nộp đợt 1 có gạch dưới và vẫn được nhận, nên
đây là khuyến cáo chứ không phải luật — nhưng không mất gì khi tuân thủ.

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

> ⚠️ **"Lần CUỐI" chứ không phải "lần tốt nhất".** BTC: *"Kết quả được dùng để
> xếp hạng là kết quả đội nộp lần cuối cùng"*. Nộp thử một cấu hình yếu ở lần
> thứ 3 là **tự hạ điểm của chính mình**, không phải thử nghiệm miễn phí.
>
> Và điểm thấy trên bảng chỉ là **50% đáp án** (Public Leaderboard); xếp hạng
> thật chấm 100% ở Private. Xem C7 của kế hoạch.

### Đừng sửa tay file CSV theo ví dụ ở trang 2 của BTC

Trang 2 viết `L01_V028, 3450, "5"` — **có khoảng trắng sau dấu phẩy**. Nhưng
chính BTC ghi *"khoảng trắng đầu/cuối được giữ nguyên, không tự động trim"*, nên
đọc bằng parser CSV chuẩn thì `answer` ra `' "5"'`, tức khoảng trắng và dấu
ngoặc kép thành **ký tự thật** trong đáp án:

```text
L01_V028, 3450, "5"   ->  answer = ' "5"'    ❌
L01_V028,3450,5       ->  answer = '5'       ✅  (dạng chuẩn trang 4-5)
```

`nop_bai` luôn ghi dạng dưới, và `--soat-zip` bắt khoảng trắng thừa như một lỗi.

---

## Mũi nhọn 1 — ba cờ mới, **mặc định tắt** (A22)

Cả ba đã nối vào `run.py` và đã đo. **Không cờ nào thắng được tập dev**, nên
mặc định tắt — bật là quyết định có ý thức, không phải mặc nhiên.

| Cờ | Bước | Đo được | Có nên bật |
| --- | --- | --- | --- |
| `--uu-tien-video 50` | 1 (mềm) | +0,0095 ở cả hai mức | 🟡 dưới ngưỡng nhiễu |
| `--thu-hep-cung` | 1 (cứng) | −0,0286 / −0,0476 | ❌ **đừng** — metadata chỉ phủ 37% ở top-50 |
| `--dedup` | 2b | đảo dấu giữa hai mức | ❌ không dùng để quyết |
| `--vlm` | 4 | chưa đo được | 🔴 xem dưới |

Đo lại bất cứ lúc nào: `python scripts\22_do_mui_nhon_1.py`

## Bước 4 — cho VLM trả lời câu Q&A

Đây là chỗ đắt nhất còn lại: **3/24 gói đề mẫu là Q&A và cả ba đang chắc chắn
0 điểm** vì `answer` là hằng số. Code đã nối sẵn, chặn ở đúng một lệnh tải model:

```powershell
ollama serve                      # nếu chưa chạy
ollama pull qwen2.5vl:3b          # ~3,2 GB. Bản 7b (~6 GB) sát trần máy 7,7 GB

.venv\Scripts\python.exe scripts\22_do_mui_nhon_1.py --vlm --so-cau 20   # ĐO TRƯỚC
.venv\Scripts\python.exe src\run.py --de de_thi --ra submission --vlm    # rồi mới nộp
```

**Đo trước khi nộp, đừng đảo thứ tự.** `tra_loi.kiem_model_nhin_duoc()` chặn
model thuần văn bản, nhưng nó không chặn được một model *nhìn được* mà *trả lời
sai định dạng* — mà sai định dạng thì 0 điểm y như bỏ trống (PHẦN C mục 4).
`--so-cau 20` chạy nhanh, đủ để nhìn ra đáp án có ngắn và chuẩn tắc không.

> Máy này chỉ có ảnh của **L21, L22, L27** (21.810/177.321 dòng có `kf_path`).
> Câu Q&A rơi vào nhóm khác thì VLM không có gì để nhìn — `gan_dap_an()` trả về
> `--tra-loi` làm đáp án dự phòng chứ **không bao giờ để trống**, vì `answer`
> rỗng làm `nop_bai.soat` chặn cả gói.

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
