# Thư mục Production OCR & ASR (Giai đoạn 1 — AIC 2026)

Thư mục này được tạo riêng (`pipeline_OCR_ASR/`) theo yêu cầu cấu trúc độc lập để phát triển và vận hành toàn bộ Pipeline trích xuất **OCR & ASR Production** mà không ảnh hưởng tới mã nguồn bên ngoài.

---

## 1. Phương án Chốt (Giai đoạn 0 -> Giai đoạn 1)

- **ASR (PhoWhisper-small)**: Đã bench đạt WER 0.4332, RTF 0.1281, timestamp hợp lệ 100%. Tốc độ xử lý ~3.3 giờ máy cho toàn bộ kho 16.6 giờ video.
- **OCR (EasyOCR / VietOCR)**: Đã chốt theo Phương án A (Local specialized models). Tích hợp lọc ROI (cắt banner header/footer) và ngưỡng tin cậy (confidence threshold) nhằm triệt tiêu 80–90% false-positive từ hình nền.
- **Chế độ Lọc Cứng (Hard Filtering / WHERE ocr_text LIKE '%...')**: Hỗ trợ khớp chính xác cho các token hiếm (tên riêng, con số, biển số xe, mã hiệu) song song với thuật toán BM25 mượt.

---

## 2. Cấu trúc thư mục

```
pipeline_OCR_ASR/
├── __init__.py
├── config.py                 # Cấu hình đường dẫn, model, ROI, confidence
├── ocr_processor.py          # Module trích xuất chữ OCR từ keyframe
├── asr_processor.py          # Module trích xuất lời nói ASR từ video
├── gop_ocr_asr.py            # Module hợp nhất OCR & ASR thành ocr_asr.parquet
├── loc_cung_token_hiem.py    # Chế độ lọc cứng token hiếm (SQL LIKE '%...')
├── run_production_pipeline.py # Script CLI chính để điều phối và vận hành
├── test_pipeline.py          # Bộ kiểm thử đơn vị (Unit tests)
└── output/                   # Nơi chứa các file Parquet sản xuất
    ├── ocr.parquet
    ├── asr.parquet
    └── ocr_asr.parquet
```

---

## 3. Hướng dẫn sử dụng

### 3.1. Chạy OCR Production
Chạy trích xuất OCR trên toàn bộ keyframes trong `master.parquet` (hoặc thử nghiệm với `--limit`):

```powershell
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe -m pipeline_OCR_ASR.run_production_pipeline --ocr
```

### 3.2. Gộp kết quả OCR & ASR
Tạo file `ocr_asr.parquet` chứa cả 177,321 dòng khung hình tương thích trực tiếp với `src/bm25.py`:

```powershell
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe -m pipeline_OCR_ASR.run_production_pipeline --gop
```

### 3.3. Thử nghiệm truy vấn (BM25 + Lọc Cứng Token Hiếm)

```powershell
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe -m pipeline_OCR_ASR.run_production_pipeline --test-query "79A-12345"
```

### 3.4. Kiểm thử (Unit Tests)

```powershell
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe -m pytest pipeline_OCR_ASR/test_pipeline.py -q
```
