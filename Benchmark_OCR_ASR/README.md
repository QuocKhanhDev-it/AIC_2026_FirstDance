# Benchmark OCR và ASR cho L29

Thư mục này chứa pipeline benchmark tách biệt cho OCR và ASR, dùng đúng dữ liệu L29 tại `D:/Study/AICChallenge` và đúng môi trường Python có sẵn tại `../.venv`.

## Trạng thái dữ liệu gán nhãn

- OCR: 60 ảnh từ 5 video; 50 ảnh có chữ và 10 negative samples `no_text` để đo false-positive.
- ASR: 20 clip x 30 giây từ 5 video, tổng cộng 10 phút.
- Nhãn OCR rõ ràng đã được sửa nháp bằng kiểm tra trực quan.
- Transcript ASR được PhoWhisper-small tạo nháp.
- Tất cả vẫn là `needs_review`. Pipeline mặc định không cho phép dùng chúng làm ground truth cho đến khi người nghe/nhìn xác nhận và đổi thành `approved`.

## Thiết lập và cache ổ D

Chạy bằng Git Bash trên Windows:

```bash
cd '/d/Projects/AIC_2026_FirstDance/BenchmarkOCRASR'
./set_up.sh --persist --verify
```

`--install` chỉ cài vào `../.venv`; script không tạo virtual environment và không cài global. `--download-models` tải snapshot thật, không cần Windows Developer Mode. Các cache model nằm dưới `D:/Library/ai_cache`, bao gồm Hugging Face, PaddleOCR, EasyOCR và VietOCR.

## Duyệt nhãn thủ công

Hai file cần sửa bằng Excel/LibreOffice (giữ mã hóa UTF-8) là:

- `eval_data/review/ocr_review.csv`: đối chiếu với ảnh/contact sheet, sửa `text_raw`, rồi đổi `review_status` thành `approved`.
- `eval_data/review/asr_review.csv`: nghe `audio_path`, sửa transcript và timestamp, rồi đổi toàn bộ dòng của clip thành `approved`.

Import và kiểm tra:

```powershell
..\.venv\Scripts\python.exe scripts\import_review_csv.py
..\.venv\Scripts\python.exe scripts\validate_labels.py --require-approved
```

Không dùng transcript PhoWhisper nháp để kết luận PhoWhisper thắng; nó chỉ giúp giảm thời gian gõ và bắt buộc phải được nghe, sửa thủ công.

## Chạy benchmark

```powershell
..\.venv\Scripts\python.exe run_ocr.py
..\.venv\Scripts\python.exe run_asr.py
..\.venv\Scripts\python.exe run_all.py
```

Để smoke test code trước khi duyệt nhãn, có thể thêm `--include-unreviewed`; kết quả đó không hợp lệ để chọn model cuối cùng.

Mỗi model chạy trong một process riêng để tránh xung đột CUDA/cuDNN giữa PaddlePaddle và PyTorch. Runner ghi lỗi hoặc OOM của một model mà không làm mất kết quả model khác.

## Mô hình và đầu ra

OCR gồm PP-OCRv5, EasyOCR, và EasyOCR detector + VietOCR recognizer. ASR gồm PhoWhisper-small, PhoWhisper-medium và Whisper-small.

Thư mục `results/` nhận:

- `ocr_results.csv`, `ocr_summary.csv/json`: CER/WER strict và normalized, IoU/recall, false-positive, latency, RAM/VRAM.
- `asr_results.csv`, `asr_summary.csv/json`: WER/CER, timestamp validity/coverage, real-time factor, RAM/VRAM.
- `plots/*.png`: biểu đồ chất lượng, tốc độ và tài nguyên.
- `summary_report.md`, `conclusion.json`: bảng tổng hợp và lựa chọn theo accuracy-first.

Nếu L29 có dưới 10 mẫu ticker cuộn thực sự, báo cáo cố ý ghi kết luận ticker là `inconclusive` thay vì suy diễn từ logo/lower-third.

## Kiểm thử

Trên máy hiện tại, Paddle báo checkpoint chạy được nhưng cuDNN runtime 9.5 khác bản 9.9 mà Paddle 3.2.0 được biên dịch cùng. Process isolation đã tránh crash với PyTorch; vẫn nên theo dõi log/OOM khi chạy đủ 60 ảnh và không bỏ qua model có trạng thái `FAILED`.

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

## OCR benchmark v2 cho BM25 — L21 + L29

V2 giữ nguyên ASR, cố định EasyOCR detector và chỉ chốt recognizer trên crop ground-truth. Metric chính là corpus WER sau đúng tokenizer BM25; Exact là tiêu chí phụ, CER chỉ còn để chẩn đoán.

### 1. Gán nhãn 100 positive + 100 negative

```powershell
..\.venv\Scripts\python.exe scripts\prepare_ocr_v2.py prepare
# Tùy chọn: sinh box/text nháp trên chính 400 frame đã lấy ngẫu nhiên; không chọn mẫu bằng score
..\.venv\Scripts\python.exe scripts\prepare_ocr_v2.py draft
# Duyệt eval_data/ocr_v2/review.csv cùng contact_sheets, sau đó:
..\.venv\Scripts\python.exe scripts\prepare_ocr_v2.py import
..\.venv\Scripts\python.exe scripts\prepare_ocr_v2.py finalize
..\.venv\Scripts\python.exe scripts\validate_ocr_v2.py --require-approved
```

Mỗi dòng được tính phải có `review_status=approved` và `second_review_status=approved`. Positive cần `bbox_xyxy`, `text_raw` và `semantic_type`; negative dùng `no_target_text` hoặc `no_text_anywhere`. Ticker không được gán positive.

### 2. Chạy recognizer-only và gate

```powershell
..\.venv\Scripts\python.exe run_ocr_v2.py recognizer
```

Sau khi duyệt contact sheet dev, điền ROI chuẩn hóa cho L21/L29 trong `configs/roi_v2.yaml`, đổi `review_status` thành `approved`, rồi chạy:

```powershell
..\.venv\Scripts\python.exe run_ocr_v2.py gate
```

Threshold chỉ được chọn trên dev. Holdout phải đạt false-positive ≤ 5% và recall ≥ 80%.

### 3. Đo giá trị tăng thêm so với metadata

```powershell
..\.venv\Scripts\python.exe scripts\prepare_ocr_queries.py prepare
# Viết lại và duyệt 40 query trong retrieval_queries_review.csv
..\.venv\Scripts\python.exe scripts\prepare_ocr_queries.py import
..\.venv\Scripts\python.exe scripts\prepare_ocr_queries.py validate
..\.venv\Scripts\python.exe run_ocr_retrieval.py --model-id vietocr_vgg_transformer
```

Chỉ chạy toàn bộ 177.321 keyframe khi kết luận là `selected_for_full_run`. Hai trạng thái dừng là `drop_ocr_channel_false_positive` và `drop_ocr_channel_no_retrieval_gain`.
