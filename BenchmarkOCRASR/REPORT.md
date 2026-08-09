# Báo cáo benchmark OCR và ASR trên dữ liệu L29

## 1. Tóm tắt điều hành

Benchmark được thực hiện để chọn một mô hình OCR và một mô hình ASR cho pipeline AIC 2026. Dữ liệu đánh giá lấy từ phần L29 do ban tổ chức cung cấp, không sử dụng các file ZIP và không mở rộng sang tập khác.

Kết quả chính:

- OCR: chạy thành công 3 mô hình trên 60 ảnh, tổng cộng 180/180 lượt suy luận có trạng thái `OK`.
- ASR: chạy thành công 3 mô hình trên 20 clip, tổng cộng 60/60 lượt suy luận có trạng thái `OK`.
- Tất cả nhãn OCR và ASR dùng để tính kết quả đã được đánh dấu `approved` sau bước duyệt thủ công.
- Không mô hình OCR nào đạt ngưỡng character accuracy 80% đã đặt ra. Mô hình gần ngưỡng nhất là `easyocr_vi_en`, đạt 76.00% trên chữ tĩnh.
- `phowhisper_small` là mô hình ASR tốt nhất: WER 0.4332, CER 0.3760, timestamp hợp lệ 100% và coverage 99.98%.
- Cấu hình đề xuất để tích hợp: `easyocr_vi_en` cho OCR và `phowhisper_small` cho ASR.

Lưu ý về cách chốt OCR: theo điều kiện đạt tuyệt đối, kết luận chính thức vẫn là chưa có model đạt ngưỡng 80%. Tuy nhiên, nếu cần chọn một cấu hình để tiếp tục phát triển pipeline, `easyocr_vi_en` là phương án thực dụng tốt nhất về tổng thể.

## 2. Mục tiêu và phạm vi

Mục tiêu của benchmark:

1. Đo chất lượng nhận dạng văn bản trên hình ảnh L29.
2. Đo chất lượng nhận dạng tiếng Việt và timestamp trên âm thanh L29.
3. Đo tốc độ và VRAM để bảo đảm mô hình có thể chạy trên RTX 2060 6 GB.
4. Chọn cấu hình đủ gọn để đưa vào pipeline xử lý dữ liệu lớn.

Phạm vi thực tế:

- Dataset nguồn: `D:/Study/AICChallenge`.
- Chỉ sử dụng phần L29.
- Video được chọn: `L29_V001`, `L29_V004`, `L29_V008`, `L29_V012`, `L29_V014`.
- OCR, ASR được benchmark độc lập; không benchmark CLIP hoặc VLM.
- Seed benchmark: `2026`.
- Chính sách lựa chọn: ưu tiên độ chính xác; nếu kết quả gần nhau mới xét latency và VRAM.

Tài liệu thiết kế ban đầu dùng L21 làm ví dụ. Pipeline và số liệu cuối trong báo cáo này đã chuyển sang đúng dữ liệu L29 theo yêu cầu thực tế.

## 3. Môi trường thực thi và quản lý cache

Benchmark chạy trên:

- Hệ điều hành: Windows 11.
- Python: 3.12.0.
- GPU: NVIDIA GeForce RTX 2060, 6144 MiB VRAM.
- Driver NVIDIA: 610.74.
- Môi trường Python: `.venv` có sẵn tại thư mục gốc dự án; không tạo môi trường mới và không cài global.

`set_up.sh` cấu hình toàn bộ cache mô hình vào:

```text
D:/Library/ai_cache
```

Các cache con gồm Hugging Face Hub, Hugging Face datasets/assets/Xet, Torch, EasyOCR, PaddleOCR, VietOCR và Whisper. Cấu hình này ngăn model/checkpoint tải vào ổ C trong quá trình chạy. PaddleOCR mobile thực tế được lưu tại:

```text
D:/Library/ai_cache/paddle/official_models/PP-OCRv5_mobile_det
D:/Library/ai_cache/paddle/official_models/latin_PP-OCRv5_mobile_rec
```

Mỗi model được chạy trong một process riêng. Cách ly process giúp tránh xung đột CUDA/cuDNN giữa PaddlePaddle và PyTorch, đồng thời nếu một model OOM hoặc lỗi thì kết quả của các model khác vẫn được giữ lại.

## 4. Chuẩn bị dữ liệu và gán nhãn

### 4.1. Dữ liệu OCR

Từ mỗi video chọn 12 mẫu, tổng cộng 60 ảnh:

| Loại mẫu | Số lượng | Mục đích |
|---|---:|---|
| `dynamic_overlay` | 37 | Phụ đề, lower-third hoặc chữ phủ động trên hình |
| `static_text` | 13 | Tiêu đề, bảng hiệu hoặc chữ đứng yên |
| `no_text` | 10 | Đo khả năng nhận nhầm chữ khi ground truth rỗng |
| **Tổng** | **60** | 12 mẫu/video × 5 video |

Mỗi nhãn OCR lưu các trường chính: `sample_id`, `video_id`, `frame_idx`, `pts_time`, `image_path`, `bbox_xyxy`, `text_type`, `text_raw`, `legibility` và `review_status`.

Quy trình gán nhãn:

1. Trích các frame ứng viên từ video L29.
2. Tạo contact sheet và crop vùng chữ để duyệt nhanh.
3. Đọc đúng phần chữ nhìn thấy trong frame, không tự hoàn thiện câu bị cắt.
4. Sửa nội dung trong `eval_data/review/ocr_review.csv`.
5. Import về `eval_data/ocr/ocr_labels.jsonl`.
6. Chỉ dùng mẫu có `review_status=approved` làm ground truth.

### 4.2. Dữ liệu ASR

Mỗi video được cắt bốn đoạn tại các mốc 120, 300, 540 và 780 giây. Mỗi đoạn dài 30 giây:

| Thuộc tính | Giá trị |
|---|---:|
| Số video | 5 |
| Clip mỗi video | 4 |
| Tổng số clip | 20 |
| Độ dài mỗi clip | 30 giây |
| Tổng thời lượng | 600 giây, tương đương 10 phút |
| Trạng thái nhãn | 20/20 `approved` |

Transcript nháp ban đầu được tạo để giảm thời gian nhập liệu, sau đó phải nghe, sửa và xác nhận thủ công. Transcript nháp không được coi là ground truth nếu chưa duyệt. Dữ liệu cuối lưu trong `eval_data/asr/transcripts.json`, gồm đường dẫn audio, mốc trong video nguồn, transcript, timestamp và trạng thái duyệt.

Trước khi chạy benchmark, toàn bộ nhãn được kiểm tra bằng:

```powershell
& 'D:\Projects\AIC_2026_FirstDance\.venv\Scripts\python.exe' `
  '.\scripts\validate_labels.py' --require-approved
```

Kết quả: `Label validation passed`.

## 5. Các mô hình được so sánh

### 5.1. OCR

| ID | Detector | Recognizer | Vai trò |
|---|---|---|---|
| `paddleocr_v5_mobile` | `PP-OCRv5_mobile_det` | `latin_PP-OCRv5_mobile_rec` | Ưu tiên tốc độ, pipeline PaddleOCR đầy đủ |
| `easyocr_det_vietocr` | EasyOCR vi/en | VietOCR `vgg_transformer` | Kỳ vọng nhận dạng tiếng Việt tốt hơn sau detection |
| `easyocr_vi_en` | EasyOCR vi/en | EasyOCR vi/en | Baseline và ứng viên cân bằng |

Tên PaddleOCR đã được kiểm tra lại từ log tải model. Cấu hình cuối thực sự dùng detector mobile, không dùng `PP-OCRv5_server_det`.

### 5.2. ASR

| ID | Checkpoint | Vai trò |
|---|---|---|
| `phowhisper_small` | `vinai/PhoWhisper-small` | Ứng viên tiếng Việt nhẹ |
| `phowhisper_medium` | `vinai/PhoWhisper-medium` | Kiểm tra đổi thêm VRAM/tốc độ lấy chất lượng |
| `whisper_small` | `openai/whisper-small` | Baseline Whisper đa ngôn ngữ |

Ba mô hình dùng cùng cấu hình generation để so sánh công bằng:

```yaml
language: vi
task: transcribe
batch_size: 1
num_beams: 5
do_sample: false
temperature: 0.0
return_timestamps: true
```

### 5.3. Luồng thực thi benchmark

Pipeline chạy theo thứ tự sau:

1. Đọc `configs/benchmark.yaml` và `configs/models.yaml`, cố định dataset, seed, model và tham số generation.
2. Kiểm tra mọi ground truth đều có trạng thái `approved`; dừng nếu còn nhãn chưa duyệt.
3. Orchestrator tạo một process độc lập cho từng model OCR. Mỗi process load đúng một model, chạy lần lượt 60 ảnh, ghi prediction, metric, latency, RAM/VRAM và trạng thái từng mẫu.
4. Sau khi process OCR kết thúc, tài nguyên được giải phóng trước khi model tiếp theo được load.
5. Quy trình tương tự được áp dụng cho ASR: mỗi model chạy 20 clip với batch size 1, trả transcript và timestamp.
6. Kết quả từng process được ghi vào `results/.parts/` trước, sau đó mới aggregate thành các file CSV/JSON chung. Cách này tránh mất toàn bộ kết quả nếu một model lỗi giữa chừng.
7. Module báo cáo tạo bảng tổng hợp, bootstrap confidence interval, biểu đồ và `conclusion.json` theo policy accuracy-first.

```text
Ground truth approved
        |
        v
Đọc config và tách process theo model
        |
        +--> OCR: 3 model × 60 ảnh --> metric OCR
        |
        +--> ASR: 3 model × 20 clip --> metric ASR
        |
        v
Aggregate CSV/JSON --> biểu đồ --> kết luận
```

## 6. Chỉ số đánh giá

### 6.1. OCR

- CER: tỷ lệ lỗi ký tự; càng thấp càng tốt.
- Character accuracy: `1 - CER`; càng cao càng tốt.
- WER: tỷ lệ lỗi từ; càng thấp càng tốt. WER có thể lớn hơn 1 khi model sinh nhiều từ thừa.
- Exact match: tỷ lệ dự đoán khớp hoàn toàn sau chuẩn hóa.
- Detection recall IoU 0.5: tỷ lệ tìm đúng vùng chữ với IoU tối thiểu 0.5.
- False-positive rate: tỷ lệ ảnh `no_text` vẫn bị model trả về chữ.
- Latency median và p95: thời gian xử lý một ảnh.
- VRAM peak: mức VRAM cao nhất quan sát được trong lượt chạy model.

### 6.2. ASR

- WER và CER: tỷ lệ lỗi từ và ký tự so với transcript thủ công.
- Khoảng tin cậy WER 95%: bootstrap 2.000 lần.
- Timestamp valid rate: tỷ lệ clip có timestamp đúng cấu trúc và thứ tự.
- Timestamp coverage: phần thời lượng clip được timestamp bao phủ.
- RTF: thời gian suy luận chia cho độ dài audio; thấp hơn là nhanh hơn. RTF dưới 1 nghĩa là xử lý nhanh hơn thời gian thực.
- VRAM peak và trạng thái OOM.

Ngưỡng cấu hình:

- OCR character accuracy mục tiêu: ít nhất 0.80.
- ASR timestamp valid rate: 1.00.
- ASR timestamp coverage: ít nhất 0.95.

## 7. Kết quả OCR

### 7.1. Chữ động và phụ đề — 37 mẫu

| Model | CER ↓ | Character accuracy ↑ | WER ↓ | Exact ↑ | Detection recall ↑ | Median / p95 | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| PaddleOCR V5 Mobile | 0.7003 | 0.2997 | 1.0261 | 0.0000 | 0.7568 | 0.049 / 0.060 s | 1.260 GB |
| EasyOCR + VietOCR | **0.5802** | **0.4198** | 0.8770 | 0.0811 | **0.9730** | 0.343 / 0.522 s | 0.754 GB |
| EasyOCR vi/en | 0.5947 | 0.4053 | **0.8432** | 0.0811 | **0.9730** | **0.189 / 0.239 s** | **0.611 GB** |

EasyOCR + VietOCR có character accuracy tốt nhất trên chữ động, hơn EasyOCR vi/en khoảng 1,45 điểm phần trăm. Đổi lại, latency median cao hơn khoảng 1,8 lần. EasyOCR vi/en có WER tốt hơn và detection recall ngang bằng, nên phù hợp hơn nếu cần cân bằng chất lượng và thông lượng.

### 7.2. Chữ tĩnh — 13 mẫu

| Model | CER ↓ | Character accuracy ↑ | WER ↓ | Exact ↑ | Detection recall ↑ | Median / p95 | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| PaddleOCR V5 Mobile | 0.4441 | 0.5559 | 0.7681 | 0.1538 | 0.9231 | **0.053 / 0.072 s** | 1.260 GB |
| EasyOCR + VietOCR | 0.4877 | 0.5123 | **0.4000** | **0.3846** | **1.0000** | 0.459 / 1.062 s | 0.754 GB |
| EasyOCR vi/en | **0.2400** | **0.7600** | 0.6637 | 0.1538 | **1.0000** | 0.201 / 0.315 s | **0.611 GB** |

EasyOCR vi/en thắng rõ theo CER và character accuracy, đồng thời nhanh hơn mô hình hybrid hơn hai lần. EasyOCR + VietOCR có WER và exact match tốt hơn, cho thấy lỗi của nó tập trung vào ký tự trong ít từ hơn; tuy nhiên tiêu chí chính của benchmark là character accuracy.

### 7.3. Ảnh không có chữ — 10 mẫu

| Model | False-positive rate ↓ | Median / p95 |
|---|---:|---:|
| PaddleOCR V5 Mobile | 0.90 | 0.046 / 0.050 s |
| EasyOCR + VietOCR | **0.80** | 0.238 / 0.320 s |
| EasyOCR vi/en | **0.80** | 0.168 / 0.213 s |

False-positive còn rất cao ở cả ba mô hình. Một phần có thể đến từ logo hoặc watermark phát sóng vẫn xuất hiện trong ảnh dù vùng nội dung mục tiêu không có chữ. Vì vậy OCR không nên tự quyết định ảnh có thông tin hữu ích chỉ dựa trên việc model trả về chuỗi khác rỗng.

### 7.4. Đánh giá ngưỡng OCR

| Model | Accuracy chữ tĩnh | Accuracy chữ động | Đạt 80%? |
|---|---:|---:|---|
| PaddleOCR V5 Mobile | 55.59% | 29.97% | Không |
| EasyOCR + VietOCR | 51.23% | **41.98%** | Không |
| EasyOCR vi/en | **76.00%** | 40.53% | Không |

Theo tiêu chí đạt tuyệt đối, chưa thể tuyên bố model OCR nào vượt benchmark. Nếu cần một model duy nhất cho bước tích hợp, EasyOCR vi/en là lựa chọn tốt nhất vì:

- Chất lượng chữ tĩnh tốt nhất và gần ngưỡng 80% nhất.
- Chất lượng chữ động chỉ thấp hơn mô hình hybrid 1,45 điểm phần trăm.
- Latency tốt hơn mô hình hybrid.
- VRAM thấp nhất trong ba model.

PaddleOCR V5 Mobile nhanh nhất, nhưng độ chính xác giảm quá nhiều nên chỉ phù hợp cho bước quét sơ bộ khi tốc độ quan trọng hơn nội dung nhận dạng.

## 8. Kết quả ASR

| Model | WER ↓ | WER 95% CI | CER ↓ | Timestamp valid ↑ | Coverage ↑ | RTF median / p95 ↓ | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| PhoWhisper Small | **0.4332** | 0.2577–0.6656 | **0.3760** | **1.00** | **0.9998** | 0.1281 / **0.2892** | **0.835 GB** |
| PhoWhisper Medium | 0.5528 | 0.3044–0.9189 | 0.4695 | **1.00** | 0.9254 | 0.2234 / 0.5267 | 2.357 GB |
| Whisper Small | 0.6656 | 0.4250–0.9540 | 0.4902 | 0.80 | 0.8773 | **0.1221** / 0.3942 | 0.835 GB |

Không model nào bị OOM. PhoWhisper Small tốt nhất về WER, CER, timestamp coverage và p95 latency. Whisper Small chỉ nhanh hơn khoảng 5% theo RTF median nhưng WER tăng từ 0.4332 lên 0.6656, đồng thời chỉ 80% timestamp hợp lệ. PhoWhisper Medium chậm hơn, dùng gần 2,36 GB VRAM và vẫn kém chính xác hơn bản Small trên tập này.

Đánh giá điều kiện timestamp:

- PhoWhisper Small: đạt validity 1.00 và coverage ≥ 0.95.
- PhoWhisper Medium: đạt validity nhưng không đạt coverage 0.95.
- Whisper Small: không đạt cả validity 1.00 lẫn coverage 0.95.

Do đó `phowhisper_small` là lựa chọn ASR cuối cùng.

## 9. Cấu hình đề xuất cuối cùng

```yaml
ocr:
  id: easyocr_vi_en
  family: easyocr
  params:
    languages: [vi, en]
    gpu: true

asr:
  id: phowhisper_small
  family: transformers_whisper
  repo_id: vinai/PhoWhisper-small

asr_generation:
  language: vi
  task: transcribe
  batch_size: 1
  num_beams: 5
  do_sample: false
  temperature: 0.0
  return_timestamps: true
```

Chính sách sử dụng đề xuất:

1. Dùng EasyOCR vi/en làm OCR mặc định.
2. Crop vùng nội dung cần tìm trước OCR; loại vùng logo/watermark cố định nếu có thể.
3. Không coi mọi chuỗi OCR khác rỗng là bằng chứng ảnh có nội dung hữu ích; cần calibrate confidence threshold trên thêm mẫu `no_text`.
4. Dùng PhoWhisper Small cho transcript và timestamp.
5. Giữ batch size 1 để ổn định trên RTX 2060 6 GB.
6. Chỉ dùng PaddleOCR Mobile như phương án quét tốc độ cao; không dùng làm OCR cuối.

Kết luận máy đọc được:

```yaml
conclusion:
  ocr:
    formal_threshold_result: failed
    selected_model_for_integration: easyocr_vi_en
    character_accuracy_static: 0.759964
    character_accuracy_dynamic: 0.405331
    latency_median_static_sec: 0.201364
    latency_median_dynamic_sec: 0.188903
    vram_peak_gb: 0.611148
  asr:
    selected_model: phowhisper_small
    wer: 0.433235
    cer: 0.376026
    timestamp_valid_rate: 1.0
    timestamp_coverage: 0.999833
    rtf_median: 0.128082
    vram_peak_gb: 0.835125
    oom_occurred: false
```

## 10. Cách chạy lại benchmark

Dùng Git Bash để cấu hình cache và tải model:

```bash
cd /d/Projects/AIC_2026_FirstDance/BenchmarkOCRASR
./set_up.sh --persist --download-models --verify --cache-root D:/Library/ai_cache
```

Sau khi môi trường đã được cấu hình, chạy bằng `.venv` của dự án:

```powershell
Set-Location 'D:\Projects\AIC_2026_FirstDance\BenchmarkOCRASR'

& 'D:\Projects\AIC_2026_FirstDance\.venv\Scripts\python.exe' `
  '.\scripts\validate_labels.py' --require-approved

& 'D:\Projects\AIC_2026_FirstDance\.venv\Scripts\python.exe' '.\run_all.py'
```

Có thể chạy riêng từng phần bằng `run_ocr.py` hoặc `run_asr.py`. Không dùng `--include-unreviewed` để đưa ra kết luận lựa chọn model.

Các đầu ra chính:

- `results/ocr_results.csv`: kết quả chi tiết từng ảnh và từng model.
- `results/ocr_summary.csv` và `.json`: tổng hợp OCR.
- `results/asr_results.csv`: kết quả chi tiết từng clip và từng model.
- `results/asr_summary.csv` và `.json`: tổng hợp ASR.
- `results/summary_report.md`: báo cáo được sinh tự động.
- `results/conclusion.json`: kết luận theo policy tự động.
- `results/plots/`: biểu đồ chất lượng, tốc độ và tài nguyên.

## 11. Kiểm tra kỹ thuật đã hoàn thành

- Validate ground truth: đạt.
- OCR: 180/180 lượt `OK`.
- ASR: 60/60 lượt `OK`.
- Unit test: 8/8 test đạt.
- `pip check`: không có dependency bị hỏng.
- Model PaddleOCR được xác nhận tải vào ổ D và dùng đúng detector mobile.

Các cảnh báo trong log:

- Không có `ccache`: chỉ ảnh hưởng khi cần biên dịch extension, không làm benchmark thất bại.
- PaddlePaddle được build với cuDNN 9.9 trong khi runtime báo cuDNN 9.5. Lần benchmark này vẫn hoàn thành toàn bộ mẫu, nhưng cần giữ process isolation và theo dõi lại nếu đổi driver hoặc PaddlePaddle.

## 12. Hạn chế và công việc tiếp theo

1. Tập OCR chỉ có 60 ảnh và chữ tĩnh chỉ có 13 mẫu; khoảng tin cậy còn rộng.
2. Nhóm `dynamic_overlay` không đồng nghĩa hoàn toàn với ticker chạy. Chưa nên dùng kết quả này để khẳng định đã đạt yêu cầu ticker riêng của cuộc thi.
3. ASR chỉ có 20 clip, tổng thời lượng 10 phút; chưa bao phủ đầy đủ nhạc nền lớn, nhiều người nói, giọng vùng miền và âm thanh kém.
4. Transcript khởi tạo từ model rồi được sửa thủ công. Dù đã `approved`, vòng đánh giá cuối nên có người thứ hai kiểm tra độc lập để giảm thiên lệch.
5. False-positive OCR rất cao; cần bổ sung mẫu âm tính và thử ROI/confidence threshold.
6. Trước khi khóa model cho vòng thi, nên bổ sung ít nhất một tập holdout mới không dùng trong quá trình chỉnh pipeline, rồi chạy lại cùng cấu hình cố định.

Đề xuất vòng tiếp theo: ưu tiên gán nhãn thêm ticker chạy thật, ảnh không có chữ nhưng có watermark/logo, và audio khó. Nếu EasyOCR vi/en vẫn không đạt 80%, thử cải thiện crop/tiền xử lý trước khi tăng độ phức tạp của model.
