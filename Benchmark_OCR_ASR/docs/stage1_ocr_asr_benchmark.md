# Benchmark Giai đoạn 1 — Chốt OCR & ASR (vai trò TV4, theo Kế hoạch v3)

## 0. Phạm vi & mục tiêu

Nhiệm vụ: **quyết định 1 model OCR + 1 model ASR** để đưa vào pipeline chính
(TV4 trong kế hoạch v3, mục 0.b). Không benchmark CLIP/VLM — các module đó
đã chốt hoặc do người khác phụ trách.

Tiêu chí "qua" theo v3 (dòng 179): OCR đúng ≥ 80% trên ticker; ASR ra
transcript có timestamp. Benchmark này đo **định lượng** để có số liệu báo
cáo, không chỉ ước lượng "đạt/không đạt".

Ràng buộc thời gian: nằm trong 48 giờ đầu (Giai đoạn 0), không kéo dài như
bản thiết kế benchmark tổng quát trước đó — cắt bỏ các mục không ảnh hưởng
trực tiếp tới quyết định chọn model (stress test dài, consistency nhiều
vòng lặp, edge case mở rộng).

## 1. Model đưa vào so sánh

### OCR — 3 ứng viên
| Model | Vai trò trong so sánh |
|---|---|
| PaddleOCR (PP-OCRv4, full pipeline detection+recognition) | Ứng viên chính, nhanh |
| PaddleOCR (detector) + VietOCR (recognizer) | Ứng viên chính, kỳ vọng đọc tiếng Việt tốt hơn |
| EasyOCR | Baseline đối chứng, để biết 2 model trên tốt hơn bao nhiêu |

### ASR — 2 ứng viên
| Model | Vai trò trong so sánh |
|---|---|
| PhoWhisper-small | Ứng viên chính, nhẹ, chạy nhanh trên 6GB VRAM |
| PhoWhisper-medium | Ứng viên chính, kỳ vọng chính xác hơn nhưng chậm hơn, kiểm tra có OOM không |

*(Không đưa PhoWhisper-large vào — đã xác nhận không chạy nổi trên RTX 2060 6GB.
Không đưa Whisper gốc OpenAI vào bản chính vì v3 đã chỉ định "PhoWhisper" —
chỉ thêm nếu bạn muốn có baseline đối chứng, đánh dấu optional bên dưới.)*

## 2. Dữ liệu test — LẤY THẲNG TỪ QUY MÔ v3 ĐÃ ĐỊNH, không mở rộng thêm

Theo mục 0.b của v3: **20 keyframe có ticker** cho OCR, **5 phút audio** cho ASR.
Benchmark này giữ đúng quy mô đó nhưng tách rõ theo 2 loại chữ (vì đây là
điểm rủi ro v3 đã cảnh báo ở Phần F):

### 2.1. Dữ liệu OCR — tách 2 nhóm bắt buộc
```
eval_data/ocr/
├── static_text/          # 10 keyframe: tiêu đề tĩnh, tên phóng viên, chữ đứng yên
│   ├── images/*.jpg       # crop sẵn vùng chữ, hoặc để nguyên full frame + toạ độ vùng
│   └── labels.json        # {"static_001.jpg": "văn bản đúng"}
│
└── ticker_scrolling/     # 10 keyframe: ticker chữ chạy dưới màn hình
    ├── images/*.jpg
    └── labels.json
```
Lấy từ video mẫu `L21_V030.mp4` (bản tin HTV9 đã xác nhận có ticker + đồng
hồ cháy vào hình, theo Phần A2 của kế hoạch v3).

### 2.2. Dữ liệu ASR — 5 phút audio, cắt thành đoạn ngắn để dễ đối chiếu
```
eval_data/asr/
├── clips/                 # cắt 5 phút thành 10 đoạn x 30s (dễ soát lỗi hơn 1 đoạn dài)
│   └── clip_001.wav ... clip_010.wav
└── transcripts.json       # {"clip_001.wav": {"text": "...", "start": 0.0, "end": 30.0}}
```
Lệnh cắt (yêu cầu đã cài `ffmpeg` sẵn trên máy — không nằm trong `setup_environment.sh`,
kiểm tra bằng `ffmpeg -version`; nếu chưa có: `sudo apt install ffmpeg` hoặc tải bản
Windows tại ffmpeg.org):
```bash
ffmpeg -i L21_V030.mp4 -ss 00:00:00 -t 30 -vn -acodec pcm_s16le -ar 16000 clip_001.wav
```
Nghe lại và tự gõ transcript đúng vào `transcripts.json` — đây là phần
duy nhất bắt buộc làm tay, không có cách tự động hóa (~1-1.5 giờ cho 5 phút
audio, vì phải nghe đi nghe lại để gõ chính xác timestamp câu).

## 3. Cấu trúc file/module

```
ocr_asr_benchmark/
├── configs/
│   └── models.yaml            # khai báo 3 OCR + 2 ASR, checkpoint path, vram_budget
│                               # (checkpoint tải qua HF_HOME/TRANSFORMERS_CACHE/TORCH_HOME
│                               #  đã set sẵn — xem move_cache_to_drive_d.md; PaddleOCR/
│                               #  Whisper cần path riêng vì không đọc các biến này)
│
├── eval_data/                 # như mục 2 ở trên
│
├── runners/
│   ├── run_ocr_bench.py       # loop 3 OCR model x 2 nhóm dữ liệu (static/ticker)
│   └── run_asr_bench.py       # loop 2 ASR model x 10 clip
│
├── utils/
│   ├── model_loader.py        # load/unload an toàn, giải phóng VRAM giữa các model
│   ├── vram_monitor.py        # đo VRAM peak
│   ├── text_metrics.py        # CER, WER (dùng jiwer)
│   └── timestamp_check.py     # verify ASR có trả về (start,end) hợp lệ, không rỗng
│
├── results/
│   ├── ocr_results.csv
│   ├── asr_results.csv
│   └── summary_report.md      # bảng cuối + khuyến nghị model, dùng để bạn báo cáo
│
└── run_all.py                 # chạy tuần tự OCR rồi ASR, xuất summary
```

## 4. Metric đo — bám sát tiêu chí "qua" của v3, không thêm phụ

### 4.1. OCR
- **Accuracy theo ký tự** = 1 − CER, báo cáo riêng theo `static_text` và `ticker_scrolling`
  → đối chiếu trực tiếp với ngưỡng "≥ 80%" mà v3 đặt ra
- **Accuracy theo từ** (1 − WER) — phụ, để tham khảo mức độ lỗi có làm sai nghĩa không
- **Thời gian xử lý / ảnh** (giây) — vì tổng dataset có thể ~200.000 keyframe (Phần A2 v3),
  cần biết tốc độ để ước tính OCR toàn bộ dataset mất bao lâu
- **VRAM peak khi chạy**

### 4.2. ASR
- **WER, CER** trên 10 clip, báo cáo trung bình + độ lệch giữa các clip
- **Timestamp có hợp lệ không**: kiểm tra output có field `(start, end)` cho từng câu,
  không bị rỗng hay lỗi định dạng (đây là điều kiện "qua" của v3, không chỉ WER thấp)
- **Real-time factor (RTF)** = thời gian xử lý / độ dài audio — quan trọng để ước tính
  thời gian ASR toàn bộ dataset (hàng trăm giờ video)
- **VRAM peak** — đặc biệt quan trọng cho PhoWhisper-medium vì sát ngưỡng 6GB,
  cần biết chắc có OOM không trước khi chốt

## 5. Yêu cầu bắt buộc trong code (do hardware RTX 2060 6GB)

1. Load 1 model tại một thời điểm, giải phóng VRAM (`del model; torch.cuda.empty_cache()`)
   trước khi load model kế tiếp.
2. Bắt lỗi OOM graceful cho PhoWhisper-medium — nếu OOM, ghi "SKIPPED - OOM" vào
   `asr_results.csv` thay vì crash, và ghi rõ trong summary report là cần chạy
   trên Kaggle/Colab nếu muốn có số liệu của model này.
3. Batch size = 1 mặc định cho ASR (tránh OOM), OCR có thể batch nhỏ 4-8 ảnh.
4. Log rõ input nào là `frame_idx` thật (không phải cột `n` trong CSV) khi ghi
   kết quả OCR ra ứng với keyframe nào — tránh đúng bẫy #2 mà v3 đã cảnh báo (Phần A5).

## 6. Output cuối — `summary_report.md` (dùng để báo cáo nhóm)

Bảng OCR:

| Model | Accuracy (static) | Accuracy (ticker) | WER | Thời gian/ảnh | VRAM peak | Đạt ngưỡng 80%? |
|---|---|---|---|---|---|---|
| PaddleOCR | ... | ... | ... | ... | ... | ✅/❌ |
| Paddle+VietOCR | ... | ... | ... | ... | ... | ✅/❌ |
| EasyOCR | ... | ... | ... | ... | ... | ✅/❌ |

Bảng ASR:

| Model | WER | CER | Timestamp hợp lệ? | RTF | VRAM peak | OOM? |
|---|---|---|---|---|---|---|
| PhoWhisper-small | ... | ... | ✅/❌ | ... | ... | Không |
| PhoWhisper-medium | ... | ... | ✅/❌ | ... | ... | Có/Không |

Cuối file bắt buộc có khối kết luận theo format cố định sau (key-value, để
dễ đọc/parse tự động nếu cần dùng ở bước sau), điền vào ngay sau khi có kết quả thật:

```yaml
conclusion:
  ocr:
    selected_model: null          # điền tên model được chọn
    meets_threshold_static: null  # true/false, ngưỡng 80%
    meets_threshold_ticker: null  # true/false, ngưỡng 80%
    fallback_policy: null         # ví dụ: "static_text_only" nếu ticker không đạt
    accuracy_static: null
    accuracy_ticker: null
    avg_latency_per_image_sec: null
    vram_peak_gb: null

  asr:
    selected_model: null
    meets_threshold_timestamp_valid: null  # true/false
    wer: null
    cer: null
    rtf: null                     # real-time factor
    vram_peak_gb: null
    oom_occurred: null            # true/false

  notes: null   # bất kỳ cảnh báo/rủi ro nào phát hiện thêm khi chạy
```
