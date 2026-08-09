# Hướng dẫn gán nhãn dữ liệu OCR & ASR — Benchmark AIC2026

## Mục đích

Chúng ta cần tự tạo đáp án đúng (ground truth) để so sánh xem model nào
đọc chữ/nghe tiếng tốt nhất. Không có việc này thì không thể biết model
nào đúng nhiều hơn — máy không tự biết đâu là "đúng", cần người xác nhận.

**Tổng thời gian ước tính: 2.5 - 3.5 giờ**, có thể chia 2 người làm song song
(1 người làm OCR, 1 người làm ASR) để rút còn ~1.5-2 giờ mỗi người.

**File nguồn dùng chung:** `L21_V030.mp4` (video mẫu bản tin HTV9, ~17 phút)

**Cần cài trước:** `ffmpeg` — kiểm tra bằng cách gõ `ffmpeg -version` trong
terminal/CMD. Nếu chưa có: Windows tải tại ffmpeg.org (giải nén, thêm vào
PATH), Mac dùng `brew install ffmpeg`, Ubuntu dùng `sudo apt install ffmpeg`.

---

## PHẦN A — Gán nhãn OCR (20 ảnh, ~1.5-2 giờ)

### Mục tiêu
Tạo 20 ảnh keyframe có chữ, kèm đáp án đúng là chữ hiển thị trên ảnh đó —
chia làm 2 nhóm 10 ảnh: **chữ tĩnh** (tiêu đề, tên phóng viên) và **ticker
chạy** (dòng chữ chạy dưới màn hình).

### Bước 1 — Tìm thời điểm có chữ trong video

Chạy lệnh sau để xuất ảnh xem trước, mỗi giây 1 ảnh:
```bash
ffmpeg -i L21_V030.mp4 -vf fps=1 -q:v 2 preview_%04d.jpg
```

Mở thư mục vừa xuất ra bằng File Explorer / Finder, xem nhanh qua các ảnh
(mỗi ảnh tương ứng 1 giây trong video, đánh số theo thứ tự). Ghi lại:
- **10 thời điểm** có chữ tĩnh rõ ràng (tiêu đề phóng sự, tên người, chữ đứng yên)
- **10 thời điểm** có ticker chữ chạy ở dưới màn hình

Ví dụ ghi chú tay: `ảnh preview_0126.jpg (giây 125-126) có tiêu đề "GIẢM LỆ PHÍ..."`.

> **Mẹo:** video này có nhiều phóng sự khác nhau ghép lại trong 1 file — cố
> gắng lấy chữ từ nhiều đoạn phóng sự khác nhau, đừng lấy 20 ảnh dồn hết vào
> 1 đoạn, để dữ liệu test đa dạng hơn.

### Bước 2 — Trích ảnh chất lượng cao tại đúng thời điểm đã ghi

Với mỗi thời điểm đã ghi ở Bước 1, chạy:
```bash
ffmpeg -ss 125.5 -i L21_V030.mp4 -frames:v 1 -q:v 1 static_001.jpg
```
Thay `125.5` bằng số giây thật, thay tên file tăng dần: `static_001.jpg` đến
`static_010.jpg` cho nhóm chữ tĩnh, `ticker_001.jpg` đến `ticker_010.jpg`
cho nhóm ticker chạy.

### Bước 3 — Gõ đáp án đúng vào file `labels.json`

Mở từng ảnh vừa trích, đọc thật kỹ dòng chữ hiển thị, gõ **chính xác từng
chữ** (kể cả dấu câu, chữ hoa/thường, dấu tiếng Việt) vào 1 file JSON theo
mẫu sau:

```json
{
  "static_001.jpg": "PHÓNG VIÊN: NGUYỄN VĂN A",
  "static_002.jpg": "GIẢM LỆ PHÍ TRƯỚC BẠ Ô TÔ TỪ 1/1/2026",
  "static_003.jpg": "HTV9 - THỜI SỰ 60 GIÂY",

  "ticker_001.jpg": "TIN NÓNG: BÃO SỐ 5 ĐANG TIẾN VÀO BIỂN ĐÔNG",
  "ticker_002.jpg": "GIÁ VÀNG HÔM NAY TĂNG NHẸ SO VỚI PHIÊN TRƯ",
  "ticker_003.jpg": "...CHIỀU MAI CÓ MƯA RẢI RÁC Ở KHU VỰC MIỀN N"
}
```

**Lưu ý quan trọng với ticker chạy:** ở đúng khung hình đó, chữ có thể đang
bị cắt giữa chừng (vì ticker đang chạy qua màn hình). Chỉ gõ đúng **những
gì nhìn thấy trong khung hình đó**, không cố đoán hay tự hoàn thiện câu —
kể cả khi câu bị cụt đầu/cụt đuôi, cứ gõ đúng như vậy.

### Kết quả nộp lại
- Thư mục ảnh: `static_001.jpg` ... `static_010.jpg`, `ticker_001.jpg` ... `ticker_010.jpg`
- File `labels.json` chứa đủ 20 dòng như mẫu trên

---

## PHẦN B — Gán nhãn ASR (10 đoạn audio 30 giây, ~1-1.5 giờ)

### Mục tiêu
Cắt 10 đoạn audio ngắn từ video, nghe và gõ lại chính xác lời đọc, kèm mốc
thời gian của từng câu.

### Bước 1 — Cắt 10 đoạn audio, rải đều trong video

Video dài ~17 phút, cắt 10 đoạn 30 giây trải đều ra (không dồn hết vào đầu
video), ví dụ cách nhau ~90 giây một lần cắt:

```bash
ffmpeg -i L21_V030.mp4 -ss 00:00:00 -t 30 -vn -acodec pcm_s16le -ar 16000 clip_001.wav
ffmpeg -i L21_V030.mp4 -ss 00:01:30 -t 30 -vn -acodec pcm_s16le -ar 16000 clip_002.wav
ffmpeg -i L21_V030.mp4 -ss 00:03:00 -t 30 -vn -acodec pcm_s16le -ar 16000 clip_003.wav
ffmpeg -i L21_V030.mp4 -ss 00:04:30 -t 30 -vn -acodec pcm_s16le -ar 16000 clip_004.wav
```
Cứ thế tăng dần `-ss` (định dạng `giờ:phút:giây`) cho đến `clip_010.wav`,
trải đều tới hết ~17 phút video.

### Bước 2 — Nghe và gõ lại lời đọc

Mở từng file `.wav` bằng trình phát nhạc bất kỳ (VLC, Windows Media Player,
hoặc Audacity nếu muốn xem waveform để dễ xác định điểm bắt đầu/kết thúc
câu). Nghe kỹ và gõ lại **đúng nguyên văn** lời đọc — không tóm tắt, không
sửa lỗi ngữ pháp của người đọc (nếu họ nói vấp thì gõ đúng như vậy).

Nếu câu khó nghe, tua lại nghe nhiều lần — không cần vội, độ chính xác quan
trọng hơn tốc độ ở bước này.

### Bước 3 — Ghi lại vào file `transcripts.json` kèm mốc thời gian

Với mỗi đoạn 30 giây, nếu có nhiều câu, tách riêng từng câu với thời điểm
bắt đầu/kết thúc **tính từ đầu đoạn clip đó** (giây 0 đến giây 30):

```json
{
  "clip_001.wav": {
    "segments": [
      {"start": 0.0, "end": 4.2, "text": "Chào mừng quý vị đến với bản tin hôm nay."},
      {"start": 4.5, "end": 9.8, "text": "Hôm nay chúng tôi sẽ đề cập đến vấn đề giao thông đô thị."}
    ]
  },
  "clip_002.wav": {
    "segments": [
      {"start": 0.0, "end": 6.1, "text": "Sáng nay tại quận 1, một vụ va chạm giao thông đã xảy ra."}
    ]
  }
}
```

Thời gian không cần chính xác tuyệt đối tới mili-giây, chỉ cần đúng trong
khoảng ±0.5 giây là đủ dùng để so sánh với kết quả model.

### Kết quả nộp lại
- 10 file audio: `clip_001.wav` ... `clip_010.wav`
- File `transcripts.json` chứa transcript + timestamp cho cả 10 clip

---

## Tổng kết nộp lại (gửi cho người tổng hợp)

```
eval_data/
├── ocr/
│   ├── static_text/
│   │   ├── images/          (10 file static_*.jpg)
│   │   └── labels.json
│   └── ticker_scrolling/
│       ├── images/          (10 file ticker_*.jpg)
│       └── labels.json
└── asr/
    ├── clips/                (10 file clip_*.wav)
    └── transcripts.json
```

Nếu có chỗ nào không chắc chắn khi gõ nhãn (nghe không rõ, chữ bị che
khuất, ticker chạy quá nhanh không đọc kịp), cứ ghi chú thêm bên cạnh trong
file JSON hoặc note riêng — không sao, quan trọng là trung thực, không đoán
bừa để lấp đầy cho đủ.
