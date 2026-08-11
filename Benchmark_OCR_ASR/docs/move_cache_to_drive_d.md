# Chuyển cache model sang ổ D (Windows)

Mặc định, HuggingFace/PaddleOCR/Whisper đều lưu checkpoint model vào ổ C
(thường tại `C:\Users\<tên_bạn>\.cache\...`). Phạm vi file này **chỉ** ép
phần cache model/checkpoint sang ổ D — không đụng tới venv hay thư mục code,
những cái đó bạn tự quyết định đặt ở đâu. Cần set biến môi trường trước khi
chạy `setup_environment.sh` hoặc bất kỳ lệnh python nào tải model.

## 1. Tạo thư mục chứa cache trên ổ D

Mở PowerShell hoặc CMD, chạy:

```powershell
mkdir D:\Library\ai_cache
mkdir D:\Library\ai_cache\huggingface
mkdir D:\Library\ai_cache\torch
mkdir D:\Library\ai_cache\paddle
mkdir D:\Library\ai_cache\whisper
```

## 2. Set biến môi trường (áp dụng cho toàn hệ thống, set 1 lần)

### Cách A — qua giao diện Windows (khuyến nghị, không mất khi tắt máy)
1. Nhấn `Win` → gõ "Environment Variables" → mở "Edit the system environment variables"
2. Chọn "Environment Variables..." → mục "User variables" → "New..."
3. Thêm lần lượt các biến sau:

| Tên biến | Giá trị |
|---|---|
| `HF_HOME` | `D:\Library\ai_cache\huggingface` |
| `HUGGINGFACE_HUB_CACHE` | `D:\Library\ai_cache\huggingface\hub` |
| `TRANSFORMERS_CACHE` | `D:\Library\ai_cache\huggingface\transformers` |
| `TORCH_HOME` | `D:\Library\ai_cache\torch` |

4. Khởi động lại terminal/VSCode để biến môi trường có hiệu lực.

### Cách B — set tạm trong 1 phiên PowerShell (mất khi đóng terminal, dùng để test nhanh)

```powershell
$env:HF_HOME = "D:\Library\ai_cache\huggingface"
$env:HUGGINGFACE_HUB_CACHE = "D:\Library\ai_cache\huggingface\hub"
$env:TRANSFORMERS_CACHE = "D:\Library\ai_cache\huggingface\transformers"
$env:TORCH_HOME = "D:\Library\ai_cache\torch"
```

## 3. Trường hợp riêng cho từng thư viện (một số không theo chuẩn HF_HOME)

### PaddleOCR
Mặc định lưu tại `~/.paddleocr`. Khi khởi tạo model trong code, truyền thẳng path:
```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    lang="vi",
    det_model_dir="D:/Library/ai_cache/paddle/det",
    rec_model_dir="D:/Library/ai_cache/paddle/rec",
    cls_model_dir="D:/Library/ai_cache/paddle/cls",
)
```
(Nếu để trống, lần đầu chạy nó tự tải vào `~/.paddleocr` bất kể set biến môi trường trên — cần set path thủ công như trên.)

### OpenAI Whisper (baseline so sánh)
Mặc định lưu tại `~/.cache/whisper`. Set qua tham số `download_root`:
```python
import whisper
model = whisper.load_model("base", download_root="D:/Library/ai_cache/whisper")
```

### VietOCR
Cần set path checkpoint thủ công trong config lúc khởi tạo (tùy theo cách agent code viết wrapper), trỏ về `D:/Library/ai_cache/vietocr`.

## 4. Kiểm tra đã áp dụng đúng chưa

Sau khi set xong, chạy thử:
```powershell
python -c "import os; print(os.environ.get('HF_HOME'))"
```
Nếu ra `D:\Library\ai_cache\huggingface` là đã set đúng. Sau đó tải thử 1 model nhỏ để verify file thực sự nằm ở ổ D:
```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-base-patch32')"
```
Kiểm tra `D:\Library\ai_cache\huggingface\hub\` có xuất hiện thư mục model chưa.

## 5. Lưu ý khi đưa cho agent code

- Yêu cầu agent **không hardcode path ổ C** trong bất kỳ đoạn code load model nào — luôn đọc qua biến môi trường hoặc argument `cache_dir`/`download_root` như trên.
- Phạm vi này chỉ áp dụng cho cache checkpoint model — venv, thư mục code, dataset là quyết định riêng của bạn, không cần đưa vào đây.
- Nếu sau này chuyển sang chạy trên Kaggle/Colab, các biến môi trường này **không áp dụng** (môi trường cloud có ổ đĩa riêng, không có khái niệm ổ D) — chỉ cần thiết cho local.
