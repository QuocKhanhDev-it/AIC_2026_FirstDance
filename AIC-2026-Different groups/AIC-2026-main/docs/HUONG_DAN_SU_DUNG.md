# 🏆 AIC 2026 - Hướng Dẫn Tổng Hợp Vận Hành & Setup

Tài liệu tinh gọn và đầy đủ nhất dành cho toàn bộ thành viên trong đội thi.

---

## 📊 1. BẢNG TỔNG HỢP TÍNH NĂNG ĐÃ HOÀN THÀNH

| STT | Tính Năng | Model / Công Nghệ | Nguồn Dữ Liệu | Tốc Độ | Trạng Thái |
|:---:|---|---|---|:---:|:---:|
| **1** | **Multi-Drive Resolver** | Smart In-Memory Hash Cache | Ổ `D:\uploads\Keyframes_L*_extracted` & `C:\` | `< 0.001s` | ✅ **SẴN SÀNG (0 lỗi 404)** |
| **2** | **Textual KIS Search** | SigLIP ViT-SO400M-14 (1152D) | `features.npy` & `faiss_cache.index` | `~0.08s` | ✅ **HOẠT ĐỘNG TỐT** |
| **3** | **TRAKE (Temporal Search)** | 5-stage DFS Sequence Matcher | Chuỗi sự kiện đa mốc thời gian | `~0.15s` | ✅ **HOẠT ĐỘNG TỐT** |
| **4** | **VQA (Multimodal RAG)** | Gemini Flash Latest | Context: FAISS + Ảnh Keyframe + Prompt | `1.5 - 3s` | ✅ **HOẠT ĐỘNG TỐT** |
| **5** | **OCR (Text in Image)** | PaddleOCR v4 (Tiếng Việt) | Tra cứu trực tiếp từ `ocr_transcripts.json` | `< 0.001s` | ✅ **HOẠT ĐỘNG TỐT** |
| **6** | **ASR (Speech to Text)** | OpenAI Whisper | Tra cứu trực tiếp từ `asr_transcripts.json` | `< 0.001s` | ✅ **HOẠT ĐỘNG TỐT** |
| **7** | **OBJECT Detection** | Pre-computed JSON + YOLOv8 | `D:\uploads\objects-aic25-b1_extracted` | `< 0.01s` | ✅ **HOẠT ĐỘNG TỐT** |
| **8** | **AIC Submission Export** | Zip Packager | Giỏ hàng người dùng chọn | Tức thì | ✅ **HOẠT ĐỘNG TỐT** |

---

## 📦 2. HƯỚNG DẪN SETUP MÁY & ĐẶT DỮ LIỆU (CHƯA CHẠY SERVER)

### 🔹 Bước 2.1: Tải Code từ GitHub
1. Vào repository: **`https://github.com/cbbtovl/AIC-2026`**
2. Bấm nút xanh **`Code`** ➔ Chọn **`Download ZIP`** (hoặc dùng lệnh `git clone https://github.com/cbbtovl/AIC-2026.git`).
3. Giải nén ra thư mục dự án (ví dụ: `D:\AIC-2026` hoặc `C:\AIC-2026`).

---

### 🔹 Bước 2.2: Danh Sách File Dữ Liệu & Vị Trí Cần Đặt
*(Tải gói Data Pack từ Google Drive hoặc copy từ USB vào máy)*:

#### A. Các file đặt TRỰC TIẾP tại thư mục gốc dự án (`AIC-2026/`):
*   `faiss_cache.index` *(File vector FAISS 1152D)*
*   `database.db` *(File cơ sở dữ liệu SQLite)*
*   `dataset_manifest.json` *(Bảng ánh xạ thời gian)*
*   `features.npy` *(Vector đặc trưng SigLIP)*

#### B. Các thư mục dữ liệu hình ảnh/âm thanh đặt tại `D:\uploads\` (hoặc `AIC-2026/uploads/`):
*   `Keyframes_L21_extracted/` ... `Keyframes_L30_extracted/` *(Chứa ảnh keyframe)*
*   `video audio/` *(Chứa các file âm thanh `Lxx_Vxxx.mp3`)*
*   `videos L21-L30/` *(Chứa các file video `Lxx_Vxxx.mp4`)*
*   `objects-aic25-b1_extracted/` *(Chứa dữ liệu vật thể JSON)*

---

### 🔹 Bước 2.3: Cài đặt Môi Trường (1 Cú Click)
1. Đảm bảo máy đã cài **Python 3.10 hoặc 3.11** (Nhớ tick chọn ô `Add Python to PATH` khi cài đặt).
2. Vào thư mục dự án, **nhấp đúp chuột vào file `setup_environment.bat`** (script sẽ tự tạo môi trường ảo `venv` và cài toàn bộ thư viện cần thiết).

---

## 🛠️ 3. HƯỚNG DẪN CHẠY THỦ CÔNG OFFLINE (VQA, ASR, OCR TRƯỚC NGÀY THI)

### 🔹 A. Hướng Dẫn Cấu Hình VQA (Lấy API Key Gemini Miễn Phí):
Tính năng VQA dùng mô hình **Gemini Flash** để hỏi đáp suy luận video. Để kích hoạt VQA, mỗi thành viên làm theo 3 bước sau:
1. **Lấy API Key miễn phí**:
   * Truy cập trang: [**`https://aistudio.google.com/app/apikey`**](https://aistudio.google.com/app/apikey) (đăng nhập bằng tài khoản Google bất kỳ).
   * Bấm nút **`Create API Key`** ➔ Chọn **`Create API key in new project`** ➔ Copy dãy mã API Key (dạng `AIzaSy...` hoặc `AQ...`).
2. **Dán Key vào file `.env`**:
   * Mở file `.env` trong thư mục dự án (hoặc đổi tên file `.env.example` thành `.env`).
   * Điền API Key vừa copy vào:
     ```env
     GEMINI_API_KEY=DÁN_MÃ_API_KEY_CỦA_BẠN_VÀO_ĐÂY
     GEMINI_MODEL=gemini-flash-latest
     ```
3. **Kiểm tra nhanh VQA xem đã hoạt động chưa**:
   Mở Terminal và chạy lệnh test thử:
   ```powershell
   .\venv\Scripts\python.exe -c "import sys; sys.path.insert(0, '.'); from services.vqa_service import interact_kisc; print(interact_kisc('a person walking'))"
   ```
   *(Nếu trả về câu trả lời phân tích là VQA đã sẵn sàng 100%)*.

---

### 🔹 B. Chạy Batch OCR (Trích xuất Chữ trên Ảnh):
Để khi thi đấu bấm vào ảnh hiện chữ ngay mà không phải chờ:
```powershell
# Quét toàn bộ ảnh keyframe và lưu kết quả vào ocr_transcripts.json:
.\venv\Scripts\python.exe scripts/batch_ocr_extract.py

# Hoặc quét thử 5 video đầu tiên:
.\venv\Scripts\python.exe scripts/batch_ocr_extract.py --limit_videos 5
```

---

### 🔹 C. Chạy Batch ASR (Trích xuất Lời thoại Âm thanh):
Để khi thi đấu bấm vào ảnh hiện phụ đề lời thoại ngay lập tức:
```powershell
# Quét toàn bộ file audio MP3 với tốc độ siêu nhanh (mô hình tiny ~5s/video):
.\venv\Scripts\python.exe scripts/batch_asr_transcribe.py --model tiny

# Hoặc chạy độ chính xác cao hơn (mô hình base):
.\venv\Scripts\python.exe scripts/batch_asr_transcribe.py --model base
```
*(Cả 2 script OCR và ASR đều tự lưu checkpoint sau mỗi video: Bạn có thể tắt máy bất cứ lúc nào, lần sau chạy tiếp nó sẽ tự quét những video còn lại mà không làm lại từ đầu).*

---

## 🚀 4. HƯỚNG DẪN KHỞI ĐỘNG SERVER

Khi đi thi, việc bật hệ thống chỉ cần chọn **1 trong 2 cách** sau:

### 🌟 Cách 1: 1-Click Nhấp đúp chuột *(Nhanh nhất)*
Vào thư mục dự án và **nhấp đúp chuột vào file `start_server.bat`**:
*   Server sẽ tự động khởi động trên cổng 8000.
*   Trình duyệt web sẽ tự động mở lên tại địa chỉ: **`http://127.0.0.1:8000`**.

### 💻 Cách 2: Chạy bằng Terminal
Mở PowerShell tại thư mục dự án và gõ:
```powershell
.\venv\Scripts\python.exe server.py
```
Sau đó mở trình duyệt (Chrome/Edge) và truy cập: **`http://127.0.0.1:8000`**.  
*(Nếu cho đồng đội dùng chung mạng Wi-Fi/LAN, đồng đội chỉ cần mở Chrome gõ `http://IP_MÁY_BẠN:8000`)*.
