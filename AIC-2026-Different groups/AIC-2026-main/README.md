# 🏆 AIC 2026 - Multimodal Retrieval Engine

Hệ thống tìm kiếm video đa phương thức tối ưu cho cuộc thi **AI Challenge (AIC) 2026**.

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688)
![SigLIP](https://img.shields.io/badge/Model-SigLIP--1152D-orange)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS--CPU-green)
![Gemini](https://img.shields.io/badge/LLM-Gemini--Flash-purple)

---

## ⚡ Khởi Động Nhanh (Quick Start)

Chỉ cần **nhấp đúp chuột** vào các file `.bat` tương ứng:

1. **Lần đầu cài đặt trên máy mới**: Nhấp đúp chuột vào [`setup_environment.bat`](file:///setup_environment.bat) *(Tự tạo venv & cài toàn bộ thư viện)*.
2. **Khởi động chạy hệ thống**: Nhấp đúp chuột vào [`start_server.bat`](file:///start_server.bat) *(Tự mở server & bật trình duyệt Web tại `http://127.0.0.1:8000`)*.

---

## 📁 Cấu Trúc Thư Mục Chuẩn (Project Structure)

```
multimodal-retrieval/
│
├── 📂 docs/                                # 📚 Tài liệu hướng dẫn sử dụng duy nhất
│   └── HUONG_DAN_SU_DUNG.md                # Hướng dẫn Setup, Chạy Batch ASR/OCR & Server
│
├── 📂 frontend/                            # 🎨 Giao diện Web thi đấu (Dark Theme)
│   └── index.html                          # Grid ảnh, Video Player, ASR/OCR, Giỏ hàng ZIP
│
├── 📂 services/                            # ⚙️ Các dịch vụ AI & Retrieval
│   ├── embedding_service.py                # SigLIP 1152D vector embedding
│   ├── trake_service.py                    # TRAKE 5-stage temporal DFS search
│   ├── vqa_service.py                      # Gemini Multimodal RAG
│   ├── asr_service.py                      # Whisper ASR
│   ├── ocr_service.py                      # PaddleOCR v4
│   ├── object_service.py                   # Object Detection & Bounding Box
│   └── export_service.py                   # Đóng gói ZIP nộp bài cho BTC
│
├── 📂 scripts/                             # 🛠️ Scripts Batch Offline & Indexing
│   ├── batch_ocr_extract.py                # Script chạy Batch OCR độc lập
│   ├── batch_asr_transcribe.py             # Script chạy Batch Whisper độc lập
│   └── build_manifest_faiss.py             # Script nạp vector SigLIP vào FAISS
│
├── 📂 exports/                             # 📥 Thư mục lưu các file ZIP nộp bài
│
├── 🌐 server.py                            # FastAPI Web Server chính (Port 8000)
├── 🔍 indexer.py                           # KIS Search Engine
├── 🗄️ database.py                          # Quản lý FAISS Vector Store & SQLite
├── ⚙️ config.py                            # Cấu hình hệ thống & Quét đa ổ đĩa D:/, C:/
│
├── 📜 requirements.txt                     # Danh sách thư viện Python
├── 🔐 .env.example                         # File mẫu cấu hình Gemini API Key
├── ⚡ setup_environment.bat                # Script 1-Click tự động cài môi trường
└── 🚀 start_server.bat                     # Script 1-Click khởi động server & mở web
```

---

## 📖 Tài Liệu Hướng Dẫn Chi Tiết (Full Guide)

👉 **[Đọc Hướng Dẫn Vận Hành & Setup Chi Tiết (HUONG_DAN_SU_DUNG.md)](docs/HUONG_DAN_SU_DUNG.md)**:
1. Bảng tổng hợp tính năng đã hoàn thành.
2. Hướng dẫn setup máy mới & vị trí đặt dữ liệu.
3. Hướng dẫn chạy Batch ASR & OCR độc lập ở nhà.
4. Hướng dẫn khởi động Server khi thi đấu.
