import os
import json
from pathlib import Path
from typing import List

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent

# Thư mục lưu trữ tệp tải lên
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Đường dẫn cơ sở dữ liệu SQLite và File cấu hình JSON
DB_PATH = BASE_DIR / "database.db"
CONFIG_FILE = BASE_DIR / "config.json"

# --- Các cấu hình mặc định ---
DEFAULT_PROVIDER = "local"  
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
DEFAULT_TEXT_EMBEDDING = "ViT-SO400M-14-SigLIP-384"
DEFAULT_VISION_MODEL = "ViT-SO400M-14-SigLIP-384"
DEFAULT_VECTOR_DIM = 1152
DEFAULT_LOCAL_WHISPER = "small"
DEFAULT_GEMINI_API_KEY = ""

# Danh sách các thư mục gốc để tự động dò tìm dữ liệu trên các ổ đĩa
DEFAULT_DATASET_ROOTS = [
    Path("D:/uploads"),
    Path("D:/"),
    BASE_DIR / "uploads",
    BASE_DIR / "data",
    BASE_DIR
]


class ConfigManager:
    def __init__(self):
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY") or DEFAULT_GEMINI_API_KEY
        self.gemini_model = os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self.provider = DEFAULT_PROVIDER
        self.text_embedding_model = DEFAULT_TEXT_EMBEDDING
        self.vision_model = DEFAULT_VISION_MODEL
        self.vector_dim = DEFAULT_VECTOR_DIM
        self.local_whisper_model = DEFAULT_LOCAL_WHISPER
        self.dataset_roots: List[str] = [str(p) for p in DEFAULT_DATASET_ROOTS]
        
        self.load_config()

    def load_config(self):
        """Đọc cấu hình từ file config.json cục bộ"""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.gemini_api_key = data.get("gemini_api_key", self.gemini_api_key)
                    self.gemini_model = data.get("gemini_model", self.gemini_model)
                    self.provider = data.get("provider", self.provider)
                    self.text_embedding_model = data.get("text_embedding_model", self.text_embedding_model)
                    self.vision_model = data.get("vision_model", self.vision_model)
                    self.local_whisper_model = data.get("local_whisper_model", self.local_whisper_model)
                    if "dataset_roots" in data and isinstance(data["dataset_roots"], list):
                        self.dataset_roots = data["dataset_roots"]
            except Exception as e:
                print(f"[Config] Không thể đọc config.json: {e}")

    def save_config(self):
        """Ghi cấu hình hiện tại xuống file config.json"""
        try:
            data = {
                "gemini_api_key": self.gemini_api_key,
                "gemini_model": self.gemini_model,
                "provider": self.provider,
                "text_embedding_model": self.text_embedding_model,
                "vision_model": self.vision_model,
                "vector_dim": self.vector_dim,
                "local_whisper_model": self.local_whisper_model,
                "dataset_roots": self.dataset_roots
            }
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print("[Config] Đã lưu cấu hình mới xuống config.json.")
        except Exception as e:
            print(f"[Config] Lỗi khi ghi file config.json: {e}")

    def get_search_roots(self) -> List[Path]:
        """Lấy danh sách các Path hợp lệ đang tồn tại trên máy"""
        roots = []
        for r_str in self.dataset_roots:
            p = Path(r_str)
            if p.exists():
                roots.append(p)
        for def_p in DEFAULT_DATASET_ROOTS:
            if def_p.exists() and def_p not in roots:
                roots.append(def_p)
        return roots


config = ConfigManager()