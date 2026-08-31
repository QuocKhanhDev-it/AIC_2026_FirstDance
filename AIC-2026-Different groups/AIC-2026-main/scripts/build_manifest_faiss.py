"""
Script nạp siêu tốc 177,321 Vector SigLIP 1152D từ features.npy + dataset_manifest.json vào SQLite & FAISS
"""
import os
import sys
import json
import time
import shutil
import pickle
import numpy as np
import faiss
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "database.db"
MANIFEST_PATH = BASE_DIR / "dataset_manifest.json"
FEATURES_PATH = BASE_DIR / "uploads" / "features.npy"
BACKUP_DIR = BASE_DIR / "backup_512d"

CACHE_INDEX_PATH = BASE_DIR / "faiss_cache.index"
CACHE_METADATA_PATH = BASE_DIR / "faiss_cache.pkl"

def backup_old_database():
    """Sao lưu database và cache cũ 512D"""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        try:
            shutil.copy2(str(DB_PATH), str(BACKUP_DIR / "database.db"))
            print(f"[Backup] Đã backup database.db cũ sang {BACKUP_DIR.name}/")
        except Exception as e:
            print(f"[Backup] Note: {e}")
    if CACHE_INDEX_PATH.exists():
        try:
            shutil.copy2(str(CACHE_INDEX_PATH), str(BACKUP_DIR / "faiss_cache.index"))
            CACHE_INDEX_PATH.unlink(missing_ok=True)
        except Exception:
            pass
    if CACHE_METADATA_PATH.exists():
        try:
            shutil.copy2(str(CACHE_METADATA_PATH), str(BACKUP_DIR / "faiss_cache.pkl"))
            CACHE_METADATA_PATH.unlink(missing_ok=True)
        except Exception:
            pass
    print("[Backup] Đã dọn dẹp FAISS cache cũ.")

def build_index():
    print("[Index] BẮT ĐẦU XÂY DỰNG INDEX SIGLIP 1152D CHO AIC 2026...")
    started_total = time.perf_counter()
    
    if not FEATURES_PATH.is_file():
        print(f"[Error] Không tìm thấy file: {FEATURES_PATH}")
        return
        
    if not MANIFEST_PATH.is_file():
        print(f"[Error] Không tìm thấy file: {MANIFEST_PATH}")
        return

    # 1. Backup và reset cache
    backup_old_database()
    
    # 2. Đọc features.npy
    print(f"[Index] Đang nạp features.npy từ {FEATURES_PATH}...")
    t0 = time.perf_counter()
    features = np.load(str(FEATURES_PATH))
    print(f"[Index] Đã nạp features.npy: shape = {features.shape}, dtype = {features.dtype} trong {time.perf_counter() - t0:.2f}s")
    
    # 3. Đọc manifest
    print(f"[Index] Đang đọc {MANIFEST_PATH}...")
    t0 = time.perf_counter()
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    records = manifest_data.get("records", [])
    print(f"[Index] Đã đọc manifest: {len(records):,} records trong {time.perf_counter() - t0:.2f}s")

    num_items = min(len(features), len(records))
    print(f"[Index] Tổng số mục cần lập chỉ mục: {num_items:,}")

    # 4. Tạo FAISS Index 1152D
    print("[Index] Đang xây dựng FAISS Index (1152D) & chuẩn hóa L2...")
    t0 = time.perf_counter()
    vec_matrix = np.ascontiguousarray(features[:num_items], dtype=np.float32)
    faiss.normalize_L2(vec_matrix)
    
    index = faiss.IndexFlatIP(1152)
    index.add(vec_matrix)
    
    faiss.write_index(index, str(CACHE_INDEX_PATH))
    print(f"[Index] Đã tạo FAISS index ({index.ntotal:,} vectors) và lưu vào {CACHE_INDEX_PATH.name} trong {time.perf_counter() - t0:.2f}s")

    # 5. Khởi tạo SQLite và nạp bảng items
    import sqlite3
    print("[Index] Đang nạp metadata vào SQLite database.db...")
    t0 = time.perf_counter()
    
    # Khởi tạo bảng mới
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    cursor = conn.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS items")
    cursor.execute("""
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            filetype TEXT NOT NULL,
            video_id TEXT,
            ordinal INTEGER,
            frame_idx INTEGER,
            pts_time REAL,
            timestamp_ms TEXT,
            extracted_text TEXT,
            description TEXT,
            embedding TEXT,
            embedding_type TEXT DEFAULT 'clip_visual',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    db_rows = []
    metadata_map = {}
    id_map = []

    for i in range(num_items):
        rec = records[i]
        db_id = i + 1
        video_id = rec.get("video_id", "")
        ordinal = rec.get("ordinal", i + 1)
        frame_idx = rec.get("frame_idx", 0)
        pts_time = float(rec.get("pts_time", 0.0))
        
        filename = f"{video_id}_{int(ordinal):03d}.jpg" if ordinal is not None else f"{video_id}.jpg"
        rel_img_path = f"{video_id}/{int(ordinal):03d}.webp"
        
        mm = int(pts_time // 60)
        ss = int(pts_time % 60)
        ms = int((pts_time - int(pts_time)) * 100)
        timestamp_ms = f"{mm:02d}:{ss:02d}.{ms:02d}"
        
        db_rows.append((
            db_id, filename, rel_img_path, "image",
            video_id, ordinal, frame_idx, pts_time, timestamp_ms,
            "", f"Keyframe {filename} at {pts_time:.2f}s",
            None, "clip_visual"
        ))
        
        metadata_map[db_id] = {
            "id": db_id,
            "filename": filename,
            "filepath": rel_img_path,
            "filetype": "image",
            "video_id": video_id,
            "ordinal": ordinal,
            "frame_idx": frame_idx,
            "pts_time": pts_time,
            "timestamp_ms": timestamp_ms,
            "extracted_text": "",
            "description": f"Keyframe {filename}",
            "embedding_type": "clip_visual"
        }
        id_map.append(db_id)

    # Insert batch vào SQLite
    cursor.executemany("""
        INSERT INTO items (
            id, filename, filepath, filetype, video_id, ordinal, frame_idx, pts_time,
            timestamp_ms, extracted_text, description, embedding, embedding_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, db_rows)
    conn.commit()
    conn.close()
    print(f"[Index] Đã nạp {len(db_rows):,} dòng vào SQLite trong {time.perf_counter() - t0:.2f}s")

    # 6. Ghi FAISS Cache metadata
    print("[Index] Đang lưu metadata cache faiss_cache.pkl...")
    t0 = time.perf_counter()
    with open(CACHE_METADATA_PATH, "wb") as f:
        pickle.dump({
            "dim": 1152,
            "id_map": id_map,
            "metadata": metadata_map
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Index] Đã ghi faiss_cache.pkl trong {time.perf_counter() - t0:.2f}s")

    print(f"[Done] HOÀN TẤT NẠP TOÀN BỘ {num_items:,} VECTORS 1152D TRONG {time.perf_counter() - started_total:.2f}s!")

if __name__ == "__main__":
    build_index()
