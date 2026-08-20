"""
gop_ocr_asr.py — Module hợp nhất kết quả OCR và ASR thành 1 DataFrame chuẩn cho BM25 Kênh 3

Output: ocr_asr.parquet chứa các cột:
- row_id (int64): ID khung ảnh trong master.parquet
- video_id (str): ID video
- kf_n (int): Số thứ tự keyframe
- ocr_text (str): Chuỗi chữ từ OCR
- asr_text (str): Chuỗi lời nói từ ASR
- text (str): Chuỗi kết hợp ocr_text + asr_text (dùng trực tiếp cho KenhVanBan.tu_bang_khung)
"""

import logging
from pathlib import Path
import pandas as pd

from pipeline_OCR_ASR.config import OCR_PARQUET_PATH, ASR_PARQUET_PATH, OCR_ASR_PARQUET_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def gop_ocr_va_asr(df_master: pd.DataFrame, df_ocr: pd.DataFrame, df_asr: pd.DataFrame, output_path: Path = OCR_ASR_PARQUET_PATH) -> pd.DataFrame:
    """Gộp df_ocr và df_asr theo row_id."""
    df_merged = df_master[["row_id", "video_id", "kf_n"]].copy()

    if df_ocr is not None and not df_ocr.empty and "ocr_text" in df_ocr.columns:
        df_merged = df_merged.merge(df_ocr[["row_id", "ocr_text"]], on="row_id", how="left")
    else:
        df_merged["ocr_text"] = ""

    if df_asr is not None and not df_asr.empty and "asr_text" in df_asr.columns:
        df_merged = df_merged.merge(df_asr[["row_id", "asr_text"]], on="row_id", how="left")
    else:
        df_merged["asr_text"] = ""

    df_merged["ocr_text"] = df_merged["ocr_text"].fillna("").astype(str)
    df_merged["asr_text"] = df_merged["asr_text"].fillna("").astype(str)

    # Ghép văn bản: ocr_text + " . " + asr_text
    def hop_nhat_text(row):
        o = row["ocr_text"].strip()
        a = row["asr_text"].strip()
        if o and a:
            return f"{o} . {a}"
        return o or a

    df_merged["text"] = df_merged.apply(hop_nhat_text, axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_parquet(output_path, index=False)
    logging.info("Đã ghi file kết hợp OCR + ASR ra: %s (%d dòng)", output_path, len(df_merged))
    return df_merged
