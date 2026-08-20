"""
loc_cung_token_hiem.py — Chế độ Lọc cứng cho Token hiếm (Hard Filtering / WHERE ocr_text LIKE '%...')

Mục đích:
Xử lý các token hiếm trong câu truy vấn KIS/Q&A (như tên riêng, biển số xe, mã số, chữ số, chữ in hoa ticker)
mà BM25 thông thường có thể bị giảm điểm do chia độ dài văn bản (length normalization) hoặc tần suất từ.

Cách hoạt động:
1. Phát hiện các token hiếm trong câu truy vấn (chứa số, mã hiệu, chữ hoa, hoặc từ ghép đặc biệt).
2. Chạy substring search trên df_ocr_asr (tương tự SQL `WHERE text LIKE '%token%'`).
3. Tăng cường điểm thưởng (bonus score) cho các ứng viên chứa chính xác token hiếm này.
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np

try:
    from src.schema import Candidate
except ImportError:
    from schema import Candidate


def bo_dau(s: str) -> str:
    """Thường hóa và bỏ dấu tiếng Việt."""
    s = unicodedata.normalize("NFD", s.strip().lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("đ", "d").replace("Đ", "d")


def phat_hien_token_hiem(query: str) -> List[str]:
    """Phát hiện token hiếm trong câu truy vấn.
    Token hiếm thường thuộc các dạng:
    - Có chữ số: '79A-12345', '2024', 'vtv1', '100k'
    - Viết hoa/tên riêng: 'Hội An', 'TP.HCM', 'Toyota'
    - Mã hiệu/ký tự đặc biệt: 'A1-02', 'ISO9001'
    - Từ ghép dài hoặc từ có độ dài >= 5
    """
    tokens = query.strip().split()
    token_hiem = []

    for t in tokens:
        # Strip trailing punctuation
        t_clean = t.strip(".,?!:;\"'()[]{}")
        if not t_clean or len(t_clean) < 2:
            continue
        
        # Chứa số
        if any(c.isdigit() for c in t_clean):
            token_hiem.append(t_clean)
        # Viết hoa chữ cái đầu hoặc toàn bộ
        elif t_clean[0].isupper() and len(t_clean) >= 3:
            token_hiem.append(t_clean)
        # Có gạch nối
        elif "-" in t_clean:
            token_hiem.append(t_clean)
        # Độ dài >= 6
        elif len(t_clean) >= 6:
            token_hiem.append(t_clean)

    return list(set(token_hiem))


class LocCungTokenHiem:
    def __init__(self, df_ocr_asr: pd.DataFrame, master: pd.DataFrame):
        self.df_ocr_asr = df_ocr_asr.copy()
        self.master = master
        
        # Chuẩn bị sẵn dạng bỏ dấu để tìm kiếm nhanh
        if "text" in self.df_ocr_asr.columns:
            self.df_ocr_asr["text_no_accent"] = self.df_ocr_asr["text"].astype(str).apply(bo_dau)
            self.df_ocr_asr["text_raw"] = self.df_ocr_asr["text"].astype(str).str.lower()
        else:
            self.df_ocr_asr["text_no_accent"] = ""
            self.df_ocr_asr["text_raw"] = ""

    def tim_kiem_loc_cung(self, query: str, top_k: int = 100, exact_bonus: float = 100.0) -> List[Candidate]:
        """Thực hiện truy vấn lọc cứng.
        
        Trả về danh sách Candidate có chứa token hiếm với bonus score cao.
        """
        token_list = phat_hien_token_hiem(query)
        if not token_list:
            return []

        # Tạo mask lọc cứng
        combined_mask = np.zeros(len(self.df_ocr_asr), dtype=bool)

        for tok in token_list:
            tok_lower = tok.lower()
            tok_no_accent = bo_dau(tok)

            # Substring match trực tiếp trên text_raw hoặc text_no_accent (SQL LIKE '%tok%')
            mask_raw = self.df_ocr_asr["text_raw"].str.contains(tok_lower, regex=False)
            mask_no_acc = self.df_ocr_asr["text_no_accent"].str.contains(tok_no_accent, regex=False)

            combined_mask = combined_mask | mask_raw.to_numpy() | mask_no_acc.to_numpy()

        df_matched = self.df_ocr_asr[combined_mask]
        if df_matched.empty:
            return []

        candidates = []
        for idx, row in df_matched.iterrows():
            rid = int(row["row_id"])
            m_row = self.master.iloc[rid]
            
            # Tính điểm bonus: cơ bản exact_bonus + thưởng cho số token hiếm trùng khớp
            hit_count = 0
            text_r = str(row["text_raw"])
            for tok in token_list:
                if tok.lower() in text_r:
                    hit_count += 1
            
            score = exact_bonus + (hit_count * 10.0)

            candidates.append(Candidate(
                row_id=rid,
                video_id=str(m_row["video_id"]),
                frame_idx=int(m_row["frame_idx"]),
                score=float(score),
                source="loc_cung_ocr_asr",
                meta={
                    "pts_time": float(m_row["pts_time"]),
                    "fps": float(m_row["fps"]),
                    "kf_n": int(m_row["kf_n"]),
                    "title": str(m_row["title"]),
                    "matched_text": str(row["text"])
                }
            ))

        # Sắp xếp điểm giảm dần và cắt top_k
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]
