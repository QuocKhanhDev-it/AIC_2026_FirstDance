"""
loc_cung_token_hiem.py — Chế độ Lọc cứng cho Token hiếm (Hard Filtering / WHERE ocr_text LIKE '%...')

Mục đích:
Xử lý các token hiếm trong câu truy vấn KIS/Q&A (như biển số xe, mã số, chữ số, tên viết tắt, mã hiệu)
mà BM25 thông thường có thể bị giảm điểm do chia độ dài văn bản (length normalization).

Quy tắc lọc an toàn:
- KHÔNG biến từ tiếng Việt thông thường (trường, truyền, phương) thành token hiếm.
- Chỉ kích hoạt khi có: chữ số (79A-12345), từ viết tắt in hoa (HTV, UNESCO), mã hiệu có dấu gạch ngang (A1-02).
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
    """Phát hiện token hiếm thực sự trong câu truy vấn.
    
    Các dạng token hiếm hợp lệ:
    1. Chứa chữ số: '79A-12345', '2024', 'vtv1', '100k', '50%'
    2. Có gạch nối/mã hiệu: 'A1-02', 'ISO-9001', 'TP.HCM'
    3. Từ viết tắt toàn bộ in hoa (>= 2 ký tự): 'HTV', 'VTV', 'CSGT', 'UNESCO'
    4. Từ bắt đầu bằng chữ hoa nếu câu truy vấn ở dạng văn bản thường (Mixed case).
    """
    tokens = query.strip().split()
    token_hiem = []
    is_all_upper = query.isupper()

    for t in tokens:
        t_clean = t.strip(".,?!:;\"'()[]{}")
        if not t_clean or len(t_clean) < 2:
            continue

        # 1. Chứa số (biển số xe, năm, số hiệu, tiền tệ, kênh)
        if any(c.isdigit() for c in t_clean):
            token_hiem.append(t_clean)
        # 2. Có ký tự nối đặc biệt
        elif "-" in t_clean or "_" in t_clean or "." in t_clean:
            token_hiem.append(t_clean)
        # 3. Viết tắt in hoa toàn bộ (HTV, VTV, CSGT, WHO)
        elif t_clean.isupper() and len(t_clean) >= 2:
            token_hiem.append(t_clean)
        # 4. Tên riêng (viết hoa chữ đầu) khi cả câu không phải viết hoa hết
        elif not is_all_upper and t_clean[0].isupper() and len(t_clean) >= 2:
            token_hiem.append(t_clean)

    # Loại bỏ trùng lặp giữ nguyên thứ tự
    seen = set()
    result = []
    for x in token_hiem:
        if x.lower() not in seen:
            seen.add(x.lower())
            result.append(x)
    return result


class LocCungTokenHiem:
    def __init__(self, df_ocr_asr: pd.DataFrame, master: pd.DataFrame):
        self.df_ocr_asr = df_ocr_asr.copy()
        self.master = master

        if "text" in self.df_ocr_asr.columns:
            self.df_ocr_asr["text_no_accent"] = self.df_ocr_asr["text"].astype(str).apply(bo_dau)
            self.df_ocr_asr["text_raw"] = self.df_ocr_asr["text"].astype(str).str.lower()
        else:
            self.df_ocr_asr["text_no_accent"] = ""
            self.df_ocr_asr["text_raw"] = ""

    def tim_kiem_loc_cung(self, query: str, top_k: int = 100, exact_bonus: float = 100.0) -> List[Candidate]:
        """Thực hiện truy vấn lọc cứng. Trả về danh sách Candidate có chứa token hiếm."""
        token_list = phat_hien_token_hiem(query)
        if not token_list:
            return []

        # Tạo mask lọc cứng
        combined_mask = np.zeros(len(self.df_ocr_asr), dtype=bool)

        for tok in token_list:
            tok_lower = tok.lower()
            tok_no_acc = bo_dau(tok)

            mask_raw = self.df_ocr_asr["text_raw"].str.contains(tok_lower, regex=False)
            mask_no_acc = self.df_ocr_asr["text_no_accent"].str.contains(tok_no_acc, regex=False)
            combined_mask = combined_mask | mask_raw.to_numpy() | mask_no_acc.to_numpy()

        df_matched = self.df_ocr_asr[combined_mask]
        if df_matched.empty:
            return []

        candidates = []
        for row in df_matched.itertuples():
            rid = int(row.row_id)
            m_row = self.master.iloc[rid]

            hit_count = 0
            text_r = str(row.text_raw)
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
                    "matched_text": str(row.text)
                }
            ))

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

