"""
objects.py — Kênh truy hồi thứ 4: vật thể đã nhận diện sẵn.

Vì sao cần file này: `Person` xuất hiện 161.352 lần, `Wok` 3.161 lần. Đếm thô
thì `Person` át tất cả và bộ lọc gần như vô nghĩa — 5 nhãn người chiếm 49,6%
toàn bộ detection. Trọng số nghịch tần suất (IDF) sửa điều đó: nhãn càng hiếm
càng đáng giá.

Nguyên tắc: CHO ĐIỂM MỀM, TUYỆT ĐỐI KHÔNG LỌC CỨNG. OpenImages không có nhãn
cho mọi khái niệm — xóa cứng là xóa mất đáp án đúng, và không thuật toán xếp
hạng lại nào cứu được.

Dựng bảng IDF (chạy một lần):
    python src/objects.py --index ./index

Dùng trong code:
    from objects import load_channel, object_score
    ch = load_channel('./index')
    diem = object_score(row_ids, ['Wok', 'Chopsticks'], ch)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MIN_SCORE = 0.5      # điểm cân bằng: phủ 89,6% keyframe, 3,4 detection/keyframe


def build_label_idf(objects: pd.DataFrame, n_keyframe: int) -> pd.DataFrame:
    """IDF theo SỐ KEYFRAME chứa nhãn, không phải số detection.

    Một keyframe có 5 người vẫn chỉ tính là 1 cho nhãn Person — nếu tính theo
    số detection thì nhãn hay xuất hiện nhiều lần trong một hình bị phạt oan.
    """
    df = objects.groupby("label")["row_id"].nunique()
    idf = np.log(n_keyframe / (df + 1.0))
    return (pd.DataFrame({"label": df.index, "n_keyframe": df.values, "idf": idf.values})
            .sort_values("idf")
            .reset_index(drop=True))


def load_channel(index_dir="./index", min_score: float = MIN_SCORE) -> dict:
    """Nạp objects + IDF một lần, dùng lại cho mọi truy vấn."""
    d = Path(index_dir)
    o = pd.read_parquet(d / "objects.parquet")
    o = o[o.score >= min_score]
    idf_path = d / "label_idf.parquet"
    if not idf_path.exists():
        raise SystemExit(f"Chưa có {idf_path}. Chạy: python src/objects.py --index {index_dir}")
    idf = pd.read_parquet(idf_path).set_index("label")["idf"]
    return {"objects": o, "idf": idf, "min_score": min_score}


def object_score(row_ids, labels_yeu_cau, channel: dict) -> np.ndarray:
    """Điểm objects cho từng row_id, cùng thứ tự với `row_ids`.

    Điểm = tổng (độ tin cậy detection × IDF của nhãn) trên các nhãn được hỏi.
    Keyframe không có nhãn nào trong danh sách -> 0, KHÔNG bị loại.
    """
    o, idf = channel["objects"], channel["idf"]
    row_ids = np.asarray(row_ids)
    sub = o[o.label.isin(labels_yeu_cau) & o.row_id.isin(row_ids)]
    if sub.empty:
        return np.zeros(len(row_ids), dtype=np.float32)
    w = sub.score.values * sub.label.map(idf).fillna(0.0).values
    tong = pd.Series(w).groupby(sub.row_id.values).sum()
    return tong.reindex(row_ids, fill_value=0.0).values.astype(np.float32)


def dem_nhan(row_ids, label, channel: dict) -> np.ndarray:
    """Số HỘP nhận diện của `label` trong từng keyframe.

    ⚠️ ĐỪNG DÙNG LÀM CÂU TRẢ LỜI CHO CÂU HỎI ĐẾM. Con số này ĐẾM THIẾU
    NGHIÊM TRỌNG — bộ nhận diện chỉ bắt vật nổi bật nhất. Đã kiểm bằng cách
    mở ảnh ra nhìn:

        L21_V001/073.jpg   detector: 1 người    thực tế: >= 4
        L21_V031/086.jpg   detector: 4 người    thực tế: > 20

    Càng đông càng thiếu, mà câu hỏi đếm của đề thi thường rơi vào cảnh đông.
    Câu hỏi đếm phải để VLM nhìn ảnh trả lời.

    Chỉ dùng làm TÍN HIỆU TƯƠNG ĐỐI: keyframe có 5 hộp `Boat` gần như chắc
    chắn nhiều thuyền hơn keyframe có 1 hộp — dù cả hai con số đều thấp hơn
    thực tế. Hữu ích để xếp hạng, vô dụng để trả lời.
    """
    o = channel["objects"]
    row_ids = np.asarray(row_ids)
    sub = o[(o.label == label) & (o.row_id.isin(row_ids))]
    return (sub.groupby("row_id").size()
            .reindex(row_ids, fill_value=0).values.astype(np.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=Path("./index"), type=Path)
    a = ap.parse_args()

    m = pd.read_parquet(a.index / "master.parquet")
    o = pd.read_parquet(a.index / "objects.parquet")
    tab = build_label_idf(o, len(m))
    tab.to_parquet(a.index / "label_idf.parquet", index=False)

    print(f"{len(tab)} nhãn -> {a.index / 'label_idf.parquet'}\n")
    print("10 NHÃN VÔ DỤNG NHẤT khi lọc (IDF thấp = có ở khắp nơi)")
    print(tab.head(10).to_string(index=False))
    print("\n10 NHÃN PHÂN BIỆT MẠNH NHẤT trong nhóm phổ biến (>= 2.000 keyframe)")
    pho_bien = tab[tab.n_keyframe >= 2000]
    print(pho_bien.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
