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
import re
import unicodedata
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


def nap_bang_nhan(f=None) -> pd.DataFrame:
    """Bảng ánh xạ tiếng Việt -> nhãn OpenImages (`dev/label_vi_en.csv`).

    Không có bảng này thì kênh objects KHÔNG BAO GIỜ kích hoạt được từ một
    truy vấn tiếng Việt — `object_score()` nhận nhãn tiếng Anh, mà đề thi ra
    tiếng Việt.

    Chỉ dịch 100 nhãn phổ biến nhất: đo trên `objects.parquet` ngưỡng 0,5
    (473 nhãn, 597.357 detection) thì **top 100 đã phủ 96,8%**.
    """
    f = Path(f) if f else Path(__file__).resolve().parent.parent / "dev" / "label_vi_en.csv"
    if not f.exists():
        raise SystemExit(f"Chưa có {f} — xem PHẦN D1.6 của kế hoạch.")
    d = pd.read_csv(f).fillna("")
    d["tu"] = d.dong_nghia.map(
        lambda s: [x.strip().lower() for x in s.split(",") if x.strip()])
    d["tu_kd"] = d.tu.map(lambda ts: [bo_dau(t) for t in ts])
    return d


def bo_dau(s: str) -> str:
    """Chuẩn hóa: thường hóa, bỏ dấu thanh và dấu mũ, `đ` -> `d`.

    Vì sao cần: người gõ nhanh trong phiên thi thường bỏ dấu, và **chính đề
    thi cũng có thể không dấu** — ví dụ trong bài báo AIC'25 là truy vấn
    `"giai phong khi hidro"`. Không chuẩn hóa thì `"ca chua"` không khớp
    `"cà chua"` và cả kênh im lặng trả 0.

    Chuẩn hóa ở ĐÂY chứ không nhồi biến thể không dấu vào CSV: một chỗ sửa,
    phủ mọi từ, và không làm phình bảng lên gấp đôi.

    `đ` phải xử riêng — nó là ký tự độc lập trong Unicode, không phải `d` cộng
    dấu, nên `NFD` không tách ra được.
    """
    s = unicodedata.normalize("NFD", s.strip().lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", s).replace("đ", "d")


def nhan_tu_truy_van(cau: str, bang: pd.DataFrame, keo_cha: bool = True) -> list[str]:
    """Truy vấn tiếng Việt -> danh sách nhãn OpenImages cho `object_score`.

    Ba quy tắc:

    1. **Khớp theo CỤM TỪ, không theo chuỗi con.** Chuỗi con rất nguy hiểm với
       tiếng Việt: "cá" nằm trong "cá nhân", "bàn" nằm trong "bàn bạc".
    2. **Cụm dài thắng cụm ngắn** (maximal munch). "bàn tay" phải ra
       `Human hand`, KHÔNG được đồng thời kéo `Table` vì chữ "bàn". Đã đo:
       không có quy tắc này thì "bàn tay người cầm dao" lôi cả `Table` vào.
    3. **Bỏ dấu CHỈ KHI truy vấn vốn đã không dấu.**

    Quy tắc 3 là chỗ dễ làm sai nhất. Bỏ dấu vô điều kiện nghe có vẻ tiện hơn,
    nhưng tiếng Việt dùng dấu để phân biệt từ, nên bỏ dấu là **gộp nhầm những
    từ khác hẳn nhau**. Đã đo, cả ba đều là lỗi thật:

        "cái ấm đun nước"        -> Cabbage   ("cai" <- "cải", không phải "cái")
        "bàn tay người cầm dao"  -> Orange    ("cam" <- "cầm", không phải quả cam)
        "ô tô màu đỏ"            -> Boat      ("do"  <- "đò",  không phải "đỏ")

    Nên: truy vấn CÓ dấu thì khớp có dấu (chính xác); truy vấn KHÔNG dấu thì
    mới hạ xuống khớp không dấu. Người gõ nhanh trong phiên thi vẫn được phục
    vụ, mà người gõ đúng chính tả không bị phạt. Đề thi cũng có thể ra không
    dấu — bài báo AIC'25 có truy vấn `"giai phong khi hidro"`.

    `keo_cha` kéo theo nhãn cha. Cần vì thứ bậc OpenImages KHÔNG tự gộp:
    `Car`, `Land vehicle`, `Vehicle` là ba nhãn riêng biệt. Quy tắc 2 nuốt mất
    cụm ngắn, nhưng `cha` bù lại đúng phần đáng bù: "xe máy" -> `Motorcycle`
    -> `Land vehicle` -> `Vehicle`.

        >>> nhan_tu_truy_van("người phụ nữ thái cà chua bên chảo", bang)
        ['Food', 'Frying pan', 'Kitchen utensil', 'Person', 'Tomato',
         'Vegetable', 'Woman', 'Wok']
    """
    khong_dau = bo_dau(cau) == cau.strip().lower()
    cot = "tu_kd" if khong_dau else "tu"
    tu = [t for t in re.split(r"[^\w]+", bo_dau(cau) if khong_dau else cau.lower()) if t]

    tra = {}
    for r in bang.itertuples():
        for t in getattr(r, cot):
            tra.setdefault(t, set()).add(r.nhan_en)

    # Thu MỌI cụm khớp được, rồi mới bỏ cụm nằm LỌT TRONG cụm khác.
    #
    # ⚠️ Đừng "ăn" token theo kiểu tham lam trái-sang-phải. Đã đo lỗi thật:
    # "chiếc ô tô màu đỏ" — "chiếc ô" (Umbrella) khớp trước ở vị trí 0-1, ăn
    # mất vị trí 1, chặn luôn "ô tô" (Car) ở vị trí 1-2. Kết quả ra
    # {Bowl, Umbrella} mà không có Car.
    #
    # Chỉ bỏ cụm nằm HẲN trong cụm khác ("bàn" trong "bàn tay"). Hai cụm chồng
    # lấn một phần thì GIỮ CẢ HAI — đó là nhập nhằng thật của tiếng Việt, và
    # `object_score` cho điểm mềm nên giữ cả hai an toàn hơn là đoán bừa.
    khop = []
    for n in range(4, 0, -1):
        for i in range(len(tu) - n + 1):
            nhan = tra.get(" ".join(tu[i:i + n]))
            if nhan:
                khop.append((i, i + n, nhan))

    ra = {x for i, j, nhan in khop
          if not any(a <= i and j <= b and (b - a) > (j - i) for a, b, _ in khop)
          for x in nhan}

    if keo_cha:
        cha = bang.set_index("nhan_en")["cha"].to_dict()
        while True:
            them = {cha[x] for x in ra if cha.get(x)} - ra
            if not them:
                break
            ra |= them
    return sorted(ra)


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
