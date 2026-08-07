"""
04_smoke_vlm.py — Bài kiểm tra 30 giây: model có THẬT SỰ nhìn thấy ảnh không?

Phải chạy cái này TRƯỚC mọi so sánh model. Lý do: một model thuần văn bản
(hoặc một model bị gọi sai cách) vẫn trả lời trôi chảy và vẫn ra một con số
"độ đúng" trông hợp lệ — nhưng nó đang đoán mò từ câu hỏi, không nhìn hình.
Con số đó vô nghĩa và không có gì trong harness phát hiện được.

Cách kiểm: dùng chính `objects.parquet` làm đáp án. Chọn keyframe có vật thể
HIẾM và độ tin cậy cao (Dog 1.00, Boat 0.98...), đưa ảnh cho model, hỏi thấy
gì. Model nhìn được thì phải nhắc tới vật thể đó.

Chạy:
    set GOOGLE_API_KEY=...
    python scripts/04_smoke_vlm.py --model gemini-3.1-flash-lite
    python scripts/04_smoke_vlm.py --model gemma-4-31b-it --n 4

Yêu cầu: biến môi trường GOOGLE_API_KEY. Không cần thư viện ngoài.
"""

import argparse, base64, json, os, re, sys, time, urllib.request
from pathlib import Path

import pandas as pd

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

# Vật thể hiếm, dễ nhận ra bằng mắt -> model nhắc tới thì chắc chắn nó nhìn thấy.
# Nhãn OpenImages là tiếng Anh, model trả lời tiếng Việt -> phải có bảng đối chiếu.
# (Bản đầu tiên của script này đi tìm chữ "Dog" trong câu trả lời tiếng Việt và
#  báo 0/4 trong khi model mô tả đúng cả 4 ảnh. Đúng loại thước đo hỏng mà chính
#  script này sinh ra để phát hiện.)
NHAN_HIEM = {
    "Dog":           ["chó", "cún"],
    "Cat":           ["mèo"],
    "Horse":         ["ngựa"],
    "Boat":          ["thuyền", "tàu", "ghe", "xuồng", "ca nô", "canô"],
    "Bicycle":       ["xe đạp"],
    "Motorcycle":    ["xe máy", "mô tô", "xe gắn máy"],
    "Airplane":      ["máy bay", "phi cơ"],
    "Train":         ["tàu hỏa", "tàu lửa", "đoàn tàu", "xe lửa", "tàu khách"],
    "Guitar":        ["ghi ta", "guitar", "ghita"],
    "Piano":         ["piano", "dương cầm"],
    "Cake":          ["bánh kem", "bánh ngọt", "bánh sinh nhật"],
    "Pizza":         ["pizza"],
    "Traffic light": ["đèn giao thông", "đèn tín hiệu", "đèn đỏ"],
}

CAU_HOI = ("Mô tả ngắn gọn những gì bạn nhìn thấy trong ảnh này bằng tiếng Việt. "
           "Liệt kê các vật thể chính. Tối đa 2 câu.")


def goi_model(model: str, key: str, anh: bytes, cau_hoi: str, timeout=60) -> tuple:
    """Trả về (text, giây). Ném exception kèm mã lỗi nếu hỏng."""
    body = {"contents": [{"parts": [
        {"inline_data": {"mime_type": "image/jpeg",
                         "data": base64.b64encode(anh).decode()}},
        {"text": cau_hoi}]}]}
    req = urllib.request.Request(
        API.format(model=model, key=key),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    giay = time.time() - t0
    try:
        return d["candidates"][0]["content"]["parts"][0]["text"].strip(), giay
    except (KeyError, IndexError):
        # bị chặn nội dung, hoặc trả về rỗng
        raise RuntimeError(f"khong co text trong phan hoi: {json.dumps(d)[:200]}")


def chon_mau(index_dir: Path, n: int) -> pd.DataFrame:
    """Keyframe có vật thể hiếm, độ tin cậy cao, VÀ có sẵn ảnh trên máy."""
    m = pd.read_parquet(index_dir / "master.parquet")
    o = pd.read_parquet(index_dir / "objects.parquet")
    m = m[m.kf_path.notna()]
    if m.empty:
        raise SystemExit("Không có keyframe nào có ảnh trên máy này. "
                         "Cần tải gói Keyframes của ít nhất một nhóm L.")
    sub = o[(o.label.isin(NHAN_HIEM.keys())) & (o.score >= 0.7) & (o.row_id.isin(m.row_id))]
    if sub.empty:
        raise SystemExit("Không tìm thấy keyframe nào có vật thể hiếm đủ tin cậy.")
    pick = sub.sort_values("score", ascending=False).drop_duplicates("label").head(n)
    return m[m.row_id.isin(pick.row_id)].merge(
        pick[["row_id", "label"]], on="row_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--index", default=Path("./index"), type=Path)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()

    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("Chưa đặt GOOGLE_API_KEY. Xem docs/01_cai_dat.md")

    mau = chon_mau(a.index, a.n)
    print(f"Model: {a.model}   |   {len(mau)} ảnh có vật thể hiếm\n")

    thay = 0
    for r in mau.itertuples(index=False):
        try:
            text, giay = goi_model(a.model, key, Path(r.kf_path).read_bytes(), CAU_HOI)
        except Exception as e:
            print(f"  {r.label:<14} LỖI: {str(e)[:90]}")
            continue
        # model có nhắc tới vật thể đó không — chấp nhận cả tên tiếng Anh lẫn
        # mọi cách gọi tiếng Việt trong NHAN_HIEM
        tu_khoa = [r.label] + NHAN_HIEM.get(r.label, [])
        trung = any(re.search(re.escape(t_), text, re.I) for t_ in tu_khoa)
        thay += trung
        print(f"  {r.label:<14} {'THẤY' if trung else 'khong nhac toi':<14} "
              f"{giay:5.1f}s  | {text[:88]}")

    print(f"\n{'='*64}")
    print(f"  Nhắc đúng vật thể: {thay}/{len(mau)}")
    if thay >= max(1, len(mau) - 1):
        print(f"  ✅ {a.model} NHÌN ĐƯỢC ảnh. Đưa vào harness so sánh được.")
    elif thay == 0:
        print(f"  ❌ {a.model} KHÔNG nhìn thấy ảnh — hoặc là model thuần văn bản,")
        print("     hoặc đang bị gọi sai cách. ĐỪNG đưa vào harness.")
    else:
        print("  ⚠️  Kết quả lẫn lộn. Tăng --n rồi xem lại từng câu trả lời.")
    print("=" * 64)


if __name__ == "__main__":
    main()
