"""
05_bench_vlm.py — Đo VLM cho Q&A trên DỮ LIỆU THẬT, có đáp án khách quan.

Vấn đề của đợt test đầu: 10 câu/ngôn ngữ nên 50% và 40% chỉ khác nhau ĐÚNG
MỘT CÂU — không phân biệt được model nào hơn. Và `--runs 1` khiến cột đồng
thuận vô nghĩa.

Script này đo BA THỨ KHÔNG CẦN ĐÁP ÁN — và cả ba đều quyết định điểm thi:

  1. TUÂN THỦ FORMAT. Theo thể lệ, `answer` sai định dạng = 0 điểm bất kể
     frame đúng. Model hay thêm chữ thừa ("6 chiếc vàng") là mất trắng.
     Đây có thể là chỉ số quan trọng hơn cả độ đúng.
  2. ĐỒNG THUẬN giữa các lượt. Model trả lời khác nhau mỗi lần thì không
     dùng được cho bài thi, dù độ đúng trung bình cao. Cần `--runs >= 3`.
  3. ĐỘ TRỄ. Ảnh hưởng trực tiếp tới số vòng thử nghiệm chạy được trong tuần.

⚠️ KHÔNG ĐO ĐỘ ĐÚNG Ở ĐÂY. Bản đầu tiên của script lấy `objects.parquet` làm
đáp án cho câu hỏi đếm — và sai. Kiểm chứng bằng cách mở ảnh ra nhìn:

    L21_V001/073.jpg  detector: 1 người   model: 4    thực tế: >= 4
    L21_V031/086.jpg  detector: 4 người   model: 13   thực tế: > 20

Bộ nhận diện ở ngưỡng 0,7 chỉ bắt vật nổi bật nhất, ĐẾM THIẾU NGHIÊM TRỌNG.
Model đúng, "đáp án" sai. Độ đúng thật chỉ đo được trên tập dev do người gán
nhãn — đó là việc của Khánh, không tự động hóa được.

Câu hỏi đếm vẫn giữ, nhưng chỉ dùng làm ĐỀ BÀI để đo format/đồng thuận/độ trễ.
Cột `khop_detector` giữ lại để tham khảo, KHÔNG PHẢI độ đúng.

Chạy:
    set GOOGLE_API_KEY=...
    python scripts/05_bench_vlm.py --models gemini-3.1-flash-lite,gemini-3.5-flash-lite --n 30 --runs 3

Kết quả ghi dần vào index/bench_vlm.csv — ngắt giữa chừng chạy lại là tiếp tục,
không mất phần đã đo (quan trọng với free tier hay bị 429).
"""

import argparse, base64, json, os, re, sys, time, urllib.error, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

# Vật thể đếm được, nhìn rõ, ít bị lẫn. Kèm cách gọi tiếng Việt cho câu hỏi.
DEM_DUOC = {
    "Person": "người", "Man": "người đàn ông", "Woman": "người phụ nữ",
    "Car": "chiếc ô tô", "Bicycle": "chiếc xe đạp", "Motorcycle": "chiếc xe máy",
    "Boat": "chiếc thuyền", "Airplane": "chiếc máy bay", "Horse": "con ngựa",
    "Dog": "con chó", "Bird": "con chim", "Flower": "bông hoa",
    "Bottle": "cái chai", "Chair": "cái ghế", "Wheel": "bánh xe",
    "Tomato": "quả cà chua", "Bowl": "cái bát", "Spoon": "cái thìa",
}

MAU_CAU_HOI = (
    "Nhìn kỹ ảnh và đếm số {vat} có trong ảnh.\n"
    "Trả lời CHỈ bằng một chữ số, không giải thích, không thêm chữ nào khác."
)


def iou(a, b) -> float:
    """box = [y_min, x_min, y_max, x_max], chuẩn hóa 0-1."""
    ay1, ax1, ay2, ax2 = [float(v) for v in a]
    by1, bx1, by2, bx2 = [float(v) for v in b]
    iy1, ix1 = max(ay1, by1), max(ax1, bx1)
    iy2, ix2 = min(ay2, by2), min(ax2, bx2)
    if iy2 <= iy1 or ix2 <= ix1:
        return 0.0
    giao = (iy2 - iy1) * (ix2 - ix1)
    hop = (ay2-ay1)*(ax2-ax1) + (by2-by1)*(bx2-bx1) - giao
    return giao / hop if hop > 0 else 0.0


def dem_sau_gop(boxes, nguong=0.5) -> int:
    """Gộp các hộp chồng nhau — bộ nhận diện hay trả nhiều hộp cho một vật."""
    giu = []
    for b in boxes:
        if all(iou(b, g) <= nguong for g in giu):
            giu.append(b)
    return len(giu)


def tao_cau_hoi(index_dir: Path, n: int, seed: int, min_score=0.7) -> pd.DataFrame:
    m = pd.read_parquet(index_dir / "master.parquet")
    o = pd.read_parquet(index_dir / "objects.parquet")
    m = m[m.kf_path.notna()]
    if m.empty:
        raise SystemExit("Máy này chưa có ảnh keyframe nào. Cần tải gói Keyframes.")

    o = o[(o.score >= min_score) & (o.row_id.isin(m.row_id)) &
          (o.label.isin(DEM_DUOC.keys()))]
    rows = []
    for (rid, lab), g in o.groupby(["row_id", "label"]):
        c = dem_sau_gop(list(g.box.values))
        if 1 <= c <= 5:                      # >5 thì chính người cũng đếm lệch
            rows.append({"row_id": rid, "label": lab, "dap_an": c})
    q = pd.DataFrame(rows)
    if q.empty:
        raise SystemExit("Không sinh được câu hỏi nào. Thử hạ --min-score.")

    # cân bằng: mỗi (nhãn, đáp án) lấy tối đa vài câu, tránh dồn vào 'Person = 1'
    q = (q.sample(frac=1.0, random_state=seed)
           .groupby(["label", "dap_an"], group_keys=False).head(3)
           .sample(frac=1.0, random_state=seed).head(n))
    return q.merge(m[["row_id", "video_id", "kf_name", "kf_path"]], on="row_id")


def hoi(model, key, anh: bytes, cau_hoi: str, temp=0.0, timeout=90, thu_lai=4):
    """Gọi API, tự lùi thời gian khi bị 429. Trả (text, giây) hoặc ném lỗi."""
    body = {"contents": [{"parts": [
        {"inline_data": {"mime_type": "image/jpeg",
                         "data": base64.b64encode(anh).decode()}},
        {"text": cau_hoi}]}],
        "generationConfig": {"temperature": temp, "maxOutputTokens": 2048}}
    data = json.dumps(body).encode()
    cho = 5
    for lan in range(thu_lai):
        req = urllib.request.Request(API.format(model=model, key=key), data=data,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            giay = time.time() - t0
            try:
                return d["candidates"][0]["content"]["parts"][0]["text"].strip(), giay
            except (KeyError, IndexError):
                return "", giay          # bị chặn / rỗng -> tính là sai format
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and lan < thu_lai - 1:
                print(f"      [{e.code}] chờ {cho}s rồi thử lại...", flush=True)
                time.sleep(cho); cho *= 2
                continue
            raise
    raise RuntimeError("het luot thu lai")


def doc_so(text: str):
    """Con số model trả về, hoặc None nếu không đọc được đáng tin.

    KHÔNG lấy con số đầu tiên. Model nhả chuỗi suy luận ("Okay, let's count...
    1. First there's a white car...") thì số đầu tiên là dấu đầu dòng `1.`,
    không phải câu trả lời. Bản đầu của script mắc đúng lỗi này: báo tra_loi=1
    cho cả 10 câu của Gemma, kéo theo cột đồng thuận giả 100%.

    Quy tắc: tin ngay khi ĐÚNG FORMAT (một con số duy nhất); nếu không thì
    chỉ nhìn đoạn CUỐI câu trả lời, nơi kết luận thường nằm.
    """
    text = text.strip()
    chu = {"không":0,"một":1,"hai":2,"ba":3,"bốn":4,"năm":5,"sáu":6,"bảy":7,
           "tám":8,"chín":9,"mười":10}
    if dung_format(text):
        return int(re.search(r"\d+", text).group())
    duoi = text[-60:]
    s = re.findall(r"\d+", duoi)
    if s:
        return int(s[-1])
    for k, v in chu.items():
        if re.search(rf"\b{k}\b", duoi, re.I):
            return v
    return None


def dung_format(text: str) -> bool:
    """Thể lệ: answer ngắn gọn, không kèm chữ định tính. Ở đây: chỉ một con số."""
    return bool(re.fullmatch(r"\W*\d+\W*", text.strip()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="danh sách, cách nhau bằng dấu phẩy")
    ap.add_argument("--index", default=Path("./index"), type=Path)
    ap.add_argument("--n", type=int, default=30, help="số câu hỏi")
    ap.add_argument("--runs", type=int, default=1, help="số lượt lặp để đo đồng thuận")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nghi", type=float, default=0.5, help="giây nghỉ giữa 2 lần gọi")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="temperature. Mặc định 0 cho tác vụ xác định. "
                         "Để 1.0 (mặc định của API) thì đồng thuận tụt mạnh.")
    a = ap.parse_args()

    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("Chưa đặt GOOGLE_API_KEY.")

    q = tao_cau_hoi(a.index, a.n, a.seed)
    models = [s.strip() for s in a.models.split(",") if s.strip()]
    print(f"{len(q)} câu hỏi đếm  ×  {len(models)} model  ×  {a.runs} lượt "
          f"= {len(q)*len(models)*a.runs} lần gọi\n")
    print("Phân bố đáp án:", dict(q.dap_an.value_counts().sort_index()), "\n")

    ket = a.index / "bench_vlm.csv"
    da_co = pd.read_csv(ket) if ket.exists() else pd.DataFrame()
    if da_co.empty:
        xong = set()
    else:
        if "temp" not in da_co:
            da_co["temp"] = 1.0          # các lượt chạy trước khi có cờ --temp
        xong = set(zip(da_co.model, da_co.temp, da_co.row_id, da_co.label, da_co.lan))

    for model in models:
        print(f"===== {model}")
        for lan in range(1, a.runs + 1):
            for i, r in enumerate(q.itertuples(index=False), 1):
                if (model, a.temp, r.row_id, r.label, lan) in xong:
                    continue
                ch = MAU_CAU_HOI.format(vat=DEM_DUOC[r.label])
                try:
                    text, giay = hoi(model, key, Path(r.kf_path).read_bytes(), ch, a.temp)
                except Exception as e:
                    print(f"  DỪNG ({model}, lượt {lan}, câu {i}): {str(e)[:80]}")
                    break
                so = doc_so(text)
                dong = {"model": model, "temp": a.temp, "lan": lan, "row_id": r.row_id,
                        "video_id": r.video_id, "kf_name": r.kf_name,
                        "label": r.label, "detector": r.dap_an, "tra_loi": so,
                        "khop_detector": int(so == r.dap_an),
                        "format": int(dung_format(text)),
                        "giay": round(giay, 2), "raw": text[:120].replace("\n", " ")}
                pd.DataFrame([dong]).to_csv(ket, mode="a", header=not ket.exists(),
                                            index=False, encoding="utf-8-sig")
                if i % 10 == 0 or i == len(q):
                    print(f"  lượt {lan}: {i}/{len(q)}", flush=True)
                time.sleep(a.nghi)
        print()

    bao_cao(ket)


def bao_cao(ket: Path):
    if not ket.exists():
        return
    d = pd.read_csv(ket)
    if "temp" not in d:
        d["temp"] = 1.0
    print("=" * 84)
    print(f"{'Model':<26}{'Format':>9}{'Đồng thuận':>13}{'Latency p50':>13}"
          f"{'p95':>8}{'~detector':>11}{'n':>6}")
    print("-" * 84)
    for (model, tmp), g in d.groupby(["model", "temp"]):
        # đồng thuận: % câu mà MỌI lượt cho cùng một câu trả lời
        dt = np.nan
        if g.lan.nunique() > 1:
            dt = (g.groupby(["row_id", "label"]).tra_loi.nunique() == 1).mean() * 100
        print(f"{model+' t='+str(tmp):<26}{g['format'].mean()*100:>8.1f}%"
              f"{('—' if np.isnan(dt) else f'{dt:.0f}%'):>13}"
              f"{g.giay.median():>12.1f}s{g.giay.quantile(.95):>7.1f}s"
              f"{g.khop_detector.mean()*100:>10.1f}%{len(g):>6}")
    print("=" * 84)
    print("  Format     = trả lời ĐÚNG một con số, không chữ thừa. Sai format = 0 điểm.")
    print("  Đồng thuận = % câu mọi lượt trả lời giống nhau. Cần --runs >= 3.")
    print("  ~detector  = trùng với bộ nhận diện. KHÔNG phải độ đúng — detector đếm thiếu.")
    print(f"\n  Chi tiết từng câu: {ket}")


if __name__ == "__main__":
    main()
