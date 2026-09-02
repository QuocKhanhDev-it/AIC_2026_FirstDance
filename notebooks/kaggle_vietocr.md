# Cell Kaggle — thử **VietOCR** trên 251 khung

Trả lời **một** câu trước khi ai đó bỏ hàng chục giờ GPU chạy lại OCR cả kho:

> Đáp án vàng của 13 câu Q&A có xuất hiện **ĐÚNG DẤU** trong OCR mới không?

A68 đo được `ocr_text` hiện tại chỉ **31% có dấu** tiếng Việt, nên đáp án
`Tà Pứa` không bao giờ khớp với `Ta Pua` mà OCR đang đọc ra. Hiện **0/13** câu
khớp đúng dấu (7/13 chỉ khớp khi bỏ dấu — mà bài nộp thì phải đúng chuỗi).

VietOCR là model nhận dạng chữ **chuyên tiếng Việt**, nên nó trả về chữ có dấu.
Cách làm mượn từ `vietnamese-news-video-ocr` của một nhóm khác: **PaddleOCR dò
vùng chữ + VietOCR đọc chữ**.

Chỉ chạy **251 ảnh** (51 khung đáp án + 200 khung ngẫu nhiên để đo tốc độ) —
vài phút, không phải hàng chục giờ.

**Add Input:** các dataset ảnh L21–L30. Không cần `aic2026-index`.
Settings → **GPU T4**.

```python
# ── THỬ VietOCR trên 251 khung ────────────────────────────────────────
import subprocess, json, glob, os, time, pathlib

def chay(lenh, im=False):
    p = subprocess.run(lenh, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not im or p.returncode != 0:
        print(p.stdout[-3000:])
    if p.returncode != 0:
        raise RuntimeError(f"ma thoat {p.returncode}: {lenh}")
    return p.stdout

chay("rm -rf /tmp/repo && git clone -q -b giai-doan-0 "
     "https://github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git /tmp/repo")
os.chdir("/tmp/repo")

# KHONG `pip -U` — no keo pillow/pandas moi vao va pha vo torchvision co san
# (da tung gay `ImportError: cannot import name '_Ink'` va hong ca phien).
chay("pip -q install --no-deps vietocr paddleocr==2.7.3")

# `--no-deps` chan duoc thu pha moi truong, nhung chan luon may goi nho ma
# paddleocr/vietocr CAN LUC IMPORT. Dò tung cai, thieu cai nao cai dung cai do.
for mod, goi in [("pyclipper", "pyclipper"), ("shapely", "shapely"),
                 ("lmdb", "lmdb"), ("gdown", "gdown"),
                 ("albumentations", "albumentations"),
                 ("rapidfuzz", "rapidfuzz"), ("skimage", "scikit-image"),
                 ("prefetch_generator", "prefetch_generator")]:
    try:
        __import__(mod)
    except ImportError:
        print("thieu", mod, "-> cai", goi)
        chay(f"pip -q install --no-deps {goi}", im=True)

# ⚠️ Kho paddle KHONG co ban `2.6.1` tran — chi co hau to `.post120` theo phien
# ban CUDA ("from versions: 2.6.1.post120, 2.6.2.post120"). Ghim sai la hong
# ngay o giay thu 16. Thu GPU truoc, khong duoc thi lui ve CPU: 251 anh cham
# van chiu duoc, chi mat phan uoc toc do.
DUNG_GPU = True
try:
    chay("pip -q install paddlepaddle-gpu==2.6.2.post120 -i "
         "https://www.paddlepaddle.org.cn/packages/stable/cu120/")
except Exception as e:
    print("khong cai duoc ban GPU, lui ve CPU:", e)
    chay("pip -q install paddlepaddle==2.6.2")
    DUNG_GPU = False

import torch
assert torch.cuda.is_available(), "Settings > Accelerator > GPU T4"
print("GPU:", torch.cuda.get_device_name(0), "| paddle GPU:", DUNG_GPU)

# ── tra anh theo video_id + kf_name (KHONG dung kf_path — duong dan may khac)
BAN_DO = {}
for p in glob.glob("/kaggle/input/**/L*_V*/*.jpg", recursive=True):
    q = pathlib.Path(p)
    BAN_DO[(q.parent.name, q.name)] = p
print(f"{len(BAN_DO):,} anh tra duoc")

KHUNG = [json.loads(l) for l in
         open("dev/khung_thu_ocr.jsonl", encoding="utf-8") if l.strip()]
co = [k for k in KHUNG if (k["video_id"], k["kf_name"]) in BAN_DO]
print(f"{len(co)}/{len(KHUNG)} khung co anh"
      f"  ({sum(1 for k in co if k['nhom']=='qa')} khung dap an)")
assert len(co) > len(KHUNG) * 0.8, "thieu qua nhieu anh — Add Input du L21-L30 chua?"

# ── model ────────────────────────────────────────────────────────────
from paddleocr import PaddleOCR
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from PIL import Image

# PaddleOCR chi DO VUNG CHU (rec=False) — phan doc chu giao cho VietOCR.
do_chu = PaddleOCR(use_angle_cls=False, lang="en",
                   use_gpu=DUNG_GPU, show_log=False)

cfg = Cfg.load_config_from_name("vgg_transformer")
cfg["device"] = "cuda:0"          # VietOCR chay torch, luon dung GPU duoc
cfg["predictor"]["beamsearch"] = False        # nhanh hon, chenh lech nho
doc_chu = Predictor(cfg)

def ocr_mot_anh(duong_dan):
    anh = Image.open(duong_dan).convert("RGB")
    hop = do_chu.ocr(duong_dan, rec=False)
    if not hop or not hop[0]:
        return ""
    ra = []
    for box in hop[0]:
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        x0, x1 = max(0, int(min(xs))), int(max(xs))
        y0, y1 = max(0, int(min(ys))), int(max(ys))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        ra.append(doc_chu.predict(anh.crop((x0, y0, x1, y1))))
    return " ".join(x for x in ra if x)

# Thu MOT anh truoc — hong o day thi hong sau 2 giay, khong phai sau 251 anh.
_thu = BAN_DO[(co[0]["video_id"], co[0]["kf_name"])]
print("thu 1 anh:", repr(ocr_mot_anh(_thu)[:120]))

# ── chay ─────────────────────────────────────────────────────────────
kq, t0 = [], time.perf_counter()
for i, k in enumerate(co, 1):
    try:
        t = ocr_mot_anh(BAN_DO[(k["video_id"], k["kf_name"])])
    except Exception as e:
        t = ""
        print(f"  loi o {k['video_id']}/{k['kf_name']}: {e}")
    kq.append({"row_id": k["row_id"], "nhom": k["nhom"], "text": t})
    if i % 50 == 0:
        print(f"  {i}/{len(co)}  {(time.perf_counter()-t0)/i:.2f} s/anh",
              flush=True)

giay = (time.perf_counter() - t0) / len(co)
with open("/kaggle/working/ocr_vietocr.jsonl", "w", encoding="utf-8") as f:
    for x in kq:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
    f.write(json.dumps({"_meta": True, "giay_moi_anh": giay,
                        "so_anh": len(co)}, ensure_ascii=False) + "\n")

print(f"\n✅ {len(kq)} khung | {giay:.2f} s/anh"
      f" -> ca kho {giay*177321/3600:.1f} gio,"
      f" chia 12 phan {giay*177321/12/3600:.1f} gio/phan")
print("\nVai ket qua dau:")
for x in kq[:5]:
    print(f"  {x['row_id']:>7}  {x['text'][:110]!r}")
```

## Sau khi chạy

Tải `ocr_vietocr.jsonl` về `index/`, rồi:

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe scripts\83_do_vietocr.py
```

Nó in bảng so **OCR cũ** với **OCR mới** trên đúng ba câu hỏi: tỷ lệ có dấu,
số câu Q&A khớp đáp án đúng dấu, và s/ảnh.

## Ngưỡng đặt TRƯỚC khi xem số

**"Khớp đúng dấu" phải ≥ 4/13** thì mới đáng chạy lại cả kho. Dưới mức đó thì
phần đáp án Q&A vẫn phải trông vào VLM, và OCR mới chỉ còn giá trị cho kênh 3 —
mà `bm25.py` đã có nhánh **không dấu** nên lợi ích ở đó nhỏ.

Ghi ngưỡng ra trước để không tự thuyết phục mình sau khi thấy con số.

## Nếu cell hỏng

`paddlepaddle-gpu` là chỗ dễ vỡ nhất — bánh xe của nó gắn hậu tố theo phiên bản
CUDA, **không có bản số trần**. Ghim `2.6.1` thì pip báo:

```
ERROR: Could not find a version that satisfies the requirement
paddlepaddle-gpu==2.6.1 (from versions: 2.6.1.post120, 2.6.2.post120)
```

Cell đã ghim `2.6.2.post120` và **tự lui về bản CPU** nếu vẫn hỏng. Chỉ 251 ảnh
nên CPU vẫn chạy được — lúc đó `DUNG_GPU=False` và **con số s/ảnh không dùng để
ước cả kho được nữa**, chỉ còn trả lời câu hỏi về dấu.

VietOCR thì chạy trên torch nên luôn dùng được GPU, bất kể paddle ở chế độ nào.
