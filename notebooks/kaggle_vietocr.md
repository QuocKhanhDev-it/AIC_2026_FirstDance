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
import subprocess, json, os, re, time, pathlib

def chay(lenh, im=False):
    p = subprocess.run(lenh, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not im or p.returncode != 0:
        print(p.stdout[-3000:])
    if p.returncode != 0:
        raise RuntimeError(f"ma thoat {p.returncode}: {lenh}")
    return p.stdout

# ⚠️ HAI URL DUOI DAY CO Y NOI CHUOI, DUNG "don dep" lai thanh mot chuoi.
# Copy tu cho da render Markdown (chat, GitHub preview) thi URL lien khoi bi
# boc thanh `[nhan](dich)`, va `sh` chet ngay: Syntax error: "(" unexpected.
# Cat lam doi thi bo tu-nhan-link khong nhan ra nua. Bam nut Raw van tot hon.
KHO_GIT = "https://" + "github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git"

chay(f"rm -rf /tmp/repo && git clone -q -b giai-doan-0 {KHO_GIT} /tmp/repo")
os.chdir("/tmp/repo")

# KHONG `pip -U` — no keo pillow/pandas moi vao va pha vo torchvision co san
# (da tung gay `ImportError: cannot import name '_Ink'` va hong ca phien).
#
# ⚠️ KHONG DUNG PaddleOCR NUA. Ba luot Kaggle chet vi chuoi phu thuoc cua no:
#   - paddlepaddle-gpu khong co ban so tran, phai ghim `.post120`
#   - thieu pyclipper (chan boi --no-deps)
#   - thieu imgaug
#   - imgaug goi `np.sctypes`, da bi bo o NumPy 2.0 ma Kaggle chay NumPy 2
# Va 797 MB paddle KHONG phai phan tra loi cau hoi. Cau hoi la "VietOCR co doc
# ra DAU khong", no khong phu thuoc ai do vung chu. EasyOCR do vung bang CRAFT,
# chay tren torch — cung ngan xep voi VietOCR, khong them runtime thu hai.
chay("pip -q install --no-deps vietocr easyocr")

TEN_GOI = {"skimage": "scikit-image", "cv2": "opencv-python-headless",
           "PIL": "pillow", "yaml": "pyyaml", "sklearn": "scikit-learn",
           "bidi": "python-bidi"}

def nhap_du(lam, toi_da=15):
    """Chay `lam()`; thieu module nao thi cai dung module do roi thu lai."""
    for _ in range(toi_da):
        try:
            return lam()
        except ModuleNotFoundError as e:
            ten = (e.name or "").split(".")[0]
            if not ten:
                raise
            goi = TEN_GOI.get(ten, ten)
            print("thieu", ten, "-> cai", goi, flush=True)
            chay(f"pip -q install --no-deps {goi}", im=True)
    raise RuntimeError("cai vong quanh qua nhieu lan — xem log ben tren")

for goi in ("python-bidi", "pyclipper", "shapely", "lmdb", "gdown",
            "albumentations", "prefetch_generator", "ninja"):
    chay(f"pip -q install --no-deps {goi}", im=True)

import torch
assert torch.cuda.is_available(), "Settings > Accelerator > GPU T4"
print("GPU:", torch.cuda.get_device_name(0))

# ── tra anh theo video_id + kf_name (KHONG dung kf_path — duong dan may khac)
KHUNG = [json.loads(l) for l in
         open("dev/khung_thu_ocr.jsonl", encoding="utf-8") if l.strip()]
CAN = {}
for k in KHUNG:
    CAN.setdefault(k["video_id"], set()).add(k["kf_name"])

# ⚠️ HAI CACH DA DO THAT VA DEU CHAM:
#     glob("/kaggle/input/**/*.jpg", recursive=True)  -> 655 giay
#     os.walk co cat nhanh duoi thu muc keyframe      -> 669 giay
# `os.walk` van LIET KE FILE cua moi thu muc no di qua, nen cat nhanh xuong
# duoi thu muc keyframe khong tiet kiem gi (thu muc keyframe von khong co thu
# muc con). Cai dat tien la 177k lan liet ke tren mount mang cua Kaggle.
#
# Cach nay KHONG liet ke thu muc keyframe mot lan nao: chi di tim THU MUC
# video, roi ghep thang duong dan vi ten file da biet san (`kf_name`).
# ⚠️ Ban truoc van cham (278s) vi mot le CHUA sua: thu muc video KHONG nam
# trong `CAN` (685/873 cai) bi coi la thu muc thuong nen van bi di vao va liet
# ke — tuc van duyet gan het 177k anh. Phai chan theo HINH DANG TEN, khong
# phai theo "co can hay khong".
LA_VIDEO = re.compile(r"^L\d\d_V\d\d\d$")

t_quet = time.perf_counter()
DUONG = {}
ngan_xep = ["/kaggle/input"]
while ngan_xep:
    try:
        with os.scandir(ngan_xep.pop()) as it:
            for e in it:
                if not e.is_dir(follow_symlinks=False):
                    continue
                if LA_VIDEO.match(e.name):
                    if e.name in CAN:
                        DUONG[e.name] = e.path
                    continue                    # can hay khong, DUNG di vao
                ngan_xep.append(e.path)
    except OSError:
        pass

BAN_DO = {}
for k in KHUNG:
    d = DUONG.get(k["video_id"])
    if d:
        duong = os.path.join(d, k["kf_name"])
        if os.path.exists(duong):
            BAN_DO[(k["video_id"], k["kf_name"])] = duong
print(f"quet {time.perf_counter()-t_quet:.0f}s | {len(DUONG)} thu muc video "
      f"| {len(BAN_DO)} anh tra duoc")

co = [k for k in KHUNG if (k["video_id"], k["kf_name"]) in BAN_DO]
print(f"{len(co)}/{len(KHUNG)} khung co anh"
      f"  ({sum(1 for k in co if k['nhom']=='qa')} khung dap an)")
assert len(co) > len(KHUNG) * 0.8, "thieu qua nhieu anh — Add Input du L21-L30 chua?"

# ── model ────────────────────────────────────────────────────────────
# EasyOCR CHI DO VUNG CHU (recognizer=False -> khong tai model nhan dang cua
# no). Phan doc chu giao cho VietOCR, vi day moi la thu dang do: VietOCR la
# model chuyen tieng Viet nen tra ve chu CO DAU.
def _nhap():
    global easyocr, Cfg, Predictor, Image
    import easyocr
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor
    from PIL import Image
nhap_du(_nhap)

do_vung = easyocr.Reader(["vi"], gpu=True, recognizer=False)

cfg = Cfg.load_config_from_name("vgg_transformer")
cfg["device"] = "cuda:0"
cfg["predictor"]["beamsearch"] = False        # nhanh hon, chenh lech nho
doc_chu = Predictor(cfg)

def ocr_mot_anh(duong_dan):
    anh = Image.open(duong_dan).convert("RGB")
    W, H = anh.size
    ngang, tu_do = do_vung.detect(duong_dan)
    vung = []
    for h in (ngang[0] if ngang else []):
        vung.append((h[0], h[2], h[1], h[3]))          # x0,y0,x1,y1
    for g in (tu_do[0] if tu_do else []):
        xs = [q[0] for q in g]; ys = [q[1] for q in g]
        vung.append((min(xs), min(ys), max(xs), max(ys)))

    ra = []
    for x0, y0, x1, y1 in vung:
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(W, int(x1)), min(H, int(y1))
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
