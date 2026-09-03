# Cell Kaggle — chạy VietOCR cho **cả kho**, chia 12 phần

Bản thử trên 251 khung (`kaggle_vietocr.md`) đo được **0,74 s/ảnh** trên T4.
Cả kho 177.321 ảnh = 36,6 giờ, chia 12 phần là **~3,0 giờ/phần** — vừa khít
phiên 6 tiếng với dư địa gấp đôi.

## Bản chia riêng cho OCR: **7 phần không cần L26**

L26 chiếm **44,9% kho** (498 video, 79.590 khung). `chia_caption` trải L26 đều
ra cả 12 phần — mỗi phần 39–43 video L26 trên ~73 — nên ai chưa tải được
dataset L26 thì **không chạy nổi phần nào**.

`chia_ocr/` tách hẳn hai nhóm (`scripts/88_chia_viec_ocr.py`):

| phần | video | khung | giờ GPU | cần L26 |
| --- | ---: | ---: | ---: | :---: |
| **A1–A7** | 53–54 | ~13.960 | **2,9** | **KHÔNG** |
| B1–B5 | 99–100 | ~15.920 | 3,3 | có |

Cân theo **số khung**, không theo số video: số khung mỗi video lệch rất mạnh
(L23 có 25 video / 2.326 khung, L25 có 88 video / 37.445 khung), chia đều theo
video thì phần rơi vào L25 chạy lâu gấp mấy lần. Xếp thùng tham lam cho các
phần lệch **dưới 1%**.

**Đổi đúng một dòng: `PHAN = "A1"` (hoặc `"B3"`…).**

⚠️ Bản chia này giờ **đã chốt**. Bài học đợt caption: chia lại khi đã có người
chạy làm ba phần hợp lệ bị đánh dấu nhầm là "sai phần". `88_` tự từ chối ghi đè
nếu `chia_ocr/` đã có nội dung.

## Trước khi chạy

* **Add Input:** các dataset ảnh chứa video của phần bạn. Không cần
  `aic2026-index`. **Phần A không cần dataset L26.**
* **Settings → Accelerator → GPU T4.**
* **Settings → Persistence → Files only** (để `/kaggle/working` sống qua các
  lần chạy lại — cần cho việc chạy tiếp).

## Chạy lại được giữa chừng

Cell ghi kết quả **ngay khi có**, mỗi 200 ảnh một lần `flush`. Phiên chết giữa
chừng thì chạy lại chính cell đó: nó đọc file cũ, bỏ qua ảnh đã xong, chạy tiếp
từ chỗ dở. Không mất giờ GPU đã bỏ ra.

```python
# ── VietOCR cho CẢ PHẦN — EasyOCR dò vùng + VietOCR đọc chữ ───────────
PHAN = "A1"          # ⚠️ ĐỔI DÒNG NÀY. A1..A7 khong can L26; B1..B5 chi L26.

import subprocess, json, os, re, time

def chay(lenh, im=False):
    p = subprocess.run(lenh, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not im or p.returncode != 0:
        print(p.stdout[-3000:])
    if p.returncode != 0:
        raise RuntimeError(f"ma thoat {p.returncode}: {lenh}")
    return p.stdout

# ⚠️ URL NOI CHUOI CO Y — copy tu cho da render Markdown thi URL lien khoi bi
# boc thanh `[nhan](dich)` va `sh` chet: Syntax error: "(" unexpected.
KHO_GIT = "https://" + "github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git"

chay(f"rm -rf /tmp/repo && git clone -q -b giai-doan-0 {KHO_GIT} /tmp/repo")
os.chdir("/tmp/repo")

# KHONG `pip -U` — no keo pillow/pandas moi vao va pha vo torchvision co san.
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

# Trong so VietOCR nam tren vocr.vn, chung chi SSL cua no hay bi tu choi tren
# Kaggle. Bo kiem chung chi cho rieng buoc tai model.
import ssl, urllib3, requests
os.environ["PYTHONHTTPSVERIFY"] = "0"
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
_goc = requests.Session.request
requests.Session.request = (
    lambda self, *a, **k: _goc(self, *a, **{**k, "verify": False}))

# ── danh sach video cua PHAN nay ─────────────────────────────────────
VIDEO = [l.strip() for l in
         open(f"chia_ocr/phan_{PHAN}.txt", encoding="utf-8") if l.strip()]
print(f"phan {PHAN}: {len(VIDEO)} video"
      f" | {'KHONG can' if str(PHAN).startswith('A') else 'CAN'} L26")

# ⚠️ DUNG `glob("/kaggle/input/**/*.jpg", recursive=True)` (655s) va dung ca
# `os.walk` (669s): ca hai LIET KE FILE cua moi thu muc chung di qua. Chan
# theo HINH DANG TEN thu muc video, ke ca thu muc KHONG can — do moi la cho
# an tien that (685/873 thu muc khong can van bi liet ke o ban truoc).
LA_VIDEO = re.compile(r"^L\d\d_V\d\d\d$")
CAN = set(VIDEO)

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
                    continue                  # can hay khong, DUNG di vao
                ngan_xep.append(e.path)
    except OSError:
        pass
print(f"quet {time.perf_counter()-t_quet:.0f}s | "
      f"{len(DUONG)}/{len(VIDEO)} thu muc video tim thay")
thieu = sorted(CAN - set(DUONG))
if thieu:
    print(f"⚠️ THIEU {len(thieu)} video, vd: {thieu[:5]}")
    print("   -> Add Input dataset chua nhung video nay roi chay lai.")
assert len(DUONG) >= len(VIDEO) * 0.9, "thieu qua nhieu video, xem tren"

# Liet ke anh — day la cho DUY NHAT phai doc noi dung thu muc keyframe.
VIEC = []
for v in sorted(DUONG):
    for f in sorted(os.listdir(DUONG[v])):
        if f.lower().endswith(".jpg"):
            VIEC.append((v, f, os.path.join(DUONG[v], f)))
print(f"{len(VIEC):,} anh trong phan nay")

# ── chay tiep tu cho do ──────────────────────────────────────────────
RA = f"/kaggle/working/ocr_vietocr_phan{PHAN}.jsonl"
XONG = set()
if os.path.exists(RA):
    for l in open(RA, encoding="utf-8"):
        try:
            d = json.loads(l)
        except json.JSONDecodeError:
            continue                       # dong cuoi bi cat khi phien chet
        if not d.get("_meta"):
            XONG.add((d["video_id"], d["kf_name"]))
    print(f"da co {len(XONG):,} anh tu lan chay truoc -> chay tiep")
CON = [x for x in VIEC if (x[0], x[1]) not in XONG]
print(f"con {len(CON):,} anh phai chay")

# ── model ────────────────────────────────────────────────────────────
def _nhap():
    global easyocr, Cfg, Predictor, Image
    import easyocr
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor
    from PIL import Image
nhap_du(_nhap)

# recognizer=False -> KHONG tai model nhan dang cua EasyOCR. No chi do VUNG,
# phan doc chu giao cho VietOCR vi VietOCR moi tra ve chu CO DAU.
do_vung = easyocr.Reader(["vi"], gpu=True, recognizer=False)

cfg = Cfg.load_config_from_name("vgg_transformer")
cfg["device"] = "cuda:0"
cfg["predictor"]["beamsearch"] = False       # nhanh hon, chenh lech nho
doc_chu = Predictor(cfg)

TOI_DA_VUNG = 40      # khung bang chu chay co the ra hang tram vung; chan lai

def ocr_mot_anh(duong_dan):
    anh = Image.open(duong_dan).convert("RGB")
    W, H = anh.size
    ngang, tu_do = do_vung.detect(duong_dan)
    vung = []
    for h in (ngang[0] if ngang else []):
        vung.append((h[0], h[2], h[1], h[3]))         # x0, y0, x1, y1
    for g in (tu_do[0] if tu_do else []):
        xs = [q[0] for q in g]; ys = [q[1] for q in g]
        vung.append((min(xs), min(ys), max(xs), max(ys)))

    cat = []
    for x0, y0, x1, y1 in vung[:TOI_DA_VUNG]:
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(W, int(x1)), min(H, int(y1))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        cat.append(anh.crop((x0, y0, x1, y1)))
    if not cat:
        return ""
    # predict_batch: mot lan goi GPU cho ca anh, thay vi mot lan moi vung.
    try:
        chu = doc_chu.predict_batch(cat)
    except Exception:
        chu = [doc_chu.predict(c) for c in cat]
    return " ".join(x for x in chu if x)

# ── chay ─────────────────────────────────────────────────────────────
# Ghi NGAY khi co, flush moi 200 anh. Phien chet giua chung thi chay lai
# chinh cell nay: no doc file cu va chay tiep. Khong mat gio GPU da bo ra.
t0, xong = time.perf_counter(), 0
with open(RA, "a", encoding="utf-8") as f:
    for i, (v, ten, duong) in enumerate(CON, 1):
        try:
            chu = ocr_mot_anh(duong)
        except Exception as e:
            chu = ""
            print(f"  loi o {v}/{ten}: {e}", flush=True)
        f.write(json.dumps({"video_id": v, "kf_name": ten, "text": chu},
                           ensure_ascii=False) + "\n")
        xong += 1
        if i % 200 == 0:
            f.flush()
            os.fsync(f.fileno())
            giay = (time.perf_counter() - t0) / i
            con_lai = (len(CON) - i) * giay / 3600
            print(f"  {i:,}/{len(CON):,}  {giay:.2f} s/anh  "
                  f"con ~{con_lai:.1f} gio", flush=True)
    f.write(json.dumps({"_meta": True, "phan": PHAN,
                        "so_anh": len(XONG) + xong,
                        "giay_moi_anh": (time.perf_counter() - t0) /
                                        max(xong, 1)}, ensure_ascii=False) + "\n")

print(f"\n✅ phan {PHAN}: {len(XONG) + xong:,} anh -> {RA}")
print(f"   {(time.perf_counter()-t0)/max(xong,1):.2f} s/anh")
```

## Sau khi chạy

Tải `ocr_vietocr_phan<N>.jsonl` từ tab **Output** về `index/ocr_vietocr/`.

⚠️ Kaggle hay đổi đuôi file khi tải — bản thử về thành `.txt`. Không sao, các
script đọc theo nội dung chứ không theo đuôi.

Khi đủ các phần thì gộp và **giữ cả OCR cũ**:

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe scripts\88_gop_ocr_vietocr.py
```

## Vì sao GỘP chứ không THAY

Đo trên 251 khung thử: OCR mới đưa tỷ lệ có dấu ở khung đáp án từ **20% lên
82%**, và hai đáp án đi từ "chỉ khớp khi bỏ dấu" sang khớp đúng chuỗi. Nhưng nó
**làm mất một con số** mà OCR cũ đọc được (`46`, câu `qa-DE1-15`).

Hai bộ hỏng ở chỗ khác nhau — cũ mất dấu, mới mất số. Thay hẳn là đánh đổi;
gộp thì không mất bên nào, và BM25 chỉ được lợi khi có thêm từ.
