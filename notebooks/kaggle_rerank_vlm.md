# Cell Kaggle — VLM chấm lại bể ứng viên (Nấc 3, bước ĐO)

Trả lời **một** câu hỏi: A54 đo khoảng trống giữa điểm thật và trần "xếp lại
hoàn hảo" là **33 điểm phần trăm**, A55 cho thấy không lấy được bằng tín hiệu
sẵn có — vậy VLM nhìn lại bức ảnh thì lấy được bao nhiêu?

**Đây là bước ĐO, chưa phải hạ tầng ngày thi.** Chạy trên Kaggle T4 chứ không
phải máy ai cả. Nếu VLM chỉ lấy được vài điểm thì cả chuyện tunnel, đồng bộ
ảnh và đường lui đều không đáng dựng.

## Chuẩn bị

Trên máy có `index/`:

```powershell
.venv\Scripts\python.exe scripts\63_xuat_be_rerank.py
```

Ra `dev/be_rerank.jsonl` (~139 KB, 52 câu × 30 ứng viên). File này **không
chứa đáp án** — cố ý, vì nó đi sang máy khác và có thể lọt vào log Kaggle.

File đã nằm trong repo nên notebook `git clone` là có. Chỉ cần Add Input các
dataset **ảnh keyframe** (L21…L29) — không cần `aic2026-index`.

Settings → Accelerator → **GPU T4 x2**.

## Cell

```python
# ── VLM chấm lại 52 câu × 30 ứng viên ────────────────────────────────
import subprocess, json, glob, os, time, pathlib

def chay(lenh):
    p = subprocess.run(lenh, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-2000:])
    if p.returncode != 0:
        raise RuntimeError(f"ma thoat {p.returncode}: {lenh}")

chay("rm -rf /tmp/repo && git clone -q -b giai-doan-0 "
     "https://github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git /tmp/repo")
os.chdir("/tmp/repo")
chay("pip -q install 'transformers>=4.45' accelerate qwen-vl-utils pillow")

import torch
assert torch.cuda.is_available(), "Settings > Accelerator > GPU T4 x2"
print("GPU:", torch.cuda.get_device_name(0))

# ── tra ảnh: video_id + kf_name, KHÔNG dùng kf_path (đường dẫn máy khác) ──
BAN_DO = {}
for p in glob.glob("/kaggle/input/**/L*_V*/*.jpg", recursive=True):
    q = pathlib.Path(p)
    BAN_DO[(q.parent.name, q.name)] = p
print(f"{len(BAN_DO):,} ảnh keyframe tra được")

CAU = [json.loads(l) for l in open("dev/be_rerank.jsonl", encoding="utf-8")]
thieu = sum(1 for c in CAU for u in c["ung_vien"]
            if (u["video_id"], u["kf_name"]) not in BAN_DO)
tong = sum(len(c["ung_vien"]) for c in CAU)
print(f"thiếu ảnh: {thieu}/{tong} ứng viên")
assert thieu < tong * 0.2, "thiếu quá 20% ảnh — Add Input đủ các dataset L2x chưa?"

# ── model ────────────────────────────────────────────────────────────
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
TEN = "Qwen/Qwen2-VL-2B-Instruct"

# fp16 chứ KHÔNG bf16: máy thi của nhóm là RTX 2060 (Turing, sm_75) không có
# bf16. Đo trên T4 (cũng Turing) để con số dùng lại được cho máy đó.
# sdpa chứ không flash_attention_2 — FA2 đòi Ampere trở lên.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    TEN, torch_dtype=torch.float16, attn_implementation="sdpa",
    device_map="cuda:0")
model.eval()
proc = AutoProcessor.from_pretrained(TEN, min_pixels=256*28*28,
                                     max_pixels=512*28*28)

# ID của "Có"/"Không" — chấm bằng LOGIT chứ không sinh chữ:
# một lượt forward, điểm liên tục, không phụ thuộc model có chịu trả lời đúng
# định dạng hay không.
CO = proc.tokenizer.encode("Có", add_special_tokens=False)[0]
KHONG = proc.tokenizer.encode("Không", add_special_tokens=False)[0]

def nhac(cau_hoi):
    return ("Mô tả cần tìm:\n" + cau_hoi.strip() +
            "\n\nẢnh này có đúng là cảnh được mô tả ở trên không? "
            "Chỉ trả lời Có hoặc Không.")

from PIL import Image

@torch.no_grad()
def cham_lo(cau_hoi, duong_dan):
    tin = [[{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": nhac(cau_hoi)}]}]
        for _ in duong_dan]
    van = [proc.apply_chat_template(t, tokenize=False,
                                    add_generation_prompt=True) for t in tin]
    anh = [Image.open(p).convert("RGB") for p in duong_dan]
    x = proc(text=van, images=anh, padding=True, return_tensors="pt").to("cuda:0")
    lg = model(**x).logits[:, -1, :].float()
    return (lg[:, CO] - lg[:, KHONG]).tolist()      # >0 nghiêng "Có"

# ── chạy ─────────────────────────────────────────────────────────────
LO = 4                    # 8 GB VRAM của máy thi chịu được; T4 16 GB dư
ra, t0, xong = [], time.perf_counter(), 0
for c in CAU:
    rid, diem = [], []
    uv = [u for u in c["ung_vien"] if (u["video_id"], u["kf_name"]) in BAN_DO]
    for i in range(0, len(uv), LO):
        lo = uv[i:i + LO]
        d = cham_lo(c["cau_hoi"],
                    [BAN_DO[(u["video_id"], u["kf_name"])] for u in lo])
        rid += [u["row_id"] for u in lo]
        diem += d
    ra.append({"id": c["id"], "row_id": rid, "diem_vlm": diem,
               "model": TEN, "dau": len(c["ung_vien"])})
    xong += len(uv)
    g = time.perf_counter() - t0
    print(f"  {len(ra):>3}/{len(CAU)}  {xong} ảnh  {g/xong:.2f} s/ảnh", flush=True)

with open("/kaggle/working/diem_vlm.jsonl", "w", encoding="utf-8") as f:
    for d in ra:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

g = (time.perf_counter() - t0) / xong
print(f"\n✅ diem_vlm.jsonl — {xong} ảnh, {g:.2f} s/ảnh")
print(f"   ngày thi: {g*30:.1f} s/câu nếu chấm top-30 (T4). "
      f"RTX 2060 chậm hơn ~2 lần.")
```

## Sau khi chạy

Tải `diem_vlm.jsonl` từ tab **Output** về `index/diem_vlm.jsonl`, rồi trên máy
có `index/`:

```powershell
.venv\Scripts\python.exe scripts\64_do_rerank_vlm.py --diem index\diem_vlm.jsonl
```

Nó đo **sáu cách dùng điểm VLM** (thay hẳn / RRF hạng / nhân / chỉ đẩy lên),
mốc là `run.py` hiện tại, báo cáo ở hai mức dung sai như mọi phép đo khác.

## Đọc kết quả

Con số duy nhất đáng quan tâm: **lấy được bao nhiêu phần trăm của +0,3067**
(khoảng trống ở bể 300 theo A54).

* Dưới ~15% → không đáng dựng hạ tầng ngày thi. Ghi lại như A53 rồi đóng hướng.
* Trên ~40% → đáng, và lúc đó mới bàn chuyện chạy ở đâu, kèm đường lui.

Và nhớ nhìn cả cột **thắng–thua–hoà**: VLM kéo tụt những câu kênh 1 vốn làm
đúng là chuyện có thật, đó là lý do bảng có cách "chỉ đẩy lên, không đẩy xuống".

## Bẫy

**Đừng đổi sang bf16.** Ví dụ Qwen-VL trên mạng hầu hết viết `bfloat16`. T4 và
RTX 2060 đều là Turing, **không có bf16** — chạy được nhưng chậm thảm, và con
số s/ảnh đo ra sẽ vô nghĩa với máy thi.

**`max_pixels` quyết định cả tốc độ lẫn VRAM.** Ảnh thu nhỏ của ta là 256px
(`49_sinh_anh_nho.py`), nên `512*28*28` là dư; hạ xuống nếu OOM trên 8 GB.

**Không có đáp án trong `be_rerank.jsonl`** — đừng "tiện tay" thêm vào để soi
kết quả trên Kaggle. Rò tập dev không crash, chỉ làm mọi con số sau đó vô nghĩa.
