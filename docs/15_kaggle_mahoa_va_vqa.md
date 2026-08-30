# 15 — Hai việc chạy trên Kaggle: mã hoá truy vấn và thử VQA

Nối tiếp [14_kaggle_sinh_truy_van_npz.md](14_kaggle_sinh_truy_van_npz.md).
Việc A là **bảo trì định kỳ**, phải chạy lại sau mỗi đợt soạn câu dev. Việc B là
**một phép thử đi/không-đi**, chạy một lần rồi kết luận.

---

## VIỆC A — Mã hoá mệnh đề truy vấn mới

### Vì sao lại phải chạy lại

`index/truy_van.npz` là **vector truy vấn mã hoá sẵn**. Có nó thì máy 7,7 GB
RAM chạy được kênh 1 mà không cần nạp model SigLIP2 — thứ đã làm treo máy nhiều
lần. Nhưng cache chỉ chứa **đúng những mệnh đề đã có lúc sinh**. Soạn câu dev
mới là cache hụt ngay, và những câu đó **im lặng biến mất khỏi mọi phép đo kênh
1** — không lỗi, không cảnh báo.

Đo lúc viết tài liệu này — và **đề sơ tuyển đợt 2 thì hụt hoàn toàn**:

```text
cache hiện có : 593 mệnh đề
tập dev cần   : 526   -> THIẾU 138 (45/231 câu)
đề đợt 2 cần :  71   -> THIẾU 71  (30/30 gói)
```

Kiểm bất cứ lúc nào — nếu số cuối khác 0 thì tới lúc chạy lại việc A:

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe -c "import numpy as np,json,sys;sys.path.insert(0,'src');import run as R;z=np.load('index/truy_van.npz',allow_pickle=True);co=set(map(str,z['cau']));d=[json.loads(l) for l in open('dev/tap_dev.jsonl',encoding='utf-8')];f=lambda c:(R.tach_su_kien(c['cau_hoi']) if c['loai']=='TRAKE' else [c['cau_hoi']]);print('chua do duoc:',sum(1 for c in d if any(t not in co for x in f(c) for t in R.tach_truy_van(x))),'/',len(d))"
```

### ⚠️ Đề thi KHÔNG còn đi theo repo nữa

Từ 28/08, `.gitignore` chặn mọi thư mục đề (`*-bo-de-thi/`, `De_Thi*/`,
`query-p*-*.txt`). Đó là chủ ý — repo công khai, đề đang thi thì không được lên
GitHub. Hệ quả: **clone repo trên Kaggle sẽ KHÔNG có thư mục đề**, khác với tài
liệu 14 hồi đề đợt 1 còn nằm trong git.

Nên bây giờ phải **đưa đề lên riêng**:

```powershell
# nén thư mục đề (vqa/ bị gitignore chặn nên file zip không lọt lên repo)
.venv\Scripts\python.exe -c "import shutil;print(shutil.make_archive('vqa/de_p2_de_ma_hoa','zip','dev/SOTUYEN2-bo-de-thi'))"
```

Upload file đó thành **Private Dataset** (~10 KB). **Không bao giờ để public** —
đây là đề đang thi.

### Các bước

Notebook mới: Internet **ON**, Accelerator **None** (CPU đủ — đây là tháp văn
bản, không phải tháp ảnh), và **Add Input** trỏ vào Private Dataset chứa đề.

```python
# 1. ma nguon — PHAI chi dinh nhanh, `main` khong co script nay
!git clone -q -b giai-doan-0 https://github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git repo
%cd repo
!pip -q install open_clip_torch pandas pyarrow

# 2. CHOT CHAN — ban `tach_su_kien` cu tach SAI so su kien, cache sinh ra se
#    mang chuoi sai LANG LE. Kiem ngay tren DE DANG DUNG, khong kiem tren mot
#    file de cu cam cung: A40 hong voi `E1 `, A44 hong voi `Canh 1:` — moi dot
#    de doi cach danh dau thi cho kiem cu lai hut dich.
import sys; sys.path.insert(0, 'src')
import run
DEM = {'query-p2-8-trake': 4, 'query-p2-21-trake': 4}   # DEM BANG MAT tu de
for ten, can in DEM.items():
    n = len(run.tach_su_kien(open(f'de_p2/{ten}.txt', encoding='utf-8').read()))
    assert n == can, f"{ten}: tach ra {n} su kien, dem bang mat la {can}"
print("OK, so su kien khop")

# 3. bung DE tu Private Dataset (repo khong con mang de theo)
!mkdir -p de_p2 && cp /kaggle/input/<ten-dataset-de>/*.txt de_p2/
import glob; assert len(glob.glob('de_p2/*.txt')) == 30, "phai co du 30 file de"
print("OK, 30 goi de")

# 4. sidecar: script doc so chieu tu day, khoi phai tai ma tran 390 MB len
import json, pathlib
pathlib.Path('index').mkdir(exist_ok=True)
json.dump({"model": "ViT-SO400M-14-SigLIP2-378", "pretrained": "webli", "chieu": 1152},
          open('index/clip_siglip2.json', 'w'))

# 5. ma hoa — ca de dot 2 lan ca tap dev
!python scripts/25_ma_hoa_truy_van.py --de de_p2 --tap-dev --fp16
```

Kiểm **ngay trên Kaggle** trước khi tải về:

```python
import numpy as np
z = np.load('index/truy_van.npz', allow_pickle=True)
print(z['vec'].shape, z['vec'].dtype)          # (N, 1152) float32
assert z['vec'].shape[1] == 1152
assert not np.isnan(z['vec']).any()
assert (np.linalg.norm(z['vec'], axis=1) > 0).all()
```

Tải `index/truy_van.npz` về, đặt đè vào `index/`, rồi chạy lại lệnh kiểm ở trên.
Phải in **`chua do duoc: 0 / <tổng>`**.

### ⚠️ Vá `tach_su_kien` XONG là cache TRAKE hỏng ngay

Cache lưu **đúng những chuỗi `tach_su_kien` sinh ra**. Sửa bộ tách là chuỗi đổi,
và những chuỗi mới đó **không có trong cache** — `run.py` dừng với
`❌ N/M chuỗi truy vấn chưa có trong cache`.

Đã cắn thật (28/08): vá A44 xong thì `query-p2-21-trake` từ 5 mệnh đề thành 4,
cache sinh trước đó thành vô dụng cho gói ấy. **Vá bộ tách rồi thì phải sinh lại
cache**, không có đường vòng.

Đường thoát tạm khi không kịp sinh lại: chạy riêng gói đó bằng cấu hình không
cần model, rồi ghép vào:

```powershell
.venv\Scripts\python.exe src\run.py --de <thu muc chi chua goi do> ^
    --ra vqa\tam --kenh objects --hop-nhat --bo-metadata
```

Yếu hơn kênh 1 nhiều, chỉ dùng để không bỏ trống gói.

### Nghiệm thu: một câu đã biết đáp án phải giữ nguyên hạng

Kiểm cấu trúc (đúng số chiều, không NaN) **không bắt được** lỗi nguy hiểm nhất:
mã hoá bằng **sai model** hoặc sai tag. Cosine tụt mà file vẫn hợp lệ hoàn toàn
— đúng cái bẫy `ViT-B-32-quickgelu`.

Phép thử bắt được: một câu dev mà kênh 1 **đang** xếp đáp án hạng 1. Cache mới
mà làm nó tụt hạng thì mã hoá sai, dù mọi kiểm tra khác đều xanh.

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe -c "import sys,json;sys.path.insert(0,'src');import run as R;from dense import KenhAnhCache;k=KenhAnhCache('./index','index/truy_van.npz');c=[json.loads(l) for l in open('dev/tap_dev.jsonl',encoding='utf-8') if 'kis-L22-105' in l][0];rs={r[0] if isinstance(r,list) else r for r in c['row_id_dung']};kq=k.tim(R.tach_truy_van(c['cau_hoi']),k=100);h=next((i+1 for i,x in enumerate(kq) if x.row_id in rs),None);print('kis-L22-105 hang khung =',h,'(phai la 1)')"
```

Đo lúc viết tài liệu (28/08, `ViT-SO400M-14-SigLIP2-378`/`webli`): **hạng 1**.
Ra số khác là **đừng dùng cache đó** — sinh lại, và soi lại xem ô sidecar có ghi
đúng `model`/`pretrained` không.

> `25_ma_hoa_truy_van.py` có cờ `--gop` để chỉ mã hoá phần thiếu. Trên Kaggle
> **không dùng được** vì repo clone về không mang theo `index/truy_van.npz`
> (`index/` không nằm trong git). Sinh lại toàn bộ cũng chỉ vài phút — mã hoá
> văn bản rẻ hơn mã hoá ảnh hàng nghìn lần.

---

## VIỆC B — Thử VQA cho câu đếm / câu hỏi màu

### Câu hỏi đang đo, viết cho hẹp

> **Cho sẵn ĐÚNG khung đáp án, VLM có đọc ra đúng đáp án không?**

Đây là **trần trên**. Bài thật còn phải tìm được khung đã. Nếu ngay cả khi đưa
đúng khung mà VLM vẫn sai phần lớn thì hướng này chết tại đây — khỏi dựng đường
ống, khỏi bàn tiếp.

### Trước khi chạy: một phát hiện làm phép thử này yếu đi

`41_cham_vqa.py` chia câu theo **câu hỏi đòi gì**, và kết quả chia làm lộ ra
chuyện đáng lo hơn cả phép đo:

| nhóm | số câu | ví dụ |
| --- | ---: | --- |
| **ĐỌC CHỮ** | 10 | *"Số hiển thị trên tấm bảng đen..."*, *"Tổng điểm của Đoàn... là bao nhiêu?"* |
| **NHÌN** | **4** | *"Có bao nhiêu con mèo được chàng trai cho ăn?"*, *"bát canh đó có màu gì?"* |

Trong **45 câu Q&A** của cả tập dev, chỉ **4 câu** thật sự đòi nhìn. Phần lớn
câu "đếm" thực ra là **đọc số trên bảng điểm, biển hiệu, màn hình** — đó là việc
của kênh 3, và VLM trả lời đúng ở đó **không nói lên điều gì mới**.

Hệ quả cho cách đọc kết quả:

* Kết luận **ÂM vẫn chắc**: 4 câu mà sai gần hết thì bỏ hướng này, khỏi bàn.
* Kết luận **DƯƠNG thì KHÔNG chắc**: 4 câu đúng hết cũng không đủ bật tính năng.
  Muốn kết luận dương phải **soạn thêm câu Q&A thuần thị giác** — đếm vật, hỏi
  màu, hỏi vị trí tương đối.

> **Một cách chia đã thử và đã bỏ.** Ban đầu tôi chia theo "đáp án có nằm trong
> văn bản OCR của khung đáp án không". Chạy thật thì hỏng ngay: đáp án `'2'`
> khớp bừa vào bất cứ đoạn OCR nào có chữ số 2, còn câu *số điện thoại* — câu
> đọc-chữ rõ nhất — lại rơi vào nhóm "OCR mù" chỉ vì OCR đọc sai. Phép thử đó đo
> **may rủi của OCR** chứ không đo **bản chất câu hỏi**.

### 1. Đóng gói ở máy local

```powershell
.venv\Scripts\python.exe scripts\40_xuat_goi_vqa.py            # xem trước
.venv\Scripts\python.exe scripts\40_xuat_goi_vqa.py --ghi --nen
```

Ra `vqa/goi_vqa.zip` (~1,1 MB, 14 câu). **Gói KHÔNG mang đáp án** — `dap_an` ở
lại `vqa/dap_an.json` trên máy này, nên không có đường nào để đáp án lọt vào lời
nhắc, kể cả do sơ ý.

Script cũng in ra những câu **rụng vì máy này chưa có ảnh** (hiện 8 câu, L26 và
L29). Cột `kf_path` nghĩa là "ảnh đã tải Ở MÁY NÀY", không phải "có keyframe" —
A5.5.

### 2. Đưa lên Kaggle

Upload `vqa/goi_vqa.zip` thành **Private Dataset**. Đây là keyframe của BTC;
thu nhỏ lại vẫn là dữ liệu của BTC. **Không bao giờ để public.**

### 3. Notebook

Accelerator: **GPU T4** (hoặc P100). Internet **ON**.

```python
!pip -q install -U transformers accelerate qwen-vl-utils bitsandbytes

import json, torch, pathlib
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# 3B vua thoai mai tren T4 16 GB o fp16. Muon 7B thi phai nap 4-bit —
# 7B fp16 ~15 GB trong so, cong activations la tran VRAM.
TEN = "Qwen/Qwen2.5-VL-3B-Instruct"
proc = AutoProcessor.from_pretrained(TEN)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    TEN, torch_dtype=torch.float16, device_map="auto")

GOC = pathlib.Path('/kaggle/input/<ten-dataset-cua-ban>')
hoi = json.loads((GOC / 'cau_hoi.json').read_text('utf-8'))

NHAC = ("Nhìn ảnh và trả lời câu hỏi bằng tiếng Việt. "
        "Chỉ viết ĐÁP ÁN NGẮN NHẤT (một số, hoặc một/vài từ chỉ màu). "
        "Không giải thích, không viết câu đầy đủ.")

ra = {}
for c in hoi:
    noi_dung = [{"type": "image", "image": str(GOC / 'anh' / t)} for t in c['anh']]
    noi_dung.append({"type": "text", "text": NHAC + "\n\n" + c['cau_hoi']})
    tin = [{"role": "user", "content": noi_dung}]

    from qwen_vl_utils import process_vision_info
    van_ban = proc.apply_chat_template(tin, tokenize=False, add_generation_prompt=True)
    anh, video = process_vision_info(tin)
    vao = proc(text=[van_ban], images=anh, videos=video,
               padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**vao, max_new_tokens=32, do_sample=False)
    tra = proc.batch_decode(
        [o[len(i):] for i, o in zip(vao.input_ids, out)],
        skip_special_tokens=True)[0].strip()
    ra[c['id']] = tra
    print(f"{c['id']:14} {tra[:60]!r}")

json.dump(ra, open('/kaggle/working/tra_loi.json', 'w'), ensure_ascii=False, indent=1)
```

`do_sample=False` (greedy) — chạy lại phải ra đúng kết quả cũ, nếu không thì
không so được hai lần chạy.

### 4. Chấm ở máy local

Tải `tra_loi.json` về `vqa/`, rồi:

```powershell
.venv\Scripts\python.exe scripts\41_cham_vqa.py vqa\tra_loi.json
```

Đọc **cột NHÌN**, không đọc con số gộp chung. Và đọc **từng dòng** — script in
cả câu trả lời thô để tự phủ quyết, vì `n` nhỏ đến mức một câu chấm nhầm đã đổi
tỷ lệ 25%.

### 5. Ghi lại dù kết quả ra sao

Thêm một mục `A<n>` vào [Ke_hoach_AIC2026_v4.md](Ke_hoach_AIC2026_v4.md) kèm số
liệu — **kể cả khi nó âm**, nhất là khi nó âm. A26 và ca `p1-15-qa` đã kết luận
câu đếm là **trần của model** chứ không phải lỗi truy hồi; phép đo này hoặc xác
nhận điều đó bằng model khác, hoặc lật nó. Cả hai đều đáng ghi.

> **Bảng xếp hạng công khai MÙ với câu Q&A.** Không có đường nào biết câu Q&A
> đúng hay sai qua điểm nộp thử. Chỉ tập dev nói được, nên đừng bao giờ "thử
> nộp để xem".
