# 17 — Chạy trên Kaggle GPU: mã hoá ảnh model thứ hai, và sinh caption

Nối tiếp [16_dua_anh_len_kaggle.md](16_dua_anh_len_kaggle.md) — ảnh phải lên
Kaggle xong mới làm được hai việc ở đây.

Hai việc **không ngang nhau về độ chắc**, và đừng làm cùng lúc:

| | Việc A — mã hoá model thứ hai | Việc B — sinh caption (kênh 5) |
| --- | --- | --- |
| Đã có script chạy được | ✅ `08_encode.py`, đã chạy thật một lần | ⚠️ `14_sinh_caption.py --backend hf`, **chưa chạy lần nào** |
| Thời gian | **chưa đo** — bậc 0 ở mục A3 cho số thật | 3,6 giờ (tập thử) → **74 giờ** (toàn kho) |
| Rủi ro | thấp — ma trận tệ thì xoá file | cao — chưa đo, có thể vô ích |
| Thứ tự | **làm trước** | chỉ làm sau khi A xong |

**Quota Kaggle: 30 giờ GPU/tuần, mỗi phiên tối đa 12 giờ.** Còn 6 ngày. Đó là
toàn bộ ngân sách — tiêu vào việc B trước là hết chỗ cho việc A.

---

## ⚠️ Điều phải biết trước cả hai việc: Kaggle chỉ có 55% kho

L26 chưa lên Kaggle. Trên máy dựng gói cũng chưa có.

| | dòng | có ảnh |
| --- | ---: | ---: |
| 9 nhóm đã lên Kaggle | 97.731 | 97.731 |
| **L26** | **79.590** | **0** |
| tổng | 177.321 | 97.731 |

Nên mọi thứ chạy trên Kaggle lúc này **chỉ phủ 55% kho**. Hệ quả cụ thể:

* Ma trận mới sẽ có **79.590 dòng vector 0**. Đó là *đúng thiết kế* của
  `08_encode.py` (không bao giờ bỏ dòng, thiếu ảnh thì ghi 0) — không phải lỗi.
* **Nhưng so nó với `clip_siglip2.npy` — vốn phủ đủ 177.321 dòng — là so sai.**
  Bể ứng viên khác nhau thì bên bể nhỏ hơn thắng vì lý do không liên quan tới
  chất lượng model. Đo được **+0,2833** cho riêng hiệu ứng này, lớn gấp bốn lần
  thứ cần đo. Bắt buộc dùng `dense.be_chung()` khi so.
* RRF thì **không sao**: hợp nhất chỉ cộng thêm bằng chứng, không lấy đi. L26
  vẫn được kênh SigLIP2 cũ phục vụ như hiện nay.

### Làm TUẦN TỰ được — đó là thiết kế sẵn, không phải đường vòng

Không cần chờ đủ dataset. `08_encode.py` **luôn ghi đủ 177.321 dòng**; dòng nào
chưa có ảnh thì để vector 0 (cosine 0 với mọi thứ = không bao giờ được truy hồi).
Nên một ma trận encode dở vẫn là **file hợp lệ, dùng ngay được** với `dense.py`.

Có thêm nhóm nào thì encode riêng nhóm đó rồi ghép — cách này đã chạy thật một
lần rồi, `clip_siglip2.json` còn ghi `"ghep_them": ["clip_siglip2_L26d_va.npy"]`:

```powershell
.venv\Scripts\python.exe scripts\08_encode.py --model ViT-gopt-16-SigLIP2-384 ^
    --pretrained webli --chi-video ds_L26.txt --out index\clip_gopt_L26.npy

.venv\Scripts\python.exe scripts\18_ghep_ma_tran.py ^
    --chinh index\clip_gopt.npy --va index\clip_gopt_L26.npy --ghi
```

`18_ghep_ma_tran.py` chỉ ghi đè những dòng **khác 0** ở ma trận vá, không bao giờ
lấy số 0 đè lên vector thật. Shape/dtype lệch là dừng, không đoán.

> ⚠️ **Đừng vá bằng cách chạy lại đúng lệnh cũ trên cùng `--out`.**
> `08_encode.py` đánh dấu dòng thiếu ảnh là **"đã xong"**, nên lượt sau bỏ qua
> chúng và không vá được gì — mà cũng không báo lỗi. Phải đi đường
> `--chi-video` + ghép.

Việc này cũng chính là cách chia một lượt encode dài thành nhiều phiên Kaggle
dưới 12 giờ: mỗi phiên một `--chi-video`, ghép ở máy local.

---

## Chuẩn bị chung — một Private Dataset chứa `index/`

Cả hai việc đều cần `master.parquet`, mà `index/` **không nằm trong git**. Clone
repo trên Kaggle sẽ không có nó.

```powershell
$env:PYTHONIOENCODING = "utf-8"
mkdir kaggle_upload\aic2026-index -Force
copy index\master.parquet kaggle_upload\aic2026-index\
```

Rồi tạo dataset (đặt `isPrivate: true` như tài liệu 16 mục 4):

```powershell
.venv\Scripts\kaggle.exe datasets create -p kaggle_upload\aic2026-index
```

Dataset này chứa **ba** file:

| file | MB | dùng cho |
| --- | ---: | --- |
| `master.parquet` | 3,5 | mọi thứ |
| `clip.npy` | 363 | `--kiem-lech-hang` |
| `trung_lap.parquet` | 1,8 | `--kiem-lech-hang` |

Đừng chép `clip_siglip2.npy` lên (390 MB) — không việc nào ở đây cần tới nó.

> **Tên thư mục mount:** đo được ở lượt chạy 30/08, Kaggle mount tại
> `/kaggle/input/datasets/<user>/<slug>/`, **không** phải `/kaggle/input/<slug>/`.
> Đó là lý do mọi cell đều `glob` tìm file thay vì ghép đường dẫn.

---

# VIỆC A — Mã hoá bằng model ảnh thứ hai

## A1. Mục đích, viết cho hẹp

Sinh `index/clip_gopt.npy` — ma trận **thứ hai**, để RRF với `clip_siglip2.npy`.
Không ghi đè gì cả. Hai ma trận sai khác nhau ở những chỗ khác nhau thì hợp nhất
mới có lời; giống hệt nhau thì RRF không thêm được gì, và đó cũng là một kết quả
đáng ghi.

Model chọn: **`ViT-gopt-16-SigLIP2-384`** / `webli`. Cùng họ SigLIP2 nên vẫn
hiểu truy vấn tiếng Việt như model đang dùng, nhưng kiến trúc và độ phân giải
khác — đủ khác để hai bên sai lệch nhau.

## A2. Thang leo — làm nhỏ trước, to sau

Không chạy một lượt 7 giờ ngay. Leo theo bậc, **mỗi bậc là một vòng trọn vẹn**:
encode → tải về → ghép → đo. Bậc nào cũng cho ra một ma trận dùng được ngay,
nên hỏng ở bậc nào thì mất đúng bậc đó.

| bậc | phạm vi | ảnh | cộng dồn | mục đích |
| ---: | --- | ---: | ---: | --- |
| **0** | `--videos 10` (1 video mỗi nhóm) | 2.634 | — | bắt sai model / sai đường dẫn / lệch hàng |
| **1** | L23 | 2.326 | 2.326 | chạy trọn vòng, kể cả tải về + ghép + đo |
| **2** | L27, L24 | 11.695 | 14.021 | đo tốc độ thật ở quy mô có nghĩa |
| **3** | L21, L30, L22 | 24.811 | 38.832 | |
| **4** | L28, L29 | 21.454 | 60.286 | |
| **5** | L25 | 37.445 | **97.731** | nhóm to nhất đã có ảnh |
| — | *L26* | *79.590* | *177.321* | *chưa lên Kaggle* |

Bậc 0 và 1 **vứt đi được** — chúng chỉ để soát. Từ bậc 2 trở đi mới tích lũy
vào ma trận thật.

### Chọn Accelerator: **T4**, KHÔNG được chọn P100

Đây là kết luận từ lượt chạy thật 30/08, **ngược với bản đầu của tài liệu này**.

P100 là compute capability **sm_60**. PyTorch cài trên Kaggle chỉ build cho
**sm_70 trở lên**:

```text
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the
current PyTorch installation.
The current PyTorch install supports sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
```

Rồi chết ở `torch.AcceleratorError: CUDA error: no kernel image is available for
execution on the device` — **sau khi đã tải xong 7,49 GB trọng số**. T4 là
sm_75, chạy được.

Lý do bản đầu khuyên P100 vẫn đúng phần nó nói (`08_encode.py` chạy trên một
GPU nên GPU thứ hai của T4 x2 nằm không) — nhưng một GPU chạy được thắng một
GPU nhanh hơn mà không chạy.

Notebook có **cell chốt GPU chạy trước khi tải trọng số**: nó đọc
`torch.cuda.get_arch_list()` và dừng hẳn nếu `sm` của máy không nằm trong đó.
Chốt này rẻ — dừng ở giây thứ vài chục thay vì sau 7,49 GB.

### ⚠️ Đổi Accelerator trong Settings KHÔNG áp cho lượt chạy do `push`

Đã đo: đổi Settings sang **T4 x2** trong trình soạn, rồi `kernels push` — lượt
chạy **vẫn ra P100**. Settings của trình soạn thuộc về *phiên nháp*, không phải
lượt chạy do API khởi động.

**Không có đường nào qua API.** Đã thử `--accelerator nvidiaTeslaT4`: lệnh
chạy trót lọt, báo *"Kernel version 3 successfully pushed"*, và lượt chạy **vẫn
ra P100**. Cờ đó bị bỏ qua âm thầm — không cảnh báo, không lỗi. SDK cũng không
lộ danh sách giá trị hợp lệ để tra.

Nên đường duy nhất chắc chắn là **chạy từ trình soạn**:

1. Mở https://www.kaggle.com/code/duyanhdz2412/aic2026-encode
2. **Settings → Accelerator → GPU T4 x2**
3. **Save Version → Save & Run All**

`kernels push` chỉ dùng để **cập nhật mã**, đừng dùng để khởi động lượt chạy
thật — nó luôn lấy GPU mặc định.

Internet **ON**. **Add Input**: `aic2026-index` + 9 dataset ảnh.

## A3. Bậc 0 — dựng notebook và soát

> **Notebook đã dựng sẵn, khỏi chép tay từng cell.** Bản đầy đủ ở
> [`notebooks/kaggle_encode.ipynb`](../notebooks/kaggle_encode.ipynb) — tải lên
> Kaggle bằng *File → Import Notebook*, hoặc đẩy thẳng:
>
> ```powershell
> .venv\Scripts\kaggle.exe kernels push -p kaggle_upload\aic2026-encode
> ```
>
> Nó đã gắn sẵn 10 dataset, bật GPU + Internet, đặt `is_private: true`.
>
> ⚠️ **`kernels push` KHÔNG chỉ lưu — nó là "Save & Run All".** Đẩy lên là
> Kaggle chạy ngay toàn bộ notebook và bắt đầu tiêu quota GPU, không hỏi gì.
> Muốn xem trước rồi mới chạy thì dùng *File → Import Notebook* trên giao diện.
>
> Lượt chạy do `push` khởi động dùng **GPU mặc định của API (T4)**, không phải
> P100 — API không chọn được loại GPU. Nên tốc độ đo được từ lượt đó là **cận
> dưới**. Muốn P100 thì vào **Settings** đổi rồi chạy lại bằng tay.
>
> Theo dõi / dừng:
>
> ```powershell
> .venv\Scripts\kaggle.exe kernels status duyanhdz2412/aic2026-encode
> ```


```python
# 1. ma nguon
!git clone -q -b giai-doan-0 https://github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git repo
%cd repo
!pip -q install open_clip_torch pandas pyarrow

# 2. index/ tu Private Dataset (git khong mang theo).
#    TIM file chu KHONG doan duong dan: ten thu muc mount trong /kaggle/input
#    KHONG phai luc nao cung bang slug dataset. Doan cung thi `cp` bao
#    "No such file or directory" roi moi cell sau do chet day chuyen.
import glob, shutil, os, pathlib
print("co trong /kaggle/input:", os.listdir('/kaggle/input'))
hit = glob.glob('/kaggle/input/**/master.parquet', recursive=True)
assert hit, "khong thay master.parquet — da Add Input dataset aic2026-index chua?"
pathlib.Path('index').mkdir(exist_ok=True)
shutil.copy(hit[0], 'index/master.parquet')
print("chep tu", hit[0])

# 3. VA DUONG DAN — buoc BAT BUOC, va la buoc de quen nhat.
#    `kf_path` trong master.parquet la duong dan TUYET DOI cua may dung index
#    (`D:\Project\...`). Tren Kaggle no khong ton tai -> `08_encode.py` thay
#    KHONG CO ANH NAO va ghi ra mot ma tran toan so 0, KHONG bao loi gi.
!python scripts/12_va_duong_dan.py --roots /kaggle/input --ghi
```

### Chốt chặn phải GỌI TÊN nhóm thiếu, không chỉ in một con số tổng

Đã cắn thật ngày 30/08, trên máy một thành viên:

```text
keyframe có ảnh    97,731 → 88,635    -9,096
344 thư mục keyframe
theo nhóm L: L21, L23, L24, L25, L27, L28, L29, L30
```

**Thiếu hẳn L22** — quên `Add Input`, không phải lỗi quyền. Nhưng bước vá đường
dẫn vẫn in dấu ✅ và ghi file bình thường; dấu hiệu duy nhất là con số `-9.096`
lẫn trong một cột. Encode tiếp thì 9.096 dòng lặng lẽ thành vector 0.

`88.635` còn "trông gần đúng" — đó mới là chỗ nguy. Nên chốt phải đối chiếu
**từng nhóm** và gọi tên cái thiếu:

```python
import pandas as pd

# Số keyframe mỗi nhóm — thuộc tính của kho BTC, không đổi theo máy.
MONG_DOI = {'L21': 7800, 'L22': 9096, 'L23': 2326, 'L24': 6781, 'L25': 37445,
            'L27': 4914, 'L28': 10683, 'L29': 10771, 'L30': 7915}

m = pd.read_parquet('index/master.parquet')
co = m[m.kf_path.notna()].video_id.str[:3].value_counts().to_dict()

thieu = [n for n in MONG_DOI if co.get(n, 0) == 0]
lech = [(n, int(co[n]), c) for n, c in MONG_DOI.items()
        if co.get(n, 0) and co[n] != c]
for n, c in sorted(MONG_DOI.items()):
    print(f"  {n}  {int(co.get(n, 0)):>7,} / {c:>7,}")

assert not thieu, f"CHUA Add Input cho nhom: {', '.join(thieu)}"
assert not lech, f"So anh lech o {lech} — dataset chua giai nen xong?"
```

Notebook [`notebooks/kaggle_encode.ipynb`](../notebooks/kaggle_encode.ipynb) đã
mang sẵn cell này.

Rồi bậc 0:

> ⚠️ **`--videos 100` KHÔNG phải phép thử nhỏ.** Nó chọn 10 video mỗi nhóm L và
> encode **trọn vẹn mọi keyframe** của chúng — đo được **25.824 ảnh**, tức **26%
> toàn bộ công việc**. Dùng `--videos 10` cho bậc 0:
>
> | `--videos` | ảnh thật | keyframe trùng lặp cho `--kiem-lech-hang` |
> | ---: | ---: | ---: |
> | **10** | **2.634** | **253** |
> | 20 | 5.242 | 437 |
> | 100 | 25.824 | 2.396 |
>
> `--kiem-lech-hang` lấy mẫu `--so-mau 200`, nên 253 là đủ. Không cần nhiều hơn.

```python
!python scripts/08_encode.py --model ViT-gopt-16-SigLIP2-384 --pretrained webli \
    --videos 10 --workers 4 --batch 32 --out /kaggle/working/thu.npy

# CHOT AN TOAN cai san trong script: kiem lech hang.
# Lech hang la loi nguy hiem nhat o day — moi vector ve sai anh, cosine van
# dep, moi kiem tra cau truc van xanh, va diem thi tut ma khong ai biet vi sao.
!python scripts/08_encode.py --kiem-lech-hang /kaggle/working/thu.npy
```

Bốn thứ phải đọc trong log bậc 0, **trước khi leo tiếp**:

| đọc gì | phải là |
| --- | --- |
| dòng `GPU:` | `Tesla P100` (hoặc T4) — ra `cpu` là hỏng, sẽ chậm gấp ~50 lần |
| số chiều | **1536** — ra 1152 là đang chạy nhầm model cũ |
| `--kiem-lech-hang` | ✅ đạt |
| **ảnh/giây** | ghi lại — đây là con số cả kế hoạch dựa vào |

`--workers 4`: Kaggle cấp 4 vCPU. Đặt 8 là các tiến trình đọc ảnh giành nhau CPU.
Tràn VRAM thì hạ `--batch` xuống 16.

### Tính thời gian thật từ ảnh/giây

```text
giờ cho phần còn lại = 97.731 / (ảnh mỗi giây) / 3600
```

Trọng số model là **7,49 GB** (đo ở lượt tải 30/08), tải mất ~40 giây.

Chưa có số đo tốc độ encode trên phần cứng Kaggle — model này
~1,1 tỷ tham số, **gấp gần ba lần** SO400M (đo được 23,9 ảnh/giây trên máy dựng
index, GPU khác). Đừng suy ra từ con số đó.

Ra dưới ~4 ảnh/giây thì tổng vượt 7 giờ. Lúc đó cân nhắc
**`ViT-L-16-SigLIP2-384`** (1024 chiều, nhẹ hơn nhiều) — vẫn là model thứ hai
độc lập, vẫn hiểu tiếng Việt, và một ma trận chạy xong đáng giá hơn một ma trận
chạy dở.

### ⚠️ `!lenh` hỏng KHÔNG làm dừng notebook — Kaggle vẫn báo COMPLETE

Bẫy nguy hiểm nhất của cả tài liệu này, và nó đã cắn thật ngay lượt chạy đầu.

Cell `!python scripts/...` mà script thoát với mã khác 0 thì **cell vẫn tính là
chạy xong**, các cell sau chạy tiếp, và cả phiên được đánh dấu **`COMPLETE`**.
Lượt 30/08 "thành công" trong khi **mọi bước encode đều chết** — không có file
`.npy` nào được sinh ra, mà `kernels status` vẫn trả về `COMPLETE`.

Nên đừng bao giờ đọc trạng thái phiên để biết có thành công không. Notebook giờ
gói mọi lệnh trong `chay()`:

```python
import subprocess

def chay(lenh):
    print("$", lenh, flush=True)
    p = subprocess.run(lenh, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"ma thoat {p.returncode}: {lenh}")
    return p.stdout
```

Nghiệm thu đúng là **đếm file sinh ra**, không phải đọc trạng thái:

```powershell
.venv\Scripts\kaggle.exe kernels output duyanhdz2412/aic2026-encode -p <thu muc>
```

### `--kiem-lech-hang` cần thêm hai file

Phép kiểm này đọc `index/clip.npy` (ma trận ViT-B/32 của BTC, 363 MB) và
`index/trung_lap.parquet`. Cả hai đã được thêm vào dataset `aic2026-index`.
Thiếu chúng thì phép kiểm chết với `FileNotFoundError` — và vì bẫy ở trên,
notebook vẫn chạy tiếp như không có gì.

## A4. Bậc 1 trở đi — encode theo nhóm rồi ghép

`--chi-video` nhận file danh sách `video_id`, mỗi dòng một id. Sinh thẳng trong
notebook để luôn khớp với `master.parquet`, khỏi phải mang file qua lại:

```python
import pandas as pd
m = pd.read_parquet('index/master.parquet')

BAC = {1: ['L23'], 2: ['L27','L24'], 3: ['L21','L30','L22'],
       4: ['L28','L29'], 5: ['L25']}

def lam(bac):
    nhom = BAC[bac]
    v = sorted(m[m.video_id.str[:3].isin(nhom) & m.kf_path.notna()].video_id.unique())
    ten = f'ds_bac{bac}.txt'
    open(ten, 'w').write('\n'.join(v) + '\n')
    print(f"bac {bac}: {nhom} -> {len(v)} video")
    return ten

ten = lam(1)
```

```python
!python scripts/08_encode.py --model ViT-gopt-16-SigLIP2-384 --pretrained webli \
    --chi-video ds_bac1.txt --workers 4 --batch 32 \
    --out /kaggle/working/clip_gopt_bac1.npy
```

Tải `clip_gopt_bac1.npy` **và `clip_gopt_bac1.json`** về máy.

> ⚠️ **KHÔNG tải `master.parquet` từ Kaggle về.** File đó đã bị bước A3.3 vá
> thành đường dẫn `/kaggle/input/...`. Đè nó lên máy local là mọi thứ đọc ảnh
> chết hàng loạt. Chỉ tải `.npy` và `.json`.

### Ghép ở máy local

Bậc 1 là bậc đầu, chưa có gì để ghép — đổi tên thành ma trận chính:

```powershell
$env:PYTHONIOENCODING = "utf-8"
copy index\clip_gopt_bac1.npy  index\clip_gopt.npy
copy index\clip_gopt_bac1.json index\clip_gopt.json
```

Từ bậc 2 trở đi thì ghép. **Xem trước rồi mới `--ghi`** — mặc định không ghi:

```powershell
.venv\Scripts\python.exe scripts\18_ghep_ma_tran.py ^
    --chinh index\clip_gopt.npy --va index\clip_gopt_bac2.npy
.venv\Scripts\python.exe scripts\18_ghep_ma_tran.py ^
    --chinh index\clip_gopt.npy --va index\clip_gopt_bac2.npy --ghi
```

Script sao lưu bản cũ thành `clip_gopt.npy.truoc_khi_ghep.npy` trước khi ghi đè.
Số dòng "mới" nó in ra phải khớp số ảnh của bậc đó ở bảng A2 — lệch là dừng lại
tìm hiểu, đừng ghép tiếp lên trên.

Kiểm độ phủ cộng dồn sau mỗi lần ghép:

```powershell
.venv\Scripts\python.exe -c "import numpy as np;a=np.load('index/clip_gopt.npy',mmap_mode='r');print('co vector:',int((np.abs(a).sum(1)>0).sum()),'/',a.shape[0],'| chieu',a.shape[1])"
```

### Sửa `so_dong`/`da_encode` trong sidecar

Sidecar `clip_gopt.json` mang số của **riêng bậc cuối**, không phải cộng dồn.
`25_ma_hoa_truy_van.py` chỉ đọc `model`/`pretrained`/`chieu` nên không sai gì,
nhưng đừng trích các số kia ra tài liệu.

## A4b. ⚠️ Cache truy vấn: PHẢI sinh lại, và là một FILE RIÊNG

Đây là bước dễ quên nhất trong cả tài liệu, vì nó không nằm trong việc "mã hoá
ảnh" mà lại chặn hoàn toàn việc dùng ma trận mới.

`index/truy_van.npz` chứa vector truy vấn mã hoá bằng **tháp văn bản của
SO400M**. Model mới có tháp văn bản khác **và số chiều khác**:

| model | chiều |
| --- | ---: |
| `ViT-SO400M-14-SigLIP2-378` (đang dùng) | 1152 |
| `ViT-gopt-16-SigLIP2-384` | **1536** |
| `ViT-L-16-SigLIP2-384` | 1024 |

Không có cách nào dùng cache cũ với ma trận mới — vector 1152 chiều không nhân
được với ma trận 1536 chiều.

**Tin tốt: chỗ này KHÔNG hỏng im lặng.** `KenhAnhCache.__init__` so số chiều và
dừng hẳn với thông báo *"Sai cặp cache/ma trận"*. Nhưng đừng để nó bắt — sinh
cache **ngay trong cùng notebook** với bước A4:

```python
!python scripts/25_ma_hoa_truy_van.py --matrix clip_gopt.npy \
    --ra index/truy_van_gopt.npz --de de_p2 --tap-dev --fp16
!cp index/truy_van_gopt.npz /kaggle/working/
```

`--matrix clip_gopt.npy` là đủ: script đọc tên model và số chiều từ **sidecar
`clip_gopt.json`** mà `08_encode.py` vừa ghi, không cần khai lại tay.

**Cùng notebook, không phải notebook khác** — model ~4 GB đã tải và nằm sẵn ở đó.
Tách ra là tải lại lần nữa, tốn quota cho đúng một việc chỉ mất vài phút.

Về máy thì **giữ cả hai file cache**, đừng đè:

```text
index/truy_van.npz        <- đi với clip_siglip2.npy   (1152)
index/truy_van_gopt.npz   <- đi với clip_gopt.npy      (1536)
```

RRF hai ma trận nghĩa là **chạy cả hai kênh**, nên cần cả hai cache cùng lúc.

> `25_ma_hoa_truy_van.py` bỏ `model.visual` trước khi mã hoá nên chỉ cần tháp
> văn bản. Về lý thuyết máy 7,7 GB có thể kham `--fp16`, nhưng ngưỡng RAM cho
> 1536 chiều là **~10 GB (ước, chưa đo)** — đừng thử ở máy chính, chốt đó sinh
> ra vì nạp model đã làm đứng máy hai lần.

## A5. Nghiệm thu

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe scripts\18_do_siglip2.py --matrix index\clip_gopt.npy
```

Rồi so RRF hai ma trận với **mốc nền là cấu hình mạnh nhất hiện có**
(`clip_siglip2.npy` một mình, có RRF kênh 3), **trên `dev/tap_de_that.jsonl`** —
không dùng `tap_dev.jsonl`, tập đó thổi phồng kênh 1 lên 2,3 lần (A45).

Đọc `bao_cao_do_nhay()`, không đọc điểm trung bình. Đảo dấu giữa hai mức dung
sai = **không kết luận được**, không phải "hơi hơn".

Ghi kết quả thành một mục `A<n>` trong
[Ke_hoach_AIC2026_v4.md](Ke_hoach_AIC2026_v4.md) — **kể cả khi âm**.

---

# VIỆC B — Sinh caption (kênh 5)

## B1. Đọc con số này trước khi quyết định làm

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe scripts\14_sinh_caption.py --uoc-tinh --backend hf --chon co-anh
```

| phạm vi | ảnh | giờ (ước) |
| --- | ---: | ---: |
| video mà `tap_de_that.jsonl` đụng tới | 8.622 | **3,6** |
| 9 nhóm đã lên Kaggle | 97.731 | 40,7 |
| toàn kho | 177.321 | 73,9 |

**Toàn kho vượt xa quota 30 giờ/tuần.** Nên chỉ có một cách vào việc hợp lý:
**chạy tập thử 8.622 ảnh, đo, rồi mới quyết**. Caption cho một phần kho ngẫu
nhiên là gần như vô dụng — nó chỉ giúp khi khung đáp án nằm trong phần đã sinh.

⚠️ 1,5 giây/ảnh là **ước, chưa đo**. Chạy `--n 40` đọc tốc độ thật trước.

## B2. Backend `hf` — vì sao phải có

`--backend ollama` cần một server Ollama; Kaggle không có.
`--backend gemini` có trần lượt/ngày, hàng chục nghìn ảnh sẽ nghẽn.

`--backend hf` nạp model **thẳng trong tiến trình** và gộp lô thật. Đường chạy
này tuần tự, **không dùng `--luong`** — nhiều thread giành một GPU thì chậm hơn
chứ không nhanh hơn. Nút tăng tốc là `--batch` và `--diem-anh`.

## B3. Notebook

Accelerator **P100** (hoặc T4). Internet **ON**. Input: `aic2026-index` + 9 dataset ảnh.

```python
!git clone -q -b giai-doan-0 https://github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git repo
%cd repo
!pip -q install -U transformers accelerate qwen-vl-utils pandas pyarrow
# chep master.parquet — dung cell TIM FILE o muc A3, dung doan duong dan
import glob, shutil, pathlib
pathlib.Path('index').mkdir(exist_ok=True)
shutil.copy(glob.glob('/kaggle/input/**/master.parquet', recursive=True)[0],
            'index/master.parquet')
!python scripts/12_va_duong_dan.py --roots /kaggle/input --ghi

# do toc do THAT truoc, 40 anh
!python scripts/14_sinh_caption.py --backend hf --chon tap:dev/tap_de_that.jsonl \
    --n 40 --batch 8
```

Đọc dòng `giây/ảnh` in ra. Nhân với 8.622 → giờ thật. Rồi chạy thật:

```python
!python scripts/14_sinh_caption.py --backend hf --chon tap:dev/tap_de_that.jsonl \
    --batch 8 --diem-anh 512 --so-chu 180
!cp index/caption.jsonl index/caption.parquet /kaggle/working/
```

`caption.jsonl` ghi **nối từng dòng** — phiên bị cắt vẫn giữ nguyên phần đã làm,
chạy lại là bỏ qua ảnh đã xong.

Tràn VRAM thì hạ `--batch` xuống 4, rồi `--diem-anh` xuống 256. Hạ `--diem-anh`
rẻ hơn nhiều, nhưng đổi lại model nhìn ảnh mờ hơn.

## B4. Nghiệm thu — đọc caption bằng MẮT trước khi đo

```powershell
.venv\Scripts\python.exe -c "import pandas as pd;d=pd.read_parquet('index/caption.parquet');print(d.caption.str.len().describe());[print('-',c[:150]) for c in d.caption.sample(15,random_state=0)]"
```

Ba thứ làm caption thành vô dụng mà **mọi kiểm tra cấu trúc đều xanh**:

* **Lẫn chữ Hán/Nhật/Hàn.** Đo được ở `qwen2.5vl:7b`: ~12% caption, và **không
  ngẫu nhiên** — dồn vào một loại cảnh (bàn dẫn chương trình tin tức) lặp qua
  nhiều khung liên tiếp. `chay_hf` xoá thẳng ký tự Hán, nhưng vẫn phải soi xem
  còn lại có ra câu tiếng Việt không.
* **Lặp vô hạn.** Từng gặp một caption 9.872 ký tự lặp cùng một câu hàng trăm
  lần. `--so-chu 180` chặn trần, nhưng caption dài bất thường vẫn đáng nghi.
* **Caption tiếng Anh.** BM25 khớp mặt chữ — caption Anh + truy vấn Việt = khớp
  đúng 0 token, kênh im lặng trả rỗng. Đúng bộ phim kênh 1 được **0,0000** vì
  CLIP mù tiếng Việt (A10).

Xem xong mới đo, bằng `bm25.KenhVanBan.tu_bang_khung` trên
`dev/tap_de_that.jsonl`.

---

## Việc này KHÔNG phải thứ đáng ưu tiên nhất — nói thẳng

A46 đo được: **video đúng chỉ nằm trong top-100 ở 19/30 gói (63%)**. Đó là trần
cứng của mọi thứ hậu xử lý.

Việc A tấn công đúng con số đó — thêm một ma trận độc lập là cơ hội kéo video
đúng vào bể. Việc B thì **chưa rõ**: caption có thể kéo 63% lên, cũng có thể chỉ
xếp lại bể sẵn có, và ta chưa có số nào để biết. Đó là lý do B chỉ được chạy ở
quy mô tập thử cho tới khi đo xong.

Tiêu 40 giờ quota vào B trước khi đo là cách chắc chắn nhất để hết 6 ngày mà
không biết mình có tiến bộ hay không.
