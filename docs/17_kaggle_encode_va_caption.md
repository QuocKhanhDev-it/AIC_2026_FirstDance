# 17 — Chạy trên Kaggle GPU: mã hoá ảnh model thứ hai, và sinh caption

Nối tiếp [16_dua_anh_len_kaggle.md](16_dua_anh_len_kaggle.md) — ảnh phải lên
Kaggle xong mới làm được hai việc ở đây.

Hai việc **không ngang nhau về độ chắc**, và đừng làm cùng lúc:

| | Việc A — mã hoá model thứ hai | Việc B — sinh caption (kênh 5) |
| --- | --- | --- |
| Đã có script chạy được | ✅ `08_encode.py`, đã chạy thật một lần | ⚠️ `14_sinh_caption.py --backend hf`, **chưa chạy lần nào** |
| Thời gian | ~3–4 giờ | 3,6 giờ (tập thử) → **74 giờ** (toàn kho) |
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

L26 lên sau thì vá riêng, không chạy lại toàn kho:

```powershell
.venv\Scripts\python.exe scripts\08_encode.py --chi-video ds_L26.txt --out index\clip_gopt_L26.npy
.venv\Scripts\python.exe scripts\18_ghep_ma_tran.py ...
```

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

Chỉ 3,5 MB. **Chỉ `master.parquet`** — đừng chép `clip*.npy` lên (390 MB, không
việc nào ở đây cần tới ma trận cũ).

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

## A2. Notebook

Accelerator **GPU T4 x2** (hoặc P100). Internet **ON**.
**Add Input**: dataset `aic2026-index` + **9 dataset ảnh** `aic2026-keyframes-l21`
… `-l30`.

```python
# 1. ma nguon
!git clone -q -b giai-doan-0 https://github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git repo
%cd repo
!pip -q install open_clip_torch pandas pyarrow

# 2. index/ tu Private Dataset (git khong mang theo)
!mkdir -p index && cp /kaggle/input/aic2026-index/master.parquet index/

# 3. VA DUONG DAN — buoc BAT BUOC, va la buoc de quen nhat.
#    `kf_path` trong master.parquet la duong dan TUYET DOI cua may dung index
#    (`D:\Project\...`). Tren Kaggle no khong ton tai -> `08_encode.py` thay
#    KHONG CO ANH NAO va ghi ra mot ma tran toan so 0, KHONG bao loi gi.
!python scripts/12_va_duong_dan.py --roots /kaggle/input --ghi
```

Bước 3 phải in ra ~97.731 ảnh. In ra 0 thì dừng lại — dataset ảnh chưa mount,
hoặc còn *processing*. Kiểm lại ngay, đừng chạy tiếp:

```python
import pandas as pd
m = pd.read_parquet('index/master.parquet')
n = m.kf_path.notna().sum()
print(f"co anh: {n:,} / {len(m):,}")
assert n > 90_000, "duong dan chua va duoc — dung lai, dung encode"
```

## A3. Chạy thử 100 video TRƯỚC

Đừng bắt đầu bằng lượt 3 giờ. Lượt thử này để bắt sai model, sai đường dẫn, sai
số chiều — những thứ mà lượt dài cũng không tự báo, chỉ tốn 3 giờ rồi mới lộ.

```python
!python scripts/08_encode.py --model ViT-gopt-16-SigLIP2-384 --pretrained webli \
    --videos 100 --out index/clip_gopt_thu.npy

# CHOT AN TOAN cai san trong script: kiem lech hang.
# Lech hang la loi nguy hiem nhat o day — moi vector ve sai anh, cosine van
# dep, moi kiem tra cau truc van xanh, va diem thi tut ma khong ai biet vi sao.
!python scripts/08_encode.py --kiem-lech-hang index/clip_gopt_thu.npy
```

Đọc tốc độ **ảnh/giây** script in ra rồi nhân lên: `97731 / (ảnh/giây) / 3600` =
số giờ thật. Vượt 12 giờ thì chia làm nhiều phiên (`--chi-video`), đừng để
Kaggle cắt ngang.

## A4. Chạy thật

```python
!python scripts/08_encode.py --model ViT-gopt-16-SigLIP2-384 --pretrained webli \
    --out /kaggle/working/clip_gopt.npy --batch 64
```

Script tự ghi checkpoint mỗi 10.000 ảnh. Phiên bị cắt thì chạy lại đúng lệnh đó.

Tải `clip_gopt.npy` **và `clip_gopt.json` cạnh nó** về `index/`.

> ⚠️ **KHÔNG tải `master.parquet` từ Kaggle về.** File đó đã bị bước A2.3 vá
> thành đường dẫn `/kaggle/input/...`. Đè nó lên máy local là mọi thứ đọc ảnh
> chết hàng loạt. Chỉ tải `.npy` và `.json`.

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

Accelerator **GPU T4 x2**. Internet **ON**. Input: `aic2026-index` + 9 dataset ảnh.

```python
!git clone -q -b giai-doan-0 https://github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git repo
%cd repo
!pip -q install -U transformers accelerate qwen-vl-utils pandas pyarrow
!mkdir -p index && cp /kaggle/input/aic2026-index/master.parquet index/
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
