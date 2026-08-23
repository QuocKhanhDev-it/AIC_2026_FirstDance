# Sinh `index/truy_van.npz` trên Kaggle — ~10 phút

*Soạn 23/08/2026. Việc này mở khoá ba phép đo đang treo, và kênh 1 cho giao diện `web/`.*

## Vì sao đáng làm trước mọi thứ khác

Model chỉ làm **đúng một việc**: biến câu chữ thành vector. Ma trận ảnh
177.321 × 1152 thì đã nằm sẵn trên đĩa mọi máy. Mà tập truy vấn là **hữu hạn
và biết trước** — 593 chuỗi.

Mã hoá 593 chuỗi đó một lần ra file **~2,5 MB**, rồi máy nào cũng chạy kênh 1
bằng numpy thuần: **không nạp model, không cần 6,5 GB RAM**. Máy 7,7 GB đang tắc
ở đúng chỗ này, và nó tắc **ba phép đo cùng lúc**:

| Đang treo | Cần gì |
| --- | --- |
| `36_do_cua_so.py` — chấm theo cửa sổ (A38) | `--cache` |
| `35_do_chuoi_trake.py` — prior khoảng cách (A39) | `--cache` |
| Kênh 1 trong giao diện `web/` | `--cache` |

## Hai điều khiến việc này nhẹ hơn tưởng

**1. KHÔNG cần tải ma trận 390 MB lên Kaggle.** Script chỉ đọc `clip_siglip2.npy`
để lấy `shape[1]`, mà sidecar 605 byte đã ghi sẵn `"chieu": 1152`. Đã vá ở A40 —
sidecar được đọc trước.

**2. KHÔNG cần GPU.** `ma_hoa()` chạy hoàn toàn trên CPU (không `.cuda()` nào,
và nó xoá `model.visual` ngay sau khi nạp). Chọn Accelerator **None** — khỏi tốn
quota 30 h GPU/tuần, và khỏi chờ hàng đợi GPU.

---

## Các bước

### 0. Tạo notebook

Kaggle → **Create** → **New Notebook**. Bên phải, mở **Settings**:

| Mục | Đặt |
| --- | --- |
| **Internet** | **On** ⚠️ |
| Accelerator | **None** (CPU) |
| Language | Python |

⚠️ **Internet phải bật**, nếu không `pip install` và tải trọng số đều chết.
Kaggle đòi **xác minh số điện thoại** mới cho bật — làm trước, đừng để tới lúc
chạy mới phát hiện.

### 1. Lấy mã nguồn

```python
!git clone --depth 1 https://github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git /kaggle/working/repo
%cd /kaggle/working/repo
!git log --oneline -1
```

Repo công khai nên clone thẳng được. `dev/` (bộ đề + tập dev) nằm trong git,
nên **không phải upload gì cả**.

### 2. Cài open_clip

```python
!pip -q install open_clip_torch==3.3.0
```

Ghim đúng phiên bản đang chạy ở máy local. Bản cũ hơn **không biết tag
`ViT-SO400M-14-SigLIP2-378`** và sẽ báo "model not found".

### 3. Dựng sidecar

`index/` không nằm trong git, nhưng script chỉ cần ba khoá:

```python
import json, pathlib
pathlib.Path("index").mkdir(exist_ok=True)
pathlib.Path("index/clip_siglip2.json").write_text(json.dumps({
    "model": "ViT-SO400M-14-SigLIP2-378",
    "pretrained": "webli",
    "chieu": 1152,
}), encoding="utf-8")
print(pathlib.Path("index/clip_siglip2.json").read_text("utf-8"))
```

`pretrained: "webli"` là **đúng trọng số** đã dựng ma trận — sidecar gốc ghi
đường dẫn safetensors của máy dựng index, nhóm đã đối chiếu và đổi về tag này
(xem A17). Ghi sai chỗ này thì vector truy vấn không cùng không gian với ma
trận ảnh, và **triệu chứng là điểm tụt chứ không phải lỗi** — loại hỏng tệ nhất.

### 4. Mã hoá

```python
!python scripts/25_ma_hoa_truy_van.py \
    --de dev/SOTUYEN1-bo-de-thi --tap-dev --gop --fp16
```

* `--de dev/SOTUYEN1-bo-de-thi` — 25 gói đề thật
* `--tap-dev` — 186 câu dev, trong đó 23 câu đề thật
* `--gop` ⚠️ — **cộng thêm** vào file có sẵn. Chạy trần thì **dựng lại từ đầu**
  và mất các chuỗi đã mã hoá trước. Cùng cái bẫy đã làm tập dev tụt 105 → 24 câu.
* `--fp16` — nạp trọng số nửa độ chính xác, tính vẫn ở fp32

Chờ vài phút (phần lớn là tải ~1,7 GB trọng số). Cuối cùng phải in ra số chuỗi
đã mã hoá — mong đợi **593**.

### 5. Kiểm ngay trên Kaggle trước khi tải về

```python
import numpy as np, json, pathlib
z = np.load("index/truy_van.npz", allow_pickle=False)
cau, vec = z["cau"], z["vec"]

print("số chuỗi :", len(cau))
print("ma trận  :", vec.shape, vec.dtype)
print("kích thước:", round(pathlib.Path("index/truy_van.npz").stat().st_size/1e6, 2), "MB")
print("ghi chú  :", json.loads(str(z["ghi_chu"])))

# Chuẩn hoá L2: MỌI vector phải có chuẩn ≈ 1. Lệch là mã hoá sai.
n = np.linalg.norm(vec, axis=1)
print(f"chuẩn: min {n.min():.4f}  max {n.max():.4f}")
assert len(cau) == len(vec) and vec.shape[1] == 1152
assert abs(n - 1).max() < 1e-3, "vector CHƯA chuẩn hoá L2 — dừng lại"
print("\n✅ hợp lệ")
```

Bốn điều phải đúng: **593 chuỗi**, **chiều 1152**, **chuẩn ≈ 1,0000**, và
`ghi_chu["model"]` là `ViT-SO400M-14-SigLIP2-378`. File **~2,5 MB**
(593 × 1152 × 4 byte = 2,73 MB thô; embedding gần như không nén được).
Vài chục MB là dấu hiệu sai.

### 6. Tải về

`index/truy_van.npz` hiện trong panel **Output** bên phải → nút tải xuống.
File ~2,5 MB, gửi qua chat cũng được.

Đặt vào `c:\Code\aic2026\index\truy_van.npz`.

---

## Nghiệm thu ở máy local — đừng bỏ bước này

File có thể tải về đúng mà vẫn **sai không gian vector**, và triệu chứng chỉ là
điểm thấp. Nên chạy một câu đã biết đáp án:

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe web\server.py --cache index\truy_van.npz
```

Mở <http://127.0.0.1:8000> → bộ đề `SOTUYEN1-bo-de-thi` → câu **`query-p1-1-kis`**.

Đáp án nhóm đã soát tay: **`L30_V046`, keyframe 95**.

| Thấy gì | Nghĩa là |
| --- | --- |
| `L30_V046` ở top-10 | ✅ cache đúng, đi tiếp |
| Không thấy trong top-100 | ❌ nhiều khả năng sai `pretrained` — dựng lại sidecar, mã hoá lại |
| Băng cảnh báo "kênh 1 đang TẮT" vẫn hiện | ❌ máy chủ không thấy file — kiểm đường dẫn |

Xong bước này thì chạy hai phép đo đang treo:

```powershell
.venv\Scripts\python.exe scripts\36_do_cua_so.py --cache index\truy_van.npz --kiem-moc
.venv\Scripts\python.exe scripts\35_do_chuoi_trake.py --cache index\truy_van.npz
```

`--kiem-moc` phải in `TRÙNG kenh.tim()`. **LỆCH thì dừng lại** — mốc nền sai thì
mọi so sánh sau đó vô nghĩa.

---

## Ba cái bẫy của riêng Kaggle

**1. Session không lưu gì.** Notebook hết giờ (9 h) hoặc bạn đóng tab là
`/kaggle/working` bay sạch. Tải file về **ngay** sau khi sinh xong, đừng để
"lát nữa".

**2. Kaggle là nơi SINH HIỆN VẬT, không phải nơi CHỨA phép đo.** Mọi con số đo
được phải chép vào `docs/Ke_hoach_AIC2026_v4.md` thành một mục `A<n>`, tái lập
được từ hiện vật. Đo trên Kaggle rồi để đó là mất.

**3. Nếu về sau upload `index/` lên Kaggle làm Dataset thì để PRIVATE.** Đó là
đặc trưng dẫn xuất từ dữ liệu thi của BTC. Và dùng **Dataset**, không phải
**Models** — trang Models (`kagglehub.model_upload`) dành cho công bố model, sai
chỗ.

---

## Cache đã sinh TRƯỚC 23/08 thì phải sinh lại

Cache chứa **đúng những chuỗi `tach_su_kien` sinh ra**. Bản trước A40 tách
`query-p1-16-trake` thành **4 mệnh đề thay vì 3** — cache cũ mang mệnh đề sai
cho mọi câu TRAKE viết theo kiểu đề thật (`E1` không dấu hai chấm).

Kiểm nhanh sau khi clone:

```python
!python -c "import sys;sys.path.insert(0,'src');import run;print(len(run.tach_su_kien(open('dev/SOTUYEN1-bo-de-thi/query-p1-16-trake.txt',encoding='utf-8').read())))"
```

Phải in **3**. In 4 là repo chưa có bản vá A40 — `git pull` lại.
