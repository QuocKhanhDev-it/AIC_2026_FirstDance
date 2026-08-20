# 09 — Đo SigLIP2 trên máy khỏe

*Máy soạn tài liệu này chỉ có **7,7 GB RAM** và đã crash nhiều lần khi nạp
`ViT-SO400M-14-SigLIP2-378`. Mọi phép đo dưới đây **cần ≥ 16 GB**, và tốt nhất
là chạy trên máy GPU của Khánh.*

Kết quả đã có: **A17** trong [Ke_hoach_AIC2026_v4.md](Ke_hoach_AIC2026_v4.md).
Đừng chạy lại chỉ để đọc con số 0,3258 — nó đã đo xong. Tài liệu này là cho
**những phép đo CHƯA làm được**.

---

## 0. Cần gì trước khi bắt đầu

| Thứ | Lấy ở đâu |
| --- | --- |
| `index/master.parquet`, `clip.npy`, `objects.parquet`, `label_idf.parquet` | Drive |
| `index/clip_siglip2.npy` + `.json` | Khánh dựng 19/08, ~390 MB |
| `dev/tap_dev.jsonl` (97 câu) | trong repo |
| `pip install -r requirements-clip.txt` | ~2,5 GB |

Lần chạy đầu **tải thêm ~4,3 GB** trọng số từ HuggingFace. Có mạng chậm thì
chạy trước một lệnh nhẹ cho nó tải xong:

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe src\dense.py "một người phụ nữ đang nấu ăn" --matrix clip_siglip2.npy --k 5
```

> ⚠️ **Sidecar `.json` phải ghi `pretrained` là TAG, không phải đường dẫn.**
> Bản Khánh gửi ghi `D:\Project\AIC_2026\models\....safetensors` — máy khác nạp
> là chết ngay ở `KenhAnh.__init__`. Sửa thành `"pretrained": "webli"`.
> `08_encode.py` nay tự làm việc này, nhưng file cũ thì phải sửa tay.

---

## 1. Kiểm ma trận trước khi tin — đừng bỏ bước nào

Ba bước này rẻ và đã bắt được lỗi thật (A11: sidecar báo 18.635 dòng trong khi
ma trận chỉ có 3.135).

```powershell
.venv\Scripts\python.exe -c @"
import numpy as np, pandas as pd
m = np.load('index/clip_siglip2.npy', mmap_mode='r')
mas = pd.read_parquet('index/master.parquet')
print('shape', m.shape, '| khop bang cai:', m.shape[0] == len(mas))
b = np.zeros(len(m), bool)
for i in range(0, len(m), 20000):
    b[i:i+20000] = np.abs(np.asarray(m[i:i+20000,:8], np.float32)).sum(1) > 0
print('dong co vector that:', int(b.sum()), '/', len(m))
x = np.asarray(m[b][:5000], np.float32)
print('chuan L2 trung vi:', float(np.median(np.linalg.norm(x, axis=1))))
"@
```

Phải ra: khớp bảng cái `True`, **177.321/177.321**, chuẩn L2 ≈ 1,0000.

**Lệch hàng** thì kiểm bằng tương quan chéo với `clip.npy`, **không cần nạp
model** — xem A17 để biết cách và con số đối chứng (bản gốc 0,7065; dịch 1 hàng
0,4383). Nếu bản gốc *không* cao hơn hẳn hai nhóm đối chứng thì **dừng lại**,
ma trận lệch hàng và mọi phép đo sau đó vô nghĩa.

---

## 2. Bốn phép đo còn nợ, theo thứ tự quan trọng

### 2.1 — Có gì cộng vào SigLIP2 mà LÃI không?

Đây là câu hỏi đắt giá nhất còn để mở.

```powershell
.venv\Scripts\python.exe scripts\18_do_siglip2.py
```

Script đã đặt **mốc nền là SigLIP2** (cấu hình mạnh nhất), và so với: CLIP,
RRF(SigLIP2+CLIP), RRF(SigLIP2+objects), RRF(SigLIP2+metadata), RRF cả bốn.

**Đọc kết quả theo A14.2:** hợp nhất chỉ lãi khi các kênh **cùng tầm chất
lượng**. Kênh nào kém SigLIP2 quá xa thì cộng vào là **pha loãng** — đã đo ba
lần. Nếu RRF thua, thử lại với trọng số trước khi kết luận:

```python
hop_nhat([kq_siglip2, kq_objects], trong_so=[1.0, 0.3])
```

### 2.2 — `dedup` trên SigLIP2 toàn kho *(A11 hẹn lại đúng phép đo này)*

A11 chưa kết luận được vì ma trận thử chỉ có L21+L22 — **đúng hai nhóm ít trùng
lặp nhất kho** (0,45% và 0,27%). Nay đo được ở nơi nó có việc để làm:

```powershell
.venv\Scripts\python.exe scripts\13_do_dedup.py --matrix clip_siglip2.npy --moi-video 3 --doi-chung
```

Xem kỹ dòng **đối chứng tiếng Anh**. Nếu tập dev bỏ nhiều gấp bội đối chứng,
script sẽ tự cảnh báo — đó là dấu hiệu kênh không đọc được truy vấn, **không
phải** dấu hiệu dedup có ích.

### 2.3 — Ta đang tự chấm chặt hơn BTC bao nhiêu?

`±2s` tương đương cửa sổ **4 giây — hẹp nhất BTC từng nhắc**. BTC nói cửa sổ
thật thường là *"3 phút hoặc 5 phút"* (A9). Nên:

```powershell
.venv\Scripts\python.exe scripts\18_do_siglip2.py --dung-sai 2 90
```

| Mức | Nghĩa |
| --- | --- |
| ±2s | **SÀN** — bảo thủ, dùng để báo cáo |
| ±90s | **TRẦN** — cửa sổ 3 phút, mức rộng BTC hay dùng |

> ⚠️ **Đừng lấy con số ±90s ra so với 0,93 của đội AIC'25.** Điểm 0,93 của họ
> đo dưới cửa sổ thật của BTC (rộng hẹp lẫn lộn), lại là thi **tương tác** có
> người lái. Hai con số ±2s và ±90s chỉ nói *khoảng dao động của chính ta*.

### 2.4 — `lan_can.py` và các mức `moi_video`

`src/lan_can.py` (A8.7 #1 — kỹ thuật lợi/công cao nhất theo bài AIC'25) chưa
bao giờ đo được vì chưa có kênh ảnh nào chạy. Nay có mốc nền thật rồi.

---

## 3. Ba cái bẫy về bộ nhớ, đã vá nhưng nên biết

**1. `dense.tim()` từng nổ RAM với ma trận float16 nhiều chiều.**
`np.asarray(self.mat) @ q` buộc numpy nâng kiểu, tức cấp phát bản float32 của
TOÀN BỘ ma trận — **817 MB mỗi truy vấn**. Đã vá bằng `_nhan()` nhân theo lô.
Với `clip.npy` (512 chiều float32) lỗi này *không* lộ ra, nên nó nằm chờ đúng
lúc đổi sang model lớn.

**2. Đừng giữ hai model trong RAM cùng lúc.** `18_do_siglip2.py` quét xong một
model rồi mới nạp model kia, chỉ giữ lại `list[Candidate]`. Viết script mới thì
giữ nguyên nếp đó.

**3. Dùng `mmap=True`.** `KenhAnh(index, matrix=..., mmap=True)` để hệ điều
hành quản lý trang nhớ thay vì nạp thẳng 390 MB.

---

## 4. Ghi kết quả về đâu

Mỗi phép đo xong thì **thêm một mục `A<n>` vào
[Ke_hoach_AIC2026_v4.md](Ke_hoach_AIC2026_v4.md)** kèm số liệu, và ghi cả thứ
đã thử mà **không** hiệu quả — phần lớn giá trị của tài liệu đó nằm ở chỗ này
(A11 dedup, A14 RRF, A15 dọn rác: cả ba đều là "đã đo, không dùng").

Bốn quy tắc bắt buộc khi báo cáo, xem `CLAUDE.md`:

- `bao_cao_do_nhay()` chứ không phải điểm trung bình
- so theo cặp, luôn kèm thắng–thua–hòa và ngưỡng nhiễu
- **mốc nền là cấu hình MẠNH NHẤT hiện có** (nay là SigLIP2)
- chỉ đổi **một** thứ mỗi lần
