# Bảng kế hoạch — encode lại CLIP trên máy có GPU

*Soạn 2026-08-12. Dành cho **một người**: chủ máy có card NVIDIA (RTX 2060
Super 8 GB). Năm máy còn lại không tham gia — xem [§6](#6-ba-điều-phải-hiểu-trước-khi-bấm-chạy).*

**Mục tiêu:** sinh ma trận embedding mạnh hơn `ViT-B/32` 512 chiều của BTC, để
**ghép RRF** với ma trận cũ. Căn cứ: A8.4 của
[Ke_hoach_AIC2026_v4.md](Ke_hoach_AIC2026_v4.md) — bước nhảy 0,86 → 0,93
điểm/câu của đội AIC'25 đến từ đúng chỗ này.

---

## 1. BẢNG KẾ HOẠCH CHÍNH

| # | Việc | Ai | Thời gian | Cần trước | Đầu ra | Xong khi |
| --- | --- | --- | --- | --- | --- | --- |
| **1** | Viết `scripts/08_encode.py` theo spec §5 | ai cũng được, **không cần GPU** | ~2 h | — | `scripts/08_encode.py` | chạy được `--help` |
| **2** | Cài `torch` bản CUDA | chủ máy GPU | ~30 ph | — | môi trường | `cuda.is_available()` → `True` |
| **3** | Loại `aic_data` khỏi Defender | chủ máy GPU | 2 ph | quyền admin | — | `Add-MpPreference` chạy xong |
| **4** | Tải **~3,5 GB** keyframe của 100 video phân tầng | chủ máy GPU | tùy mạng | — | ảnh trên đĩa | 100/100 video có ảnh |
| **5** | Chạy thử 100 video | chủ máy GPU | **~4 ph** | 1,2,3,4 | `clip_siglip2_thu.npy` | in ra tốc độ thật |
| **6** | **Kiểm lệch hàng `row_id`** | chủ máy GPU | ~1 ph | 5 | báo cáo | cặp trùng lặp vẫn trùng |
| **7** | **Soạn tập dev trên đúng 100 video đó** | **Khánh** | ~1 ngày | 4 | 30–50 truy vấn | có đáp án chuẩn |
| **8** | Đo 3 cấu hình A/B/C | Khánh | ~2 h | 6, 7 | bảng điểm | C hoặc B thắng A |
| **9** | Tải nốt **30,5 GB** keyframe | chủ máy GPU | tùy mạng | **8 phải thắng** | ảnh trên đĩa | 873/873 video |
| **10** | Chạy toàn kho | chủ máy GPU | **~2 h** | 9 | `clip_siglip2.npy` 409 MB | `(177321, 1152)` |
| **11** | Kiểm lệch hàng lần cuối + đẩy Drive | chủ máy GPU | ~15 ph | 10 | Drive + `.json` | cả nhóm tải được |

> **Việc 1, 2, 3, 4 làm song song được NGAY HÔM NAY.**
> **Việc 7 (tập dev) là đường găng** — không có nó thì việc 9–11 là đốt điện mù.
> **Việc 9 không được bắt đầu trước khi việc 8 thắng.**

---

## 2. BẢNG CHỐT MODEL

GFLOPs tính từ `vision_cfg` thật của `open_clip 3.3.0`, không chép ở đâu về.

| Model | Chiều | Ảnh | GFLOPs | so L/14 | Tiếng Việt | Ước ảnh/s | Toàn kho | VRAM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ViT-L-14-quickgelu` | 768 | 224 | 81 | 1,0× | ✗ | ~110 | ~27 ph | ~2 GB |
| `ViT-L-16-SigLIP2-384` | 1024 | 384 | ~191 | 2,4× | ✓ | ~47 | ~1 h | ~3 GB |
| ⭐ **`ViT-SO400M-14-SigLIP2-378`** | **1152** | 378 | **~347** | 4,3× | **✓** | **~26** | **~1 h 55** | ~3–4 GB |
| `ViT-H-14-378-quickgelu` `dfn5b` | 1024 | 378 | 503 | 6,2× | ✗ | ~18 | ~2 h 45 | ~4–5 GB |
| `ViT-gopt-16-SigLIP2-384` | 1536 | 384 | ~700+ | 9× | ✓ | ~13 | ~3 h 45 | **~6 GB** ⚠ |

### Vì sao chọn `ViT-SO400M-14-SigLIP2-378` + `webli`

| # | Lý do | Sức nặng |
| --- | --- | --- |
| 1 | **Nhận thẳng truy vấn tiếng Việt.** `ViT-L/14` và `ViT-H/14` chỉ biết tiếng Anh → phải dịch truy vấn trước khi encode, thêm một khâu và một chỗ hỏng. Đề thi ra bằng tiếng Việt | ★★★ |
| 2 | **1152 chiều = đúng model SigLIP2 của bài báo.** Trong `open_clip` chỉ dòng SO400M cho ra 1152 | ★★ |
| 3 | **Rẻ hơn `ViT-H/14` 1,45×** (347 vs 503 GFLOPs) mà chiều lại cao hơn | ★★ |
| 4 | **Đã có sẵn nửa "CLIP".** Bài fuse CLIP + SigLIP2; ta có CLIP (`ViT-B/32` của BTC — yếu hơn nhưng **miễn phí và đã kiểm chứng**). Chỉ cần encode SigLIP2 là đủ cặp → **không phải chạy hai model** | ★★★ |

| Loại | Vì sao không |
| --- | --- |
| `ViT-gopt` | ~1,1 tỷ tham số, riêng trọng số fp16 đã 4–5 GB trên card 8 GB → phải hạ batch rất thấp, mất hết lợi thế |
| `ViT-H-14-378` `dfn5b` | Tái lập đúng bài báo được, chỉ tốn thêm ~1 h — nhưng **không giải quyết vấn đề tiếng Việt** |

> ⚠️ Cột "ước ảnh/s" là **ƯỚC TÍNH** (giả định ~9 TFLOPS fp16 hiệu dụng), chưa
> đo. Việc 5 sẽ thay bằng số đo thật. **Lệch quá 2 lần → gần như chắc chắn đang
> chạy nhầm torch bản CPU** (xem §4 hàng đầu).

---

## 3. BẢNG BA CHỐT AN TOÀN — BẮT BUỘC

| # | Chốt | Hỏng thế nào | Luật cứng | Kiểm bằng |
| --- | --- | --- | --- | --- |
| **A** | **Lệch hàng `row_id`** | Gặp ảnh hỏng rồi **bỏ qua** dòng đó → **mọi dòng sau dịch một bậc**, không có lỗi nào báo. Đúng loại lỗi ta mất cả Giai đoạn 0 để loại trừ | Thiếu ảnh thì **GHI VECTOR 0**, tuyệt đối không bỏ dòng. Chép `01_build_index.py:121` | `index/trung_lap.parquet` — xem dưới |
| **B** | **Sai biến thể checkpoint** | Bẫy A6 lặp lại: `ViT-B-32` với trọng số cần `-quickgelu` làm cosine tụt **0,9913 → 0,9513**, không lỗi nào | `assert` tên tag; text encoder lấy từ **đúng lượt** `create_model_and_transforms` | ghi model + pretrained ra `.json` kèm |
| **C** | **Quên chuẩn hóa L2** | `M @ q` không còn là cosine. `.npy` của BTC chuẩn hóa sẵn nên ta chưa từng phải nghĩ tới | Chuẩn hóa trước khi lưu — chép `01_build_index.py:196-198` | `np.linalg.norm(M,axis=1)` ≈ 1,0 |

### Phép kiểm chốt A — rẻ và mạnh

`index/trung_lap.parquet` đã có sẵn danh sách cặp keyframe **cùng video,
cosine ≥ 0,99** dưới `ViT-B/32`. Hai ảnh thật sự giống nhau thì **model nào
cũng thấy giống**.

| Kết quả trên ma trận mới | Kết luận |
| --- | --- |
| Cặp đó vẫn cosine cao | ✅ hàng khớp, đi tiếp |
| Cặp đó hết giống nhau | ❌ **đã lệch hàng — DỪNG NGAY** |

Không cần thêm dữ liệu gì. Đây là lý do việc 6 phải chạy trước việc 9.

---

## 4. BẢNG RỦI RO

| Rủi ro | Dấu hiệu sớm | Phương án | Mức |
| --- | --- | --- | --- |
| **Chạy nhầm torch bản CPU** | tốc độ ~0,5 ảnh/s thay vì ~26 | Bản `+cpu` **vẫn chạy bình thường**, chỉ chậm 100× và không báo gì. Kiểm bằng lệnh ở §5 việc 2 | 🔴 hay gặp nhất |
| **Lệch hàng `row_id`** | phép kiểm §3A thất bại | Ghi vector 0, không bỏ dòng | 🔴 chết người |
| **Không có tập dev** | không đo được gì | Encode xong vẫn **không biết có nên dùng**. Việc 7 là đường găng | 🔴 chặn thật |
| Dùng `bfloat16` | chậm hơn cả fp32 | Turing **không có** bf16 phần cứng → dùng `torch.float16` | 🟡 |
| `CUDA out of memory` | lỗi nổ ngay lập tức | Hạ `--batch` 16 → 8 → 4. **Không âm thầm nên không nguy hiểm** | 🟢 |
| Tải 30,5 GB rồi mới biết vô ích | — | Việc 5–8 chặn trước, chỉ cần 3,5 GB | 🟢 đã chặn |
| CPU giải nén JPEG thành nút cổ chai | GPU nhàn, tốc độ thấp | `DataLoader` nhiều `num_workers` | 🟡 |
| Mất điện giữa lượt chạy 2 giờ | — | Checkpoint mỗi ~10.000 dòng | 🟡 |
| Quên ghi lại model đã dùng | 3 tuần sau không ai biết ma trận sinh bằng gì | Ghi `.json` kèm — chốt B | 🟡 |

---

## 5. BẢNG LỆNH THEO TỪNG VIỆC

| Việc | Lệnh | Phải thấy gì |
| --- | --- | --- |
| **2** | `pip uninstall -y torch`<br>`pip install torch --index-url https://download.pytorch.org/whl/cu124`<br>`pip install -r requirements-clip.txt` | cài xong không lỗi |
| **2 kiểm** | `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` | `True NVIDIA GeForce RTX 2060 SUPER`<br>**ra `False` thì DỪNG** |
| **3** | `Add-MpPreference -ExclusionPath "C:\Code\aic_data"` | *(cần admin)* |
| **5** | `python scripts\08_encode.py --model ViT-SO400M-14-SigLIP2-378 --pretrained webli --videos 100 --batch 16 --out index\clip_siglip2_thu.npy` | in tốc độ thật ảnh/giây |
| **6** | `python scripts\08_encode.py --kiem-lech-hang index\clip_siglip2_thu.npy` | cặp trùng lặp vẫn trùng |
| **10** | `python scripts\08_encode.py --model ViT-SO400M-14-SigLIP2-378 --pretrained webli --batch 16 --fp16 --out index\clip_siglip2.npy` | `(177321, 1152)` |

> 🟢 **CẬP NHẬT 16/08 — việc 8 đã có câu trả lời sớm, đo trên CPU.** Không cần
> chờ GPU nữa: `ViT-B-16-SigLIP2-256` (bản **nhỏ nhất** họ SigLIP2, chạy được
> trên CPU) đã encode 11 video và so ba cấu hình trên cùng bể —
>
> | Cấu hình | ±2s | ±15s |
> | --- | --- | --- |
> | CLIP + tiếng Việt | 0,0095 | 0,0857 |
> | CLIP + bản dịch tay | 0,8190 | 0,8952 |
> | **SigLIP2 + tiếng Việt** | **0,8571** | **0,9429** |
>
> **Thắng 21/21 câu**, không thua câu nào. Xem A10.3 của kế hoạch.
>
> → **Việc 9 và 10 nay được phép chạy.** Điều kiện "chỉ tải 30,5 GB sau khi
> việc 8 thắng" đã thoả. Dùng `ViT-SO400M-14-SigLIP2-378` như đã chốt ở §2 —
> nó mạnh hơn bản vừa thử nhiều.

### Việc 8 — ba cấu hình phải so

| Cấu hình | Là gì | Ý nghĩa nếu thắng |
| --- | --- | --- |
| **A** | `ViT-B/32` của BTC *(có sẵn, miễn phí)* | mốc nền |
| **B** | SigLIP2 đơn thuần | model mới đủ mạnh để dùng một mình |
| **C** | **RRF(A, B)** | ghép có lợi — đúng kết quả đội AIC'25 |

> 🛑 **BẮT BUỘC KHÓA BỂ ỨNG VIÊN — nếu không, việc 8 ra kết luận NGƯỢC.**
>
> `clip.npy` có đủ **177.321 dòng thật**. Ma trận chạy thử SigLIP2 chỉ encode
> vài nghìn dòng, phần còn lại là **vector 0** (cosine 0, không bao giờ được
> truy hồi). So thẳng A với B là so *"tìm trong 177 nghìn"* với *"tìm trong
> vài nghìn"* — B thắng vì bể nhỏ hơn, **không liên quan gì tới chất lượng
> model**.
>
> Đo thật trên tập dev 12 câu, **cùng một ma trận `clip.npy`**, chỉ đổi bể:
>
> | Bể ứng viên | Điểm |
> | --- | --- |
> | đầy đủ 177.321 keyframe | **0,5167** |
> | thu hẹp 2.328 keyframe | **0,8000** |
> | | **+0,2833 thuần ảo giác** |
>
> Để so sánh: toàn bộ lợi ích đội AIC'25 thu được khi thêm SigLIP2 là
> **+0,07** điểm/câu (A8.2). **Ảo giác lớn gấp 4 lần thứ cần đo.**
>
> Cách đúng — `src/dense.py` đã có sẵn:
>
> ```python
> from dense import KenhAnh, be_chung
> a_kenh = KenhAnh("./index")                                  # ViT-B/32
> b_kenh = KenhAnh("./index", matrix="clip_siglip2_thu.npy")   # SigLIP2 thử
> be = be_chung(a_kenh, b_kenh)        # chỉ dòng CẢ HAI đều encode thật
>
> a = cham(dev, lambda c: a_kenh.tim(c.cau_hoi, k=100, be=be))
> b = cham(dev, lambda c: b_kenh.tim(c.cau_hoi, k=100, be=be))
> print(so_sanh_cap(a, b, "ViT-B/32", "SigLIP2"))
> ```
>
> Và **câu hỏi dev phải có đáp án nằm trong bể đó** — soạn từ đúng các video
> đã encode, nếu không thì không câu nào tìm được.

> ⚠️ **Cách đo SAI thường gặp:** encode 2.000 ảnh rải rác rồi đo. Truy hồi là
> xếp hạng trên **toàn kho** — chỉ 2.000 ứng viên thì bài toán dễ hẳn, số đẹp
> giả tạo, không suy ra được gì. Phải encode **trọn vẹn** các video trong tập
> để có bài toán khép kín thật ở quy mô 1/9.
>
> ⚠️ 100 video phải **lấy phân tầng theo nhóm L** (mỗi nhóm 10), không ngẫu
> nhiên — L26 chiếm 57% kho nên ngẫu nhiên sẽ ra tập lệch hẳn (A2).

---

## 6. Ba điều phải hiểu trước khi bấm chạy

| # | Điều | Vì sao |
| --- | --- | --- |
| 1 | **`clip.npy` cũ GIỮ NGUYÊN — không xóa, không ghi đè** | Toàn bộ bằng chứng Giai đoạn 0 (873/873, 0 lệch chỉ số thật) là bằng chứng cho **ma trận đó**. Và đội AIC'25 **không thay** CLIP bằng SigLIP2 — họ **ghép hai cái**. Điểm đến từ chỗ ghép |
| 2 | **`01_build_index.py` KHÔNG được sửa** | Nó đọc `.npy` BTC phát sẵn theo từng video, không encode gì. Việc encode là script riêng → nâng cấp là việc **cộng thêm**, không có đường nào làm hỏng cái đang chạy. Ma trận mới tệ thì xóa file là xong |
| 3 | **MỘT máy GPU, không phải năm máy chia nhau** | Đo thật: `ViT-B/32` trên CPU 4 luồng = **12 ảnh/s** → 177k ảnh mất **4,1 h**; `ViT-L/14` nặng gấp ~6 → **~78 h/máy**. Năm máy CPU vẫn thua một GPU. Kết quả là **một file `.npy`** chia qua Drive — y hệt cách BTC phát `clip.npy` |

---

## 7. BẢNG BÀN GIAO

| File | Cỡ | Nội dung | Đi đâu |
| --- | --- | --- | --- |
| `index/clip_siglip2.npy` | 409 MB | `(177321, 1152)` fp16, đã chuẩn hóa L2 | **Drive** |
| `index/clip_siglip2.json` | nhỏ | model, pretrained, ngày chạy, tốc độ đo, số dòng vector 0 | **Drive** |
| `master.parquet` | — | **không đổi** | không gửi lại |

- **Không đẩy git** — repo là public, `.gitignore` đã chặn `*.npy`.
- Người khác chỉ cần tải thêm 409 MB: cùng `row_id`, cùng thứ tự, ghép thẳng
  vào RRF.

---

## 8. SPEC `scripts/08_encode.py`

| Cờ | Ý nghĩa |
| --- | --- |
| `--model` | tên `open_clip`, vd `ViT-SO400M-14-SigLIP2-378` |
| `--pretrained` | vd `webli` |
| `--videos N` | chỉ encode N video, **phân tầng theo nhóm L** (mặc định: tất cả) |
| `--batch N` | mặc định 16 |
| `--fp16` / `--fp32` | lưu float16 (mặc định) hay float32 |
| `--out` | đường dẫn `.npy` |
| `--kiem-lech-hang FILE` | chạy riêng phép kiểm §3A, **không encode** |

| # | Ràng buộc bắt buộc |
| --- | --- |
| 1 | Lặp **theo đúng thứ tự `row_id`** của `master.parquet`, **không glob thư mục** |
| 2 | Thiếu/hỏng ảnh → **ghi vector 0**, đếm và in ra cuối, **không bỏ dòng** |
| 3 | `assert` model + pretrained; ghi ra `.json` kèm |
| 4 | Chuẩn hóa L2 trước khi lưu |
| 5 | `torch.autocast('cuda', dtype=torch.float16)` + `torch.no_grad()` |
| 6 | `DataLoader` nhiều `num_workers` — nếu không, CPU giải nén JPEG thành nút cổ chai chứ không phải GPU |
| 7 | In **tốc độ thật ảnh/giây** để thay ước tính ở §2 |
| 8 | Checkpoint mỗi ~10.000 dòng — chạy 2 h mà mất điện thì không phải làm lại từ đầu |
