# Tổng kết Giai đoạn 0 — dữ liệu, và hướng đi các giai đoạn sau

*2026-08-09. Gộp kết quả kiểm chứng của 5 máy.*

---

## PHẦN 1 — GIAI ĐOẠN 0 ĐÃ XONG

### 1.1 Con số kết luận

| | Kết quả |
| --- | --- |
| Độ phủ kiểm chứng | **872 / 873 video (99,9%)** — đủ 10/10 nhóm L |
| Script 02 — ảnh keyframe ↔ dòng CSV | **870 / 872 đạt (99,8%)** |
| Script 03 — vector CLIP ↔ dòng CSV | **865 / 872 đạt (99,2%)** |
| Mẫu chưa giải thích được | **1** (`L25_V004`, keyframe cuối video) |

Ngưỡng để qua Giai đoạn 0 là 95%. Ta ở **99,8%**. Bảng cái đúng.

Chỉ thiếu `L25_V001` — máy giữ L25 không có file video đó.

### 1.2 Bốn con số phải nhớ

| Con số | Giá trị | Vì sao quan trọng |
| --- | --- | --- |
| **% có object JSON** | **100,0%** | objects là kênh dùng được, không phải kênh phụ |
| **Trung vị mật độ keyframe** | **55 frame** | ~1,8 giây ở 30fps → vẫn phải trích dày cho TRAKE |
| **fps tồn tại** | **25.0 / 26.44 / 29.97 / 30.0** | CẤM hardcode fps |
| **Tổng keyframe** | **177.321** trên 873 video | ma trận CLIP 347 MB, vừa RAM |

### 1.3 Ba điều đã chứng minh được, không phải phỏng đoán

**a) `row_id` như nhau trên mọi máy.** Đối chiếu 5 lô kết quả với
`master.parquet` máy này — **765/765 dòng trùng khít** cả `video_id`, `kf_n`,
`frame_idx` lẫn `fps`.

**b) Cả pipeline tái lập được.** Máy giữ L23+L26+L27 chạy lại **nhóm L21** —
nhóm máy khác đã chạy từ đầu. Trùng khít **29/29 ở mọi cột, kể cả `cosine`
tới 4 chữ số thập phân.** Hai máy, hai ổ đĩa, cùng chuỗi ffmpeg → CLIP, ra
đúng một con số. Nghĩa là **ngưỡng hiệu chuẩn trên một máy áp thẳng sang máy
khác được**, không cần hiệu chuẩn chéo.

**c) Rủi ro fps đã đóng.** Cả 26,44 (1 video) lẫn 29,97 (30 video) đều kiểm
và đạt. `L24_V044` — video 26,44 fps duy nhất — được xác nhận **hai lần độc
lập** trên hai keyframe khác nhau, cho tỷ số 26,4380 và 26,4331.

### 1.4 Hai cái bẫy phát hiện trong quá trình kiểm chứng

**A5.6 — Keyframe trùng lặp.** 11,83% keyframe toàn kho có bản sao cosine
≥ 0,99 trong cùng video. Phân bố cực lệch: **L25 chiếm 49,82%**, mọi nhóm
khác 0,27–2,16%. Bản sao nằm liền nhau (cảnh tĩnh kéo dài), trung vị cách 5
keyframe. Video tệ nhất `L25_V085`: 408/599 keyframe có bản sao; có cặp
cosine đúng **1,0000**.

**A5.7 — Cụm frame liên tiếp.** 10.845 cặp keyframe cách nhau ≤ 2 frame
(6,12% kho), nằm ở 745/873 video. 132 video có cụm này ở 5 keyframe **cuối**.
Cách 1 frame ở 30fps là **33 ms** — `ffmpeg -ss` không định vị nổi.

Hai hiện tượng này **không phải lỗi dữ liệu**. Nhưng chúng làm phép kiểm
chứng báo động giả, và — quan trọng hơn — chúng là **rủi ro điểm số thật**.
Xem PHẦN 2.

### 1.5 Ba lần script báo động giả, và bài học

| Lần | Báo | Thực tế |
| --- | --- | --- |
| L22, 3 mẫu "KHÔNG KHỚP" | tương quan pixel 0,675 | đồ họa chuyển cảnh; đồng hồ trên hình đọc cùng `18:36:43` → đúng giây |
| L25, 8 mẫu `LECH_INDEX` | không đúng hạng 1 | keyframe trùng lặp, cosine giữa chúng tới 1,0000 |
| L26, 4 mẫu `LECH_INDEX` | không đúng hạng 1 | y hệt trên |

**Bài học đã đưa vào code:** ngưỡng tuyệt đối luôn thua so sánh tương đối.
`02_verify.py` thêm `KHOP_YEU` (biên độ so dòng kề); `03_verify_CLIP.py` đổi
thứ tự phán quyết — **cosine tuyệt đối trước, thứ hạng sau** — vì lệch chỉ số
thật nghĩa là `clip[row_id]` là vector của một **cảnh khác**, khi đó cosine
tụt xuống 0,3–0,7 chứ không thể ≥ 0,95.

---

## PHẦN 2 — HAI RỦI RO CHI PHỐI THIẾT KẾ CÁC GIAI ĐOẠN SAU

### 2.1 Keyframe trùng lặp → có thể mất điểm dù trả lời đúng cảnh

`Answer_KIS` nộp **một** `frame_idx` cụ thể. Nếu frame đáp án nằm trong cụm
20 bản sao, hệ thống rất dễ trả về một thành viên khác — **đúng cảnh, đúng
nội dung, sai `frame_idx`, 0 điểm**.

Xác suất một keyframe bất kỳ rơi vào cụm: **~12% toàn kho, ~50% nếu rơi vào
L25** (L25 chiếm 21% số keyframe).

**Cách xử lý rẽ đôi theo câu trả lời của BTC cho câu 0.a:**

| BTC trả lời | Việc phải làm |
| --- | --- |
| Chấp nhận cửa sổ `[s,e]` | Gộp cụm còn **một đại diện** → giải phóng slot trong top-100. Rủi ro biến mất. |
| Đòi đúng `frame_idx` | Với ứng viên top, **nộp nhiều thành viên của cụm**. Cụm trung vị 5 phần tử, ta có 100 slot — trả giá được. |

> **Câu 0.a giờ là câu quan trọng nhất chưa có lời đáp.** Trước đây nó chỉ
> ảnh hưởng module trích dày cho TRAKE. Giờ nó quyết định **chiến lược nộp
> bài của cả KIS lẫn Q&A**. Nếu BTC chưa trả lời, xây cả hai đường — phần
> chung (bảng cụm `index/trung_lap.parquet`) dùng được cho cả hai.

### 2.2 Mật độ keyframe → TRAKE vẫn bị trần điểm

Trung vị 55 frame, p90 150, chỉ 12,62% cặp cách ≤ 10 frame. Xác suất keyframe
có sẵn rơi trúng cửa sổ 10 frame là **14,6%** → R-Score trần ~0,15 dù thuật
toán dóng hàng hoàn hảo.

**Module trích dày vẫn là đường găng.** Không có cách nào vòng qua.

Ngoại lệ đáng chú ý: `L24_V044` có 42 keyframe trong 35,18 giây, và 745/873
video có ít nhất một cụm frame liên tiếp. Ở những chỗ đó dữ liệu gốc đã dày
sẵn — trích dày nên **bỏ qua** chúng để tiết kiệm.

### 2.3 Kho lệch nặng về L26

L26 chiếm **498/873 video (57%)** và 79.590/177.321 keyframe (45%).

**Hệ quả cho tập dev của Khánh:** lấy mẫu ngẫu nhiên sẽ ra 57% câu hỏi từ
L26. Phải **lấy mẫu phân tầng theo nhóm L**, nếu không toàn bộ tinh chỉnh ở
Giai đoạn 3 tối ưu cho một nhóm duy nhất.

---

## PHẦN 3 — TÌNH TRẠNG BỐN KÊNH TRUY HỒI

| Kênh | Dữ liệu | Trạng thái | Chặn bởi |
| --- | --- | --- | --- |
| **1. CLIP** | 100% (177.321 × 512) | ✅ sẵn sàng | — |
| **2. BM25 metadata** | title 100%, description 99,7% (955 ký tự) | ✅ sẵn sàng | — |
| **3. BM25 OCR/ASR** | chưa chạy | 🟡 đã bench, chưa sản xuất | ROI + ngưỡng |
| **4. Objects + IDF** | 100%, 1.122.384 detection | ✅ sẵn sàng | — |

**Ba trong bốn kênh có thể xây ngay hôm nay** — không chờ tải dữ liệu, không
chờ ai.

### 3.1 Kênh OCR/ASR — kết quả bench và kết luận

**ASR: chốt PhoWhisper Small.** WER 0,4332, coverage 0,9998, timestamp hợp lệ
1,00, RTF 0,1281. Toàn kho 16,6 giờ máy, chia 5 máy còn **3,3 giờ**.

Lý do chọn **không phải** "Medium kém hơn" — khoảng tin cậy 95% của hai model
chồng nhau từ 0,3044 đến 0,6656, với 20 mẫu không kết luận được. Lý do là
Small **rẻ hơn 2,8× VRAM và 1,7× thời gian mà không có bằng chứng Medium tốt
hơn**.

WER 0,43 nghe tệ nhưng ASR chịu sai số tốt: một bản tin 5 phút có hàng trăm
từ, 57% đúng vẫn thừa từ đặc trưng cho BM25.

**OCR: ba việc phải làm trước khi chốt model.**

1. **Chữ động (ticker) — bỏ.** WER 0,84–1,03, Exact 0,00–0,08. Không cứu
   được. Kế hoạch v4 PHẦN F đã dự đoán đúng.
2. **False positive 80–90% trên ảnh không có chữ.** Đây là chốt chặn: chạy
   thẳng lên 177k keyframe thì phần lớn index OCR là rác. Phải cắt ROI vùng
   tiêu đề + ngưỡng confidence, rồi **đo lại**.
3. **Chọn model theo WER/Exact, không phải CER.** BM25 khớp theo **từ** — sai
   một ký tự là thành token khác. Theo CER thì `easyocr_vi_en` thắng (0,2400
   vs 0,4877); theo **WER thì `easyocr_det_vietocr` thắng ngược** (0,4000 vs
   0,6637), và Exact 0,3846 vs 0,1538. Lỗi của VietOCR **dồn cục** — một số
   dòng sai hẳn, còn lại đúng nguyên. Đó chính là thứ BM25 cần.

   *Cảnh báo: chỉ 13 mẫu chữ tĩnh. Chênh lệch 5/13 với 2/13 là ba mẫu — chưa
   đủ chắc. Bench lại ≥ 50 mẫu sau khi cắt ROI.*

**Một câu đáng đo trước khi bỏ 17 giờ máy:** OCR có thêm được gì so với
metadata không? `title` phủ 100% và `description` trung bình 955 ký tự đã nói
chủ đề rồi. Giá trị riêng của OCR nằm ở chữ dưới màn hình (tên người phỏng
vấn, địa danh) — thứ metadata không có. Đo trên tập dev; nếu không thêm được
gì thì bỏ cả kênh.

---

## PHẦN 4 — LÀM GÌ TIẾP

### 4.1 Làm được ngay, không chờ ai

| # | Việc | Ai | Chặn bởi |
| --- | --- | --- | --- |
| 1 | Ma trận CLIP + text encoder (`ViT-B-32-quickgelu`) | TV1 | — |
| 2 | `src/objects.py` — IDF + `object_score` | TV2 | — |
| 3 | BM25 metadata | TV3 | — |
| 4 | Bảng cụm trùng lặp từ `index/trung_lap.parquet` | TV1 | — |
| 5 | Hợp nhất RRF 3 kênh (chưa cần OCR) | TV1 | 1, 2, 3 |

Xong 5 việc này là có **hệ thống tìm kiếm chạy được**, đo được trên tập dev.
Kênh OCR ghép vào sau, không chặn.

### 4.2 Chờ câu trả lời của BTC

**Câu 0.a — BTC chấp nhận cửa sổ `[s,e]` hay đòi đúng `frame_idx`?**

Đây là câu chi phối nhiều nhất. Nó quyết định:
- Chiến lược nộp bài KIS và Q&A (gộp cụm hay nộp nhiều thành viên) — mục 2.1
- Độ rộng cửa sổ module trích dày cho TRAKE
- Cách Khánh soạn đáp án tập dev

Nếu quá hạn mà chưa có đáp, **mặc định giả thiết khắt khe hơn** (đòi đúng
`frame_idx`) và xây theo hướng đó — đường đó bao được cả hai trường hợp.

### 4.3 Việc người, không phải việc máy

| # | Việc | Ghi chú |
| --- | --- | --- |
| 1 | Máy giữ L25 kiểm lại `L25_V004` với vài keyframe **giữa** video | Giải quyết mẫu duy nhất còn treo (A5.7) |
| 2 | Máy giữ L23+L26+L27 tải lại gói `Keyframes_L21` | Thiếu 8 file ảnh |
| 3 | Cắt ROI + ngưỡng confidence cho OCR, đo lại false positive | Chốt chặn của kênh 3 |
| 4 | Bench lại OCR ≥ 50 mẫu, lấy **WER/Exact** làm chỉ số chính | Sau khi có ROI |
| 5 | Khánh soạn tập dev — **lấy mẫu phân tầng theo nhóm L** | Tránh 57% câu hỏi rơi vào L26 |
| 6 | Chốt bảng tên thành viên duy nhất | Còn treo từ Giai đoạn 0 |

### 4.4 Điều KHÔNG cần lo nữa

- ~~Bảng cái sai~~ — chứng minh trên 99,9% kho
- ~~fps lạ~~ — cả 26,44 lẫn 29,97 đã kiểm và đạt
- ~~`row_id` lệch giữa các máy~~ — đo 765/765
- ~~Số liệu các máy không so sánh được~~ — cả pipeline tái lập, cosine trùng
  tới 4 chữ số thập phân
- ~~Objects là kênh nhiễu~~ — 100% phủ, 1,12 triệu detection

---

## Nguồn số liệu

Mọi con số trong tài liệu này đo được, không ước lượng. Chạy lại bằng:

```powershell
python scripts\07_gop_kiem_chung.py
```

Dữ liệu thô: [dev/verify/](../dev/verify/) — kết quả của 5 máy.
Chi tiết: [dev/so_lieu_giai_doan_0.md](../dev/so_lieu_giai_doan_0.md).
Kế hoạch đầy đủ: [docs/Ke_hoach_AIC2026_v4.md](Ke_hoach_AIC2026_v4.md).
