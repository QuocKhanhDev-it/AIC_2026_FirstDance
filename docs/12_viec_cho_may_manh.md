# Việc cho máy mạnh — mở khoá phần đang tắc của hệ thống

*Soạn 21/08/2026. Gửi người giữ máy ≥ 16 GB RAM (hoặc chạy trên Colab/Kaggle).*

---

## Bối cảnh 30 giây

Điểm leaderboard: **5,4**, đội cao nhất 12,2. Đã tăng được từ 0,8 qua ba bước —
đổi sang kênh ảnh SigLIP2 (3,8), rồi cho Gemini xếp lại top-20 (4,8) và top-50
(5,4). **Sau đó chững hẳn**: bốn lượt nộp liên tiếp không tăng, một lượt còn tụt.

Nguyên nhân đã khoanh được: trần 5,4 **không phá được bằng cách cải thiện khâu
xếp lại** — phần còn thiếu nằm ở **bể ứng viên**, tức ở kênh 1 (SigLIP2). Nếu
SigLIP2 không đưa khung đúng vào top-100 thì không bộ xếp lại nào cứu được.

Mà mọi việc động tới kênh 1 đều **tắc ở máy 7,7 GB**: không nạp nổi
`ViT-SO400M-14-SigLIP2-378` (~3,5 GB trọng số), đã làm treo máy hai lần.

**Vì thế cần máy của bạn.** Việc 1 dưới đây mở khoá cả năm việc còn lại và chỉ
mất ~10 phút.

---

## VIỆC 1 — Mã hoá sẵn vector truy vấn *(ưu tiên tuyệt đối, ~10 phút)*

Model chỉ làm **đúng một việc**: biến câu chữ thành vector. Ma trận ảnh
177.321 × 1152 thì đã có sẵn trên mọi máy. Mà tập truy vấn là **hữu hạn và biết
trước** — 24 câu đề + 115 câu dev = **296 chuỗi** (đã tính cả các mệnh đề do
`tach_truy_van` cắt ra).

Mã hoá một lần ra file vài trăm KB, rồi **mọi máy trong nhóm chạy được kênh 1
mà không cần nạp model**.

```powershell
git pull origin giai-doan-0
.venv\Scripts\python.exe scripts\25_ma_hoa_truy_van.py `
    --de dev\THUNGHIEM-bo-de-thi --tap-dev
```

Ra `index/truy_van.npz`. **Gửi lại file đó cho nhóm** — vài trăm KB, gửi qua chat
được.

Máy yếu dùng nó như sau, không nạp model gì:

```powershell
.venv\Scripts\python.exe src\run.py --de <thư mục đề> --cache index\truy_van.npz
```

> Nếu máy bạn cũng chật, thêm `--fp16` (nạp trọng số nửa độ chính xác, tự giải
> phóng tháp ảnh vì mã hoá văn bản không dùng tới). Script có chốt RAM riêng,
> sẽ dừng an toàn thay vì treo máy.

---

## VIỆC 2 — Đo TRAKE *(3/24 gói, chưa từng đo được lần nào)*

Tập dev vừa có **13 câu TRAKE** (L24, L27, L30 — đã soát bằng mắt, mô tả khớp
ảnh). Trước đó chỉ có 3 câu nên `run.dung_trake()` — thứ quyết định điểm của cả
một dạng truy vấn — chưa bao giờ được đo.

```powershell
.venv\Scripts\python.exe scripts\22_do_trake.py
.venv\Scripts\python.exe scripts\26_do_don_cuc_trake.py
```

Script thứ hai so **ba chính sách chống dồn cục**. Đã chạy trên máy yếu với ứng
viên kênh 3 và ra **0,0000 cả ba** — kênh 3 không tìm được sự kiện TRAKE nào, nên
phép đo không kết luận được gì. Với ứng viên kênh 1 thì mới có điểm để so.

**Vì sao đáng làm:** bài nộp hiện tại có **47/100 dòng** ở `query-p1-18-trake` và
33/100 ở `query-p1-4-trake` chứa cặp sự kiện cách nhau **dưới 100 frame** — ví dụ
thật `L23_V013,0,1,2,2298`, tức ba sự kiện cách nhau 0,03 giây. Chắc chắn vô
nghĩa, nhưng chưa có bằng chứng nào để đổi mặc định.

Báo về: bảng điểm ba chính sách ở hai mức dung sai.

---

## VIỆC 3 — Đo lọc xếp tầng SigLIP2 + OCR **trên tập dev**

Đây là kỹ thuật đã đưa điểm 3,8 → 5,4 **nhưng chưa bao giờ đo được trên tập
dev** — nó được dò thẳng bằng leaderboard, khác mọi cấu hình khác trong dự án.

Quy luật đã đo được cả hai phía, cần kiểm chứng lại trên dev:

```
XẾP LẠI trong bể kênh 1 đã chọn   →  +1,6 điểm   (3,8 → 5,4)
THAY THẾ bằng ứng viên mới        →  −0,4 điểm   (5,4 → 5,0)
```

Cách đo: lấy top-100 của SigLIP2 cho từng câu dev, cho kênh 3 (OCR+ASR) xếp lại
top-50, chấm bằng `cham_diem.bao_cao_do_nhay`. Nếu dev **không** tái lập được
mức tăng này thì tập dev đang mù thêm một lần nữa (đã mù hai lần: A19, A20) — và
đó cũng là thông tin quan trọng.

---

## VIỆC 4 — Phân rã truy vấn bằng LLM *(đòn bẩy lớn nhất còn lại)*

Đề thi dài **63 từ / 2,4 mệnh đề**, trong khi text encoder SigLIP2 chỉ nhận
**64 token**. Hiện đang cắt theo dấu câu rồi lấy điểm cao nhất — thô.

Thử: dùng LLM (Gemini, khoá đã có trong `.env`) viết lại mỗi truy vấn thành
**3 mệnh đề thị giác ngắn ≤ 20 từ**, mỗi mệnh đề một góc nhìn:

1. cảnh tổng thể / không gian
2. hành động của nhân vật
3. vật thể đặc trưng cận cảnh

Rồi mã hoá cả ba (Việc 1 đã có sẵn công cụ: `--them "..."`) và gộp điểm:

```
Score(i) = max_j cos(v_i, q_j) + λ · Σ_j cos(v_i, q_j)      với λ ≈ 0,1–0,2
```

**Đo trên tập dev trước khi nộp.** Mốc nền là `tach_truy_van` hiện tại, không
phải "không cắt gì".

---

## VIỆC 5 — Hợp nhất hai model ảnh qua bản dịch

Số liệu của ta đã ủng hộ hướng này, trên cùng một bể ứng viên:

| | ±2s |
| --- | ---: |
| CLIP ViT-B/32 + tiếng Việt | 0,0095 |
| CLIP ViT-B/32 + **bản dịch tay sang Anh** | **0,8190** |
| SigLIP2 + tiếng Việt | 0,8571 |

Hai không gian vector này bù trừ nhau. Cách hợp nhất **đúng quy luật đã đo**
(chỉ xếp lại, không thay ứng viên):

1. Lấy **top-100 của SigLIP2** làm bể cố định.
2. Dịch truy vấn sang tiếng Anh bằng LLM.
3. Tính cosine của model tiếng Anh **trên đúng 100 ứng viên đó**.
4. Chuẩn hoá min-max cả hai điểm về `[0,1]` **trên 100 ứng viên** rồi cộng:
   `α · Ŝ_siglip2 + (1−α) · Ŝ_anh`, dò `α ≈ 0,6–0,7`.

Bước 4 chuẩn hoá **trong phạm vi 100 ứng viên** chính là cách né vấn đề thang
điểm mà `rrf.py` nêu (cosine 0,25–0,40 so với BM25 không chặn trên).

---

## VIỆC 6 — Xếp lại bằng ảnh, **với đủ keyframe**

Đã thử trên máy yếu: **5,4 → 5,2, tệ đi**. Nhưng nguyên nhân nhiều khả năng là
**hiện vật của máy, không phải của kỹ thuật**:

| | |
| --- | --- |
| Máy yếu chỉ có ảnh của L21/L22/L24/L27/L30 | **21% toàn kho** |
| Trong top-50 của 18 gói KIS | 537/900 khung có ảnh (60%), lệch mạnh: có gói 50/50, có gói 2/50 |
| Sau khi xếp lại bằng ảnh, hạng 1 là khung **có ảnh** | **13/18** (trước đó 10/18) |

Bộ xếp lại **chỉ đẩy lên được thứ nó nhìn thấy**, nên nó đẩy 21% kho lên trên
79% còn lại — thiên vị theo *"máy nào đã tải nhóm nào"*, không theo chất lượng.

**Nếu máy bạn có đủ keyframe toàn kho**, chạy lại và so:

```powershell
.venv\Scripts\python.exe scripts\30_xep_lai_thi_giac.py `
    --nguon firstdance6.zip --de dev\THUNGHIEM-bo-de-thi `
    --ra submission_thigiac --nen thigiac_du_anh.zip
```

Nếu vẫn tệ đi khi đã đủ ảnh thì kết luận mới chắc: xếp lại bằng thị giác không
giúp, và ta khép hướng đó lại.

**Trước khi chạy, cho nhóm biết máy bạn đang có ảnh của những nhóm L nào** —
con số đó quyết định đọc kết quả thế nào.

---

## Kỷ luật đo — xin giữ đúng, vì cả dự án dựa vào nó

* **Mốc nền là cấu hình MẠNH NHẤT hiện có**, không phải cái tiện tay.
* **Chỉ đổi một thứ mỗi lần.** Đã vấp nhiều lần: đổi hai thứ rồi quy công nhầm.
* Dùng `cham_diem.bao_cao_do_nhay()` — nó chấm ở **hai mức dung sai** và tự kết
  luận `✅ ON DINH` / `🟡 YEU` / `❌ DAO DAU`. **Đảo dấu giữa hai mức = không kết
  luận được**, không phải "hơi hơn".
* Báo **thắng–thua–hoà kèm ngưỡng nhiễu**, không chỉ điểm trung bình.
* Đo được gì mới thì **thêm một mục `A<n>` vào `docs/Ke_hoach_AIC2026_v4.md`**,
  kèm cả thứ đã thử mà **không** hiệu quả — phần lớn giá trị của tài liệu đó nằm
  ở chỗ này. Hiện đã có 11 kỹ thuật bị phép đo bác, xem
  [11_tom_tat_cho_tu_van_ngoai.md](11_tom_tat_cho_tu_van_ngoai.md) mục 5.

## Báo kết quả về như thế nào

1. `index/truy_van.npz` (Việc 1) — gửi file, cả nhóm dùng chung.
2. Với mỗi việc: dán **nguyên văn output** của script, đừng tóm tắt thành
   "có vẻ tốt hơn".
3. Nếu một việc chạy không được: dán **nguyên văn thông báo lỗi**. Đừng sửa
   quanh nó — mấy chốt đó dựng ra để chặn đúng những lỗi im lặng đã cắn thật.
