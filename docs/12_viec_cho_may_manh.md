# Việc cho máy mạnh — mở khoá phần đang tắc của hệ thống

*Soạn 21/08/2026. Gửi người giữ máy ≥ 16 GB RAM (hoặc chạy trên Colab/Kaggle).*

---

## Bối cảnh 30 giây

Điểm leaderboard: **6,2** (cập nhật 21/08 tối, xem A36), đội cao nhất 12,2. Đã
tăng được từ 0,8 qua ba bước — đổi sang kênh ảnh SigLIP2 (3,8), rồi cho Gemini
xếp lại top-20 (4,8) và top-50 (5,4), **chững lại 4 lượt**, rồi vá 3 đáp án
Q&A đã xác minh bằng mắt đưa lên **6,2** (A36).

Nguyên nhân đã khoanh được: trần 5,4 **không phá được bằng cách cải thiện khâu
xếp lại** — phần còn thiếu nằm ở **bể ứng viên**, tức ở kênh 1 (SigLIP2). Nếu
SigLIP2 không đưa khung đúng vào top-100 thì không bộ xếp lại nào cứu được.

Mà mọi việc động tới kênh 1 đều **tắc ở máy 7,7 GB**: không nạp nổi
`ViT-SO400M-14-SigLIP2-378` (~3,5 GB trọng số), đã làm treo máy hai lần.

**Vì thế cần máy của bạn.** Việc 1 dưới đây mở khoá cả năm việc còn lại và chỉ
mất ~10 phút.

> 🔴 **CẬP NHẬT 21/08 tối — cả 6 việc dưới đây đã xong trong ngày, xem A29-A34
> trong `Ke_hoach_AIC2026_v4.md`.** Đã thử nộp thật tổ hợp RRF(A30)+xếp lại
> ảnh(A33) — kết quả **~5,0, TỆ HƠN 5,4**, dù cả hai đều đo dương trên dev.
> Nguyên nhân: RRF được đo bằng câu KHÔNG tách, nhưng đề thật bị `run.py` tách
> câu (đề thật dài 63 từ, dev chỉ 12/60 câu đủ dài để lộ ra khác biệt này —
> đúng lỗi mù đã cảnh báo ở A19/A20, lần thứ ba). Đã nộp lại bản AN TOÀN (đúng
> công thức A27/A28) để phục hồi mốc. Đã soạn thêm 20 câu dev DÀI và đo lại
> với n=32: **RRF trung tính trên câu dài (⚪ không đổi gì)** — xác nhận
> **không nên bật `--hop-nhat` cho bài nộp thật**. **Bài học: không ráp nhiều
> kỹ thuật mới đo riêng lẻ vào MỘT lượt nộp thật — đổi một thứ, nộp, đối
> chiếu.** Xem A34. **Bản an toàn đã nộp lại đạt 6,2 — CAO HƠN mốc 5,4 cũ**
> (A36), nhờ vá 3 đáp án Q&A đã xác minh bằng mắt. Mốc nền từ nay là 6,2.
> Đã thêm cờ `--hop-nhat-chi-cau-ngan` vào `run.py` (RRF tự tắt cho câu dài,
> A35) — chưa nộp thật, để dành cho lượt nộp riêng tiếp theo.
>
> **Cập nhật thêm: xếp lại bằng ẢNH một mình nộp thật chỉ được 4,2 — TỆ NHẤT
> trong 3 cấu hình đã thử, dù trên dev nó là tín hiệu đẹp nhất (A37).** Kết
> luận: **xếp lại bằng CHỮ vẫn là lựa chọn mặc định duy nhất đáng tin** (3/3
> lần đúng trên leaderboard thật). Tạm dừng đầu tư thêm vào xếp lại bằng ảnh.

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

## VIỆC 2 — Đo TRAKE — ✅ **XONG (21/08)**

Tập dev nay có **23 câu TRAKE** (L24, L26, L27, L30). Đo cả hai script bằng
`index/truy_van.npz` (không cần nạp model):

```powershell
.venv\Scripts\python.exe scripts\22_do_trake.py --cache index\truy_van.npz
.venv\Scripts\python.exe scripts\26_do_don_cuc_trake.py --cache index\truy_van.npz
```

(script 26 vừa được thêm cờ `--cache` — bản cũ chỉ đo được trên kênh 3 và ra
0,0000 cả ba biến thể, không kết luận được gì).

**Kết quả — xem A29 trong `Ke_hoach_AIC2026_v4.md`:**

* `run.dung_trake()` thật (D_THAT) **thắng cả ba biến thể giả lập** — không
  cần sửa gì ở khâu lắp ráp.
* RRF kênh1+4 vào TRAKE **lỗ**, đúng quy luật đã biết ở A28 — không trộn kênh
  4 vào TRAKE.
* Chốt chống dồn cục "xét từng cặp" (đề xuất ở đây trước đó) bắt sạch 100%
  dồn cục nhưng **làm điểm tệ đi** (−0,058 ở ±2s, −0,116 ở ±15s) — **không
  thêm vào `run.py`**. Nghi ngờ ban đầu ("47/100 dòng dồn cục chắc chắn vô
  nghĩa") **bị chính phép đo bác**: ép rải đều mất nhiều hơn được.

**Không còn việc gì cần làm ở đây** — giữ nguyên `run.dung_trake()` hiện tại.

---

## VIỆC 3 — Đo lọc xếp tầng SigLIP2 + OCR trên tập dev — ✅ **XONG (21/08), kết quả GÂY LO NGẠI — xem A31**

Kết quả: kỹ thuật ăn +1,6 điểm trên leaderboard chỉ được **+0,01 (🟡 YẾU,
không vượt nhiễu)** trên 60 câu KIS dev — **lần mù thứ ba của tập dev** (sau
A19, A20), lần này chênh lệch tới 160x. Xem phân tích đầy đủ ở A31
`Ke_hoach_AIC2026_v4.md`: không đủ bằng chứng để kết luận kỹ thuật này ảo,
cũng không đủ để tin dev đang đo đúng khâu xếp lại — **khuyến nghị: giữ
nguyên bài nộp, nhưng đừng tinh chỉnh tiếp khâu xếp lại dựa trên dev.**

Đã đo thêm: `RRF(1,3,w=0,1)` (A30) làm bể ứng viên rồi mới xếp lại =
0,4367/0,5833, nhỉnh hơn xếp lại một mình (0,4267/0,5833) nhưng vẫn 🟡 YẾU.
Hai kỹ thuật **không triệt tiêu nhau** — có thể dùng cùng lúc cho lượt nộp
sau, xem A31.

---

## VIỆC 4 — Phân rã truy vấn bằng LLM — ✅ **ĐO XONG (21/08), BỊ BÁC trên câu ngắn — xem A32**

`scripts/32_do_phan_ra_llm.py --cache index/truy_van.npz --lam 0.15 --fp16`.
Kết quả trên 60 câu KIS dev: **−0,0533/−0,0733, ✅ ỔN ĐỊNH theo chiều xấu đi**
— phân rã làm tệ hơn, không tốt hơn.

**Nhưng ĐỪNG đóng hẳn hướng này.** Câu dev (~15-20 từ) ngắn hơn đề thật (63
từ) tới 3 lần — `tach_truy_van` trên dev gần như không cắt gì, nên phép đo
này chỉ trả lời được "phân rã có hại cho câu ngắn không" (có), không trả lời
được câu đang cần: "phân rã có giúp câu DÀI (đúng vấn đề ban đầu) không".
Việc còn lại nếu muốn theo tiếp: đo trên **24 câu đề mẫu** (`dev/THUNGHIEM-bo-de-thi`,
dài đúng kiểu đề thật) thay vì tập dev — script đã viết sẵn, chỉ cần trỏ
nguồn câu hỏi khác.

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

## VIỆC 6 — Xếp lại bằng ảnh, với đủ keyframe — ✅ **XONG (21/08), KẾT QUẢ TỐT NHẤT — xem A33**

Máy này có đủ 100% keyframe (177.321/177.321). Đo trên 60 câu KIS dev:
**+0,0300/+0,0267, ✅ ỔN ĐỊNH, 9 thắng–1 thua–50 hoà** — cấu hình xếp lại
**tốt nhất** đo được trên dev hôm nay (hơn cả xếp lại bằng chữ +0,01 và
RRF+chữ +0,02). **Xác nhận: máy cũ thiếu ảnh (21%) mới là nguyên nhân làm
điểm tệ đi 5,4→5,2, không phải bản thân kỹ thuật.**

**Khuyến nghị: đưa `scripts/30_xep_lai_thi_giac.py` vào bài nộp chính**, chạy
trên máy đủ ảnh toàn kho. Đây là ứng viên mạnh nhất cho lượt nộp tiếp theo.

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

---

> ➡️ **Đợt 2 đã soạn xong: [13_lenh_cho_may_manh_dot2.md](13_lenh_cho_may_manh_dot2.md).**
> Hai phép đo (cửa sổ A38, prior khoảng cách TRAKE A39), lần đầu có **23 câu đề
> thật do BTC viết** trong tập dev để làm thước.
