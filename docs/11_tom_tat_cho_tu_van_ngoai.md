# Tóm tắt hệ thống để hỏi tư vấn ngoài — AIC HCMC 2026, đội FirstDance

*Cập nhật 21/08/2026. Dán trọn file này khi hỏi một trợ lý khác. Phần quan trọng
nhất là **mục 5 — những gì đã bị phép đo bác**: không có nó thì lời tư vấn sẽ lặp
lại đúng những thứ đã thử và thất bại.*

---

## 1. Bài toán

Truy hồi video cho AI Challenge HCMC 2026, vòng sơ tuyển. Ba dạng truy vấn:

* **KIS** — tìm khung hình khớp một đoạn mô tả tiếng Việt (18/24 gói đề)
* **Q&A** — như KIS, nhưng nộp kèm một chuỗi `answer` (3/24 gói)
* **TRAKE** — nộp N khung theo đúng thứ tự thời gian của N sự kiện (3/24 gói)

**Kho:** 873 video, 177.321 keyframe, 129,8 giờ. Đề thi tiếng Việt, dài trung
bình **63 từ / 2,4 mệnh đề** mỗi truy vấn.

**Chấm:** `Final Score = trung bình R@{1,5,20,50,100}`, mỗi gói nộp tối đa 100
dòng. Không có điểm phạt — dòng thứ 100 vẫn đáng 0,2. Public Leaderboard chỉ chấm
**50% đáp án**; xếp hạng cuối chấm 100% và lấy **lượt nộp CUỐI**, không phải lượt
tốt nhất.

**Điểm hiện tại: 5,4.** Đội cao nhất **12,2**.

---

## 2. Lịch sử điểm — mỗi bước đổi đúng một thứ

| Lượt | Thay đổi | Điểm |
| --- | --- | ---: |
| #1 | kênh objects đơn lẻ | 0,8 |
| #2 | RRF(objects, OCR) + cắt truy vấn theo câu | 2,6 |
| #3 | **đổi sang kênh ảnh SigLIP2** | 3,8 |
| #5–7 | sửa/phá đáp án Q&A (3 gói) | 3,8 — **không đổi** |
| #8 | Gemini xếp lại **top-20** ứng viên KIS bằng OCR/ASR | **4,8** |
| #9 | Gemini xếp lại **top-50** | **5,4** |
| #10 | xếp lại top-100 | 5,4 — bão hoà |
| #11 | top-50 + lọc cứng OCR toàn kho | 5,0 — **tệ đi** |
| #12 | top-50 + thêm tiêu đề video vào bằng chứng | 5,4 — không đổi |

---

## 3. Hạ tầng

Không dùng Milvus / Elasticsearch / Postgres. Brute-force `numpy`: quét 177k
vector mất **16,7 ms** — đã đo. Dữ liệu là `master.parquet` + các file `.npy`.

**Ràng buộc phần cứng nghiêm trọng:** máy đang phát triển có **7,7 GB RAM**,
không nạp nổi model SigLIP2 (~3,5 GB trọng số) — đã làm treo máy hai lần. Ma
trận ảnh 177.321 × 1152 thì có sẵn trên đĩa. Đã dựng cơ chế **cache vector truy
vấn** (mã hoá sẵn trên máy khác) để lách, nhưng chưa chạy được vì máy hiện chỉ
còn 0,4–1,0 GB trống.

---

## 4. Năm kênh truy hồi và điểm đo trên tập dev (115 câu tiếng Việt)

| Kênh | Điểm ±2s | Ghi chú |
| --- | ---: | --- |
| **1 — ảnh, SigLIP2 SO400M** | **0,3258** | `ViT-SO400M-14-SigLIP2-378`, 1152 chiều, phủ 100% kho |
| 1 — ảnh, CLIP ViT-B/32 | **0,0000** | mù tiếng Việt: vector câu Việt còn xa bản dịch Anh hơn cả một câu vô quan |
| 3 — OCR + ASR | 0,1183 | 165k khung có OCR (93,2%), 137k có ASR (77,4%) |
| 4 — objects + IDF | 0,0417 | OpenImages, có bảng ánh xạ nhãn Việt–Anh |
| 2 — metadata | 0,0000 | cấp video, không phân biệt được khung trong cùng video |
| 5 — caption VLM | chưa chạy | code xong, chưa sinh dữ liệu |

**Cấu hình đang dùng:** SigLIP2 lấy 100 ứng viên → Gemini 3.1 flash-lite xếp lại
top-50 bằng OCR/ASR của chính các khung đó (chỉ đẩy lên khi có bằng chứng rõ,
không bỏ ứng viên nào).

---

## 5. ĐÃ THỬ VÀ BỊ PHÉP ĐO BÁC — đừng gợi ý lại

Mỗi dòng dưới đây là một phép đo có đối chứng, chấm ở hai mức dung sai, báo
thắng–thua–hoà kèm ngưỡng nhiễu.

| # | Kỹ thuật | Kết quả |
| --- | --- | --- |
| 1 | **Khử trùng lặp keyframe** trước khi cắt top-K | đảo dấu giữa hai mức dung sai; ghép với ràng buộc đa dạng thì **tệ đi ổn định** (0 thắng – 7 thua) |
| 2 | **RRF thô** hợp nhất nhiều kênh ngang hàng | −0,0144, ổn định. Lặp lại ở ba bối cảnh khác nhau, đều lỗ |
| 3 | **Hợp nhất hai tầng** (chọn video trước, rồi xếp khung) | tệ đi |
| 4 | **"Nearby frame"** — chèn khung lân cận vào danh sách nộp | tệ đi ổn định. Lý do: BTC chấm theo **cửa sổ rộng 4 giây–5 phút**, nên khung cách đáp án ±2s **đã được tính đúng rồi**; chèn nó vào là tiêu một trong 100 chỗ để mua lại thứ đã có |
| 5 | **Ràng buộc đa dạng** (mỗi video ≤ 2 slot trong top-5) | tệ đi |
| 6 | **Thu hẹp cấp video bằng metadata** (top-50 video, cắt cứng) | −0,0286. Metadata chỉ phủ **37,1%** số câu ở top-50 → cắt cứng là vứt 63% |
| 7 | **Dìm trọng số kênh phụ** trong RRF (0,3 thay vì 1,0) | tệ đi ổn định khi hai kênh ngang tầm |
| 8 | **Lọc cứng OCR trên toàn kho** (đưa khung mới vào top) | **−0,4 điểm trên leaderboard thật** |
| 9 | Xếp lại sâu hơn top-50 (top-100) | không đổi |
| 10 | Thêm **tiêu đề video** vào bằng chứng cho bộ xếp lại | không đổi |
| 11 | Sửa **đáp án Q&A** cho đúng (đã xác minh bằng mắt) | không đổi điểm public — 3 gói Q&A **không nằm trong 50% được chấm public** |

### Quy luật rút ra từ 11 phép đo trên

> **Kênh yếu chỉ được XẾP LẠI những gì kênh mạnh đã chọn — không được THÊM ứng
> viên mới, không được THAY thế.**

Đo được cả hai phía, cùng một kênh OCR, cùng một model, cùng truy vấn:

```
XẾP LẠI trong bể SigLIP2 đã chọn   →  +1,6 điểm   (3,8 → 5,4)
THAY THẾ bằng ứng viên mới          →  −0,4 điểm   (5,4 → 5,0)
```

---

## 6. Chỗ đang nghẽn

**Ba lần liên tiếp không đổi điểm** (top-100, tiêu đề video, và trước đó là mọi
thay đổi Q&A). Trần 5,4 **không phá được bằng cách cho bộ xếp lại nhiều bằng
chứng hơn về cùng những ứng viên đó**. Phần còn thiếu nằm ở **bể ứng viên** —
tức ở kênh 1 (SigLIP2) — chứ không ở khâu xếp lại.

Nói cách khác: nếu SigLIP2 không đưa khung đúng vào top-100, không bộ xếp lại nào
cứu được, và đưa khung mới vào bằng OCR thì đã đo là lỗ.

---

## 7. Câu hỏi muốn hỏi tư vấn ngoài

1. **Làm sao cải thiện bể ứng viên của kênh 1?** Truy vấn tiếng Việt dài 63 từ,
   2,4 mệnh đề, trong khi text encoder SigLIP2 chỉ nhận **64 token**. Hiện đang
   cắt theo câu rồi lấy điểm cao nhất trên từng keyframe. Có cách nào tốt hơn để
   biểu diễn một truy vấn dài–nhiều mệnh đề trong một không gian vector ảnh–văn
   bản không? (query decomposition, multi-vector retrieval, ColBERT-style late
   interaction, HyDE…?)

2. **Có model ảnh–văn bản nào mạnh hơn `ViT-SO400M-14-SigLIP2-378` cho tiếng
   Việt** mà chạy được trên GPU tiêu dùng / Colab free? (đã cân nhắc: CLIP
   ViT-B/32 = 0,0000 vì mù tiếng Việt)

3. **Dịch truy vấn sang tiếng Anh rồi dùng model mạnh tiếng Anh** — đo trên tập
   con: CLIP + bản dịch tay = 0,8190 so với SigLIP2 + tiếng Việt = 0,8571 (cùng
   bể ứng viên). Có nên hợp nhất **hai model ảnh** (SigLIP2 tiếng Việt + một
   model tiếng Anh mạnh với truy vấn đã dịch) không? Lưu ý quy luật ở mục 5 —
   hợp nhất chỉ được phép **xếp lại**, không được thay.

4. **Kênh caption** (VLM sinh mô tả tiếng Việt cho từng keyframe rồi tìm bằng
   BM25) — đáng đầu tư không, khi kênh văn bản OCR+ASR hiện chỉ được 0,1183 so
   với 0,3258 của kênh ảnh?

5. **TRAKE** — 3/24 gói, chưa đo được lần nào vì thiếu RAM. Hiện lắp N khung bằng
   quy hoạch động trên ứng viên của từng sự kiện con, ép tăng dần theo thời gian.
   Có kỹ thuật nào tốt hơn cho việc dóng một chuỗi sự kiện vào một video?

6. **Q&A** — 3/24 gói, không nằm trong phần được chấm public nên không đo được
   bằng leaderboard. Đo trên tập dev (42 câu, chấm ở khung đáp án đúng): trả lời
   từ OCR/ASR bằng Gemini đúng **31,0%** khớp chuỗi chính xác, **57,1%** nếu tính
   khớp lỏng. Các câu hỏi **đếm** và **màu sắc** trượt hết vì OCR không chứa
   thông tin đó. Có cách nào rẻ hơn VLM cho nhóm câu đó không?

---

## 8. Kỷ luật đo của đội — xin giữ khi tư vấn

* Không bật tính năng nào mặc định trước khi nó thắng trên tập dev.
* Chấm ở **hai mức dung sai**; đảo dấu giữa hai mức = không kết luận được.
* Luôn báo **thắng–thua–hoà** kèm ngưỡng nhiễu, không chỉ điểm trung bình.
* Mốc nền phải là **cấu hình mạnh nhất hiện có**, không phải cái tiện tay.
* **Chỉ đổi một thứ mỗi lần.**

Lời khuyên kiểu *"BTC thường chấm khớp từ khoá hoặc F1"* đã bị bác — quy định
BTC nói *"so sánh chính xác về mặt ngữ nghĩa"* (tr.2) và *"so sánh dưới dạng
chuỗi chính xác"* (tr.8), không nói F1.
