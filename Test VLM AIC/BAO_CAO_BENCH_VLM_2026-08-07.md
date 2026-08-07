# Báo cáo bench VLM cho Q&A — AIC 2026 (mục 0.c, đợt 3)

**Ngày chạy:** 2026-08-07 · **Harness:** `vlm_bench.py` · **Bộ câu hỏi:** `questions.json`
(18 câu, 6 loại: count/ocr/color/name/object/action, trên 2 video `L23_V001` +
`L27_V001`, đủ 2 ngôn ngữ vi/en) · **temperature = 0** (bắt buộc, xem D0.3 đợt 2
trong `Ke_hoach_AIC2026_v4.md`).

Mục tiêu: chạy hết các nguồn model miễn phí khả dụng — Gemini free tier,
OpenRouter `:free`, và Ollama local đã cài — để có thêm dữ liệu chốt model
Q&A. File thô nằm ở `runs/2026-08-07_gemini/`, `runs/2026-08-07_openrouter/`,
`runs/2026-08-07_ollama/` (mỗi thư mục có `raw.jsonl` + `report.md` do
harness tự sinh).

**Đã đổi so với kế hoạch giữa chừng:** nhóm Ollama đang chạy trên máy
**không có GPU** — mỗi lượt gọi mất 100–230 giây (CPU), làm RAM tụt còn 1,2 GB
trống trong lúc `minicpm-v:8b` đang chạy. Đã dừng theo yêu cầu của người dùng
sau khi `qwen2.5vl:3b` chạy xong trọn vẹn và `minicpm-v:8b` mới được 15/18
câu. `llama3.2-vision:11b` và `moondream` (đã cài) **chưa được test chính
thức** trong đợt này. `qwen2.5vl:7b` không cài được nên không có trong danh
sách.

---

## 1. Bảng tổng hợp toàn bộ model đã chạy

| Model | Nguồn | Lang | Đúng | Định dạng | Đồng thuận | p50 (s) | Lỗi/tổng | Ghi chú |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **gemini-3.1-flash-lite** | Gemini API | vi | **67% (12/18)** | 94% | **100%** | 5.5 | 0/54 | Dẫn đầu, ổn định nhất |
| gemini-3.1-flash-lite | Gemini API | en | 50% (9/18) | 100% | 100% | 5.3 | 0/54 | |
| gemini-3.5-flash-lite | Gemini API | vi | 61% (11/18) | 96% | 91% | 5.1 | 0/54 | Xem cảnh báo §3 |
| gemini-3.5-flash-lite | Gemini API | en | 56% (10/18) | 94% | 93% | 5.1 | 0/54 | |
| gemini-3.6-flash | Gemini API | vi | 56% (5/9) | 20% | 70% | 5.1 | 29/54 | Hết quota giữa chừng |
| gemini-3.6-flash | Gemini API | en | 0% (0/1) | 0% | 0% | — | 54/54 | Hết quota gần như ngay |
| gemma-4-26b-a4b | OpenRouter free | vi | 47% (8/17) | 82% | 100% | 11.1 | 1/18 | Dùng được, chậm hơn |
| gemma-4-26b-a4b | OpenRouter free | en | 38% (6/16) | 88% | 100% | 16.6 | 2/18 | |
| gemma-4-31b | OpenRouter free | vi/en | 0% | 0% | 0% | — | **18/18** | Rate-limit upstream 100% |
| nemotron-nano-12b-vl | OpenRouter free | vi | 12% (1/8) | 88% | 100% | 5.4 | 10/18 | Đa số timeout ~125s |
| nemotron-nano-12b-vl | OpenRouter free | en | 0% (0/5) | 100% | 100% | 5.3 | 13/18 | |
| qwen2.5vl-3b (Ollama) | Local CPU | vi | 61% (11/18) | 89% | 100%* | **170.3** | 0/18 | Đúng khá cao, quá chậm |
| minicpm-v:8b (Ollama) | Local CPU | vi | 20% (3/15)† | 80%† | — | ~150 | dừng giữa | †chưa hoàn tất, không tính điểm |
| llama3.2-vision:11b | Local CPU | — | — | — | — | — | — | Chưa chạy (dừng sớm) |
| moondream | Local CPU | — | — | — | — | — | — | Chưa chạy chính thức, xem §5 |

\* `runs=1` cho toàn bộ OpenRouter và Ollama trong đợt này (giới hạn quota /
thời gian CPU) → cột "đồng thuận" ở các nhóm đó không đo được sự ổn định
thật, chỉ có nghĩa với Gemini (`runs=3`).

---

## 2. Ba nhóm, ba câu chuyện khác nhau

### A. Gemini free tier — chạy trọn vẹn (324/324 lượt, không model nào bị khoá hẳn)

`gemini-3.1-flash-lite` **tiếp tục dẫn đầu**, khớp với hai đợt đo trước trong
kế hoạch: đúng 67% (vi), định dạng gần tuyệt đối, và quan trọng nhất —
**đồng thuận 100% trên cả 108 lượt gọi** (18 câu × 3 lần × 2 ngôn ngữ). Đây
là lần thứ ba con số này được xác nhận độc lập.

`gemini-3.6-flash` **hết quota gần như ngay lập tức** — 83/108 lượt lỗi.
Đúng như cảnh báo trong D0.3: model này không đáng tin cho khối lượng gọi
lớn trên free tier.

### B. OpenRouter free — chỉ 1/3 model dùng được

`gemma-4-26b-a4b` là model OpenRouter duy nhất hoạt động ổn định (0-2 lỗi/18),
nhưng độ đúng (38-47%) và tốc độ (p50 11-17s) đều kém hơn rõ rệt so với
Gemini trực tiếp.

`gemma-4-31b` **thất bại 100%** — lỗi trả về nguyên văn:
`"google/gemma-4-31b-it:free is temporarily rate-limited upstream"`. Đây
chính xác là lỗi đã ghi trong kế hoạch (PHẦN D0.3, mục 4: *"gemma-4-31b chạy
tốt qua Google API — trong khi qua OpenRouter nó thất bại"*). Xác nhận lại:
**đừng test model này qua OpenRouter, chỉ test qua Google AI Studio trực
tiếp nếu muốn đánh giá đúng năng lực của nó.**

`nemotron-nano-12b-vl` đa số timeout ở mốc ~125 giây rồi lỗi (23/36 lượt),
số ít trả lời được thì đúng rất thấp (0-12%). Không dùng được ở trạng thái
hạ tầng hiện tại.

### C. Ollama local — bị chặn bởi phần cứng, không phải bởi model

Máy chạy benchmark **không có GPU** (không có `nvidia-smi`). Một lượt gọi
VLM qua Ollama mất 100–230 giây trên CPU, so với 5 giây qua Gemini API —
chậm hơn **20-40 lần**. `qwen2.5vl:3b` cho độ đúng khá tốt so với kích
thước (61%, ngang `gemini-3.5-flash-lite`), nhưng ở tốc độ này, chạy 100 câu
× nhiều vòng thử nghiệm trong 4 tuần là không khả thi trên máy này.

`minicpm-v:8b` bị dừng giữa chừng theo yêu cầu (RAM chỉ còn 1,2 GB trống,
có nguy cơ ảnh hưởng các ứng dụng khác đang mở). 15/18 câu đã trả lời cho
tín hiệu sơ bộ **kém hơn** `qwen2.5vl:3b` (đúng 3/15 ≈ 20%) — không đủ dữ
liệu để kết luận chắc chắn, chỉ nêu để tham khảo.

---

## 3. Một phát hiện cần lưu ý — mâu thuẫn với đợt đo trước

Đợt đo trước (kế hoạch, D0.3 đợt 2) kết luận `gemini-3.5-flash-lite`
**"KHÔNG tái lập được, kể cả ở temperature=0"** — 7/10 câu đổi đáp án giữa
các lần gọi, dựa trên **một câu hỏi đếm vật thể đông** (đếm 10-17 người/xe)
lặp lại 3 lần.

Đợt đo này (18 câu đa dạng loại, mỗi câu 3 lần, temp=0) đo được đồng thuận
**91-93%** cho cùng model — cao hơn nhiều so với con số cũ.

**Đây không hẳn là mâu thuẫn thật** — hai phép đo không cùng loại câu hỏi:
đợt cũ chỉ đo một dạng khó nhất (đếm vật thể đông, dễ dao động), đợt này
gồm cả câu dễ tái lập (màu sắc, tên) lẫn câu đếm số lượng nhỏ (2-5). Xem
bảng theo loại câu hỏi trong `runs/2026-08-07_gemini/report.md` — loại
`count` của `gemini-3.5-flash-lite` chỉ đạt 75%, thấp hơn `color`/`name`
(100%). Có khả năng độ bất ổn của model này **tập trung ở câu đếm đông**,
không phải toàn bộ output.

> **Khuyến nghị:** đừng vội phục hồi `gemini-3.5-flash-lite` làm ứng viên
> chính dựa trên đồng thuận 91-93% này. Cần thêm một đợt test riêng, tập
> trung ≥10 câu đếm vật thể đông (>10 đối tượng), trước khi kết luận lại.

---

## 4. Khuyến nghị chốt (mục 0.c)

Theo đúng thứ tự ưu tiên đã thống nhất trong kế hoạch (định dạng → đồng
thuận → độ đúng → tốc độ/khối lượng):

| Vai trò | Model | Lý do |
| --- | --- | --- |
| **Chính** | `gemini-3.1-flash-lite`, temp=0, gọi trực tiếp Google AI Studio | Dẫn đầu nhất quán qua **3 đợt đo độc lập**; đồng thuận 100%; định dạng 94-100%; p50 ~5,3s |
| **Dự phòng #1** (khi model chính hết quota trong ngày) | `gemini-3.5-flash-lite` | Nhanh, định dạng tốt — nhưng cần test thêm câu đếm đông trước khi tin hoàn toàn (§3) |
| **Dự phòng #2** (khi cả hai Gemini hết quota) | `gemma-4-26b-a4b` qua OpenRouter | Model OpenRouter duy nhất còn sống; đúng thấp hơn nhưng không lỗi |
| **Dự phòng offline cuối cùng** | `qwen2.5vl:3b` qua Ollama | Chỉ nên dùng nếu mất mạng hoàn toàn, VÀ chỉ nếu chạy trên máy có GPU — trên CPU hiện tại quá chậm để dùng thật |

**Loại khỏi danh sách ứng viên** (dựa trên số đo, không phải suy đoán):

- `gemini-3.6-flash` — quota free cạn quá nhanh để tin cậy.
- `gemma-4-31b` qua OpenRouter — rate-limit upstream 100%, phải test lại qua
  Google API nếu muốn dùng.
- `nemotron-nano-12b-vl` — đa số timeout, hạ tầng OpenRouter chưa ổn định
  cho model này.

**Không đủ dữ liệu, cần chạy bổ sung khi máy rảnh hơn:**

- `minicpm-v:8b`, `llama3.2-vision:11b`, `moondream` (Ollama) — nên chạy lại
  khi có GPU hoặc trên máy khác trong nhóm, vì trên CPU hiện tại chi phí
  thời gian quá cao so với giá trị thông tin thu được.

---

## 5. Ghi chú thêm — moondream (quan sát sơ bộ, không có trong file kết quả)

Trước đợt chạy chính thức, có một lượt kiểm tra nhanh (không lưu lại được
raw data vì bị dọn trước khi quyết định giữ) cho `moondream` (1.8B) trên
17/18 câu: model **không tuân theo được prompt JSON tiếng Việt** — output là
chuỗi lặp từ vô nghĩa kiểu `"không biết không xác chắc chắn rằng..."`, định
dạng đúng 0/17. Đây chỉ là quan sát tham khảo, không phải kết quả đã kiểm
chứng đầy đủ như các model khác trong báo cáo này — nhưng đủ để dự đoán
`moondream` nhiều khả năng không đạt ngưỡng dùng được, do kích thước quá nhỏ
so với yêu cầu prompt.

---

## 6. Giới hạn của đợt đo này (đọc trước khi trích dẫn số liệu)

1. **18 câu vẫn dưới ngưỡng ≥50 câu** mà kế hoạch yêu cầu trước khi so sánh
   model một cách chắc chắn (D0.3, "Việc tiếp theo", mục 1).
2. **OpenRouter và Ollama chỉ chạy `runs=1`** (do giới hạn quota ngày /
   thời gian CPU) — không đo được độ ổn định (đồng thuận) cho các model đó,
   chỉ có con số cho Gemini.
3. Hàm so khớp đáp án dùng khớp chuỗi chặt (`match()` trong `vlm_bench.py`)
   — một vài câu trả lời đúng về ý nhưng dài hơn ground-truth (vd.
   `"sprinting to finish"` thay vì `"sprinting"`) bị chấm sai. Không ảnh
   hưởng tới xếp hạng tương đối giữa các model vì áp dụng đồng nhất, nhưng
   con số đúng tuyệt đối có thể thấp hơn thực tế một chút.
4. Bộ câu hỏi hiện chỉ trải trên 2 video (`L23_V001`, `L27_V001`), thuộc
   nhóm L23/L27 — chưa đại diện cho toàn bộ 10 nhóm L, đặc biệt chưa có câu
   nào từ nhóm **L26** (chiếm 57% kho dữ liệu, xem PHẦN A2 trong kế hoạch).

## 7. Việc tiếp theo

1. Mở rộng `questions.json` lên ≥50 câu, lấy mẫu phân tầng theo nhóm L
   (ưu tiên thêm câu từ L26) trước khi so sánh model lần cuối.
2. Test riêng `gemini-3.5-flash-lite` với ≥10 câu đếm vật thể đông để giải
   quyết mâu thuẫn ở §3.
3. Test `gemma-4-31b` qua Google AI Studio trực tiếp (không qua OpenRouter).
4. Nếu có máy có GPU trong nhóm, chạy lại đủ 5 model Ollama (bao gồm
   `llama3.2-vision:11b`, `moondream`) để có bộ so sánh local đầy đủ.
5. Chốt chính thức: **model = `gemini-3.1-flash-lite`, ngôn ngữ = vi,
   temperature = 0**, theo dữ liệu hiện có — điền vào mục "Chốt" trong
   `runs/2026-08-07_gemini/report.md` nếu nhóm đồng ý.
