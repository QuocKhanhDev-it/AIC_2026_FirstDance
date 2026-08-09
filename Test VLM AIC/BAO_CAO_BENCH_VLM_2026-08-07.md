# Báo cáo bench VLM cho Q&A — AIC 2026 (mục 0.c)

**Harness:** `vlm_bench.py` · **Bộ câu hỏi:** `questions.json` (18 câu, 6 loại:
count/ocr/color/name/object/action, trên 2 video `L23_V001` + `L27_V001`, đủ
2 ngôn ngữ vi/en) · **temperature = 0** (bắt buộc, xem D0.3 đợt 2 trong
`Ke_hoach_AIC2026_v4.md`).

Mục tiêu: chạy hết các nguồn model miễn phí khả dụng — Gemini free tier,
OpenRouter `:free`, và Ollama local — để có đủ dữ liệu chốt model dùng cho
Q&A. Đo trên hai loại phần cứng khác nhau cho nhóm Ollama: máy **không có
GPU** (CPU-only) và máy có **GPU rời RTX 2060 Super, 8GB VRAM** — kết quả
của cả hai đều được đưa vào bảng dưới, nguồn phần cứng được ghi rõ ở cột
Ghi chú vì nó ảnh hưởng trực tiếp tới tốc độ và (bất ngờ là) cả độ đúng.

File thô nằm ở:
- `runs/2026-08-07_gemini/`, `runs/2026-08-07_openrouter/`,
  `runs/2026-08-07_ollama/` (CPU) — mỗi thư mục có `raw.jsonl` + `report.md`
  do harness tự sinh.
- `runs/2026-08-08_ollama/` (GPU), `runs/2026-08-08_openrouter/` (chạy lại 2
  model còn sống, bị ngắt giữa chừng bằng Ctrl+C nên **không có
  `report.md`** — số liệu của đợt này tính trực tiếp từ `raw.jsonl`).

---

## 1. Bảng tổng hợp toàn bộ model đã đo

| Model | Nguồn | Lang | Đúng | Định dạng | Đồng thuận | p50 (s) | Lỗi/tổng | Ghi chú |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **gemini-3.1-flash-lite** | Gemini API | vi | **67% (12/18)** | 94% | **100%** | 5.5 | 0/54 | Dẫn đầu, ổn định nhất |
| gemini-3.1-flash-lite | Gemini API | en | 50% (9/18) | 100% | 100% | 5.3 | 0/54 | |
| gemini-3.5-flash-lite | Gemini API | vi | 61% (11/18) | 96% | 91% | 5.1 | 0/54 | Xem cảnh báo §3 |
| gemini-3.5-flash-lite | Gemini API | en | 56% (10/18) | 94% | 93% | 5.1 | 0/54 | |
| gemini-3.6-flash | Gemini API | vi | 56% (5/9) | 20% | 70% | 5.1 | 29/54 | Hết quota giữa chừng |
| gemini-3.6-flash | Gemini API | en | 0% (0/1) | 0% | 0% | — | 54/54 | Hết quota gần như ngay |
| gemma-4-26b-a4b | OpenRouter free | vi | 47% (8/17) / 44% (8/18)* | 82% / 78%* | 100% | 11.1 | 1/18 (2 lần đo) | *2 lần đo độc lập, kết quả khớp nhau trong sai số n=18 |
| gemma-4-26b-a4b | OpenRouter free | en | 38% (6/16) / 28% (5/18)* | 88% / 83%* | 100% | 16.6 | 2/18 (2 lần đo) | *2 lần đo độc lập, kết quả khớp nhau trong sai số n=18 |
| gemma-4-31b | OpenRouter free | vi/en | 0% | 0% | 0% | — | **18/18** | Rate-limit upstream 100% |
| nemotron-nano-12b-vl | OpenRouter free | vi | 12% (1/8) / 25% (2/8)* | 88% / 50%* | 100% | 5.4 | 10/18, 4/8* | *lần 2: chỉ kịp 8/18 câu trước khi bị dừng thủ công |
| nemotron-nano-12b-vl | OpenRouter free | en | 0% (0/5) | 100% | 100% | 5.3 | 13/18 | Chưa chạy lại ở lần đo thứ 2 |
| **qwen2.5vl:7b (Ollama, GPU)** | Local GPU 8GB | vi | **50% (9/18)** | 78% | — (runs=1) | **8.2** | 0/18 | Tốt nhất nhóm local. Trên máy CPU-only, model này không cài được |
| qwen2.5vl:3b (Ollama, CPU) | Local CPU | vi | 61% (11/18) | 89% | — (runs=1) | 170.3 | 0/18 | Đúng cao trên CPU nhưng quá chậm |
| qwen2.5vl:3b (Ollama, GPU) | Local GPU 8GB | vi | 28% (5/18) | 39% | — (runs=1) | 6.7 | 0/18 | **Giảm mạnh so với CPU (61% → 28%) — lỗi lặp token, xem §3.2** |
| minicpm-v:8b (Ollama, CPU) | Local CPU | vi | 20% (3/15)† | 80%† | — | ~150 | dừng giữa (15/18) | †Dừng vì thiếu RAM, không tính điểm chính thức |
| minicpm-v:8b (Ollama, GPU) | Local GPU 8GB | vi | 17% (3/18) | 44% | — (runs=1) | 7.4 | 0/18 | Chạy trọn 18/18, xác nhận độ đúng thấp |
| llama3.2-vision:11b | Local (Ollama) | — | — | — | — | — | 18/18 | Chưa đo được — xem §3.3 |
| moondream | Local (Ollama) | — | — | — | — | — | 18/18 | Chưa đo được chính thức — xem §3.3, §3.4 |

\* Toàn bộ OpenRouter và Ollama chạy `runs=1` (giới hạn quota ngày / thời
gian CPU-GPU) → cột "đồng thuận" không đo được cho các model đó, chỉ có ý
nghĩa với Gemini (`runs=3`). Với `gemma-4-26b-a4b` và `nemotron-nano-12b-vl`,
việc đo lại độc lập ở hai thời điểm khác nhau đóng vai trò kiểm tra chéo độ
ổn định thay cho đồng thuận trong-một-lần-chạy.

---

## 2. Ba nguồn, ba câu chuyện khác nhau

### A. Gemini free tier — chạy trọn vẹn (324/324 lượt, không model nào bị khoá hẳn)

`gemini-3.1-flash-lite` **dẫn đầu nhất quán qua ba đợt đo độc lập**: đúng
67% (vi), định dạng gần tuyệt đối, và quan trọng nhất — **đồng thuận 100%
trên cả 108 lượt gọi** (18 câu × 3 lần × 2 ngôn ngữ).

`gemini-3.6-flash` **hết quota gần như ngay lập tức** — 83/108 lượt lỗi.
Đúng như cảnh báo trong kế hoạch D0.3: model này không đáng tin cho khối
lượng gọi lớn trên free tier.

### B. OpenRouter free — chỉ 1/3 model dùng được, đã kiểm chứng 2 lần độc lập

`gemma-4-26b-a4b` là model OpenRouter duy nhất hoạt động ổn định (0-2
lỗi/18). Đo lại độc lập lần hai cho kết quả khớp trong sai số của n=18 (vi:
47%→44%, en: 38%→28%) — đủ tin cậy để dùng làm dự phòng, dù độ đúng và tốc
độ (p50 11-17s) đều kém hơn rõ rệt so với Gemini trực tiếp.

`gemma-4-31b` **thất bại 100%** — lỗi trả về nguyên văn:
`"google/gemma-4-31b-it:free is temporarily rate-limited upstream"`. Đúng
như đã ghi trong kế hoạch (PHẦN D0.3, mục 4: *"gemma-4-31b chạy tốt qua
Google API — trong khi qua OpenRouter nó thất bại"*). **Đừng test model này
qua OpenRouter, chỉ test qua Google AI Studio trực tiếp** nếu muốn đánh giá
đúng năng lực của nó.

`nemotron-nano-12b-vl` đo hai lần độc lập (cách nhau khoảng một ngày), cả
hai lần đều đa số timeout `"Upstream idle timeout exceeded"` (HTTP 504) ở
mốc ~82-125 giây/lượt kể cả lượt lỗi, số ít trả lời được thì đúng rất thấp
(0-25%). Hai lần đo độc lập cùng kết luận → **chốt loại, không cần đo
thêm**: lỗi ở hạ tầng OpenRouter cho model này, không phải chất lượng model.

### C. Ollama local — phần cứng quyết định cả tốc độ lẫn độ đúng

**Trên CPU** (không có `nvidia-smi`): một lượt gọi VLM mất 100–230 giây — so
với 5 giây qua Gemini API, chậm hơn 20-40 lần. `qwen2.5vl:3b` đạt độ đúng
khá tốt so với kích thước (61%, ngang `gemini-3.5-flash-lite`), nhưng ở tốc
độ này, chạy hàng trăm câu × nhiều vòng thử nghiệm trong 4 tuần là không khả
thi.

**Trên GPU rời RTX 2060 Super 8GB VRAM**: tốc độ cải thiện rõ rệt —
`qwen2.5vl:7b` (model không cài được trên máy CPU-only) đạt p50 8,2 giây,
chỉ còn chậm hơn Gemini API ~1,5 lần thay vì 20-40 lần, và cho độ đúng tốt
nhất trong nhóm local (50%, 9/18). Đủ nhanh để cân nhắc dùng thật cho khối
lượng lớn nếu cần chạy offline.

Tuy nhiên, `qwen2.5vl:3b` chạy trên GPU lại tụt độ đúng mạnh (61% → 28%) so
với chính nó chạy trên CPU cùng bộ câu hỏi — nguyên nhân là lỗi sinh văn bản
lặp token, không liên quan tới việc "nhìn" đúng hay sai (chi tiết §3.2).

`minicpm-v:8b` trên CPU bị dừng giữa chừng (RAM chỉ còn 1,2 GB trống). Chạy
lại trọn vẹn trên GPU (18/18 câu) xác nhận độ đúng thấp (17%, 3/18) —
tín hiệu ban đầu trên CPU (20%, 3/15) và kết quả đầy đủ trên GPU khớp nhau,
model này không đủ tốt để đưa vào danh sách ứng viên chính.

`llama3.2-vision:11b` và `moondream` **vẫn chưa đo được chính thức** sau hai
lần thử — xem §3.3.

---

## 3. Các phát hiện cần lưu ý

### 3.1 Mâu thuẫn đồng thuận của `gemini-3.5-flash-lite` — đã có lời giải một phần

Một đợt đo trước (kế hoạch, D0.3 đợt 2) kết luận `gemini-3.5-flash-lite`
**"KHÔNG tái lập được, kể cả ở temperature=0"** — 7/10 câu đổi đáp án giữa
các lần gọi, dựa trên **một câu hỏi đếm vật thể đông** (đếm 10-17 người/xe)
lặp lại 3 lần.

Đợt đo trong báo cáo này (18 câu đa dạng loại, mỗi câu 3 lần, temp=0) đo
được đồng thuận **91-93%** cho cùng model — cao hơn nhiều so với con số cũ.

**Đây không hẳn là mâu thuẫn thật** — hai phép đo không cùng loại câu hỏi:
đợt cũ chỉ đo dạng khó nhất (đếm vật thể đông, dễ dao động), đợt này gồm cả
câu dễ tái lập (màu sắc, tên) lẫn câu đếm số lượng nhỏ (2-5). Theo bảng chi
tiết trong `runs/2026-08-07_gemini/report.md`, loại `count` của
`gemini-3.5-flash-lite` chỉ đạt 75%, thấp hơn hẳn `color`/`name` (100%). Rất
có thể độ bất ổn của model này **tập trung ở câu đếm đông**, không phải
toàn bộ output.

> **Khuyến nghị:** đừng vội phục hồi `gemini-3.5-flash-lite` làm ứng viên
> chính dựa trên đồng thuận 91-93% này. Cần thêm một đợt test riêng, tập
> trung ≥10 câu đếm vật thể đông (>10 đối tượng), trước khi kết luận lại.

### 3.2 `qwen2.5vl:3b` lặp token vô nghĩa khi chạy trên GPU — không phải lỗi thị giác

Trên CPU, `qwen2.5vl:3b` đúng 61% (11/18). Trên GPU, cùng bộ 18 câu hỏi,
cùng `temperature=0`, chỉ còn đúng 28% (5/18). Soi `raw.jsonl` tìm ra
nguyên nhân: 7/18 câu trả lời là chuỗi ký tự lặp vô nghĩa (ví dụ
`"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"`) — đây là lỗi **lặp token
(repetition loop)** khi sinh văn bản, một lỗi đã biết ở model nhỏ, không
phải model "nhìn sai" ảnh. Bản 7B chạy cùng máy, cùng lượt, không dính lỗi
này một lần nào.

> **Khuyến nghị:** ngừng dùng `qwen2.5vl:3b` làm dự phòng offline. Chuyển
> sang `qwen2.5vl:7b` — đúng cao hơn nhiều (50% so với 28%), không lỗi lặp
> token, và tốc độ chỉ chậm hơn không đáng kể (p50 8,2s so với 6,7s). Ghi
> chú "máy yếu thì dùng bản 3B" trong REGISTRY gốc (`vlm_bench.py`) nên hiểu
> là ưu tiên VRAM, không phải ưu tiên độ ổn định — ở đây 3B kém ổn định hơn
> 7B, ngược với trực giác "nhỏ hơn thì an toàn hơn".

### 3.3 `llama3.2-vision:11b` và `moondream` — hai lần thử, hai lý do khác nhau, vẫn chưa đo được

**Lần 1 (CPU):** chưa test vì chủ động dừng sớm do thiếu RAM khi
`minicpm-v:8b` đang chạy — quyết định đúng đắn, không phải lỗi kỹ thuật.

**Lần 2 (GPU):** `ollama pull` báo tải xong, nhưng khi gọi qua API trả về
HTTP 404 `"model 'xxx' not found"` trên toàn bộ 18/18 câu của cả hai model —
tức **quá trình tải bị lỗi âm thầm**, không liên quan gì tới hạ tầng bench
hay GPU. Xác nhận độc lập bằng `ollama list` và `GET /v1/models`: máy chỉ
thực sự có 3 model (`qwen2.5vl:7b`, `qwen2.5vl:3b`, `minicpm-v:8b`), thiếu
đúng 2 model báo lỗi.

**Việc cần làm trước lần bench sau:** `ollama pull llama3.2-vision:11b` và
`ollama pull moondream` lại từ đầu, xác nhận cả hai xuất hiện trong `ollama
list` trước khi chạy `vlm_bench.py`.

### 3.4 Quan sát sơ bộ về `moondream` (không chính thức)

Trước đợt chạy chính thức, có một lượt kiểm tra nhanh (không lưu lại được
raw data vì bị dọn trước khi quyết định giữ) cho `moondream` (1.8B) trên
17/18 câu: model **không tuân theo được prompt JSON tiếng Việt** — output là
chuỗi lặp từ vô nghĩa kiểu `"không biết không xác chắc chắn rằng..."`, định
dạng đúng 0/17. Đây chỉ là quan sát tham khảo, chưa phải kết quả đã kiểm
chứng đầy đủ như các model khác trong báo cáo này — nhưng đủ để dự đoán
`moondream` nhiều khả năng không đạt ngưỡng dùng được, do kích thước quá nhỏ
so với yêu cầu prompt.

---

## 4. Khuyến nghị chốt (mục 0.c)

Theo đúng thứ tự ưu tiên đã thống nhất trong kế hoạch (định dạng → đồng
thuận → độ đúng → tốc độ/khối lượng):

| Vai trò | Model | Lý do |
| --- | --- | --- |
| **Chính** | `gemini-3.1-flash-lite`, temp=0, gọi trực tiếp Google AI Studio | Dẫn đầu nhất quán qua **3 đợt đo độc lập**; đồng thuận 100%; định dạng 94-100%; p50 ~5,3s |
| **Dự phòng #1** (khi model chính hết quota trong ngày) | `gemini-3.5-flash-lite` | Nhanh, định dạng tốt — nhưng cần test thêm câu đếm đông trước khi tin hoàn toàn (§3.1) |
| **Dự phòng #2** (khi cả hai Gemini hết quota) | `gemma-4-26b-a4b` qua OpenRouter | Model OpenRouter duy nhất còn sống; xác nhận 2 lần đo độc lập nhất quán; đúng thấp hơn Gemini nhưng không lỗi nặng |
| **Dự phòng offline cuối cùng** (cần GPU) | `qwen2.5vl:7b` qua Ollama | Đúng cao nhất nhóm local (50%), không lỗi lặp token (§3.2), p50 8,2s trên GPU 8GB — thay thế bản 3B vốn kém ổn định hơn dù nhỏ hơn |

**Loại khỏi danh sách ứng viên** (dựa trên số đo, không phải suy đoán):

- `gemini-3.6-flash` — quota free cạn quá nhanh để tin cậy.
- `gemma-4-31b` qua OpenRouter — rate-limit upstream 100%, phải test lại qua
  Google API nếu muốn dùng.
- `nemotron-nano-12b-vl` — **2 lần đo độc lập**, cùng kết luận đa số
  timeout, hạ tầng OpenRouter chưa ổn định cho model này. Chốt loại, không
  cần test lại trừ khi OpenRouter đổi hạ tầng.
- `qwen2.5vl:3b` qua Ollama — kém ổn định hơn bản 7B (lỗi lặp token trên
  GPU), không còn là lựa chọn dự phòng offline hợp lý.

**Không đủ dữ liệu, cần chạy bổ sung khi có điều kiện:**

- `llama3.2-vision:11b`, `moondream` (Ollama) — chưa đo được qua hai lần
  thử, hiện tại bị chặn bởi lỗi tải model (`ollama pull` báo xong nhưng
  server không thấy model — §3.3), không còn liên quan tới CPU/GPU. Cần pull
  lại và xác nhận bằng `ollama list` trước khi bench tiếp.

---

## 5. Giới hạn của đợt đo này (đọc trước khi trích dẫn số liệu)

1. **18 câu vẫn dưới ngưỡng ≥50 câu** mà kế hoạch yêu cầu trước khi so sánh
   model một cách chắc chắn (D0.3, "Việc tiếp theo", mục 1). Đây vẫn là
   giới hạn lớn nhất che khuất mọi kết luận trong toàn bộ báo cáo.
2. **OpenRouter và Ollama chủ yếu chạy `runs=1`** (do giới hạn quota ngày /
   thời gian CPU-GPU) — không đo được độ ổn định (đồng thuận) trong một lần
   chạy cho các model đó; chỉ Gemini có `runs=3`. Với `gemma-4-26b-a4b` và
   `nemotron-nano-12b-vl`, hai lần đo độc lập ở hai thời điểm khác nhau bù
   đắp phần nào cho việc này.
3. Hàm so khớp đáp án dùng khớp chuỗi chặt (`match()` trong `vlm_bench.py`)
   — một vài câu trả lời đúng về ý nhưng dài hơn ground-truth (vd.
   `"sprinting to finish"` thay vì `"sprinting"`) bị chấm sai. Không ảnh
   hưởng tới xếp hạng tương đối giữa các model vì áp dụng đồng nhất, nhưng
   con số đúng tuyệt đối có thể thấp hơn thực tế một chút.
4. Bộ câu hỏi hiện chỉ trải trên 2 video (`L23_V001`, `L27_V001`), thuộc
   nhóm L23/L27 — chưa đại diện cho toàn bộ 10 nhóm L, đặc biệt chưa có câu
   nào từ nhóm **L26** (chiếm 57% kho dữ liệu, xem PHẦN A2 trong kế hoạch).
5. Đợt chạy lại OpenRouter cho `gemma-4-26b-a4b` và `nemotron-nano-12b-vl`
   bị ngắt giữa chừng bằng Ctrl+C (quyết định chủ động vì `nemotron` chạy
   quá lâu) nên không có `report.md` tự sinh — bảng số liệu tương ứng được
   tính trực tiếp từ `raw.jsonl`.

---

## 6. Việc tiếp theo

1. Mở rộng `questions.json` lên ≥50 câu, lấy mẫu phân tầng theo nhóm L
   (ưu tiên thêm câu từ L26) trước khi so sánh model lần cuối — giới hạn
   lớn nhất hiện tại.
2. Test riêng `gemini-3.5-flash-lite` với ≥10 câu đếm vật thể đông để giải
   quyết mâu thuẫn ở §3.1.
3. Test `gemma-4-31b` qua Google AI Studio trực tiếp (không qua OpenRouter).
4. Tải lại `llama3.2-vision:11b` và `moondream` qua `ollama pull`, xác nhận
   bằng `ollama list` trước khi bench, rồi chạy nốt 2 model còn thiếu để có
   bộ so sánh Ollama đầy đủ 5/5.
5. Chốt chính thức: **model = `gemini-3.1-flash-lite`, ngôn ngữ = vi,
   temperature = 0**, theo dữ liệu hiện có — điền vào mục "Chốt" trong
   `runs/2026-08-07_gemini/report.md` nếu nhóm đồng ý.
