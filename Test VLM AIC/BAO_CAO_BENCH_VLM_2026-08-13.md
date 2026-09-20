# Báo cáo bench VLM cho Q&A — AIC 2026 (mục 0.c, đợt 2)

**Nối tiếp** `BAO_CAO_BENCH_VLM_2026-08-07.md`. Đợt này thực hiện đúng 4 việc đã
chốt ở mục "Việc tiếp theo" của báo cáo trước, cộng thêm dữ liệu mới (3 video
`L26_V003`, `L26_V083`, `L28_V012`) do người dùng bổ sung.

**Harness:** `vlm_bench.py` (đã thêm entry `gemma-4-31b-it` gọi thẳng Google AI
Studio) · **temperature = 0**.

File thô: `runs/2026-08-13_dense_count/`, `runs/2026-08-13_gemma_google/`,
`runs/2026-08-13_ollama_<model>/` — mỗi thư mục có `raw.jsonl` + `report.md`.

---

## 0. Tóm tắt 4 việc đã làm

| # | Việc | Kết quả |
| --- | --- | --- |
| 1 | Mở rộng `questions.json` lên ≥50 câu, ưu tiên L26 | **Xong — 52 câu**, 5 video (xem §1) |
| 2 | Test riêng `gemini-3.5-flash-lite` với ≥10 câu đếm vật thể đông (>10) | **Xong** — xác nhận model **yếu hẳn** ở dạng câu này (§2) |
| 3 | Test `gemma-4-31b` qua Google AI Studio trực tiếp | **Xong** — chạy được, không rate-limit, nhưng format vẫn kém (§3) |
| 4 | Test cả 5 model Ollama | **4/5 xong** — `llama3.2-vision:11b` lỗi hạ tầng, không phải lỗi model (§4) |

**Phát hiện quan trọng nhất của đợt này:** trên bộ câu hỏi lớn hơn (52 câu, đa
dạng nguồn hơn), `qwen2.5vl:7b` qua Ollama đạt **52-58% đúng — cao hơn cả**
`gemma-4-31b-it` (35-40%) và ngang ngửa các model Gemini free tier ở lần đo
trước. Model local rẻ nhất lại đang là ứng viên dự phòng offline mạnh nhất,
không chỉ là "phương án cuối cùng". Chi tiết ở §5.

---

## 1. Bộ câu hỏi mở rộng — 52 câu, 5 video

| Video | Số câu | Ghi chú nội dung |
| --- | --- | --- |
| `L23_V001` | 10 | Giải đua xe đạp (giữ nguyên từ đợt 1) |
| `L27_V001` | 8 | Du lịch Châu Đốc — Núi Sam (giữ nguyên từ đợt 1) |
| `L26_V003` | 12 | **Mới** — chương trình nấu ăn "Món ngon mỗi ngày" (Ajinomoto), tập 1 |
| `L26_V083` | 12 | **Mới** — cùng chương trình, tập khác, 2 người dẫn |
| `L28_V012` | 10 | **Mới** — phim tài liệu "Đời người": khảo cổ, bảo tàng, sông nước An Giang |

Phân bố theo loại: color 12 · object 12 · ocr 9 · count 8 · action 6 · name 5.

L26 chiếm 24/52 câu (46%) — ưu tiên cao nhất theo đúng yêu cầu, dù chưa bằng tỷ
trọng 57% của L26 trong toàn kho dữ liệu (do L23/L27 giữ nguyên để so sánh
xuyên-đợt được).

Ground truth cho toàn bộ câu mới được điền tay sau khi xem trực tiếp từng
frame (không có nhãn tự động). Một lỗi ground truth đã phát hiện và sửa —
xem §6.1.

---

## 2. Việc 2 — `gemini-3.5-flash-lite` trên câu đếm vật thể đông (>10)

Bộ `questions_count_dense.json`, 10 câu, toàn bộ từ `L23_V001` (nguồn cảnh
đông duy nhất trong kho dữ liệu hiện có — đoàn đua xe đạp + khán giả). `runs=3`
cho cả vi/en, đúng phương pháp đo đồng thuận như báo cáo trước.

**Ground truth cho dạng câu này khó hơn hẳn phần còn lại**: ảnh đông người
chồng lấn không có nhãn khách quan. Với vật thể tĩnh đếm rõ (banner, cờ, VĐV
tách biệt) — gt là một số, đếm tay cẩn thận. Với đám đông chồng lấn — gt là
**một dải vài số liền kề** (model đúng nếu rơi vào dải), thay vì giả vờ có độ
chính xác tuyệt đối mà chính công cụ phát hiện vật thể tự động của dự án này
cũng không đạt được (xem `scripts/05_bench_vlm.py`: "trên 5 vật thì chính
người cũng đếm lệch").

| Model | Lang | Đúng | Định dạng | Đồng thuận | p50 (s) |
| --- | --- | --- | --- | --- | --- |
| `gemini-3.5-flash-lite` | vi | **40% (4/10)** | 100% | 70% | 3.24 |
| `gemini-3.5-flash-lite` | en | 40% (4/10) | 97% | 67% | 3.39 |

**Kết luận cho §3.1 của báo cáo trước — mâu thuẫn đã được giải quyết dứt
điểm:** `gemini-3.5-flash-lite` đúng **40%** ở dạng đếm đông, thấp hơn hẳn
61% đo được trên bộ 18 câu hỗn hợp trước đó, và thấp hơn hẳn 91-93% đồng thuận
đo được trên bộ câu dễ hơn. Điều này **xác nhận** giả thuyết đã nêu: model bất
ổn **tập trung ở câu đếm đông**, không phải toàn bộ output.

Nhìn vào các câu sai: model không nói "không biết" (đúng theo prompt), mà đoán
số nhỏ hơn nhiều so với thực tế (`DC01`: đáp án 13 → model đoán 6; `DC07`: đáp
án 12-13 → model đoán 30, tức lần này lại đoán *cao* hơn) — cho thấy model
**không có chiến lược đếm nhất quán** ở ngưỡng vật thể lớn, đoán ngẫu nhiên
theo cả hai hướng chứ không phải luôn đếm thiếu.

> **Khuyến nghị chốt cho §3.1:** **KHÔNG phục hồi** `gemini-3.5-flash-lite`
> làm ứng viên chính chỉ vì đồng thuận cao — với câu đếm đông (>10 đối tượng),
> độ đúng chỉ 40%. Nếu bộ câu hỏi thi thật có nhiều câu đếm đông, model chính
> vẫn nên là `gemini-3.1-flash-lite` (chưa đo lại ở dạng này, cần làm ở đợt
> sau) hoặc chấp nhận rủi ro đã biết trước với `gemini-3.5-flash-lite`.

---

## 3. Việc 3 — `gemma-4-31b` qua Google AI Studio trực tiếp

Thêm entry `gemma-4-31b-it` vào `REGISTRY` dùng backend `gemini` (cùng endpoint
`generateContent`, khác với entry OpenRouter `gemma-4-31b` đã có). Chạy trên
toàn bộ 52 câu, `runs=1` (không phải 3 — xem lý do ở §6.2), `--rpm 20`.

| Model | Lang | Đúng | Định dạng | Đồng thuận | p50 (s) | p95 (s) | Lỗi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gemma-4-31b-it` | en | **40% (21/52)** | 29% | 100%* | 17.14 | 52.09 | 0/52 |
| `gemma-4-31b-it` | vi | 35% (18/52) | 13% | 100%* | 25.66 | 102.52 | 0/52 |

\* "Đồng thuận 100%" ở đây không có ý nghĩa thật — cột này chỉ phản ánh
`runs=1` (một lần gọi tự nhiên "đồng thuận" với chính nó), không phải model ổn
định.

**Xác nhận đúng như dự đoán trong kế hoạch (D0.3 #4): model chạy được, không
bị rate-limit** — 0 lỗi API trên cả 104 lượt gọi, khác hẳn 100% thất bại khi
gọi qua OpenRouter ở đợt trước. Nhưng:

- **Format vẫn là điểm yếu chí mạng**, đúng như phát hiện cũ: chỉ 13-29% câu
  trả lời đúng khuôn JSON/≤4 từ. Phần lớn model xả nguyên chuỗi suy luận từng
  bước bằng **tiếng Anh** dù được yêu cầu trả lời tiếng Việt, dài tới hàng
  trăm từ, bị cắt ở 1000 ký tự (`raw` trong `raw.jsonl`).
- Độ trễ cao và dao động mạnh: p50 17-26s, p95 lên tới **102 giây** (đợt vi) —
  chậm hơn Gemini 4-5 lần và không ổn định.
- Độ đúng 35-40% nghe có vẻ khá, nhưng đây là đúng theo khớp chuỗi **sau khi**
  model tình cờ trả lời ngắn gọn — không đại diện cho khả năng dùng thật
  trong hệ thống 100 dòng, vì phần lớn câu trả lời không parse được thành
  answer sạch (`parse_error: "không phải JSON"`).

> **Khuyến nghị:** giữ nguyên kết luận cũ — `gemma-4-31b-it` **bị loại vì
> format, không phải vì độ đúng**. Ngay cả khi gọi đúng đường (Google API,
> không rate-limit), vấn đề cốt lõi là model không tuân thủ prompt constraint,
> không phải do hạ tầng. Không đưa vào danh sách ứng viên chính hay dự phòng.

---

## 4. Việc 4 — 5 model Ollama trên bộ 52 câu

`runs=1` mỗi câu (Ollama không giới hạn quota nhưng tốc độ CPU/GPU vẫn là nút
thắt cho `runs` cao trên 52 câu × 2 ngôn ngữ). GPU: RTX 2060 Super 8GB VRAM.

### 4.1 Bảng tổng hợp — 4/5 model chạy được

| Model | Lang | Đúng | Định dạng | p50 (s) | p95 (s) |
| --- | --- | --- | --- | --- | --- |
| **`qwen2.5vl:7b`** | en | **58% (30/52)** | **88%** | 3.62 | 5.42 |
| **`qwen2.5vl:7b`** | vi | **52% (27/52)** | 79% | 8.23 | 8.58 |
| `minicpm-v:8b` | en | 19% (10/52) | 62% | 4.25 | 5.00 |
| `minicpm-v:8b` | vi | 15% (8/52) | 54% | 7.23 | 7.99 |
| `qwen2.5vl:3b` | vi | 10% (5/52) | 13% | 6.56 | 6.84 |
| `qwen2.5vl:3b` | en | 8% (4/52) | 15% | 3.03 | 3.30 |
| `moondream` | vi/en | **0% (0/52)** | **0%** | 3.2-3.6 | 6.8-7.9 |
| `llama3.2-vision:11b` | — | **không chạy được** | — | — | — |

### 4.2 `qwen2.5vl:7b` — vượt trội rõ rệt, kể cả so với đợt đo trước

Đây là kết quả đáng chú ý nhất đợt này. Trên bộ 18 câu cũ, `qwen2.5vl:7b` chỉ
đạt 50%. Trên bộ 52 câu đa dạng hơn (nhiều OCR/tên thương hiệu từ L26), model
đạt **52-58%**, format tuân thủ 79-88% — áp sát mức của Gemini free tier
(94-100%) và **vượt xa** `gemma-4-31b-it` (13-29%) dù chạy hoàn toàn offline,
miễn phí, trên GPU 8GB phổ thông.

Theo loại câu hỏi (vi): `name` 100%, `color`/`ocr` 58-67%, `count` 50%,
`object` 42%, `action` 0% (xem §6.3 — vấn đề chung của mọi model, không riêng
gì `qwen2.5vl:7b`).

### 4.3 `qwen2.5vl:3b` trên GPU — lỗi lặp token còn NẶNG HƠN đợt trước

Đợt 1 đo được lỗi lặp token (`"@@@@..."`) ở 7/18 câu (39%) trên GPU. Đợt này,
trên bộ 52 câu, tỷ lệ lỗi lặp token là **43/52 = 83%** ở CẢ hai ngôn ngữ —
tệ hơn nhiều so với ước tính trước. Kết luận cũ **được củng cố mạnh**: model
này gần như không dùng được trên GPU 8GB của máy hiện tại.

### 4.4 `minicpm-v:8b` — xác nhận lại mức đúng thấp

15-19% đúng, khớp với xu hướng thấp đã thấy ở đợt 1 (17% trên 18 câu). Không
đưa vào danh sách ứng viên.

### 4.5 `moondream` — 0% chính thức, xác nhận quan sát không chính thức trước đó

Đợt 1 chỉ có quan sát sơ bộ (không lưu raw data). Đợt này đo chính thức đầy
đủ 52 câu × 2 ngôn ngữ = 104 lượt, **0% đúng, 0% đúng định dạng** ở cả hai
ngôn ngữ. Xác nhận dứt điểm: model 1.8B quá nhỏ để tuân thủ prompt JSON có
ràng buộc phức tạp, dù tốc độ rất nhanh (p50 ~3.2-3.6s, nhanh nhất trong nhóm
Ollama). **Loại khỏi danh sách ứng viên**, không cần đo lại.

### 4.6 `llama3.2-vision:11b` — lỗi hạ tầng, KHÔNG PHẢI lỗi model

Không chạy được một câu nào. Log lỗi từ chính Ollama server:

```text
llama-server process has terminated: exit status 1: error loading model:
unknown model architecture: 'mllama'
error loading model: unknown model architecture: 'mllama'
```

`ollama list` xác nhận model đã tải đầy đủ (7.8GB, digest hợp lệ) — đây
**không phải lỗi tải model** như nghi ngờ ở đợt 1, mà là **bản Ollama server
đang chạy trên máy (0.32.6) không hỗ trợ kiến trúc `mllama`** mà
`llama3.2-vision` dùng. Đây là giới hạn của phiên bản phần mềm, cần nâng cấp
Ollama lên bản mới hơn để giải quyết — việc này ngoài phạm vi bench nên
**chưa thực hiện**, cần xác nhận với người dùng trước khi đổi phiên bản phần
mềm hệ thống.

> **Việc cần làm:** nâng cấp Ollama (`ollama --version` hiện tại: 0.32.6) rồi
> chạy lại đúng lệnh: `python vlm_bench.py --questions questions.json
> --models llama3.2-vision-ollama --langs vi en --runs 1 --out
> runs/llama32vision_retry`.

---

## 5. Bảng xếp hạng tổng hợp — tất cả model đã đo (đợt 1 + đợt 2)

⚠️ **Lưu ý khi so sánh:** các con số dưới đây đo trên **hai bộ câu hỏi khác
nhau** (18 câu cũ vs 52 câu mới) và **số lần chạy khác nhau** (`runs=3` cho
Gemini/Gemma, `runs=1` cho Ollama) — không so sánh tuyệt đối được, chỉ dùng để
nhìn xu hướng.

| Model | Nguồn | Bộ câu | Đúng (vi) | Định dạng | p50 (s) |
| --- | --- | --- | --- | --- | --- |
| `gemini-3.1-flash-lite` | Gemini API | 18 câu, runs=3 | 67% | 94% | 5.5 |
| `gemini-3.5-flash-lite` | Gemini API | 18 câu, runs=3 | 61% | 96% | 5.1 |
| **`qwen2.5vl:7b` (Ollama, GPU)** | **Local** | **52 câu, runs=1** | **52%** | **79%** | **8.2** |
| `gemma-4-26b-a4b` | OpenRouter free | 18 câu, runs=1 | 44-47% | 78-82% | 11-17 |
| `gemma-4-31b-it` | **Google API trực tiếp** | 52 câu, runs=1 | 35% | 13% | 25.7 |
| `qwen2.5vl:3b` (Ollama, CPU) | Local | 18 câu, runs=1 | 61% | 89% | 170.3 |
| `minicpm-v:8b` (Ollama, GPU) | Local | 52 câu, runs=1 | 15% | 54% | 7.2 |
| `qwen2.5vl:3b` (Ollama, GPU) | Local | 52 câu, runs=1 | 10% | 13% | 6.6 |
| `gemini-3.5-flash-lite` (**đếm đông >10**) | Gemini API | 10 câu, runs=3 | 40% | 100% | 3.2 |
| `moondream` (Ollama) | Local | 52 câu, runs=1 | 0% | 0% | 3.6 |
| `llama3.2-vision:11b` (Ollama) | Local | — | **lỗi hạ tầng** | — | — |
| `nemotron-nano-12b-vl`, `gemma-4-31b` (qua OpenRouter) | OpenRouter free | — | đã loại từ đợt 1 | — | — |

**Model chính vẫn là `gemini-3.1-flash-lite`** (chưa đo lại ở đợt này — ứng
viên ổn định nhất qua nhiều đợt độc lập, xem báo cáo gốc). Thay đổi lớn nhất
so với đợt 1 là ở **hạng mục dự phòng offline**: `qwen2.5vl:7b` giờ có dữ liệu
mạnh hơn (bộ câu lớn, đa dạng hơn) để tin tưởng làm phương án offline chính,
thay vì chỉ là "lựa chọn cuối cùng".

---

## 6. Các phát hiện cần lưu ý

### 6.1 Một lỗi ground truth tự phát hiện qua đối chiếu nhiều model (Q51)

`Q51` (L28_V012, "Người đàn ông trong khung hình cận cảnh đang làm gì?") ban
đầu có gt = "nhìn ra sông" — điền tay dựa trên **1 trong 3 khung hình** dùng
cho câu hỏi (khung giữa, cận cảnh mặt nghiêng). Khi soát lại vì cả 3 model
(`qwen2.5vl:7b`, `gemma-4-31b-it`) đều **độc lập** trả lời "chụp ảnh"/"đang
dùng máy ảnh" cho câu này, xem lại khung hình đầu (389.jpg) mới thấy người
đàn ông đang đi bộ, tay cầm máy ảnh rõ ràng — ground truth ban đầu sai vì chỉ
xem 1/3 khung hình khi soạn câu.

**Đã sửa** gt thành "đi dạo"/"walking" (hành động nhất quán nhất qua cả 3
khung). **Chưa chạy lại bench** sau khi sửa (chi phí chạy lại cả 7 model không
tương xứng với 1 câu) — số liệu "action: 0%" trong toàn bộ báo cáo này **vẫn
tính trên gt cũ, sai**, nên tỷ lệ đúng thật của loại `action` nhỉnh hơn một
chút so với số hiển thị.

> **Bài học quy trình:** khi soạn câu hỏi từ 3 khung hình, phải xem **cả 3**
> trước khi chốt đáp án, không chỉ xem khung đại diện. Việc n>1 model độc lập
> cùng trả lời khác gt là tín hiệu đáng tin để soát lại gt, đúng tinh thần
> "câu sai — soi tay trước khi kết luận" của chính harness này.

### 6.2 Vì sao `gemma-4-31b-it` chỉ chạy `runs=1` thay vì `runs=3`

Batch cũ (18 câu) đã xác nhận format 0% qua 3 đợt đo độc lập (OpenRouter 2
lần + Google API 1 lần trong `Ke_hoach_AIC2026_v4.md`). Chạy `runs=3` trên bộ
52 câu (312 lượt gọi, p50 17-26s) sẽ tốn ~1.5-2 giờ chỉ để xác nhận lại một
kết luận đã biết trước với độ tin cậy cao — không tương xứng chi phí. `runs=1`
đủ để xác nhận format vẫn kém ở quy mô câu hỏi lớn hơn, đa dạng hơn.

### 6.3 Loại câu "action" — 0% ở CẢ BA model đo (không riêng model nào)

`qwen2.5vl:7b`, `gemma-4-31b-it`, `minicpm-v:8b` đều đạt đúng **0%** ở loại
`action` trên cả hai ngôn ngữ. Soát tay 6 câu action cho thấy đây **không
hẳn là model không hiểu hành động** — phần lớn model mô tả đúng ý nhưng:

- Trả lời **thành câu dài** thay vì ≤4 từ (`"Chiên miếng thịt cuộn."` thay vì
  `"chiên"`) → format sai → tính là sai theo luật khớp chuỗi chặt, dù đúng
  bản chất.
- Dùng từ đồng nghĩa không có trong danh sách gt hẹp (`"ghi chú"` thay vì
  `"viết"`/`"ghi chép"`; `"đang cố gắng tăng tốc"` thay vì `"rút đích"`).
- Một trường hợp là lỗi ground truth thật (Q51, đã sửa ở §6.1).

> **Khuyến nghị:** nếu câu hỏi thi thật có dạng `action`, cần đầu tư prompt
> engineering riêng để ép model trả lời đúng 1 động từ ngắn, và/hoặc mở rộng
> danh sách từ đồng nghĩa được chấp nhận trong gt — đây là vấn đề của cách đo,
> không hẳn là model kém ở khả năng nhận diện hành động.

### 6.4 Vấn đề console encoding khi soát log tay (ghi chú kỹ thuật)

Máy chạy bench dùng Python trên Windows với console mặc định codepage 1252,
không in được tiếng Việt có dấu ra terminal khi debug bằng `print()` trực
tiếp (harness `vlm_bench.py` đã tự xử lý việc này bằng
`sys.stdout.reconfigure(encoding="utf-8")`, nhưng script debug tay thì
không). Không ảnh hưởng tới kết quả bench (ghi file luôn đúng UTF-8), chỉ gây
lỗi khi debug — ghi chú lại để lần sau đỡ mất thời gian.

### 6.5 SSL certificate — chặn toàn bộ gọi mạng ở đợt đầu phiên này

Trước khi bench được, `GEMINI_API_KEY` không hoạt động do lỗi
`CERTIFICATE_VERIFY_FAILED` khi Python (`requests`/`certifi`) gọi
`generativelanguage.googleapis.com` — không phải lỗi key. Nguyên nhân: máy có
phần mềm (antivirus/proxy) chèn chứng chỉ TLS riêng mà Windows tin cậy nhưng
`certifi` (bundle CA riêng của Python) thì không — PowerShell/`curl` (dùng kho
chứng chỉ Windows) gọi được bình thường, chỉ Python bị chặn. Đã sửa bằng
`pip install pip-system-certs` (tự động vá `ssl` module dùng kho chứng chỉ hệ
thống). Ghi chú lại vì lỗi này sẽ tái xuất hiện trên máy khác nếu không cài
gói này trước.

---

## 7. Giới hạn của đợt đo này

1. **Ground truth cho câu đếm đông (§2) và một phần câu hỏi mới (§1) là điền
   tay, không có nhãn khách quan** — khác biệt căn bản so với việc chấm bằng
   detector tự động (vốn cũng không đáng tin cho >5 vật, xem
   `scripts/05_bench_vlm.py`). Với đám đông chồng lấn, gt dùng dải giá trị
   thay vì một số chính xác — đã ghi rõ trong `questions_count_dense.json`
   (trường `note` từng câu).
2. **`runs=1` cho Ollama và `gemma-4-31b-it`** (so với `runs=3` cho Gemini) —
   không đo được đồng thuận cho các model này, chỉ đo được độ đúng một lần
   chạy. Không nên coi con số đúng của các model này ổn định như Gemini.
3. **Một lỗi ground truth đã phát hiện và sửa (Q51) nhưng chưa chạy lại
   bench** — số liệu loại `action` trong báo cáo này bị đánh giá thấp hơn
   thực tế một chút (xem §6.1, §6.3).
4. **`llama3.2-vision:11b` chưa đo được** do lỗi phiên bản Ollama — 4/5 chứ
   không phải 5/5 model local đã có dữ liệu.
5. Bộ câu hỏi 52 câu vẫn dưới ngưỡng ≥50 câu **cho mỗi lần so sánh cụ thể**
   nhưng vẫn là một bộ mẫu nhỏ so với quy mô đề thi thật (hàng trăm câu) —
   giữ nguyên giới hạn đã nêu ở báo cáo đợt 1.
6. Nhóm `L28_V012` (phim tài liệu) là **loại nội dung mới, khác hẳn** các
   nhóm trước (không phải thể thao/du lịch/ẩm thực) — 10 câu chưa đủ để kết
   luận riêng cho loại nội dung này.

---

## 8. Việc tiếp theo (đợt 1 — đã cập nhật trạng thái)

1. ~~Nâng cấp Ollama rồi chạy lại `llama3.2-vision:11b`~~ — **đã làm, xem §10:
   không sửa được, lỗi thượng nguồn của Ollama.**
2. Chạy lại bộ `action` (6 câu) sau khi mở rộng danh sách từ đồng nghĩa được
   chấp nhận trong gt — **chưa làm** (việc tiếp theo).
3. ~~Đo `gemini-3.1-flash-lite` trên bộ câu đếm đông >10 đối tượng~~ — **đã
   làm, xem §9.**
4. Nếu quyết định dùng `qwen2.5vl:7b` làm phương án offline chính thức, đo
   thêm `runs=3` trên một tập con câu hỏi để có số liệu đồng thuận — **chưa
   làm** (việc tiếp theo).
5. Mở rộng `questions.json` thêm từ các nhóm L còn thiếu — **chưa làm** (việc
   tiếp theo).

---

## 9. Đo lại `gemini-3.1-flash-lite` (đợt 2, theo yêu cầu người dùng)

### 9.1 Trên bộ đếm vật thể đông (>10, 10 câu, runs=3) — đã xong

| Model | Lang | Đúng | Định dạng | Đồng thuận | p50 (s) |
| --- | --- | --- | --- | --- | --- |
| `gemini-3.1-flash-lite` | vi | 40% (4/10) | 100% | 100% | 8.08 |
| `gemini-3.1-flash-lite` | en | 50% (5/10) | 100% | 100% | 7.68 |

So với `gemini-3.5-flash-lite` trên cùng bộ câu (§2: 40%/40%, format 97-100%,
**đồng thuận chỉ 67-70%**): `gemini-3.1-flash-lite` đúng **tương đương** (cùng
mức ~40-50%) nhưng **đồng thuận giữ nguyên 100%** — đúng với đặc tính đã biết
của model này qua mọi đợt đo (báo cáo gốc, `Ke_hoach_AIC2026_v4.md` D0.3 đợt
2). **Kết luận:** cả hai bản Gemini Flash-Lite đều **yếu như nhau** ở dạng đếm
đông (~40-50%, thấp hơn hẳn mức 61-67% đo được ở câu dễ) — đây là điểm yếu
**chung của dòng Flash-Lite**, không phải điểm yếu riêng của bản `3.5`. Ứng
viên chính `gemini-3.1-flash-lite` **không tránh được** nhược điểm này, nhưng
vẫn đáng tin hơn nhờ đồng thuận tuyệt đối 100% — khi sai, nó **sai nhất
quán** (dễ phát hiện và xử lý bằng logic ứng dụng, ví dụ đặt ngưỡng nghi ngờ
khi câu hỏi thuộc dạng đếm), khác với `3.5` vốn **sai ngẫu nhiên giữa các lần
gọi** (không dò được bằng cách gọi lại).

### 9.2 Trên bộ 52 câu chính (runs=3, để so sánh công bằng với các model khác)

**Đang chạy nền lúc viết báo cáo này** — bộ này 312 lượt gọi, gặp một số lượt
timeout mạng (`Read timed out`, không phải 429 quota) khiến tốc độ chậm hơn
dự kiến. Sẽ cập nhật bảng kết quả vào báo cáo này ngay khi hoàn tất; không
trì hoãn phần quyết định ở §11 vì dữ liệu ở §9.1 cùng dữ liệu lịch sử (18 câu,
báo cáo gốc: 67%/50% đúng, 94-100% format, 100% đồng thuận qua 3 đợt độc lập)
đã đủ để ra quyết định.

---

## 10. Thử lại `llama3.2-vision:11b` sau khi nâng Ollama — vẫn không chạy được

Ollama trên máy đã **tự động cập nhật** lên bản mới nhất trong lúc chờ
(0.32.6 → **0.32.9**, xác nhận qua `ollama --version` và `GET /api/version`).
Chạy lại smoke-test 1 câu — **vẫn lỗi y hệt**:

```text
llama-server process has terminated: exit status 1: error loading model:
unknown model architecture: 'mllama'
```

Tra cứu xác nhận **đây không phải lỗi cấu hình máy này**: theo
[GitHub issue #16547](https://github.com/ollama/ollama/issues/16547) của
chính kho `ollama/ollama`, **"new engine" của Ollama từ bản 0.30.0 trở đi
chưa hỗ trợ kiến trúc `mllama`** — đúng như release notes chính thức của
Ollama ghi ("llama3.2-vision is not yet supported"). Bản 0.32.9 (mới nhất
hiện có, xác nhận qua GitHub Releases) **vẫn nằm trong khoảng bị ảnh
hưởng**, không có bản mới hơn để nâng cấp thêm.

**Đường duy nhất còn lại** là **hạ cấp** Ollama xuống bản trước 0.30.0 (một
số người dùng báo bản 0.6.0 chạy được, theo cùng chuỗi issue trên GitHub) —
nhưng **không khuyến nghị làm việc này**: hạ cấp có thể phá vỡ khả năng chạy
`qwen2.5vl:7b`/`qwen2.5vl:3b`/`minicpm-v:8b` (dùng kiến trúc mới hơn, có thể
cần bản Ollama mới) — tức đánh đổi ứng viên local **đang thắng rõ rệt**
(qwen2.5vl:7b, 52-58% đúng, xem §4.2) để cứu một model **chưa từng được đo
là có đúng tốt hay không**. Không đáng.

> **Chốt cho `llama3.2-vision:11b`:** loại khỏi danh sách ứng viên vĩnh viễn
> trên hạ tầng Ollama hiện tại, không phải vì độ đúng kém (chưa đo được) mà
> vì **xung đột phiên bản không giải quyết được** mà không đánh đổi model tốt
> hơn. Nếu vẫn muốn đánh giá model này, cách duy nhất còn lại là chạy qua
> vLLM/HuggingFace `transformers` trực tiếp thay vì Ollama — công sức dựng hạ
> tầng riêng, ngoài phạm vi bench hiện tại.

---

## 11. Quyết định: dùng model FREE hay TRẢ PHÍ làm chính?

### 11.1 Bối cảnh từ chính kế hoạch dự án

`docs/Ke_hoach_AIC2026_v4.md` (mục D0.3 #6, viết từ trước) đã tự đặt câu hỏi
này và **để ngỏ**: *"Ba model đã chết vì rate-limit trong một đợt test 20
lượt. Bài thi cần chạy 100 câu × nhiều lần trong 4 tuần — quota free sẽ không
đủ. Phải quyết định sớm: trả phí, hay chạy model local."* Đúng vậy — **ngay
trong lúc đo lại `gemini-3.1-flash-lite` cho báo cáo này (§9.2), bench đã dính
timeout/chậm** vì đang dùng key free-tier, một minh chứng sống cho đúng vấn đề
mà kế hoạch đã cảnh báo trước.

**Một ràng buộc quan trọng khác chưa có câu trả lời**: đề bài đã hỏi BTC liệu
**vòng sau có cấm gọi API hay không** (mục 0.a, "đã gửi, chờ trả lời" —
`vlm_bench.py` vẫn còn để trống dòng "Phương án dự phòng offline" trong mọi
report tự sinh, chờ câu trả lời này). **Nghĩa là bất kể chọn free hay trả
phí, vẫn PHẢI giữ một phương án offline chạy được** — quyết định ở đây chỉ là
chọn **model gọi API chính**, không thay thế việc giữ `qwen2.5vl:7b` (Ollama)
làm dự phòng.

### 11.2 Chi phí thực đo — không phải ước lượng suông

Tính từ token thật đo được trên chính bộ câu hỏi của dự án này (trung bình
**3.557 token vào / 36,5 token ra** mỗi lượt gọi, 2-3 ảnh + prompt, đo trên 60
lượt gọi thật của `gemini-3.1-flash-lite`), đối chiếu giá API trả phí (tháng
8/2026):

| Model | Giá (vào/ra, mỗi 1M token) | Chi phí 1 lượt gọi | 100 câu × 1 lần | 100 câu × 2 ngôn ngữ × 3 lần | 4 tuần test nặng (~20.000 lượt) |
| --- | --- | --- | --- | --- | --- |
| **`gemini-3.1-flash-lite`** (đang dùng) | $0,25 / $1,50 | **$0,00094** | **$0,09** | **$0,57** | **~$19** |
| `gemini-3.5-flash-lite` | $0,30 / $2,50 | $0,00116 | $0,12 | $0,70 | ~$23 |
| GPT-5-mini *(ước tính, chưa test trong dự án)* | $0,125 / $1,00 | $0,00048 | $0,05 | $0,29 | ~$10 |
| Claude Haiku 4.5 *(ước tính, chưa test trong dự án)* | $1,00 / $5,00 | $0,00374 | $0,37 | $2,24 | ~$75 |

*(Giá GPT-5-mini/Claude Haiku lấy từ nguồn ngoài dự án — xem cuối mục — và
dùng LẠI số token đo trên Gemini để ước tính, do hai nhà cung cấp này chưa
từng được benchmark thật trong dự án. Cách tính token ảnh của mỗi nhà cung
cấp khác nhau nên con số này chỉ mang tính so sánh tỷ lệ, không chính xác
tuyệt đối.)*

**Kết luận về chi phí: hoàn toàn không đáng kể ở MỌI kịch bản**, kể cả kịch
bản "test nặng suốt 4 tuần, 20.000 lượt gọi" (nhiều hơn hẳn khối lượng thực tế
đã dùng — toàn bộ 2 đợt bench của dự án này tới giờ cộng lại mới khoảng 1.500
lượt gọi Gemini). Cả bốn lựa chọn đều rẻ hơn một cốc cà phê cho tới rẻ hơn một
bữa ăn cho **toàn bộ** khối lượng công việc còn lại của cuộc thi.

### 11.3 Quyết định

> ## **Khuyến nghị: TRẢ PHÍ — bật billing cho đúng model đang dùng
> (`gemini-3.1-flash-lite`), không đổi sang nhà cung cấp khác.**

**Vì sao trả phí, không giữ free:**

1. **Chi phí không phải là yếu tố quyết định** — dưới $20 cho cả 4 tuần theo
   ước tính rộng rãi nhất (§11.2), trong khi cái giá của việc **hết quota
   giữa lúc thi** là mất điểm thật (Q&A: trả lời sai/không trả lời = 0 điểm,
   PHẦN C của kế hoạch). Đánh đổi vài trăm nghìn đồng lấy rủi ro mất điểm thi
   là không hợp lý.
2. **Tốc độ tăng 4-5 lần** khi gọi thẳng Google API có billing so với free
   tier bị rate-limit (đã ghi nhận trong kế hoạch gốc: `3.5-flash-lite` 7,5s
   → 1,4s khi hết nghẽn hạ tầng trung gian; hôm nay tự dính đúng vấn đề này ở
   §9.2). Với 100 câu × nhiều vòng thử trong 4 tuần, chênh lệch là hàng giờ
   đồng hồ công sức đội, theo đúng ước tính đã có sẵn trong kế hoạch.
3. **Không cần đổi model, không cần benchmark lại từ đầu.**
   `gemini-3.1-flash-lite` đã là ứng viên chính qua **nhiều đợt đo độc lập**
   (báo cáo gốc + §9.1 hôm nay): đồng thuận 100% xuyên suốt, format 94-100%,
   đúng cao nhất nhóm Gemini free (61-67% ở câu dễ). Bật billing cho đúng
   model này là thay đổi **rẻ nhất, ít rủi ro nhất** có thể làm — không phải
   "chuyển sang GPT-4o" hay thử nghiệm nhà cung cấp mới chưa kiểm chứng.
4. **GPT-5-mini và Claude Haiku RẺ HƠN hoặc GẦN BẰNG** về lý thuyết (GPT-5-mini
   ước tính rẻ nhất) nhưng **chưa có một dòng dữ liệu benchmark thật nào**
   trong dự án này — độ đúng tiếng Việt, độ tuân thủ JSON, độ ổn định đều là
   ẩn số. Đổi sang một trong hai model này tốn thời gian bench lại từ đầu
   (đúng quy trình đã làm cho 7 model trong 2 báo cáo này) chỉ để tiết kiệm
   vài đô-la — không tương xứng, trừ khi `gemini-3.1-flash-lite` bị BTC cấm
   dùng vì lý do khác.

**Việc cụ thể cần làm:**

1. Bật billing (thẻ thanh toán) cho project Google AI Studio đang dùng —
   không cần đổi `GEMINI_API_KEY`, cùng một key sẽ tự chuyển từ free tier
   sang trả phí khi bật billing (theo cơ chế Google AI Studio).
2. Đặt cảnh báo ngân sách (budget alert) trong Google Cloud Console ở mức
   thấp (ví dụ $20-30) — đủ dư so với ước tính §11.2 nhưng chặn được rủi ro
   dùng sai (vòng lặp vô hạn gọi API do lỗi code, v.v.).
3. **Vẫn giữ nguyên `qwen2.5vl:7b` (Ollama) làm phương án dự phòng offline**
   — không phải vì free, mà vì đây là câu trả lời duy nhất nếu BTC xác nhận
   cấm gọi API ở vòng sau (mục 0.a, chưa có trả lời). Đây là quyết định
   **song song**, không phải thay thế cho quyết định trả phí ở trên.
4. Theo dõi chi phí thật qua 1-2 tuần đầu, đối chiếu với ước tính §11.2 —
   nếu lệch xa (ví dụ do prompt dài hơn dự kiến, hoặc `runs` cao hơn), điều
   chỉnh lại ngân sách dự trù.

**Nguồn giá tham khảo** (chốt tại thời điểm viết báo cáo, 13/8/2026 — giá API
thay đổi theo thời gian, cần kiểm tra lại trước khi trích dẫn về sau):
[Gemini API pricing 2026 — Morph](https://www.morphllm.com/gemini-api-pricing),
[Gemini API Pricing — CostGoat](https://costgoat.com/pricing/gemini-api),
[OpenAI API Pricing — BenchLM](https://benchlm.ai/openai/api-pricing),
[GPT-5-mini pricing — PricePerToken](https://pricepertoken.com/pricing-page/model/openai-gpt-5-mini),
[Anthropic API Pricing — CloudZero](https://www.cloudzero.com/blog/claude-pricing/).
