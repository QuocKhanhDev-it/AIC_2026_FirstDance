# KẾ HOẠCH v4 — AI Challenge HCMC 2026, Vòng Sơ tuyển

*Thay thế v3. Giữ nguyên khung 4 giai đoạn, cấu trúc 2 mũi nhọn, và triết lý
"kiểm tra giả định trước khi build".*

**Khác v3 ở đâu:** PHẦN A của v3 đo trên 6 file CSV / 8 file `.npy` / 1 video
và được đánh dấu "không cần tranh luận lại". Giờ đã dựng xong bảng cái, PHẦN A
được đo lại trên **toàn bộ 873 video / 177.321 keyframe**. Ba con số sai lệch
đáng kể, một kết luận bị đảo ngược. Xem [PHẦN G](#phần-g--so-sánh-v3--v4).

---

## PHẦN A — SỰ THẬT VỀ DỮ LIỆU

*Đo trên **toàn bộ** 873 video, 177.321 keyframe, 1.122.384 detection,
176.448 khoảng cách keyframe. Không phải ước lượng từ mẫu.*

*Nguồn: `index/master.parquet`, `index/objects.parquet`, `index/clip.npy`
dựng bằng `scripts/01_build_index.py`, kiểm chứng bằng `02_verify.py` và
`03_verify_CLIP.py`.*

### A0. Bảng cái đã được chứng minh là đúng

Trước mọi con số khác: liên kết giữa các nguồn đã được máy kiểm chứng, không
phải tin bằng mắt.

| Chứng minh điều gì | Cách | Kết quả |
| --- | --- | --- |
| Ảnh keyframe thứ *i* ↔ dòng thứ *i* CSV | ffmpeg trích frame tại `pts_time`, so tương quan pixel + biên độ với dòng kề | **83/83 KHỚP** (L21+L22+L29) |
| Vector CLIP thứ *i* ↔ dòng thứ *i* CSV | encode lại frame, so cosine + xếp hạng trong video | **82/83 đúng hạng 1**, cosine 0,951-0,999 |

Phép thứ hai v3 không yêu cầu, nhưng cần thiết: vector CLIP nằm trong file
`.npy` riêng, lệch hàng ở đó thì kiểm ảnh không phát hiện được — mà TV1 dùng
trực tiếp vector này.

*Giới hạn:* mới kiểm được trên 60 video (L21+L22) vì các nhóm khác chưa có video gốc.
Mỗi máy tải thêm nhóm L nào thì chạy lại hai script đó cho nhóm đó.

### A1. Mật độ keyframe — v3 sai gấp đôi, nhưng kết luận vẫn đứng

| Chỉ số | v3 (6 file CSV) | **Đo toàn bộ** |
| --- | --- | --- |
| Khoảng cách 2 keyframe liên tiếp | trung vị 109 frame | **trung vị 55 frame** |
| p90 / max | 175 / 210 | **150 / 211** |
| % cặp cách ≤ 10 frame | 1,15% | **12,62%** |
| Xác suất keyframe có sẵn rơi trúng cửa sổ 10 frame | ~9% | **14,6%** |

> **→ Vẫn BẮT BUỘC có module trích xuất frame dày. Vẫn là đường găng.**
> 14,6% nghĩa là R-Score trần ~0,15 dù thuật toán dóng hàng hoàn hảo. Kết
> luận của v3 đúng; chỉ độ lớn thay đổi.

**Nhưng con số này đổi cách TV2 làm module.** Khoảng cách trung vị 55 frame
(~1,8s ở 30fps) chứ không phải 109 (~4,4s) — cửa sổ cần trích hẹp hơn, chi
phí thấp hơn ước tính của v3. **TV2 phải tự đo lại trên máy mình**, đừng lấy
mốc "3,1 giây CPU cho 300 frame" của v3 vì mốc đó đo trên một máy khác.

### A2. Kho dữ liệu lệch nặng về một nhóm L

873 video, phân bố **rất không đều**:

| Nhóm | L21 | L22 | L23 | L24 | L25 | **L26** | L27 | L28 | L29 | L30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Số video | 29 | 31 | 25 | 43 | 88 | **498** | 16 | 24 | 23 | 96 |

**L26 chiếm 57% toàn bộ kho.** Đây là điều v3 không thấy.

| Chỉ số | Giá trị |
| --- | --- |
| Keyframe / video | min 24, trung vị 163, max 632 |
| Độ dài video | min 30s, trung vị 317s (~5,3 phút), max 2735s (~45,6 phút) |
| Tổng keyframe | **177.321** (v3 ước ~200.000 — sát) |
| Ma trận CLIP float32 | **347 MB** — vừa RAM |

> **Hệ quả cho tập dev của Khánh:** lấy mẫu ngẫu nhiên sẽ ra 57% câu hỏi từ
> L26. Phải **lấy mẫu phân tầng** theo nhóm L, nếu không tập dev đo sai và
> toàn bộ tinh chỉnh ở Giai đoạn 3 tối ưu cho một nhóm duy nhất.

### A3. Metadata là kênh truy hồi mạnh — v3 đúng, và mạnh hơn v3 nghĩ

| Trường | % video có nội dung | Độ dài trung bình |
| --- | --- | --- |
| `title` | **100,0%** | 63 ký tự |
| `description` | **99,7%** | **955 ký tự** |
| `keywords` | 99,8% | 285 ký tự |

**99,7% mô tả là tiếng Việt có dấu.** Description trung bình gần 1000 ký tự —
đây là mỏ vàng cho BM25, giàu hơn nhiều so với những gì v3 suy ra từ một file.

Với TRAKE, **sai video = 0 điểm ngay lập tức**, nên lọc đúng video là việc
sinh lời cao nhất. Giữ nguyên ưu tiên cao của v3.

### A4. Objects — v3 KẾT LUẬN SAI, phải nâng lên kênh chính

v3 kết luận *"nhiễu nặng, hạ ưu tiên, chỉ dùng cộng/trừ điểm mềm"* dựa trên
**một file JSON duy nhất**. Đo toàn bộ 177.321 file:

| Ngưỡng | Số detection | Mỗi keyframe | % keyframe có ≥1 |
| --- | --- | --- | --- |
| ≥ 0,3 | 1.122.384 | 6,3 | **95,0%** |
| ≥ 0,5 | 597.357 | 3,4 | **89,6%** |
| ≥ 0,7 | 276.577 | 1,6 | **73,2%** |

Ngay ở ngưỡng khắt khe 0,7 vẫn còn **73% keyframe có ít nhất một vật thể đáng
tin**. Đây không phải kênh phụ.

**514 nhãn khác nhau**, đã là nhãn chữ đọc được trong `detection_class_entities`
— không cần bảng ánh xạ MID→tên (v3 đúng ở điểm này).

Nhưng có một cái bẫy phân bố:

| Nhóm nhãn | % tổng detection |
| --- | --- |
| 5 nhãn người (Person, Clothing, Human face, Man, Woman) | **49,6%** |
| Nhãn hiếm (< 20.000 lần) | 36,7% |

> **Lọc theo `Person` gần như vô nghĩa** — một nửa kho có nhãn đó. Giá trị
> phân biệt nằm ở nhãn hiếm. Bắt buộc dùng **trọng số nghịch tần suất (IDF)**,
> không đếm thô. Xem [PHẦN D1.6](#phần-d16--cách-nâng-objects-lên-kênh-chính).
>
> **→ Objects là KÊNH CHÍNH thứ 4, ngang hàng BM25 OCR/ASR. Vẫn cho điểm mềm,
> vẫn tuyệt đối không lọc cứng** (lý do lọc mềm của v3 vẫn đúng: OpenImages
> không có nhãn cho mọi khái niệm, xóa cứng là xóa mất đáp án).

### A5. Sáu cái bẫy kỹ thuật

**1. `.npy` là per-video**, `(n_keyframe, 512)`, `float16`, **đã chuẩn hóa L2**
→ cosine = dot product; phải tự ghép index toàn cục.
*Xác nhận:* norm của vector nguồn = 0,9995–1,0005, trung bình 1,0000. v3 đúng.
Script hiện chuyển sang `float32` khi ghép (chuẩn hóa L2 trên `float16` mất
độ chính xác, và matmul `float32` trên CPU nhanh hơn).

**2. Nộp cho BTC là `frame_idx`, KHÔNG phải cột `n`.** Nhầm cột này → toàn bộ
bài nộp sai. Bảng cái giữ cả hai (`frame_idx` và `kf_n`) với tên khác nhau rõ
ràng để không nhầm.

**3. fps có BỐN giá trị, không phải hai.**

```text
25.0    26.44    29.97    30.0
```

v3 ghi "25 và 30 lẫn lộn" — thiếu `26.44` và `29.97`. Cả hai đều là số lẻ:
`29.97` là NTSC drop-frame, `26.44` thì không theo chuẩn nào. Mọi phép quy
đổi giây↔frame **phải nhận `fps` làm tham số, cấm hardcode**. Khác biệt tồn
tại ngay trong cùng nhóm L21 (`V001` = 30,0; `V003` = 25,0).

**31 video có fps lạ, và CHƯA MÁY NÀO tải video gốc của chúng:**

```text
fps 26.44: 1 video -> L24_V044
fps 29.97: 30 video -> L25_V004, L25_V005, L25_V008, L25_V013, L25_V014, L25_V017, L25_V021, L25_V022, L25_V025, L25_V030, L25_V031, L25_V034, L25_V039, L25_V040, L25_V043, L25_V048, L25_V049, L25_V052, L25_V057, L25_V058, L25_V061, L25_V066, L25_V067, L25_V070, L25_V074, L25_V077, L25_V078, L25_V084, L25_V085, L25_V088
```

`26.44` gần như chắc chắn là **variable frame rate**. Với VFR thì
`pts_time × fps` có thể không ra đúng cách BTC đánh số frame, mà `frame_idx`
lại là giá trị nộp bài. Chưa kiểm được vì chưa có `.mp4` nào trong số này.

> **Việc phải giao:** đảm bảo 31 video trên nằm trong phần tải của một máy
> nào đó, rồi chạy `02_verify.py` riêng cho chúng. Đây là nhóm rủi ro cao
> nhất còn lại của bảng cái. Xem PHẦN H mục 8.

**4. ~~Tên file object JSON không liên tục~~ — SAI, đã bác bỏ.**

v3 viết *"tên file object không liên tục (001, 005, 008, 009, 011, 015...)
→ không phải keyframe nào cũng có object"*. Đếm chính xác: **177.321 file JSON
= đúng 177.321 dòng CSV**, khớp 1-1 từng video, đánh số liên tục 001, 002,
003... Cái "001, 005, 008" mà v3 thấy là số hiệu **video** (`L21_V004` không
tồn tại), không phải số hiệu keyframe.

**"Đếm khớp" không đủ để kết luận — đã kiểm cả TÊN.** Số lượng bằng nhau vẫn
có thể là ghép lệch. Ba phép kiểm bổ sung:

| Phép kiểm | Kết quả |
| --- | --- |
| Video có tên JSON không liên tục `001..n` | **0 / 873** |
| Lệch tên giữa ảnh keyframe và object JSON (L21+L22) | **0 / 16.896** |
| Lệch giữa tên JSON và cột `n` (toàn bộ) | **0 / 177.321** |

Kiểm thẳng trên đĩa với chính `L26_V383` — video mà v3 trích dẫn: **157 file,
tên `001.json` → `157.json`, liên tục hoàn toàn**.

Vẫn khớp theo **tên file** thay vì chỉ số cho an toàn — nhưng không còn phải
"xử lý thiếu một cách êm", vì không thiếu.

**5. MỚI — đường dẫn trong `master.parquet` là TUYỆT ĐỐI.**

`kf_path`, `obj_path`, `video_path` trỏ vào ổ đĩa của máy dựng index. Mô hình
chia dữ liệu nhiều máy (xem [B4](#b4-mô-hình-chia-dữ-liệu-nhiều-máy)) làm ba
cột này đứt trên máy khác, trong khi mọi cột còn lại vẫn chạy — **lỗi im lặng**.
Cách xử lý ở [docs/01_cai_dat.md](01_cai_dat.md).

**6. MỚI — model CLIP phải là `ViT-B-32-quickgelu`.**

Xem A6.

### A6. Chiều vector = 512 → CLIP **ViT-B/32**, bản **QuickGELU**

v3 viết đúng: text encoder phải đúng ViT-B/32, nếu không vector không cùng
không gian. Nhưng chưa đủ — phải là tag **`ViT-B-32-quickgelu`**.

Model OpenAI CLIP dùng hàm kích hoạt QuickGELU. `open_clip` nạp `ViT-B-32`
thường **vẫn chạy bình thường**, chỉ in một dòng cảnh báo lẫn giữa hàng chục
dòng log:

```text
UserWarning: QuickGELU mismatch between final model config (quick_gelu=False)
and pretrained tag 'openai' (quick_gelu=True)
```

Đo trên 29 video L21 (nhóm đã biết chắc bảng cái đúng):

| Tag | Cosine với vector BTC |
| --- | --- |
| `ViT-B-32` | 0,9513 |
| **`ViT-B-32-quickgelu`** | **0,9913** |

> **Nếu TV1 dùng nhầm tag, hệ thống vẫn trả kết quả trông hợp lý, chỉ kém hơn
> — và không ai biết vì sao điểm dev thấp.** Đúng loại lỗi im lặng mà PHẦN E
> cảnh báo. Đặt assert kiểm tra tag khi khởi tạo encoder.

Ghi chú: gói BTC phát là `clip-features-**32**` — bản CLIP nhỏ nhất. **Cần
kiểm tra trang tải xem có gói từ model lớn hơn không** (ViT-L/14, 768 chiều).
Nếu không có, phương án dự phòng ở PHẦN F vẫn là tự encode lại 177k ảnh.

### A7. MỚI — nội dung kho lệch về ẩm thực, không phải giao thông

v3 lấy ví dụ xuyên suốt là bản tin và giao thông. Thống kê nhãn thật:

Đo theo **số keyframe chứa nhãn** (ngưỡng ≥ 0,5), không phải số detection —
một khung hình chợ rau sinh ra hàng chục hộp `Tomato` nên đếm detection sẽ
thổi phồng:

| Nhãn | Số keyframe | % kho |
| --- | --- | --- |
| **Đồ ăn (Food)** | **26.519** | **14,96%** |
| Bàn (Table) | 10.798 | 6,09% |
| Rau củ (Vegetable) | 3.482 | 1,96% |
| Thuyền (Boat) | 2.596 | 1,46% |
| **Cà chua (Tomato)** | **2.533** | **1,43%** |
| Xe đạp (Bicycle) | 1.147 | 0,65% |
| **Ô tô (Car)** | **1.593** | **0,90%** |
| Chảo (Wok) | 1.452 | 0,82% |
| Đũa (Chopsticks) | 1.857 | 1,05% |

**`Food` xuất hiện ở 15% toàn kho, `Car` chỉ 0,9% — gấp 16 lần.** Cà chua vẫn
nhiều hơn ô tô nhưng chỉ 1,6 lần, không phải 2,5 lần như bản đầu tính theo số
detection. Chảo, đũa, salad, bông cải đều lọt top 60.

> **Hệ quả:** Khánh phải **xem trước vài video của nhiều nhóm L** trước khi
> viết 30–50 truy vấn dev. Soạn theo giả định "tin tức giao thông" sẽ ra tập
> dev không đại diện, và toàn bộ tinh chỉnh ở Giai đoạn 3 tối ưu cho phân bố
> sai. Đây là cách hỏng âm thầm của phụ thuộc **#3** ở PHẦN E.

---

## PHẦN B — QUYẾT ĐỊNH HẠ TẦNG

### B1. Không dùng Supabase / Postgres / Milvus / Elasticsearch — ĐÃ KIỂM CHỨNG

Giữ nguyên lập luận của v3, nay có bằng chứng thực tế: truy vấn ghép 1,1 triệu
detection với 177 nghìn keyframe bằng DuckDB chạy trong **vài trăm mili-giây**
trên máy cá nhân, không cần server.

| Loại dữ liệu | Lưu bằng | Truy vấn bằng |
| --- | --- | --- |
| Vector CLIP | một `.npy` float32 nạp thẳng RAM (347 MB) | NumPy `M @ q` |
| Bảng index / objects / OCR / ASR / metadata | **Parquet** | **DuckDB** |
| Ảnh keyframe, video | để nguyên trên đĩa | lưu đường dẫn trong bảng index |

### B2. Không dùng C++ / pybind11 / ctypes

Giữ nguyên v3. Nút cổ chai thật là decode video và inference model.

### B3. Cấu trúc thư mục — đã dựng

```text
aic2026/
  scripts/    00_discover.py  01_build_index.py  02_verify.py  03_verify_CLIP.py
  index/      master.parquet  clip.npy  objects.parquet  problems.csv  *_report
  src/        dense.py  retrieval.py  scoring.py  objects.py  run.py
  dev/        số liệu, checklist, queries_draft.md, dev.json
  docs/       hướng dẫn thành viên
  cache/      frame dày đã decode
  submissions/
```

### B4. Mô hình chia dữ liệu nhiều máy

**Đã chốt:** mỗi máy tải và xử lý một tập video + keyframe tương ứng.

**Ước lượng theo THỜI LƯỢNG, không theo số video.** L21 và L22 là bản tin HTV
dài 1095–1178s/video, gấp **2,05–2,20 lần** trung bình kho (535s). Ngoại suy
theo số video phóng đại đúng chừng đó — bản đầu của mục này ghi 136 GB, sai
gần gấp đôi.

Đo trên 60 video đã tải: video **110 KB/giây**, keyframe **187 KB/ảnh**.
Toàn kho có **129,8 giờ** video và 177.321 keyframe:

| Thành phần | Cả 873 video |
| --- | --- |
| Videos | **~52 GB** |
| Keyframes | **~33 GB** |
| csv + clip + objects + media-info | ~2 GB (đã có đủ trên mọi máy) |
| **Tổng** | **~85 GB** |

> **85 GB vừa một ổ ngoài 1 TB.** Nếu chỉ vì dung lượng thì **không cần chia
> 5 máy** — và bỏ được mô hình chia sẽ loại luôn hai rủi ro kèm theo: đường
> dẫn tuyệt đối đứt (A5.5) và "không ai chạy verify phần mình" (PHẦN F).
>
> Nhưng dung lượng không phải lý do duy nhất. Trích dày và OCR trên 129,8 giờ
> video là khối lượng CPU lớn; chia máy là chia **thời gian tính toán**, không
> chỉ chia chỗ chứa. Cân nhắc lại theo thực tế phần cứng của nhóm.

Gợi ý chia theo nhóm L, cân bằng số video:

| Máy | Nhóm L | Số video | Video | Keyframes |
| --- | --- | --- | --- | --- |
| A | L26 (chia đôi, phần 1) | ~250 | ~27 GB | ~12 GB |
| B | L26 (chia đôi, phần 2) | ~248 | ~27 GB | ~12 GB |
| C | L30 + L25 | 184 | ~20 GB | ~9 GB |
| D | L24 + L22 + L21 | 103 | ~11 GB | ~5 GB |
| E | L23 + L28 + L29 + L27 | 88 | ~10 GB | ~4 GB |

**Ba quy tắc bắt buộc để mô hình này không vỡ:**

1. **Mọi máy đều có đủ `index/`** (csv/clip/objects/metadata phủ 100% từ đầu).
   Chỉ ảnh và video là chia. Nghĩa là mọi máy **truy vấn được toàn bộ kho**,
   chỉ không mở được ảnh/video ngoài phần mình giữ.
2. **Mỗi máy chạy `02_verify.py` + `03_verify_CLIP.py` trên phần mình giữ**
   rồi báo kết quả về. Bảng cái chỉ được coi là đúng khi đủ 873 video được
   phủ, không phải chỉ 29. Gửi đúng **hai file** — `index/verify_report.csv`
   (script 02) và `index/verify_clip*.csv` (script 03) — bỏ vào
   `dev/verify/<nhóm_L>/`, rồi `python scripts/07_gop_kiem_chung.py` ra bảng
   độ phủ. **Không gửi `master.parquet`/`clip.npy`/`objects.parquet`:**
   mỗi bộ 395 MB và giống hệt nhau trên mọi máy trừ ba cột đường dẫn.

   *Vì sao gộp được:* `00_discover.py` duyệt theo `sorted(video_id)` và
   `01_build_index.py` đánh `row_id` tuần tự, nên `row_id ↔ (video_id, kf_n)`
   là như nhau ở mọi máy có đủ 873 file CSV. **Đã đo thật** — đối chiếu 23
   dòng L29 trong `verify_clip.csv` của máy khác với `master.parquet` máy này:
   23/23 trùng khít cả `kf_n` lẫn `frame_idx`. Đây là điều làm cho A5.5 (đường
   dẫn tuyệt đối khác nhau) **không** phá được việc gộp kết quả: đường dẫn
   khác máy, nhưng `row_id` thì không.
3. **Ai giữ phần nào thì làm việc nặng phần đó.** Trích dày, OCR, decode video
   chạy trên máy giữ dữ liệu; kết quả xuất ra Parquet (`ocr.parquet`,
   `asr.parquet`, `dense_*.parquet`) rồi gộp qua Drive. Parquet nhẹ, video nặng
   — chỉ chuyển cái nhẹ.

> **Cái này thay đổi lịch trình:** TV2 (đường găng) và TV4 (OCR) giờ là **việc
> phân tán**, không phải một người làm hết. Cần chốt sớm ai giữ nhóm L nào.

---

## PHẦN C — CƠ CHẾ ĐIỂM CHI PHỐI MỌI THIẾT KẾ

*Giữ nguyên v3, không có gì thay đổi.*

`Final Score = trung bình R@{1, 5, 20, 50, 100}` — hàm bậc thang theo thứ
hạng của **câu đúng đầu tiên**:

| Câu đúng đầu tiên ở hạng | Final Score |
| --- | --- |
| 1 | **1,00** |
| 2 – 5 | 0,80 |
| 6 – 20 | 0,60 |
| 21 – 50 | 0,40 |
| 51 – 100 | 0,20 |
| > 100 | 0,00 |

Bốn hệ quả bắt buộc code theo:

1. **Luôn nộp đủ 100 câu.** Không có điểm phạt. Câu thứ 100 vẫn đáng 0,2.
2. **Top-5 phải ĐA DẠNG.** Ràng buộc cứng: *mỗi video ≤ 2 slot trong top-5;
   top-20 trải trên ≥ 8 video khác nhau.*
3. **TRAKE có điểm phân số** → không bao giờ bỏ trống một khoảnh khắc.
4. **Q&A: `answer` sai → 0 điểm bất kể frame đúng.** Ưu tiên độ chắc chắn của
   câu trả lời hơn độ chính xác của frame.

---

## PHẦN D — LỘ TRÌNH

### GIAI ĐOẠN 0 — Bảng cái & chốt giả định

#### D0.1 — Dựng bảng cái: **XONG**

| Câu hỏi phải trả lời | Trạng thái |
| --- | --- |
| Ảnh keyframe thứ *i* có đúng dòng thứ *i* CSV? | ✅ **Đúng** — 83/83 trên L21+L22+L29 |
| Bao nhiêu % keyframe có object JSON? | ✅ **100%** |
| Mật độ keyframe toàn bộ có ~109 frame? | ✅ **Không — 55 frame** |

Output đủ: `master.parquet` (177.321 dòng), `clip.npy` (177321×512),
`objects.parquet` (1.122.384 detection), `problems.csv`.

`problems.csv` có 844 dòng nhưng **toàn bộ là `lech_so_keyframe`** — do chưa
tải ảnh, không phải lỗi ghép. **Không có dòng `lech_so_vector` nào**, đây mới
là loại lỗi nghiêm trọng.

**Còn thiếu:** kiểm tra chéo bằng mắt (mỗi người mở 3 dòng `KHOP`, đối chiếu
tận mắt). Đây là lần **duy nhất** nên kiểm bằng mắt — để tin vào công cụ.

#### D0.2 — Chốt giả định

| # | Việc | Người | Trạng thái |
| --- | --- | --- | --- |
| 0.a | Gửi BTC câu hỏi cửa sổ `[s,e]` | Khánh | ✅ **đã gửi**, chờ trả lời |
| 0.b | Chốt OCR + ASR tiếng Việt | TV4 | ⬜ chưa |
| 0.c | Chốt VLM cho Q&A | TV5 | 🟡 **đang chạy** — xem D0.3 |
| 0.d | Phân bố nhãn objects | TV2 | ✅ **xong** — `index/label_top60.csv`, 514 nhãn |
| 0.e | **MỚI** — chia dữ liệu, chốt ai giữ nhóm L nào | cả nhóm | ⬜ chưa |

#### D0.3 — Chốt VLM cho Q&A: kết quả đợt 1

Harness của Khánh, **10 câu × 2 ngôn ngữ, `--runs 1`**:

| # | Model | Đúng (vi/en) | Format đúng | Latency p50 |
| --- | --- | --- | --- | --- |
| 1 | `gemini-3.1-flash-lite` | **50% / 40%** | **100%** | ~8–10s |
| 2 | `gemini-3.5-flash-lite` | 40% / 40% | **100%** | **~7,5s** |
| 3 | `gemma-4-26b-a4b` | 30% / 40% | 90% | ~14–15s |

Chưa đủ dữ liệu để xếp hạng:

| Model | Vấn đề |
| --- | --- |
| `gemini-3.6-flash` | hết quota giữa chừng (24/60), phần `vi` đạt 50% (4/8) |
| `nemotron-nano-12b-vl` | chỉ xong 9/20 lượt, bị 429 giữa chừng |
| `gemma-4-31b` | 0/20, rate-limit thượng nguồn — **chưa hề được đánh giá** |

**Ba nhận xét trước khi chốt:**

**1. Cỡ mẫu quá nhỏ để xếp hạng.** 10 câu/ngôn ngữ nghĩa là 50% vs 40% chỉ
khác nhau **đúng một câu**. Với n=10, khoảng tin cậy 95% của tỷ lệ 50% trải
từ ~19% đến ~81% — ba model hiện **không phân biệt được về mặt thống kê**.
Đừng chốt model dựa trên bảng này.

**2. Cột "Đồng thuận" hiện vô nghĩa** vì `--runs 1`. Đây là chỉ số quan trọng
nhất của harness: model trả lời khác nhau giữa các lượt thì không dùng được
cho bài thi, dù độ đúng trung bình cao.

**3. Điểm đáng chú ý nhất không phải thứ hạng, mà là TRẦN.** Cả ba model đều
ở khoảng **30–50%**. Khoảng cách giữa các model (10–20 điểm) **nhỏ hơn khoảng
cách tới mức dùng được**. Nghĩa là:

> Đừng tối ưu việc chọn model. Đầu tư vào **ngữ cảnh đưa vào model** —
> 3–5 frame trong cửa sổ ±2s + đoạn ASR tương ứng, đúng như thiết kế ở
> Giai đoạn 2, Bước 4. Đó mới là chỗ có thể
> đẩy 40% lên mức thi đấu được.

**4. Format đúng 100% có thể quan trọng hơn độ đúng.** Theo PHẦN C: `answer`
sai định dạng → 0 điểm bất kể frame đúng. `gemma-4-26b-a4b` thỉnh thoảng thêm
màu vào số (`"6 chiếc vàng"`, `"5 yellow"`) — vi phạm quy tắc "tối đa 4 từ,
không kèm chữ định tính". Hai model Gemini giữ format tuyệt đối. Đây là lợi
thế thật, không phải chi tiết nhỏ.

#### Mở rộng danh sách model — kiểm tra ĐIỀU KIỆN CẦN trước khi test

Bài Q&A đưa **3–5 ảnh frame + câu hỏi** vào model. Nghĩa là model **bắt buộc
phải nhìn được ảnh**. Đây là bộ lọc đầu tiên, và nó loại luôn một số cái tên
hay được nhắc tới:

| Nhà | Dòng model | Nhìn được ảnh? | Ghi chú |
| --- | --- | --- | --- |
| Google | Gemini Flash / Flash-Lite | ✅ có | đang test, format 100% |
| Google | Gemma (open weights) | ⚠️ tùy bản | chỉ bản có hậu tố thị giác mới nhìn được ảnh |
| NVIDIA | Nemotron **-vl** | ✅ có | hậu tố `vl` = vision-language |
| OpenAI | GPT-4o / GPT-5 dòng đa phương thức | ✅ có | trả phí, quota ổn định |
| **DeepSeek** | **deepseek-chat, deepseek-reasoner (V3/R1)** | ❌ **KHÔNG** | **thuần văn bản — không dùng được cho Q&A ảnh** |
| DeepSeek | deepseek-vl2 | ✅ có | nhưng ít nhà cung cấp phục vụ qua API |
| Anthropic | Claude dòng Opus/Sonnet | ✅ có | trả phí |
| Qwen | Qwen-VL | ✅ có | mã nguồn mở, chạy local được |

> **Cảnh báo về DeepSeek:** model chủ lực của DeepSeek (`deepseek-chat`,
> `deepseek-reasoner`) là **thuần văn bản, không nhận ảnh**. Đưa vào harness
> sẽ hoặc lỗi, hoặc tệ hơn: model đoán mò từ mỗi câu hỏi mà không thấy hình,
> ra một con số độ đúng trông có vẻ hợp lệ nhưng vô nghĩa. Chỉ `deepseek-vl2`
> mới nhìn được ảnh, và không phải nhà cung cấp nào cũng phục vụ nó.

**Quy tắc trước khi thêm bất kỳ model nào vào harness:**

1. Lọc theo **modality = image+text** trên trang model của OpenRouter. Đừng
   thêm theo tên nghe quen.
2. **Chạy một lượt thử với ảnh có nội dung rõ ràng** và hỏi "trong ảnh có gì".
   Model trả lời chung chung không dính dáng tới ảnh → nó không thật sự nhìn
   thấy. Đây là bài kiểm tra 30 giây, làm trước khi chạy 60 lượt.
3. Ghi lại **ngày test** — ID model và quota thay đổi liên tục.

**Kiểm chứng ID model trước khi chạy.** Tên model đổi nhanh; đừng chép ID từ
tài liệu cũ. Lấy ID trực tiếp từ danh sách model của nhà cung cấp tại thời
điểm chạy.

#### Kết quả đợt 2 — đo trực tiếp qua Google API, `--runs 3`

Harness: `scripts/05_bench_vlm.py`. 10 câu hỏi đếm trên keyframe thật, mỗi
model 3 lượt. **Không đo độ đúng** (lý do ở phần đính chính D1.6 — bộ nhận
diện đếm thiếu nên không làm đáp án được). Đo ba thứ khách quan:

| Model | temp | Format | **Đồng thuận** | p50 | p95 |
| --- | --- | --- | --- | --- | --- |
| **`gemini-3.1-flash-lite`** | **0** | **100%** | **100%** | **1,7s** | 3,7s |
| `gemini-3.1-flash-lite` | 1,0 | 100% | 50% | 2,8s | 6,2s |
| `gemini-3.5-flash-lite` | **0** | 100% | **30%** | 1,3s | 1,9s |
| `gemini-3.5-flash-lite` | 1,0 | 100% | 40% | 1,4s | 1,7s |
| `gemini-3.6-flash` | 1,0 | 94,7% | 70% | 7,0s | 11,4s |
| `gemma-4-31b-it` | 0 | **0%** | 90% | 12,3s | 19,8s |

**Bốn kết luận, theo thứ tự quan trọng:**

**1. `temperature = 0` là bắt buộc.** API mặc định `temperature = 1,0` — đó là
lý do đồng thuận thấp trong mọi đợt test trước, không phải model kém.
`gemini-3.1-flash-lite` nhảy từ **50% lên 100%** khi hạ về 0, và nhanh hơn.
Chi phí sửa: một dòng `generationConfig`.

**2. `gemini-3.5-flash-lite` KHÔNG tái lập được, kể cả ở `temperature = 0`.**
Đây là phát hiện quyết định.

*Về nguyên nhân:* nhiều khả năng là hạ tầng phía nhà cung cấp (gộp lô, định
tuyến MoE) chứ **không phải bản thân model kém hơn**. Ghi rõ điều này để sau
này nếu cần `3.5` thì biết là đi **thử lại**, không phải đã loại vĩnh viễn.
Quyết định không đổi — tính tái lập là yêu cầu thật của bài thi. Cùng ảnh, cùng câu hỏi, 3 lượt:

```text
                    lượt 1  lượt 2  lượt 3
gemini-3.1-flash-lite (temp 0)          gemini-3.5-flash-lite (temp 0)
  Bicycle      12    12    12             Bicycle      10    12    11
  Chair         8     8     8             Chair        14    14    11
  Man          13    13    13             Man          10    13    13
  Motorcycle   13    13    13             Motorcycle   16    15    17
  Person        4     4     4             Person        4     5     4
  -> 10/10 giống hệt                      -> 7/10 câu đổi đáp án
```

Bài thi nộp file rồi chấm — pipeline cho kết quả khác nhau mỗi lần chạy thì
không gỡ lỗi được, không tin được, và không tái lập được điểm dev.
**Chọn `gemini-3.1-flash-lite`.**

Lưu ý: bảng của đợt 1 (OpenRouter) cho `3.5` gần bằng `3.1` về độ đúng và
nhanh hơn — nhìn vào đó dễ chọn `3.5`. Chỉ số đồng thuận mới lộ ra khác biệt
thật, và nó ngược chiều.

**3. Gemma bị loại vì format, không phải vì độ đúng.** `gemma-4-31b-it` tuân
thủ format **0%** — nó nhả nguyên chuỗi suy luận bằng tiếng Anh:

```text
"Okay, let's count the cars in the image.  1. Scanning the road from left to
 right: there's a white car here: `{"point": [410, 414], "label"...
```

Theo PHẦN C, `answer` sai định dạng = 0 điểm bất kể frame đúng. Chậm gấp 7
lần cũng không giúp gì. Vẫn có thể cứu bằng prompt ép format chặt hơn, nhưng
đó là công sức đổ vào model đang thua ở mọi mặt khác.

**4. `gemma-4-31b` chạy tốt qua Google API** — trong khi qua OpenRouter nó
thất bại 0/20 vì *"temporarily rate-limited upstream"*. **Model không tệ, nó
chưa từng được đánh giá.** Bài học chung: lỗi hạ tầng của nhà trung gian dễ
bị đọc nhầm thành model kém.

**Và một phát hiện về hạ tầng:** gọi thẳng Google nhanh hơn OpenRouter **4–5
lần** (`3.5-flash-lite`: 7,5s → 1,4s). Với 100 câu × nhiều vòng thử nghiệm
trong 4 tuần, đây là hàng giờ đồng hồ.

**Việc tiếp theo cho 0.c** — theo thứ tự:

1. **Tăng cỡ mẫu lên ≥ 50 câu** trước khi so sánh model. Dùng chính tập dev
   Q&A của Khánh khi có (15 câu là chưa đủ, cần mở rộng).
2. Chạy lại `gemini-3.1-flash-lite` và `gemini-3.5-flash-lite` với `--runs 3`
   để đo đồng thuận thật.
3. **Thử với ngữ cảnh thật** (3–5 frame + ASR), không phải chỉ 1 frame — vì
   đó mới là input của hệ thống cuối.
4. Thử lại `gemma-4-31b` và `nemotron-nano-12b-vl` lúc khác (rate-limit
   thượng nguồn thường hết sau vài chục phút–vài giờ).
5. Thử lại `gemini-3.6-flash` vào ngày mới (quota Gemini free reset theo ngày).
6. **Chốt phương án quota.** Ba model đã chết vì rate-limit trong một đợt test
   20 lượt. Bài thi cần chạy 100 câu × nhiều lần — quota free sẽ không đủ.
   Phải quyết định sớm: trả phí, hay chạy model local.

---

### GIAI ĐOẠN 1 — Bốn kênh nguyên liệu + thước đo

#### TV1 — Kênh 1: truy hồi CLIP

- Nạp `clip.npy` (float32, 347 MB) vào RAM, cosine = `M @ q`.
- Text encoder **`ViT-B-32-quickgelu`** — xem A6. **Đặt assert kiểm tra tag.**
- Dịch/viết lại truy vấn sang tiếng Anh, sinh 3–5 biến thể.
- ⚠️ Trả về `frame_idx`, không phải `kf_n`.
- ⚠️ Mọi hàm giây↔frame nhận `fps` làm tham số (**4 giá trị**, xem A5.3).

#### TV2 — Module trích xuất frame dày *(đường găng)*

```python
extract_dense(video_id, center_frame, radius_sec, stride) -> [(frame_idx, image)]
```

- `ffmpeg -ss <t> -t <d> -i <video>` — seek **trước** `-i`.
- Cache theo `(video_id, cửa sổ)` trong `cache/`.
- **Tự đo chi phí trên máy mình**, đừng dùng mốc 3,1s/300 frame của v3.
- **Chạy phân tán:** mỗi máy trích cho nhóm L mình giữ, xuất Parquet, gộp qua
  Drive.

#### TV3 — Kênh 2: BM25 trên metadata cấp video

Giữ nguyên v3, ưu tiên cao. Dữ liệu giàu hơn v3 nghĩ (A3): description trung
bình 955 ký tự, 99,7% tiếng Việt có dấu.

#### TV4 — Kênh 3: OCR ticker + ASR audio

Giữ nguyên v3. **Chạy phân tán** theo nhóm L mình giữ.

#### TV2 — Kênh 4: Objects *(MỚI, nâng từ kênh phụ)*

Xem [PHẦN D1.6](#phần-d16--cách-nâng-objects-lên-kênh-chính) bên dưới.

#### TV5 — Khung pipeline & chuẩn I/O

Giữ nguyên v3.

```python
Candidate = {video_id, frame_idx, score, source, meta}
Answer_KIS   = {video_id, frame_idx}
Answer_QA    = {video_id, frame_idx, answer}
Answer_TRAKE = {video_id, [frame_idx_1..frame_idx_N]}
```

#### Khánh — Thước đo + tập dev

Giữ nguyên v3, thêm **hai ràng buộc mới**:

1. **Lấy mẫu phân tầng theo nhóm L** (A2 — L26 chiếm 57%, ngẫu nhiên sẽ lệch).
2. **Xem trước video của nhiều nhóm L** trước khi viết truy vấn (A7 — nội dung
   lệch về ẩm thực, không phải giao thông).

---

### PHẦN D1.6 — Cách nâng objects lên kênh chính

*Sáu việc, ước tính 1,5 ngày, không phải một tuần.*

#### 1. Chọn ngưỡng: **0,5**

| Ngưỡng | Phủ keyframe | Detection/keyframe | Đánh giá |
| --- | --- | --- | --- |
| 0,3 | 95,0% | 6,3 | nhiều nhiễu ở đuôi |
| **0,5** | **89,6%** | **3,4** | **điểm cân bằng** |
| 0,7 | 73,2% | 1,6 | sạch nhưng mất 27% keyframe |

Dựng lại `objects.parquet` ở ngưỡng 0,5 **không cần dựng lại bảng cái**:

```powershell
python scripts\01_build_index.py --out .\index --objects-only --min-obj-score 0.5
```

Giữ luôn bản 0,3 để thử nghiệm — chi phí chỉ là một file 45 MB.

#### 2. Tính IDF cho 514 nhãn — **việc quan trọng nhất**

`Person` xuất hiện 161.352 lần, `Wok` 3.161 lần. Đếm thô thì `Person` át tất
cả. Trọng số nghịch tần suất sửa điều đó:

```python
# src/objects.py
import numpy as np, pandas as pd

def build_label_idf(objects: pd.DataFrame, n_keyframe: int) -> pd.Series:
    """IDF theo số keyframe chứa nhãn, không phải số detection."""
    df = objects.groupby("label")["row_id"].nunique()
    return np.log(n_keyframe / (df + 1))
```

Giá trị thật đã tính (`index/label_idf.parquet`, 514 nhãn):

| Nhãn | Số keyframe chứa | IDF |
| --- | --- | --- |
| Clothing | 86.819 | **0,714** |
| Person | 82.431 | **0,766** |
| Boat | 3.497 | 3,926 |
| Wok | 2.531 | 4,249 |
| Car | 2.371 | 4,314 |
| Chopsticks | 1.857 | **4,558** |

Chênh **6 lần** giữa nhãn phổ biến và nhãn hiếm. Lưu ý IDF của `Person` là
**0,77 chứ không phải ~0,1** — `log(177321/82432)` không thể nhỏ hơn thế. Nhãn
người vẫn đóng góp một chút, chỉ là ít hơn nhãn hiếm nhiều lần.

#### 3. Bảng ánh xạ Việt → Anh

Truy vấn thi đấu là tiếng Việt, nhãn là tiếng Anh. Không cần dịch cả 514 nhãn:
**top ~150 nhãn phủ khoảng 95% detection**, dịch tay chừng đó là đủ, mất ~1 giờ.

Lưu `dev/label_vi_en.csv` với cột `nhan_en, nhan_vi, dong_nghia` (nhiều từ
tiếng Việt cho một nhãn: `Wok` → "chảo, chảo gang, chảo sâu lòng").

Lưu ý thứ bậc OpenImages không tự gộp: `Car`, `Land vehicle`, `Vehicle` là
**ba nhãn riêng biệt**. Phải khai báo quan hệ cha-con thủ công cho các cụm
quan trọng, nếu không truy vấn "ô tô" sẽ bỏ sót.

#### 4. Hàm cho điểm

```python
def object_score(row_ids, labels_yeu_cau, objects, idf, min_score=0.5):
    """Điểm mềm, KHÔNG lọc cứng. Trả về mảng cùng thứ tự row_ids."""
    sub = objects[(objects.score >= min_score) &
                  (objects.label.isin(labels_yeu_cau))]
    w = sub.assign(w=sub.score * sub.label.map(idf))
    return w.groupby("row_id")["w"].sum().reindex(row_ids, fill_value=0.0).values
```

Nguyên tắc của v3 giữ nguyên: **cho điểm mềm, tuyệt đối không lọc cứng.**
OpenImages không có nhãn cho mọi khái niệm — xóa cứng là xóa mất đáp án đúng
mà không thuật toán nào cứu lại được.

#### 5. Đưa vào RRF như kênh thứ tư ngang hàng

| Kênh | Nguồn | Mạnh ở |
| --- | --- | --- |
| CLIP cosine | ma trận TV1 | bối cảnh thị giác tổng thể |
| BM25 metadata | TV3 | chủ đề, tên riêng, địa danh |
| BM25 OCR + ASR | TV4 | chữ trên hình, lời dẫn, con số |
| **Objects + IDF** | **TV2** | **vật thể cụ thể, đếm số lượng** |

> ⚠️ **Đính chính — objects KHÔNG trả lời được câu hỏi đếm.**
>
> Bản đầu của mục này viết "objects là kênh duy nhất cho biết số lượng, đặc
> biệt mạnh cho câu hỏi đếm". **Sai.** Kiểm bằng cách mở ảnh ra nhìn:
>
> | Ảnh | Detector (≥0,7) | VLM | Thực tế |
> | --- | --- | --- | --- |
> | `L21_V001/073.jpg` | 1 người | 4 | ≥ 4 |
> | `L21_V031/086.jpg` | 4 người | 13 | > 20 |
>
> Bộ nhận diện chỉ bắt vật nổi bật nhất nên **đếm thiếu nghiêm trọng**, và
> càng đông càng thiếu — đúng loại cảnh mà câu hỏi đếm hay rơi vào.
>
> Số hộp vẫn dùng được làm **tín hiệu tương đối để xếp hạng** (5 hộp `Boat`
> gần như chắc chắn nhiều thuyền hơn 1 hộp), nhưng **câu hỏi đếm phải để VLM
> nhìn ảnh trả lời**. Xem cảnh báo trong `src/objects.py::dem_nhan()`.

#### 6. Đo trên tập dev, giữ hay bỏ theo số

Đúng nguyên tắc Giai đoạn 3 của v3: **chỉ giữ cái nào tăng điểm đo được.**
Đo Final Score trước/sau khi thêm kênh objects. Nếu không tăng thì hạ lại
thành kênh phụ — nhưng lần này là quyết định dựa trên số đo, không phải suy
đoán từ một file JSON.

---

### GIAI ĐOẠN 2 — Ba dạng truy vấn

*Giữ nguyên cấu trúc 2 mũi nhọn của v3. Chỉ đổi: objects lên kênh chính,
và Bước 4 Q&A nhấn mạnh ngữ cảnh (theo D0.3).*

#### Mũi nhọn 1 — Textual KIS & Q&A (TV1, TV3, TV5)

**Bước 1 — Thu hẹp cấp video.** BM25 metadata → top-50 video.

**Bước 2 — Truy hồi đa kênh, hợp nhất bằng Reciprocal Rank Fusion.** Bốn kênh
ở bảng trên. RRF an toàn hơn weighted-sum vì không cần chuẩn hóa thang điểm
giữa cosine (0,2–0,35) và BM25 (không chặn trên).

**Bước 3 — Tinh chỉnh vị trí frame** *(chỉ khi BTC xác nhận cửa sổ hẹp ở 0.a)*.

**Bước 4 — Sinh câu trả lời (riêng Q&A).**

- **Không đưa 1 frame duy nhất.** Câu hỏi dạng đếm cần chuỗi frame.
- Đưa vào VLM: **3–5 frame trong cửa sổ ±2s + đoạn ASR tương ứng + câu hỏi**.
- Ép output ngắn, chuẩn tắc (số viết bằng chữ số, màu một từ, **tối đa 4 từ,
  không kèm chữ định tính**).
- Trong 100 câu nộp: giữ nhiều `frame_idx` khác nhau nhưng **cùng một `answer`**
  nếu VLM tự tin.

> Theo D0.3, **đây là nơi quyết định điểm Q&A, không phải việc chọn model.**
> Chênh lệch giữa các model là 10–20 điểm; chênh lệch giữa "1 frame" và
> "3–5 frame + ASR" nhiều khả năng lớn hơn thế.

#### Mũi nhọn 2 — TRAKE (TV2, TV4, Khánh)

*Giữ nguyên toàn bộ 5 bước của v3.*

**Bước 1 —** Bóc tách sự kiện thành N truy vấn con, giữ đúng thứ tự.
**Bước 2 —** Chốt video trước (BM25 metadata + CLIP tổng hợp) → 1–3 video.
**Bước 3 —** Cho điểm mềm bằng `object_score()`, không lọc cứng.
**Bước 4 —** Trích dày trong vùng ứng viên (`stride=1..2`, vùng ~30s).
**Bước 5 —** Dóng hàng thời gian bằng quy hoạch động, O(N·K), NumPy thuần.
DP top-M (beam) ra 100 chuỗi, bắt buộc khác nhau ở ≥ 2/N vị trí.

---

### GIAI ĐOẠN 3 — Xếp hạng lại & đóng gói

*Giữ nguyên v3.* Đo baseline trên tập dev → thử từng chiến thuật → **chỉ giữ
cái nào tăng điểm đo được** → ràng buộc đa dạng top-K → kiểm trên tập test
chưa từng nhìn.

---

## PHẦN E — BẢN ĐỒ PHỤ THUỘC

| # | Cầu nối | Nếu đứt | Trạng thái |
| --- | --- | --- | --- |
| **1** | **Bảng cái → tất cả** | mọi module sai mà không ai biết tới tuần 4 | ✅ **đã chứng minh đúng** |
| **2** | **TV2 (trích dày) → Bước 4 TRAKE** | TRAKE trần điểm ở ~0,15 | ⬜ chặn bởi tải video |
| **3** | **Khánh (tập dev) → toàn bộ GĐ3** | rerank thành đoán mò | ⬜ chặn bởi tải video |
| 4 | TV3 (BM25 metadata) → Bước 2 TRAKE | sai video → 0 điểm | ✅ dữ liệu đã đủ (100%) |
| 5 | 0.c (VLM) → Bước 4 Q&A | `answer` sai định dạng → 0 điểm | 🟡 đang test |
| 6 | TV1 (ma trận CLIP) → cả 2 mũi nhọn | cả hai mũi nhọn tắc | ✅ dữ liệu đã đủ (100%) |
| **7** | **MỚI — tải đủ video/keyframe → #2, #3** | **hai phụ thuộc quan trọng nhất đều tắc** | ⬜ **9,5% đã kiểm chứng** |

**Đường găng:**
`Bảng cái ✅ → tải dữ liệu ⬜ → TV2 (trích dày) → Khánh (DP + tập dev) → GĐ3`

> **Nút thắt đã dịch chuyển.** Ở v3, đường găng bắt đầu từ bảng cái. Bảng cái
> nay đã xong và được chứng minh. Nút thắt mới là **tải dữ liệu**: chỉ 83/873
> video có keyframe và video gốc, mà cả #2 lẫn #3 — hai phụ thuộc nặng nhất —
> đều cần chúng. Đây là việc cần dồn người vào ngay.

---

## PHẦN F — RỦI RO CÒN LẠI

| Rủi ro | Dấu hiệu sớm | Phương án |
| --- | --- | --- |
| **Chia dữ liệu nhiều máy làm đứt đường dẫn tuyệt đối** | `kf_path` trỏ file không tồn tại | Thống nhất đường dẫn dữ liệu giống nhau trên mọi máy, hoặc remap (xem A5.5). Riêng việc **gộp kết quả** thì an toàn: `row_id` như nhau trên mọi máy (đã đo 23/23 trên L29) |
| **Không ai chạy verify cho nhóm L mình giữ** | mới 83/873 video được kiểm chứng, 3/10 nhóm L | Bắt buộc mỗi máy chạy `02`+`03` cho phần mình, báo kết quả |
| BTC dùng cửa sổ `[s,e]` hẹp cho cả KIS | câu trả lời 0.a | Trích dày cho cả mũi nhọn 1 |
| Tập dev lệch về L26 hoặc lệch chủ đề | Khánh soạn xong, kiểm phân bố | Lấy mẫu phân tầng theo nhóm L (A2), xem trước nội dung (A7) |
| **Quota VLM free không đủ cho bài thi** | 3/6 model chết vì rate-limit ở đợt test 20 lượt | Chốt sớm: trả phí hay chạy model local |
| OCR ticker sai nhiều | kết quả 0.b < 80% | Chỉ OCR vùng tiêu đề tĩnh, bỏ ticker chạy |
| Batch 2 định dạng khác batch 1 | BTC công bố | Chạy lại `00`+`01` là biết ngay |
| CLIP ViT-B/32 quá yếu | điểm dev KIS thấp | Tự encode lại 177k ảnh bằng model mạnh hơn (cần GPU) |
| ~~Ảnh keyframe không khớp thứ tự CSV~~ | — | ✅ **đã loại trừ** — 83/83 khớp |

---

## PHẦN G — SO SÁNH v3 → v4

| Hạng mục | v3 | v4 |
| --- | --- | --- |
| Cơ sở của PHẦN A | 6 CSV, 8 npy, 1 video | **873 video, 177.321 keyframe** |
| Mật độ keyframe | 109 frame, 1,15% cặp ≤10 | **55 frame, 12,62%** |
| Trần R-Score TRAKE | ~0,09 | **~0,15** |
| Objects | "nhiễu nặng, kênh phụ, 2 giờ ở tuần 2" | **kênh chính thứ 4, IDF, 1,5 ngày** |
| Tên file object JSON | "không liên tục" | **liên tục 1-1, 100% phủ** |
| fps | "25 và 30" | **25 / 26,44 / 29,97 / 30** |
| Model CLIP | "ViT-B/32" | **`ViT-B-32-quickgelu`** (kèm bằng chứng) |
| Phân bố nhóm L | không nêu | **L26 chiếm 57% → lấy mẫu phân tầng** |
| Nội dung kho | giả định tin tức/giao thông | **lệch ẩm thực: cà chua > ô tô ×2,5** |
| Metadata | "mạnh, đang bị bỏ quên" | **xác nhận: 955 ký tự/video, 99,7% có dấu** |
| Hạ tầng Parquet+DuckDB | quyết định trên lý thuyết | **đã kiểm chứng thực tế** |
| Chia dữ liệu | không nêu | **PHẦN B4 — mô hình nhiều máy** |
| Đường găng | bắt đầu từ bảng cái | **bảng cái xong; nút thắt là tải dữ liệu** |
| Chốt VLM | chưa làm | **đợt 1 xong, kèm cảnh báo cỡ mẫu** |
| Kiểm chứng bảng cái | 1 phép (ảnh↔CSV) | **2 phép** (thêm CLIP↔CSV) |

---

## PHẦN H — VIỆC LÀM NGAY

1. **Cả nhóm — chốt ai giữ nhóm L nào** và bắt đầu tải. Đây là nút thắt duy
   nhất hiện tại; hai phụ thuộc nặng nhất (#2, #3) đều chờ nó.
2. **Mỗi máy — chạy `02_verify.py` + `03_verify_CLIP.py`** cho phần mình giữ
   ngay sau khi tải xong, báo kết quả về. Chưa đủ 873 video thì bảng cái mới
   chỉ được chứng minh trên 9,5%.
3. **Khánh — xem trước video của ≥ 4 nhóm L khác nhau** trước khi viết tập dev
   (A7). Song song: chờ trả lời 0.a từ BTC.
4. **TV5 — mở rộng harness VLM lên ≥ 50 câu, `--runs 3`**, và test với ngữ
   cảnh thật (3–5 frame + ASR). Chốt phương án quota.
5. **TV2 — dựng `src/objects.py`** (IDF + `object_score`) theo D1.6. Việc này
   **không chờ tải dữ liệu** vì objects đã phủ 100% từ đầu — làm được ngay.
6. **TV1 — dựng ma trận CLIP + text encoder** với tag `ViT-B-32-quickgelu` và
   assert kiểm tra. Cũng **không chờ tải dữ liệu**.
7. **Cả nhóm — chốt một bảng tên thành viên duy nhất** (TV1..TV5 + Khánh).
8. **Giao 31 video fps lạ cho một máy cụ thể** (A5.3) và chạy `02_verify.py`
   riêng cho chúng. Nhóm rủi ro cao nhất còn lại của bảng cái — nếu `26.44`
   là variable frame rate thì `frame_idx` của những video đó có thể lệch, mà
   đó chính là giá trị nộp bài.

> Mục 5 và 6 là hai việc **làm được ngay hôm nay** trong khi chờ tải dữ liệu.
> Đừng để cả nhóm ngồi chờ băng thông.
