# KẾ HOẠCH v4 — AI Challenge HCMC 2026, Vòng Sơ tuyển

*Thay thế v3. Giữ nguyên khung 4 giai đoạn, cấu trúc 2 mũi nhọn, và triết lý
"kiểm tra giả định trước khi build".*

**Khác v3 ở đâu:** PHẦN A của v3 đo trên 6 file CSV / 8 file `.npy` / 1 video
và được đánh dấu "không cần tranh luận lại". Giờ đã dựng xong bảng cái, PHẦN A
được đo lại trên **toàn bộ 873 video / 177.321 keyframe**. Ba con số sai lệch
đáng kể, một kết luận bị đảo ngược. Xem [PHẦN G](#phần-g--lịch-sử-sửa-đổi).

**Bản sửa 4.1 (2026-08-12) — nguồn bằng chứng mới:** đọc được bài báo hệ thống
của một đội **mùa AIC'25**, tức đúng cuộc thi này năm trước. Đây là lần đầu ta
có số liệu từ bên ngoài nhóm. Nó **trả lời câu hỏi 0.a** mà ta treo suốt Giai
đoạn 0, xác nhận hai quyết định hạ tầng, và lộ ra hai lỗ hổng thật trong kế
hoạch. Xem [PHẦN A8](#a8--bằng-chứng-ngoài-bài-báo-hệ-thống-aic25).

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
| Ảnh keyframe thứ *i* ↔ dòng thứ *i* CSV | ffmpeg trích frame tại `pts_time`, so tương quan pixel + biên độ với dòng kề | **871/873 đạt** (99,8%) |
| Vector CLIP thứ *i* ↔ dòng thứ *i* CSV | encode lại frame, so cosine + xếp hạng trong video | **863/873 đạt** (98,9%) |

Phép thứ hai v3 không yêu cầu, nhưng cần thiết: vector CLIP nằm trong file
`.npy` riêng, lệch hàng ở đó thì kiểm ảnh không phát hiện được — mà TV1 dùng
trực tiếp vector này.

*Phạm vi:* **873/873 video — 100%**, đủ cả 10/10 nhóm L. Mỗi máy tải thêm nhóm L nào thì chạy lại hai script đó
cho nhóm đó rồi gửi hai file kết quả về (xem `scripts/07_gop_kiem_chung.py`).

6 mẫu `NGHI_NGO` của script 03 đều **đúng hạng 1** với cách biệt dương
(+0,03…+0,16), cosine 0,919–0,947 — khác biệt tiền xử lý JPEG/resize, không
phải lệch chỉ số.

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

> **Đã đo trên máy này (14/08, `python src/trich_day.py --do-luong`):**
> **~284 ms/khung** (61 khung trong 17,3s, `L23_V014`, mỗi khung một tiến
> trình `ffmpeg` riêng) — **chậm hơn ~28 lần** mốc "10 ms/khung" của v3. Chi
> phí khởi động tiến trình `ffmpeg` trên Windows là phần lớn, không phải
> decode. Không đổi kết luận (vẫn bắt buộc trích dày), nhưng đổi cách ước
> lượng CPU ở B4: 129,8 giờ video × nhiều truy vấn con × nhiều khung/truy vấn
> ở ~0,28s/khung là đáng kể — nếu chi phí này chặn tiến độ, hướng tối ưu rõ
> nhất là gộp nhiều `-frames:v` vào MỘT lệnh ffmpeg cho cả cửa sổ thay vì một
> tiến trình mỗi khung. **Mỗi máy nên tự chạy lại lệnh trên** vì phần cứng
> khác nhau — con số này không đại diện cho máy khác.
>
> **Máy thứ hai đo lại (14/08) — và đã KIỂM CHỨNG hướng tối ưu nói trên.**
> Cùng `--do-luong` trên máy khác: **199 ms/khung** (61 khung / 12,1s,
> `L21_V001`). Cùng bậc với 284 ms, nên đây là tính chất của cách gọi ffmpeg
> chứ không phải của một máy cụ thể.
>
> Đo trực tiếp giả thuyết "chi phí nằm ở khởi động tiến trình, không phải
> decode" trên cùng một cửa sổ 61 khung liên tiếp:
>
> | Cách gọi | Thời gian | Mỗi khung |
> | --- | --- | --- |
> | 61 tiến trình `ffmpeg` riêng *(cách hiện tại)* | 10,28 s | **169 ms** |
> | **1 tiến trình cho cả cửa sổ** (`-ss` + `-t` + `image2pipe`) | **1,18 s** | **19 ms** |
>
> **Nhanh gấp 8,7 lần.** Và 19 ms/khung khớp lại với mốc "~10 ms/khung" của
> v3 — nhiều khả năng mốc v3 đo bằng cách gộp, còn cách gọi từng khung mới là
> thứ làm chậm 20–28 lần.
>
> → Giả thuyết của TV2 **đúng**, và đây là tối ưu đáng làm vì `trich_day` nằm
> trên đường găng (PHẦN E #2). Ràng buộc khi cài: `-ss` + `-t` trả về **một
> luồng ảnh nối tiếp**, phải tách theo header từng ảnh; và **cache theo từng
> frame vẫn giữ nguyên** — chỉ đổi cách lấy khi cache trượt, không đổi cấu
> trúc cache.

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
> không đếm thô. Xem [PHẦN D1.6](#phần-d16--kênh-objects-đã-cài-xong-srcobjectspy).
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

**31 video có fps lạ. ĐÃ KIỂM CẢ HAI LOẠI, đều đạt:**

```text
fps 26.44:  1 video -> L24_V044                    ✅ 1/1  ĐẠT
fps 29.97: 30 video -> L25_V004, L25_V005, L25_V008, L25_V013, L25_V014, L25_V017, L25_V021, L25_V022, L25_V025, L25_V030, L25_V031, L25_V034, L25_V039, L25_V040, L25_V043, L25_V048, L25_V049, L25_V052, L25_V057, L25_V058, L25_V061, L25_V066, L25_V067, L25_V070, L25_V074, L25_V077, L25_V078, L25_V084, L25_V085, L25_V088    ✅ 17/30  ĐẠT
```

**`26.44` đã hết là bẫy.** Máy giữ L24 tải được `L24_V044` và chạy cả hai
script: pixel corr **0,9997**, CLIP **hạng 1** cosine 0,9891. Nghi vấn VFR
không thành hiện thực — `frame_idx / pts_time` đo được là **26,4380**, khớp
`26.44` trong CSV.

Mẫu này khắc nghiệt hơn mọi mẫu khác: keyframe `003.jpg` ứng với `frame_idx=5`,
tức **0,19 giây**, mà bài kiểm pixel vẫn tách được khỏi hai dòng kề (biên độ
0,148). Bất kỳ chỗ nào làm tròn 26,44 → 25 hoặc 30 đều đã làm mẫu này trượt.

Hóa ra `L24_V044` cũng là **video dày keyframe nhất kho** — 42 keyframe trong
35,18 giây, ba cái cuối cách nhau đúng **1 frame**. Đây là ngoại lệ ở đầu kia
của phân bố so với trung vị 55 frame.

**`29.97` cũng đã hết là bẫy.** Máy giữ L25 kiểm 17/30 video nhóm này:
script 02 cho 13 mẫu corr **0,9810–0,9999**, toàn bộ `KHOP`; script 03 cho
11 video, cosine **0,9878–0,9979**, không mẫu nào lệch chỉ số.

> **Cả hai loại fps lạ đều đã được chứng minh.** Rủi ro fps — thứ v3 lẫn bản
> đầu của v4 coi là nguy hiểm nhất — nay đã đóng. Chỗ nguy hiểm nhất còn lại
> của bảng cái là **keyframe trùng lặp**, xem A5.6 ngay dưới.

### A5.6 — KEYFRAME TRÙNG LẶP: cái bẫy mới, nguy hiểm hơn fps

Phát hiện khi soi 8 dòng bị `03_verify_CLIP.py` báo `LECH_INDEX` ở lô L25.
Hóa ra **không có lệch chỉ số nào** — chúng là keyframe trùng lặp. Nhưng lần
truy ngược đó lộ ra một đặc điểm của kho mà cả v3 lẫn v4 đều chưa biết.

Đo trên toàn bộ 177.321 keyframe, so cosine từng keyframe với mọi keyframe
khác **trong cùng video**:

| Ngưỡng | Số keyframe có bản sao | Tỷ lệ |
| --- | --- | --- |
| ≥ 0,999 | 9.994 | 5,64% |
| ≥ 0,995 | 16.824 | 9,49% |
| **≥ 0,99** | **20.975** | **11,83%** |
| ≥ 0,98 | 28.859 | 16,28% |

Phân bố **cực kỳ lệch theo nhóm L** (ngưỡng 0,99):

| Nhóm | Tỷ lệ keyframe có bản sao |
| --- | --- |
| **L25** | **49,82%** |
| L27 / L30 / L28 / L26 | 2,16% / 2,15% / 2,11% / 2,04% |
| L23 / L24 / L29 / L21 / L22 | 1,59% → 0,27% |

L25 lệch **23 lần** so với nhóm cao thứ nhì. Video tệ nhất `L25_V085`: 408/599
keyframe có bản sao (68,1%). Có cặp cosine đúng **1,0000** — vector giống hệt.

Các bản sao **nằm liền nhau**, tức cảnh tĩnh kéo dài chứ không rải rác:
trung vị cách nhau 5 keyframe, 72,2% cách ≤ 10. Chỉ 6,1% cách > 50 keyframe
(đồ họa/logo lặp lại). Cụm điển hình có 5 bản sao, p90 là 19, lớn nhất 125.

**Vì sao đây là rủi ro điểm số, không chỉ là chuyện dữ liệu.**

`Answer_KIS` nộp một `frame_idx` cụ thể. Nếu frame đáp án nằm trong một cụm
20 keyframe gần như y hệt, hệ thống ta rất dễ trả về **một thành viên khác
của cụm** — đúng cảnh, đúng nội dung, nhưng **sai `frame_idx` → 0 điểm**.
Xác suất một keyframe bất kỳ rơi vào cụm là **~12% toàn kho, ~50% nếu câu hỏi
rơi vào L25**. L25 chiếm 21% số keyframe của kho.

**Cách xử lý phụ thuộc hoàn toàn vào câu trả lời của BTC cho câu hỏi 0.a**
(BTC chấp nhận cửa sổ `[s,e]` hay đòi đúng một `frame_idx`):

| BTC trả lời | Việc phải làm |
| --- | --- |
| Chấp nhận cửa sổ `[s,e]` | Gộp cụm trùng lặp lại còn **một đại diện**, giải phóng slot trong top-100 cho ứng viên khác. Rủi ro biến mất. |
| Đòi đúng `frame_idx` | Với ứng viên top, **nộp nhiều thành viên của cụm**. Ta có 100 slot, cụm trung vị chỉ 5 phần tử — trả giá được. Nhưng phải cân với ràng buộc đa dạng ở PHẦN C. |

> **Câu 0.a vừa tăng hẳn mức quan trọng.** Trước đây nó chỉ ảnh hưởng module
> trích dày cho TRAKE. Giờ nó quyết định luôn chiến lược nộp bài của KIS và
> Q&A. Nếu BTC chưa trả lời, **xây cả hai đường** và chọn sau — phần chung
> (bảng cụm trùng lặp) dùng được cho cả hai.

**Việc làm được ngay:** bảng cụm đã dựng sẵn tại `index/trung_lap.parquet`
(cột `row_id`, `max_cos`). Dựng lại bằng đoạn ở cuối D1.6.

**Hệ quả thứ hai — phép thử thứ hạng của script 03 không dùng được ở L25.**
Xếp hạng giữa các vector giống nhau tới 1,0000 chỉ là nhiễu. Script đã sửa:
thứ tự phán quyết giờ xét **cosine tuyệt đối trước, thứ hạng sau**. Lý do:
lệch chỉ số thật nghĩa là `clip[row_id]` là vector của một **cảnh khác**, khi
đó cosine tụt xuống 0,3–0,7 chứ không thể ≥ 0,95. Nên cosine cao đã đủ kết
luận ghép đúng, bất kể thứ hạng. Phán quyết mới `KHOP_TRUNG_LAP` dành cho
trường hợp cosine ≥ 0,95 nhưng hạng > 1 vì dòng thắng là bản sao.

---

### A5.7 — CỤM FRAME LIÊN TIẾP: vì sao phép kiểm chứng báo động giả

**Kết luận: 0 lệch chỉ số thật trên 873/873 video.** Mọi cảnh báo `LECH_INDEX`
đều truy ngược được về tính chất của kho, không phải lỗi ghép.

Ba tính chất gây báo động giả, đã đo trên toàn kho:

| Hiện tượng | Số đo | Hệ quả |
| --- | --- | --- |
| Cặp keyframe cách ≤ 2 frame | 10.845 cặp (6,12%), ở 745/873 video | hai ảnh gần như y hệt, thứ hạng đảo ngẫu nhiên |
| Keyframe **trùng hệt `frame_idx`** với dòng liền trước | **614 keyframe** (0,35%), ở 192 video | không phép đo nào phân biệt được |
| Cụm dày ở 5 keyframe cuối video | 132 video | ffmpeg seek mép file kém chính xác |

**Vì sao không còn là rủi ro tính điểm:** theo A8.1, `frame_idx` chỉ cần **rơi
trong khoảng chuẩn**. Hai dòng trùng `frame_idx` thì nộp cái nào cũng trúng.

**Vì sao vẫn phải biết:** chúng **chiếm slot trong top-100**. Đó là lý do có
`src/dedup.py` (xem PHẦN C mục 6).

> **Bài học phương pháp — thứ tự phán quyết.** `03_verify_CLIP.py` từng xếp
> *thứ hạng* trước *cosine tuyệt đối*, nên báo động giả hàng loạt. Lệch chỉ số
> THẬT thì `clip[row_id]` trỏ sang cảnh khác hẳn và cosine rơi về 0,3–0,7 —
> nên **cosine ≥ 0,95 chứng minh ghép đúng bất kể thứ hạng**. Đảo lại thứ tự
> phán quyết là hết báo động giả. Sinh ra phán quyết `KHOP_TRUNG_LAP`.

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

### A8 — Bằng chứng ngoài: bài báo hệ thống AIC'25

*Nguồn: Nguyen et al., "Vortex: Multi-Modal Fusion System for Intelligent Video
Retrieval", arXiv:2606.19682v1, 18/06/2026. Đội FocusOnFun, ĐH KHTN — ĐHQG-HCM.
Đạt **79,6/88 (90,5%)** vòng Sơ tuyển AIC'25 và xếp loại Xuất sắc vòng Chung kết.*

Mọi mục A1–A7 ở trên đo từ dữ liệu của chính ta. A8 là loại bằng chứng khác:
**một đội mạnh đã thi đúng cuộc thi này năm ngoái và công bố họ làm gì.** Phải
đọc nó khác cách đọc số liệu của mình — họ không có nghĩa vụ chứng minh gì cả,
và ở dưới có một chỗ số liệu của họ nói ngược lại lời họ viết.

#### A8.1 — CÂU 0.a ĐÃ CÓ ĐÁP ÁN (điều quan trọng nhất của cả mục này)

Mục Evaluation Metrics của bài, nguyên văn:

> Textual-KIS required both the video name and frame index **to fall within the
> reference range**, Visual-QA required correct video, frame, and textual answer,
> and Temporal Alignment granted **partial credit proportional to the number of
> correctly matched frames within the allowed tolerance**.

**Đáp án chuẩn là một KHOẢNG, không phải một frame.** Ta treo câu này suốt Giai
đoạn 0 và xếp nó là câu hỏi mở giá trị cao nhất. Ba hệ quả:

1. **A5.6 và A5.7 chính thức hết là rủi ro tính điểm.** 11,83% keyframe trùng
   lặp, 614 keyframe cùng `frame_idx` — nộp một trong hai dòng đều rơi vào cùng
   một khoảng. Ta đã kết luận vậy bằng suy luận; giờ có văn bản mùa trước.
2. **TRAKE có điểm từng phần, theo *số sự kiện khớp*.** Nên chiến thuật đúng là
   **nộp đủ N sự kiện kể cả khi chỉ chắc vài cái**. Bỏ trống chắc chắn 0; đoán
   sai cũng 0; nên luôn đoán. Điều này v4 chưa nói rõ — mục 3 của PHẦN C chỉ
   viết "không bao giờ bỏ trống một khoảnh khắc" mà không nêu lý do là điểm
   từng phần tính theo số sự kiện.
3. **Bước 3 của Mũi nhọn 1 ("tinh chỉnh vị trí frame") tụt ưu tiên.** Nó được
   viết là *"chỉ khi BTC xác nhận cửa sổ hẹp ở 0.a"* — điều kiện đó nhiều khả
   năng KHÔNG xảy ra.

> ⚠️ **Đây là luật AIC'25, không phải AIC'26.** Vẫn phải để Khánh hỏi BTC. Khác
> biệt là: trước đây ta *bị chặn*, giờ ta *có mặc định hợp lý để thiết kế theo*
> và chỉ cần xác nhận lại.

Công thức chấm trong bài trùng khít PHẦN C: `R@k = max R-Score trong top-k`,
điểm cuối = trung bình của R@{1, 5, 20, 50, 100}.

#### A8.2 — Bảng điểm của họ nói ngược lại lời họ viết

Bảng 1 của bài, cột "Điểm/câu" là do ta thêm:

| Vòng | Điểm | Số câu | **Điểm/câu** | Module tích hợp |
| --- | --- | --- | --- | --- |
| 1 | 20,6 | 24 | **0,86** | CLIP đơn thuần |
| 2 | 27,8 | 30 | **0,93** | + SigLIP2, hợp nhất RRF |
| 3 | 31,2 | 35 | **0,89** | + Temporal Search + Relevance Feedback |

Bài viết *"performance progressively improved as additional modules were
integrated"*. Nhưng điểm **tổng** tăng chủ yếu vì **số câu tăng**. Chuẩn hóa
theo câu thì vòng 3 **thấp hơn** vòng 2.

Có thể câu vòng sau khó hơn — nên đây không phải bằng chứng rằng Temporal
Search và Relevance Feedback vô dụng. Điểm cần nhớ là: **bài không có một
ablation nào.** Mọi tuyên bố về đóng góp của từng module đều không được kiểm
chứng có đối chứng.

**Hai điều rút ra, và chúng kéo ngược chiều nhau:**

- **CLIP đơn thuần đã đạt ~0,86/câu.** Một baseline CLIP làm tử tế lấy gần hết
  điểm. Điều này củng cố PHẦN E: ưu tiên số 1 là kênh CLIP chạy đúng và nhanh,
  không phải gom cho đủ kênh.
- **Nhưng giá trị biên của kênh 2–4 là chưa ai chứng minh**, kể cả đội 90,5%
  này. Nên GIAI ĐOẠN 3 giữ nguyên kỷ luật *"chỉ giữ cái nào tăng điểm đo được"*
  — và đó là lý do tập dev của Khánh vẫn là phụ thuộc #3.

#### A8.3 — Xác nhận hai quyết định hạ tầng, phủ định một

| Quyết định của ta | Họ làm gì | Kết luận |
| --- | --- | --- |
| RRF thay vì weighted-sum | RRF, `k = 60` | ✅ **Trùng.** Cả hằng số cũng trùng |
| Không dùng Milvus/Elasticsearch | **Có dùng** cả hai + Redis | ✅ **Vẫn giữ quyết định của ta** — xem B1 |
| CLIP ViT-B/32 512 chiều của BTC | ViT-L-14 (DFN5B) 1024 + SigLIP2 1152 | ❌ **Ta đang yếu hơn hẳn** |

#### A8.4 — Hai lỗ hổng thật trong kế hoạch v4

**Lỗ hổng 1 — thiếu kênh mô tả cảnh (caption).** Bốn kênh của v4 là CLIP, BM25
metadata, BM25 OCR/ASR, objects+IDF. **Không có caption sinh bởi VLM.** Mà hai
câu Textual KIS trong ví dụ của họ được giải nhờ đúng thứ đó:

- *"hang động có hình khắc động vật"* → trúng nhờ ASR + caption sinh tự động
- *"công trình dạng vòng elip bằng gạch đất nung"* → trúng nhờ scene description

Metadata cấp video (A3) mô tả **cả video**, không mô tả **cảnh trong keyframe**.
Objects cho nhãn rời rạc chứ không cho quan hệ ("vòng elip", "bằng gạch"). Nên
đây là một khoảng trống thật, không phải trùng lặp với kênh sẵn có.

Họ dùng **một model Qwen2.5-VL-3B-Instruct làm cả OCR lẫn captioning** — đổi
lấy hai kênh bằng một lần chạy. Bench OCR/ASR hiện tại của Khánh so EasyOCR /
PaddleOCR / VietOCR, **không cái nào biết viết caption**.

**Lỗ hổng 2 — CLIP ViT-B/32 của BTC là sàn, không phải trần.** Bước nhảy
0,86 → 0,93 điểm/câu của họ đến từ **thêm một model thứ hai mạnh hơn rồi RRF
hai cái**. PHẦN F đã liệt kê rủi ro "CLIP ViT-B/32 quá yếu" nhưng xếp nó ở thế
bị động (*"nếu điểm dev thấp thì..."*). Bài báo cho thấy nên xem đây là **cơ
hội chủ động**, và ta đã có sẵn hạ tầng để làm: chia nhóm L cho 5–6 máy, và đã
chứng minh pipeline tái lập chính xác giữa các máy (A0 — cosine trùng tới 4 chữ
số thập phân, 29/29 mẫu).

#### A8.5 — OCR mạnh hơn ta tưởng, nhưng phải dùng như BỘ LỌC

**3 trong 5 ví dụ thực chiến của họ được giải bằng OCR**, gồm cả câu TRAKE khó
nhất:

| Câu | Chuỗi OCR | Vì sao thắng |
| --- | --- | --- |
| tkis-02 | `hidro` | chữ trong câu hỏi trắc nghiệm hiện trên màn hình |
| vkis-07 | `DI TICH KIM LONG` | chữ khắc trên hiện vật |
| trake-03 | `PHI 1 0 BRU` | **bảng tỷ số** — chốt được thời điểm bàn thắng |

Nhưng để ý cú pháp họ gõ: `/filter all ocr{hidro}` — **LỌC, không phải xếp
hạng.** v4 đang xếp OCR làm một kênh BM25 hòa vào RRF. Với token hiếm, lọc dứt
khoát hơn hẳn: `hidro` có ở 3 khung hình thì lọc trả đúng 3, còn BM25 hòa RRF
có thể bị ba kênh kia dìm xuống dưới hạng 20.

**Cần CẢ HAI chế độ**, không phải chọn một. Đây là ràng buộc thiết kế cho
Kênh 3, ghi vào GIAI ĐOẠN 1.

#### A8.6 — Đây là cuộc thi TƯƠNG TÁC, có người ngồi lái

Điều chỉnh nhận thức lớn nhất. Vòng Chung kết chấm bởi Jury Board, gợi ý mở
dần, thí sinh gõ truy vấn trực tiếp trong phiên thi. Toàn bộ phần Relevance
Feedback, "nearby frame", Temporal Search mode của họ là **giao diện** — và
điểm đến từ **một con người thao tác nhanh**.

Công việc của ta tới giờ hoàn toàn là đánh chỉ mục ngoại tuyến. v4 không có một
dòng nào về giao diện. Nếu AIC'26 cũng tương tác thì **tốc độ và độ tiện của UI
là một phần của điểm số**, không phải phần thưởng thêm.

→ Thêm vào danh sách hỏi BTC, cùng câu 0.a. Xem [GIAI ĐOẠN 2](#giai-đoạn-2--ba-dạng-truy-vấn).

#### A8.7 — Bốn kỹ thuật nên lấy, xếp theo giá trị / công sức

| # | Kỹ thuật | Công sức | Vì sao đáng |
| --- | --- | --- | --- |
| 1 | **"Nearby frame"** — từ một kết quả, duyệt keyframe liền kề theo thời gian | vài chục dòng trên `master.parquet` | Xuất hiện ở 2/5 ví dụ của họ, là bước xoay chuyển của cả câu Q&A lẫn câu TRAKE |
| 2 | **OCR làm bộ lọc** (A8.5) | có sẵn dữ liệu | 3/5 ví dụ |
| 3 | **Temporal re-rank** (Algorithm 1) | ~20 dòng | TRAKE + KIS có bối cảnh trước/sau |
| 4 | **Rocchio feedback** | ~10 dòng NumPy | Chỉ có giá trị nếu thi tương tác (A8.6) |

Vì sao "nearby frame" đứng đầu: câu Q&A của họ cho gợi ý về **nguyên liệu**,
còn đáp án nằm ở **bước cắt** vài giây sau. Truy hồi ngữ nghĩa đưa tới lân cận;
đi bộ theo thời gian đưa tới đích. Cùng lý do với A1 (trung vị mật độ keyframe
55 frame — lân cận là chỗ đáng đi bộ).

#### A8.8 — Khử trùng lặp: cách rẻ để lấy một phần lợi ích của họ ngay

Họ không dùng keyframe của BTC mà tự trích lại bằng AutoShot + lọc theo
`rel_diff > 0,4` (chuẩn L2 giữa embedding frame hiện tại và keyframe giữ gần
nhất). **Bộ lọc đó thực chất là khử dư thừa** — đúng thứ A5.6 đo được ở kho của
ta: 11,83% keyframe trùng ở cosine ≥ 0,99, riêng L25 tới 49,82%.

Ta **không cần AutoShot và không cần trích lại gì**: `clip.npy` đã nằm sẵn
trong RAM, tính `rel_diff` giữa các keyframe liên tiếp là một phép trừ vector.

**Vì sao đáng làm, tính theo điểm:** điểm lấy `max R-Score trong top-k`, mà
`R@1` là 1/5 tổng điểm. Nếu top-5 bị 5 bản sao gần như y hệt của cùng một
khoảnh khắc chiếm chỗ, ta phí 4 slot mà không tăng cơ hội trúng. Việc này cộng
hưởng với ràng buộc đa dạng đã có ở PHẦN C mục 2 — nhưng ràng buộc đó tính theo
`video_id`, **không bắt được bản sao trong CÙNG một video**.

`index/trung_lap.parquet` (dựng ở Giai đoạn 0) đã có sẵn `max_cos` từng
keyframe tới bản giống nhất cùng video. Nguyên liệu có rồi.

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

**Đội AIC'25 ở A8 dùng Milvus + Elasticsearch + Redis. Ta vẫn không đổi.** Đây
là chỗ ta cố ý làm khác họ, nên phải nói rõ vì sao — đo trên máy này:

```text
clip.npy: (177321, 512) float32 = 346,3 MB
Quét TOÀN BỘ + lấy top-100:      16,7 ms/truy vấn   (NumPy vét cạn)
Nếu thêm SigLIP2 1152 chiều:     779 MB RAM         (vẫn nạp thẳng được)
```

HNSW của Milvus sinh ra cho hàng chục triệu vector. Ở 177 nghìn, nó đổi vài
mili-giây mà người dùng không cảm nhận được lấy một tầng hạ tầng phải cài, phải
đồng bộ giữa 6 máy, và **phải debug lúc đang thi**. Vét cạn còn cho *đúng* top-k
chứ không xấp xỉ.

> **Ngưỡng đảo quyết định:** nếu số keyframe vượt ~2 triệu (batch 2 lớn bất
> thường), hoặc nếu ta thêm ≥ 3 model embedding cùng lúc, hãy đo lại. Dưới mức
> đó thì vét cạn thắng về mọi mặt trừ độ "trông có vẻ hiện đại".

Riêng Elasticsearch: thứ ta cần từ nó là **lọc theo OCR/metadata** (A8.5). DuckDB
làm được bằng một câu `WHERE ... LIKE`/full-text trên Parquet.

### B2. Không dùng C++ / pybind11 / ctypes

Giữ nguyên v3. Nút cổ chai thật là decode video và inference model.

### B3. Cấu trúc thư mục

*Sửa ở 4.1: bản trước liệt kê `src/` như thể **đã dựng**, nhưng đó là danh sách
**dự kiến** — thực tế lúc đó chỉ có `objects.py`. Dưới đây tách rõ hai cột.*

```text
aic2026/
  scripts/    00_discover  01_build_index  02_verify  03_verify_CLIP
              04_smoke_vlm  05_bench_vlm  06_tim  07_gop_kiem_chung  08_encode
              09_trich_day_batch  10_contact_sheet
  index/      master.parquet  clip.npy  objects.parquet  trung_lap.parquet
              label_idf.parquet  problems.csv  *_report
  dev/        số liệu, checklist, verify/
  docs/       hướng dẫn thành viên
  cache/      frame dày đã decode
  submissions/
```

| `src/` | Trạng thái | Là gì |
| --- | --- | --- |
| `schema.py` | ✅ | `Candidate`, `AnswerKIS/QA/TRAKE` |
| `dense.py` | ✅ | Kênh 1 — truy hồi vector ảnh, **không dính model nào** |
| `rrf.py` | ✅ | Hợp nhất **N** danh sách + ràng buộc đa dạng |
| `dedup.py` | ✅ | Gộp bản sao cùng video (A5.6, PHẦN C mục 6) |
| `lan_can.py` | ✅ | Đi bộ theo thời gian (A8.7 #1) |
| `thoi_gian.py` | ✅ | Xếp hạng lại theo chuỗi (A8.7 #3) |
| `objects.py` | ✅ | Kênh 4 — objects + IDF |
| `trich_day.py` | ✅ | **TV2 — trích frame dày** quanh một khoảnh khắc (Bước 4 TRAKE); trả `list[KhungDay]`, KHÔNG phải `Candidate` (không có row_id thật, xem docstring) |
| `bm25.py` | ⬜ | Kênh 2 + 3 — chưa có |
| `run.py` | ⬜ | Đường ống đầu-cuối — chưa có |

| `tests/` | Trạng thái | Chốt chặn cho |
| --- | --- | --- |
| `test_dense.py` | ✅ 7/7 qua | **Bẫy A6** — sai biến thể model không ném lỗi, chỉ tụt điểm âm thầm |
| `moc_dense.json` | ✅ | Mốc cố định: truy vấn chuẩn → `row_id` phải không đổi |
| `test_trich_day.py` | ✅ 7/7 qua | `frame_idx` đúng dãy yêu cầu + khớp pixel với ảnh keyframe gốc (corr ≥ 0,95, cùng ngưỡng `02_verify.py`) + cache không gọi lại ffmpeg |

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
   là như nhau ở mọi máy có đủ 873 file CSV. **Đã đo thật** — đối chiếu hai lô
   của máy khác (23 dòng L29, 139 dòng L24+L30, 64 dòng L25+L28, 539 dòng
   L23+L26+L27) với `master.parquet` máy này: **765/765 trùng khít** cả
   `video_id`, `kf_n`, `frame_idx` lẫn `fps`.

   Bằng chứng mạnh hơn nữa: máy giữ L23+L26+L27 chạy lại **cả nhóm L21** —
   nhóm máy này đã chạy. Kết quả trùng khít 29/29 ở **mọi cột**, kể cả
   `cosine` tới 4 chữ số thập phân. Hai máy, hai ổ đĩa, cùng ffmpeg + CLIP,
   ra đúng cùng một con số. Không chỉ `row_id` tái lập được — **cả pipeline
   tái lập được.** Đây là điều làm cho A5.5 (đường
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

*Công thức giữ nguyên v3 và nay được bài báo AIC'25 xác nhận nguyên văn (A8.1).
Phần **hệ quả** thì có thay đổi: đáp án chuẩn là một KHOẢNG, không phải một
frame.*

`Final Score = trung bình R@{1, 5, 20, 50, 100}`, với
`R@k = max(R-Score của 100 kết quả đầu, xét tới hạng k)`. Với KIS/Q&A thì
R-Score là 0 hoặc 1, nên rút gọn thành hàm bậc thang theo thứ hạng của **câu
đúng đầu tiên**:

| Câu đúng đầu tiên ở hạng | Final Score |
| --- | --- |
| 1 | **1,00** |
| 2 – 5 | 0,80 |
| 6 – 20 | 0,60 |
| 21 – 50 | 0,40 |
| 51 – 100 | 0,20 |
| > 100 | 0,00 |

**Sáu hệ quả bắt buộc code theo:**

1. **Luôn nộp đủ 100 câu.** Không có điểm phạt. Câu thứ 100 vẫn đáng 0,2.
2. **Top-5 phải ĐA DẠNG.** Ràng buộc cứng: *mỗi video ≤ 2 slot trong top-5;
   top-20 trải trên ≥ 8 video khác nhau.*
3. **TRAKE: nộp ĐỦ N sự kiện, kể cả khi chỉ chắc vài cái.** *(sửa ở 4.1)* Điểm
   TRAKE tính **từng phần theo số sự kiện khớp** trong dung sai cho phép
   (A8.1). Bỏ trống chắc chắn 0; đoán sai cũng 0 — nên **luôn đoán**. Không bao
   giờ nộp một chuỗi thiếu vị trí.
4. **Q&A: `answer` sai → 0 điểm bất kể frame đúng.** Ưu tiên độ chắc chắn của
   câu trả lời hơn độ chính xác của frame.
5. **MỚI — `frame_idx` chỉ cần rơi TRONG khoảng chuẩn.** *(A8.1)* Không cần
   trúng chính xác một frame. Đây là lý do A5.6 (keyframe trùng lặp) và A5.7
   (cụm frame liên tiếp, 614 keyframe cùng `frame_idx`) **không còn là rủi ro
   tính điểm** — hai dòng cạnh nhau đều rơi cùng một khoảng.
6. **MỚI — khử trùng lặp TRONG cùng video trước khi cắt top-K.** *(A8.8)* Ràng
   buộc ở mục 2 tính theo `video_id` nên **không bắt được bản sao trong cùng một
   video**. Mà A5.6 đo được 11,83% keyframe có bản sao cùng video ở
   cosine ≥ 0,99 (L25: 49,82%). Năm bản sao chiếm hết top-5 là phí 4 slot.

---

## PHẦN D — LỘ TRÌNH

### GIAI ĐOẠN 0 — Bảng cái & chốt giả định

#### D0.1 — Dựng bảng cái: **XONG**

| Câu hỏi phải trả lời | Trạng thái |
| --- | --- |
| Ảnh keyframe thứ *i* có đúng dòng thứ *i* CSV? | ✅ **Đúng** — 871/873, phủ **100%** kho |
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

#### D0.3 — Chốt VLM cho Q&A: **XONG (đợt 2)**

*Báo cáo đầy đủ: `Test VLM AIC/BAO_CAO_BENCH_VLM_2026-08-13.md`. Dưới đây chỉ
là kết luận đã chốt.*

**Chốt `gemini-3.1-flash-lite`, `temperature = 0`.**

| Kết luận | Bằng chứng |
| --- | --- |
| **`temperature = 0` là BẮT BUỘC** | API mặc định 1,0 — đó là lý do đồng thuận thấp ở mọi đợt trước, không phải model kém. Hạ về 0: đồng thuận **50% → 100%**, lại còn nhanh hơn |
| **`3.5-flash-lite` không tái lập được** | ngay ở `temp = 0` vẫn **7/10 câu đổi đáp án** giữa 3 lượt. `3.1` thì 10/10 giống hệt. Nghi do hạ tầng nhà cung cấp chứ không phải model kém — cần thì **thử lại**, chưa loại vĩnh viễn |
| **Gemma loại vì FORMAT, không phải độ đúng** | `gemma-4-31b-it` tuân thủ format **0%** — nhả nguyên chuỗi suy luận. PHẦN C mục 4: sai định dạng = 0 điểm |
| **Gọi thẳng Google nhanh hơn OpenRouter 4–5 lần** | 7,5s → 1,4s. Với 100 câu × nhiều vòng trong 4 tuần là hàng giờ |

**Ba điều quan trọng hơn việc chọn model:**

1. **Trần độ đúng ~30–50% ở MỌI model.** Khoảng cách giữa các model (10–20
   điểm) **nhỏ hơn khoảng cách tới mức dùng được**. Đừng tối ưu việc chọn
   model — đầu tư vào **ngữ cảnh đưa vào** (3–5 frame ±2s + đoạn ASR), đúng
   thiết kế ở Giai đoạn 2 Bước 4.
2. **Cỡ mẫu 10 câu không xếp hạng được gì.** Với n=10, khoảng tin cậy 95% của
   tỷ lệ 50% trải từ ~19% đến ~81%. Cần ≥ 50 câu — dùng chính tập dev Q&A.
3. **Quota free sẽ không đủ.** Ba model đã chết vì rate-limit trong một đợt
   test 20 lượt. Bài thi cần 100 câu × nhiều lần. **Phải chốt: trả phí hay
   chạy local.**

> **Bộ lọc bắt buộc trước khi thêm bất kỳ model nào vào harness:** model phải
> **nhìn được ảnh**. Cạm bẫy cụ thể: `deepseek-chat` và `deepseek-reasoner` là
> **thuần văn bản** — đưa vào harness thì model đoán mò từ câu hỏi mà không
> thấy hình, ra một con số độ đúng trông hợp lệ nhưng vô nghĩa. Chỉ
> `deepseek-vl2` mới nhìn được ảnh.
>
> Kiểm 30 giây: đưa một ảnh nội dung rõ ràng, hỏi "trong ảnh có gì". Trả lời
> chung chung không dính tới ảnh → nó không thật sự nhìn thấy.

---

### GIAI ĐOẠN 1 — Năm kênh nguyên liệu + thước đo

*Bản 4.1: kênh 4 → **năm** kênh (thêm caption VLM, A8.4), cộng hai việc mới
không chờ tải dữ liệu (khử trùng lặp, embedding thứ hai).*

> **Thứ tự ưu tiên, theo A8.2:** CLIP đơn thuần đã đạt ~0,86 điểm/câu ở đội
> AIC'25. **Kênh 1 chạy đúng và nhanh quan trọng hơn gom đủ năm kênh.** Đừng để
> kênh 5 chặn kênh 1.

#### TV1 — Kênh 1: truy hồi CLIP

- Nạp `clip.npy` (float32, 347 MB) vào RAM, cosine = `M @ q`.
- Text encoder **`ViT-B-32-quickgelu`** — xem A6. **Đặt assert kiểm tra tag.**
- Dịch/viết lại truy vấn sang tiếng Anh, sinh 3–5 biến thể.
- ⚠️ Trả về `frame_idx`, không phải `kf_n`.
- ⚠️ Mọi hàm giây↔frame nhận `fps` làm tham số (**4 giá trị**, xem A5.3).
- **MỚI — vét cạn, đừng cài thư viện ANN.** Đo được 16,7 ms cho toàn bộ 177k
  vector + lấy top-100 (B1). Faiss/HNSW ở cỡ này là hạ tầng thừa.

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

**MỚI (4.1) — phải có HAI chế độ, không chỉ một** *(A8.5)*:

| Chế độ | Dùng khi | Cài đặt |
| --- | --- | --- |
| **Lọc cứng** | truy vấn chứa chuỗi hiếm, đọc được từ màn hình | `WHERE ocr_text LIKE '%hidro%'` trên Parquet, trả **mọi** khung khớp, bỏ qua RRF |
| Kênh BM25 hòa RRF | truy vấn mô tả chung | như v3 |

Lý do: 3/5 ví dụ thực chiến của đội AIC'25 được giải bằng **lọc** OCR — kể cả
câu TRAKE khó nhất, chốt bằng bảng tỷ số `PHI 1 0 BRU`. Với token hiếm, hòa vào
RRF có thể bị ba kênh kia dìm xuống dưới hạng 20; lọc thì không.

> **Gợi ý cho bench của Khánh:** vì có chế độ lọc, **recall của OCR quan trọng
> hơn precision** ở những chuỗi hiếm — đọc sót chữ `hidro` là mất hẳn đường
> thắng, còn đọc thừa vài chuỗi rác thì chế độ lọc không bao giờ chạm tới. Đây
> là một lý do nữa để bỏ CER và đo bằng truy hồi.

#### TV2 — Kênh 4: Objects *(nâng từ kênh phụ ở v3)*

Xem [PHẦN D1.6](#phần-d16--kênh-objects-đã-cài-xong-srcobjectspy) bên dưới.

#### Kênh 5: Caption / mô tả cảnh bằng VLM *(MỚI ở 4.1 — A8.4)*

**Lỗ hổng thật trong v4.** Metadata (A3) mô tả *cả video*; objects cho *nhãn
rời rạc*. Không kênh nào diễn tả được **quan hệ trong một cảnh** — mà đó chính
là dạng truy vấn KIS hay gặp:

> *"công trình dạng vòng elip bằng gạch đất nung"* — `Building` trong objects
> không phân biệt được elip với vuông; metadata cấp video không nhắc tới.

Đội AIC'25 giải đúng dạng này bằng caption sinh tự động.

**Cách rẻ nhất: một model làm hai việc.** Họ dùng **Qwen2.5-VL-3B-Instruct**
cho *cả OCR lẫn captioning* trong một lần chạy. Bench hiện tại của Khánh so
EasyOCR / PaddleOCR / VietOCR — **không cái nào biết viết caption**.

- **Việc cần làm:** thêm Qwen2.5-VL-3B vào bench OCR như một ứng viên, chấm
  bằng chính thước đo truy hồi đã dựng ở `retrieval_v2.py`.
- **Nếu nó thắng:** một lần chạy ra hai kênh, và bài toán chi phí đổi hẳn.
- **Nếu thua ở OCR:** vẫn có thể chạy riêng cho caption trên tập con.
- ⚠️ **Đừng để kênh 5 chặn kênh 1–4.** Đây là kênh thêm, không phải đường găng.
- ⚠️ 177.321 keyframe × một model 3B là chi phí lớn. **Đo trên 200 ảnh trước**,
  nhân lên, rồi mới quyết — đúng cách đã làm với OCR/ASR.

#### TV1 + TV2 — Hai việc MỚI làm được NGAY, không chờ tải dữ liệu

**(a) Khử trùng lặp trong cùng video** *(A8.8, PHẦN C mục 6)*

Nguyên liệu đã có: `clip.npy` trong RAM và `index/trung_lap.parquet` (đã dựng ở
Giai đoạn 0, chứa `max_cos` từng keyframe tới bản giống nhất cùng video).

```python
# src/dedup.py — gộp keyframe gần trùng thành cụm, giữ một đại diện
# Không cần AutoShot, không cần trích lại gì: chỉ so vector đã có.
def gom_cum(clip, master, nguong=0.99): ...   # -> cot 'cum_id'
def loc_top_k(ung_vien, k, moi_cum_toi_da=1): ...
```

Áp **sau khi xếp hạng, trước khi cắt top-K**, để không mất ứng viên ở khâu
truy hồi. Kiểm hiệu quả trên tập dev của Khánh — nếu không tăng điểm thì bỏ,
theo kỷ luật của GIAI ĐOẠN 3.

**(b) Đo thử một embedding thứ hai** *(A8.4 lỗ hổng 2)*

Bước nhảy 0,86 → 0,93 điểm/câu của đội AIC'25 đến từ đúng chỗ này. Nhưng
**đừng encode cả 177k ảnh trước khi biết nó có lợi không**:

1. Lấy ~2.000 keyframe (phân tầng theo nhóm L, theo A2).
2. Encode bằng SigLIP2 *hoặc* CLIP ViT-L-14, đo trên tập dev của Khánh.
3. So ba cấu hình: **ViT-B/32 đơn thuần** / model mới đơn thuần / **RRF hai cái**.
4. Chỉ khi RRF thắng rõ mới chạy toàn kho — chia nhóm L cho 5–6 máy, đúng mô
   hình PHẦN B4 vừa dùng thành công ở Giai đoạn 0.

RAM không phải rào cản: thêm SigLIP2 1152 chiều là 779 MB, vẫn nạp thẳng (B1).

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

### PHẦN D1.6 — Kênh objects: **ĐÃ CÀI XONG** (`src/objects.py`)

Còn lại đúng **một** việc: bảng ánh xạ Việt → Anh (xem cuối mục).

| Quyết định | Chốt | Lý do |
| --- | --- | --- |
| Ngưỡng detection | **0,5** | phủ 89,6% keyframe, 3,4 detection/keyframe. Ngưỡng 0,3 nhiễu đuôi, 0,7 mất 27% keyframe |
| Trọng số | **IDF theo số KEYFRAME chứa nhãn**, không theo số detection | một ảnh có 5 người vẫn chỉ tính 1 cho `Person`; tính theo detection thì nhãn hay lặp trong một hình bị phạt oan |
| Cách dùng | **cho điểm mềm, TUYỆT ĐỐI không lọc cứng** | OpenImages không có nhãn cho mọi khái niệm — xóa cứng là xóa mất đáp án đúng, không thuật toán nào cứu lại |

Giá trị thật ở `index/label_idf.parquet` (514 nhãn): `Clothing` 0,714 /
`Person` 0,766 ... `Wok` 4,249 / `Chopsticks` 4,558 — **chênh 6 lần** giữa nhãn
phổ biến và nhãn hiếm.

> ⚠️ **Đính chính — objects KHÔNG trả lời được câu hỏi đếm.**
>
> Bản đầu viết "objects là kênh duy nhất cho biết số lượng, đặc biệt mạnh cho
> câu hỏi đếm". **Sai.** Kiểm bằng cách mở ảnh ra nhìn:
>
> | Ảnh | Detector (≥0,7) | VLM | Thực tế |
> | --- | --- | --- | --- |
> | `L21_V001/073.jpg` | 1 người | 4 | ≥ 4 |
> | `L21_V031/086.jpg` | 4 người | 13 | > 20 |
>
> Bộ nhận diện chỉ bắt vật nổi bật nhất nên **đếm thiếu nghiêm trọng**, càng
> đông càng thiếu — đúng loại cảnh mà câu hỏi đếm hay rơi vào. Số hộp vẫn dùng
> được làm **tín hiệu tương đối để xếp hạng**, nhưng **câu hỏi đếm phải để VLM
> nhìn ảnh trả lời**. Xem `src/objects.py::dem_nhan()`.

**Việc còn lại — `dev/label_vi_en.csv` (chưa có).** Truy vấn thi đấu là tiếng
Việt, nhãn là tiếng Anh. Không cần dịch cả 514 nhãn: **top ~150 nhãn phủ ~95%
detection**, dịch tay chừng đó mất ~1 giờ. Cột `nhan_en, nhan_vi, dong_nghia`
(nhiều từ Việt cho một nhãn: `Wok` → "chảo, chảo gang, chảo sâu lòng").

⚠️ Thứ bậc OpenImages **không tự gộp**: `Car`, `Land vehicle`, `Vehicle` là ba
nhãn riêng. Phải khai báo quan hệ cha-con thủ công cho các cụm quan trọng, nếu
không truy vấn "ô tô" sẽ bỏ sót.

---

### GIAI ĐOẠN 2 — Ba dạng truy vấn

*Giữ nguyên cấu trúc 2 mũi nhọn của v3. Chỉ đổi: objects lên kênh chính,
và Bước 4 Q&A nhấn mạnh ngữ cảnh (theo D0.3).*

#### Mũi nhọn 1 — Textual KIS & Q&A (TV1, TV3, TV5)

**Bước 1 — Thu hẹp cấp video.** BM25 metadata → top-50 video.

**Bước 2 — Truy hồi đa kênh, hợp nhất bằng Reciprocal Rank Fusion.** Bốn kênh
ở bảng trên. RRF an toàn hơn weighted-sum vì không cần chuẩn hóa thang điểm
giữa cosine (0,2–0,35) và BM25 (không chặn trên).

**Bước 2b — MỚI (4.1): khử trùng lặp rồi mới cắt top-K.** Xem PHẦN C mục 6 và
A8.8. Đặt *sau* RRF, *trước* khi cắt.

**Bước 3 — ~~Tinh chỉnh vị trí frame~~ — TỤT ƯU TIÊN.** *(4.1)* Bước này viết
là *"chỉ khi BTC xác nhận cửa sổ hẹp ở 0.a"*. Theo A8.1, luật AIC'25 chấm
`frame_idx` **rơi trong một khoảng**, nên điều kiện đó nhiều khả năng không xảy
ra. **Chưa làm bước này cho tới khi BTC trả lời khác đi.** Công sức chuyển sang
Bước 3b.

**Bước 3b — MỚI: đi bộ theo thời gian ("nearby frame").** *(A8.7 — kỹ thuật
đáng giá nhất trên mỗi đơn vị công sức)*

```python
# src/lan_can.py
def lan_can(video_id, frame_idx, so_buoc=10) -> list[Candidate]:
    """Các keyframe liền kề theo thời gian trong cùng video.
    master.parquet sắp theo (video_id, frame_idx) -> chỉ là trượt con trỏ."""
```

Vì sao đáng: câu Q&A trong ví dụ của đội AIC'25 cho gợi ý về **nguyên liệu**,
còn đáp án nằm ở **bước cắt** vài giây sau. Truy hồi ngữ nghĩa đưa tới lân cận;
đi bộ đưa tới đích. Cùng lý do với A1 — trung vị mật độ 55 frame nghĩa là lân
cận đủ dày để đi bộ có ý nghĩa.

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

*Giữ nguyên 5 bước của v3, thêm Bước 2b ở bản 4.1.*

**Bước 1 —** Bóc tách sự kiện thành N truy vấn con, giữ đúng thứ tự.
**Bước 2 —** Chốt video trước (BM25 metadata + CLIP tổng hợp) → 1–3 video.

**Bước 2b — MỚI: xếp hạng lại theo thời gian, làm TRƯỚC khi trích dày.**
*(A8.7 #3 — Algorithm 1 của bài AIC'25)*

Ba truy vấn độc lập (trước / hiện tại / sau), gộp điểm cao nhất theo `video_id`:

```python
# src/thoi_gian.py  —  O(K log K), khoảng 20 dòng
def xep_lai_theo_thoi_gian(R_truoc, R_hien_tai, R_sau):
    tot_truoc, tot_sau = {}, {}
    for vid, diem in R_truoc: tot_truoc[vid] = max(tot_truoc.get(vid, 0), diem)
    for vid, diem in R_sau:   tot_sau[vid]   = max(tot_sau.get(vid, 0), diem)
    ra = [(vid, fid, d + tot_truoc.get(vid, 0) + tot_sau.get(vid, 0))
          for vid, fid, d in R_hien_tai]
    return sorted(ra, key=lambda x: -x[2])
```

Video chứa **trọn chuỗi** sự kiện được cộng dồn điểm nên nổi lên đầu. Rẻ hơn
hẳn quy hoạch động ở Bước 5 và chạy được trên **toàn kho**, nên dùng nó để
**thu hẹp vùng trước**, rồi mới trích dày.

Dùng được cả cho Mũi nhọn 1 khi truy vấn KIS mô tả bối cảnh trước/sau.

**Bước 3 —** Cho điểm mềm bằng `object_score()`, không lọc cứng.
**Bước 4 —** Trích dày trong vùng ứng viên (`stride=1..2`, vùng ~30s).
**Bước 5 —** Dóng hàng thời gian bằng quy hoạch động, O(N·K), NumPy thuần.
DP top-M (beam) ra 100 chuỗi, bắt buộc khác nhau ở ≥ 2/N vị trí.

> **Nhắc lại PHẦN C mục 3:** TRAKE chấm **từng phần theo số sự kiện khớp**
> (A8.1). Chuỗi nào chưa chắc vẫn phải điền đủ N vị trí — điền sai bằng bỏ
> trống, mà điền đúng thì được điểm.

#### MỚI (4.1) — Giao diện: nếu thi tương tác thì đây là một phần của điểm

*Xem A8.6. v4 không có một dòng nào về việc này — đó là thiếu sót.*

Vòng Chung kết AIC'25 là **người ngồi lái**: gợi ý mở dần, thí sinh gõ truy vấn
trong phiên thi, Jury Board chấm. Nếu AIC'26 giữ thể thức đó thì tốc độ thao
tác của người dùng **là điểm số**, không phải tiện nghi.

**Chưa chốt được cho tới khi BTC trả lời** (gộp vào cùng câu 0.a). Nhưng nếu có
thì đây là mức tối thiểu, xếp theo thứ tự đáng làm:

| Ưu tiên | Thành phần | Ghi chú |
| --- | --- | --- |
| 1 | Ô tìm kiếm + lưới keyframe xếp hạng | có ảnh keyframe là dựng được |
| 2 | **Nút "khung lân cận"** | Bước 3b — rẻ nhất, lợi nhất |
| 3 | **Ô lọc OCR** | A8.5 — 3/5 ví dụ thực chiến |
| 4 | Ba ô Trước / Hiện tại / Sau | gọi thẳng Bước 2b |
| 5 | Rocchio feedback: `qm = αq₀ + β·TB(thích) − γ·TB(không thích)` | ~10 dòng NumPy, **chỉ có nghĩa nếu thi tương tác** |

> **Cảnh báo phạm vi:** đây là nơi dễ đốt thời gian nhất trong cả kế hoạch. Một
> ô nhập + lưới ảnh + hai nút là đủ. **Không dựng hệ thống tài khoản, không
> lịch sử phiên, không đăng nhập.** Nhắc lại A8.2: chưa ai chứng minh các module
> tương tác cộng điểm — điểm/câu của đội AIC'25 còn *giảm* ở vòng thêm chúng.

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
| **7** | **MỚI — tải đủ video/keyframe → #2, #3** | **hai phụ thuộc quan trọng nhất đều tắc** | ✅ **100% đã kiểm chứng** |

**Đường găng:**
`Bảng cái ✅ → tải dữ liệu ⬜ → TV2 (trích dày) → Khánh (DP + tập dev) → GĐ3`

> **Nút thắt đã dịch chuyển.** Ở v3, đường găng bắt đầu từ bảng cái. Bảng cái
> nay đã xong và được chứng minh. Nút thắt mới là **tải dữ liệu**: chỉ 387/873
> video có keyframe và video gốc, mà cả #2 lẫn #3 — hai phụ thuộc nặng nhất —
> đều cần chúng. Đây là việc cần dồn người vào ngay.

---

## PHẦN F — RỦI RO CÒN LẠI

| Rủi ro | Dấu hiệu sớm | Phương án |
| --- | --- | --- |
| **Chia dữ liệu nhiều máy làm đứt đường dẫn tuyệt đối** | `kf_path` trỏ file không tồn tại | Thống nhất đường dẫn dữ liệu giống nhau trên mọi máy, hoặc remap (xem A5.5). Riêng việc **gộp kết quả** thì an toàn: `row_id` như nhau trên mọi máy (đã đo 765/765 trên năm lô) |
| **Không ai chạy verify cho nhóm L mình giữ** | 873/873 video được kiểm chứng — 100% | Bắt buộc mỗi máy chạy `02`+`03` cho phần mình, báo kết quả |
| ~~BTC dùng cửa sổ `[s,e]` hẹp cho cả KIS~~ | — | 🟢 **hạ cấp** — luật AIC'25 chấm `frame_idx` rơi **trong một khoảng** (A8.1). Vẫn hỏi BTC để xác nhận, nhưng không còn chặn thiết kế |
| **MỚI — thi tương tác mà ta không có giao diện** | BTC công bố thể thức Chung kết | Hỏi BTC cùng câu 0.a. Nếu có: dựng UI tối thiểu ở GIAI ĐOẠN 2 (A8.6), tuyệt đối không phình |
| **MỚI — thiếu kênh mô tả cảnh** | truy vấn tả *quan hệ* trong cảnh ("vòng elip bằng gạch") đều trượt trên tập dev | Kênh 5 — Qwen2.5-VL làm cả OCR lẫn caption (A8.4). Đo trên 200 ảnh trước khi cam kết |
| **MỚI — chép module của đội AIC'25 mà không đo** | tốn tuần cho Rocchio/Temporal rồi điểm dev không nhúc nhích | Bài của họ **không có ablation nào**, và điểm/câu còn *giảm* ở vòng thêm hai module đó (A8.2). Mọi thứ lấy từ A8 phải qua tập dev mới được giữ |
| Tập dev lệch về L26 hoặc lệch chủ đề | Khánh soạn xong, kiểm phân bố | Lấy mẫu phân tầng theo nhóm L (A2), xem trước nội dung (A7) |
| **Quota VLM free không đủ cho bài thi** | 3/6 model chết vì rate-limit ở đợt test 20 lượt | Chốt sớm: trả phí hay chạy model local |
| OCR ticker sai nhiều | kết quả 0.b < 80% | Chỉ OCR vùng tiêu đề tĩnh, bỏ ticker chạy |
| Batch 2 định dạng khác batch 1 | BTC công bố | Chạy lại `00`+`01` là biết ngay |
| **CLIP ViT-B/32 quá yếu** *(nâng mức — A8.4)* | **không cần chờ dấu hiệu**: đội AIC'25 dùng ViT-L-14 1024 chiều **+ SigLIP2** 1152 chiều, và bước nhảy 0,86 → 0,93 điểm/câu của họ đến từ đúng đây | Đổi từ thế bị động sang **chủ động**: thử trên ~2.000 ảnh trước (GIAI ĐOẠN 1, việc *(b)*), chỉ chạy toàn kho nếu RRF hai model thắng rõ trên tập dev |
| ~~Ảnh keyframe không khớp thứ tự CSV~~ | — | ✅ **đã loại trừ** — 871/873 khớp, 0 lệch thật |

---

## PHẦN G — LỊCH SỬ SỬA ĐỔI

**v3 → v4.** PHẦN A của v3 đo trên 6 CSV / 8 `.npy` / 1 video; v4 đo lại trên
**toàn bộ 873 video, 177.321 keyframe**. Ba con số sai lệch đáng kể và một kết
luận bị đảo ngược: mật độ keyframe 109 → **55 frame**; fps "25 và 30" →
**25 / 26,44 / 29,97 / 30**; model "ViT-B/32" → **`ViT-B-32-quickgelu`**; và
objects từ "nhiễu nặng, kênh phụ" → **kênh chính thứ tư**.

### G2 — Sửa đổi v4 → v4.1 (nguồn: bài báo AIC'25, xem A8)

| Hạng mục | v4 | **v4.1** |
| --- | --- | --- |
| Loại bằng chứng | chỉ đo từ dữ liệu của ta | **thêm kết quả thật của một đội mùa trước** |
| Câu 0.a (cửa sổ `[s,e]`) | câu hỏi mở giá trị cao nhất, đang chặn | **có đáp án**: `frame_idx` chỉ cần rơi trong một khoảng |
| A5.6 / A5.7 | rủi ro tính điểm chưa rõ mức | **hết là rủi ro tính điểm** |
| TRAKE | "không bỏ trống khoảnh khắc nào" | **điểm từng phần theo số sự kiện → luôn điền đủ N** |
| Số kênh | 4 | **5** (thêm caption VLM) |
| Kênh OCR | một chế độ (BM25 hòa RRF) | **hai chế độ** — thêm lọc cứng cho token hiếm |
| Bước 3 Mũi nhọn 1 | tinh chỉnh vị trí frame | **tụt ưu tiên**; thay bằng "khung lân cận" |
| Model embedding | ViT-B/32 của BTC; "yếu" là rủi ro bị động | **cơ hội chủ động**: thử model thứ hai + RRF trên 2.000 ảnh |
| Khử trùng lặp | không có | **bước 2b**, dùng `clip.npy` sẵn có |
| Xếp hạng theo thời gian | chỉ có DP ở Bước 5 TRAKE | **thêm Bước 2b `O(K log K)`** chạy được toàn kho |
| Giao diện | **không nhắc tới** | **một mục riêng** — có thể là một phần của điểm |
| Milvus/Elasticsearch | loại, lập luận lý thuyết | **loại, có đối chứng**: 16,7 ms vét cạn |

> **Cách đọc G2:** mọi dòng đều đến từ *một* bài báo *một* đội *một* mùa. Bài
> không có ablation (A8.2). Nên đây là **giả thuyết có căn cứ**, không phải sự
> thật đã đo — phải qua tập dev của Khánh mới được giữ, y như mọi thứ khác ở
> GIAI ĐOẠN 3.

---

## PHẦN H — VIỆC LÀM NGAY

*Giai đoạn 0 đã đóng (873/873, 0 lệch chỉ số thật). Mọi việc dưới đây thuộc
Giai đoạn 1.*

### H1. Đường găng — chặn mọi thứ khác

| # | Việc | Ai | Vì sao chặn |
| --- | --- | --- | --- |
| **1** | **Tập dev ~60 câu, phân tầng 10 nhóm L** | **cả nhóm, Khánh gộp** | Bản 4.1 thêm 6 giả thuyết từ bài báo AIC'25 và **không cái nào được giữ nếu không đo được**. Việc 8 của kế hoạch GPU cũng chặn ở đây |

Hiện có **13 câu / 2 nhóm L**. Mỗi người soạn **6 câu cho mỗi nhóm L mình
giữ** — chỉ máy giữ gói `Keyframes_*` mới mở được ảnh gốc để kiểm lại. Quy
trình đầy đủ: [07_lam_tap_dev.md](07_lam_tap_dev.md).

```powershell
python scripts\10_contact_sheet.py --nhom <L của mình> --thua 10
python scripts\10_contact_sheet.py --tra <row_id...> --mo
# soạn vào dev/tap_dev_thanh_vien/tap_dev_<nhóm>.jsonl
python src\tap_dev.py --gop dev\tap_dev_thanh_vien\*.jsonl --file dev\tap_dev.jsonl
python src\tap_dev.py --file dev\tap_dev.jsonl --no-cum
python src\tap_dev.py --file dev\tap_dev.jsonl --kiem
```

### H2. Làm được ngay, không chờ ai

| # | Việc | Ai | Ghi chú |
| --- | --- | --- | --- |
| 2 | **`src/bm25.py`** — kênh 2 (metadata) + kênh 3 (OCR/ASR) | TV3 | metadata đã phủ 100%, 955 ký tự/video. Kênh 3 cần **hai chế độ**: lọc cứng cho token hiếm + BM25 hòa RRF (A8.5) |
| 3 | **`dev/label_vi_en.csv`** — dịch top ~150 nhãn | TV2 | ~1 giờ, mở khóa nốt kênh 4 (D1.6) |
| 4 | **Tối ưu `trich_day`: gộp cả cửa sổ vào MỘT lệnh ffmpeg** | TV2 | đã đo: **nhanh gấp 8,7 lần** (169 → 19 ms/khung). Nằm trên đường găng TRAKE |
| 5 | **Commit script vá `kf_path`** | Khánh | máy nào tải `index/` từ Drive cũng gặp (A5.5); đừng để mỗi người viết lại |
| 6 | **Gán nhãn 400 mẫu `ocr_v2` + điền `roi_v2.yaml`** | TV4 | đang **0/400**. ROI: ranh giới là **dải chữ chạy cuối cùng**, KHÔNG phải "bỏ nửa dưới" — băng rôn tiêu đề có liên quan tới hình |
| 7 | **`src/run.py`** — đường ống đầu-cuối | TV5 | chỉ đáng viết khi đã có ≥ 2 kênh |

### H3. Chờ tập dev xong

| # | Việc | Ai |
| --- | --- | --- |
| 8 | Đo A/B/C cho SigLIP2 — **nhớ `be_chung()`**, xem [06 §5](06_ke_hoach_encode_GPU.md) | Khánh |
| 9 | Đo `dedup.py` / RRF / `lan_can.py` — giữ hay bỏ theo số | TV1 |
| 10 | Mở rộng bench VLM lên ≥ 50 câu, test với ngữ cảnh thật | TV5 |

### H4. Việc người, không tự động hóa được

| # | Việc | Ai |
| --- | --- | --- |
| 11 | Gửi BTC **ba câu**: 0.a (khoảng hay frame chính xác), 0.d (TRAKE điểm từng phần), **0.e (Chung kết có thi tương tác không)** — 0.e quyết định có dựng giao diện | Khánh |
| 12 | Chốt phương án quota VLM: trả phí hay chạy local | TV5 |
| 13 | Chốt một bảng tên thành viên duy nhất | cả nhóm |
| 14 | Máy giữ L23+L26+L27 tải lại gói `Keyframes_L21` (thiếu 8 file ảnh) | — |

> **Kỷ luật cho toàn bộ bản 4.1:** mọi thứ lấy từ bài báo AIC'25 là **một bài
> báo, một đội, một mùa, không ablation** (A8.2). Dựng thì dựng, nhưng **chỉ
> giữ cái nào tăng điểm đo được trên tập dev**. Đó là lý do việc 1 quan trọng
> hơn việc 2–10 cộng lại.
