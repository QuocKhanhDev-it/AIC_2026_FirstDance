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

> 🛑 **BỊ BÁC BỞI TRẢ LỜI CỦA BTC (15/08) — xem A9.** Con số 14,6% và "trần
> R-Score ~0,15" tính trên giả định **cửa sổ rộng 10 frame**, lấy từ ví dụ
> `[500,510]` trong thể lệ. BTC xác nhận cửa sổ thật là **4 giây đến 5 phút**.
> Ở cửa sổ ≥ 10 giây, keyframe có sẵn phủ **100%**. Bảng trên giữ lại làm lịch
> sử; **đừng dùng nó để ước lượng trần điểm nữa.**

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
>
> **ĐÃ CÀI (H2 #4). Đo lại sau khi cài: 199 → ~48 ms/khung, nhanh ~4 lần.**
>
> ⚠️ **Bẫy khi tự đo lại — LƯỢT ĐẦU LUÔN CHẬM.** Ba lượt liên tiếp cùng một
> lệnh cho **90 → 51 → 45 ms/khung**. Lượt đầu tốn thêm ~3 giây chỉ để đọc
> vùng video từ đĩa; lượt sau đọc từ bộ đệm hệ điều hành. **Bỏ lượt đầu, lấy
> từ lượt hai trở đi** — không thì sẽ kết luận tối ưu chỉ nhanh 2 lần trong
> khi thực tế 4 lần.
>
> *Một giả thuyết đã bị bác:* nghi `-vcodec png` tốn CPU nén nên thử `bmp`.
> Đo thật: PNG **33,8** ms/khung, BMP **28,6** — chỉ hơn 15%, dù luồng BMP
> nặng gấp 4,6 lần (161 MB so với 35 MB). **Không đáng đổi.** Chi phí thật
> nằm ở seek và decode, không phải ở mã hóa ảnh trung gian.

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

### A9 — BTC TRẢ LỜI (15/08): cửa sổ rộng 4 giây–5 phút, không phải 10 frame

*Đây là bằng chứng mạnh nhất ta có — nguồn chính thức, không phải suy luận từ
bài báo hay đo gián tiếp. Nó **bác bỏ một con số nền** của kế hoạch.*

#### A9.1 — Nguyên văn ba câu trả lời

| Hỏi | BTC trả lời |
| --- | --- |
| Độ rộng cửa sổ `[s,e]` | *"Chắc có thể là **3 phút hoặc 5 phút** (an toàn cho BTC), **ngắn nhất 10s hoặc 4s** tuỳ trường hợp"* |
| Nguồn sinh đáp án | *"Keyframe chỉ là **giải pháp mẫu**, cái đáp án **có thể nằm giữa 2 frame**"* |
| Đề mẫu / GT | *"Sẽ có **query mẫu**, không có GT hoặc chỉ có **GT từ các năm trước**"* |

Ví dụ `[500,510]` trong thể lệ khiến cả nhóm hiểu cửa sổ là **10 frame**. Sai
hai bậc độ lớn: đơn vị thực tế là **giây tới phút**.

#### A9.2 — Đo lại tác động: từ 14,6% lên 86–100%

Cửa sổ rộng `W` giây trượt (không chứa keyframe nào) chỉ khi nó nằm **trọn
trong một khoảng trống** giữa hai keyframe. Đo trên 176.448 khoảng trống thật:

```text
trung vị 2,16s | trung bình 2,65s | p90 5,57s | p99 6,76s | LỚN NHẤT 8,0s
```

| Cửa sổ | ≈ frame @25fps | Keyframe có sẵn phủ được |
| --- | --- | --- |
| 0,4s *(10 frame — giả định cũ)* | 10 | **13,9%** |
| **4s** *(hẹp nhất BTC nêu)* | 100 | **86,1%** |
| **10s** | 250 | **100%** |
| 30s – 5 phút | 750 – 7.500 | **100%** |

**Khoảng trống lớn nhất trong TOÀN KHO chỉ 8,0 giây.** Nên mọi cửa sổ từ 10
giây trở lên **chắc chắn** chứa ít nhất một keyframe ta đã có.

#### A9.3 — Ba hệ quả, theo thứ tự quan trọng

**1. Trần điểm TRAKE ~0,15 KHÔNG còn đúng.** Nó tính trên cửa sổ 10 frame. Với
dung sai thật, keyframe có sẵn đã phủ 86–100%, nên trần bị giới hạn bởi **chất
lượng truy hồi**, không phải mật độ keyframe. PHẦN E phụ thuộc #2 phải sửa.

**2. `trich_day` ĐỔI LÝ DO TỒN TẠI, không mất đi.** Nó không còn cần để *bắn
trúng cửa sổ* — nhưng BTC nêu một lý do khác, cụ thể hơn và ta chưa tính tới:

> *"Các trường hợp thường gặp là về bài toán **đếm số lượng**, 1 cái frame có
> thể không đếm được hết... Ví dụ 1 em bé được bế bởi 4 người liên tiếp trong
> bản tin, dựa trên keyframe thì chỉ có 3 nên không đảm bảo."*

Tức trích dày cần cho **nhìn thấy nội dung**, không phải cho **trúng chỉ số**.
Phạm vi hẹp hơn (chỉ câu Q&A dạng đếm và TRAKE nhiều sự kiện) nhưng ở đó thì
không thay thế được. Ăn khớp với đo đạc riêng của ta ở D1.6: bộ nhận diện đếm
thiếu nghiêm trọng, **câu hỏi đếm phải để VLM nhìn nhiều frame trả lời**.

**3. Việc khó chuyển hẳn sang CHỌN ĐÚNG KHOẢNH KHẮC.** Khi trúng cửa sổ gần
như miễn phí, điểm số phụ thuộc gần như hoàn toàn vào truy hồi: đúng video, và
đúng vùng trong video. Đó là kênh 1–5 và RRF, **không phải** ffmpeg.

> ⚠️ **Vẫn còn một ẩn số:** BTC nói "3 phút hoặc 5 phút... ngắn nhất 10s hoặc
> 4s **tuỳ trường hợp**" — tức độ rộng **thay đổi theo câu**, và ta không biết
> phân bố. Ở mức 4s vẫn còn 13,9% trượt. Nên **đừng bỏ hẳn trích dày**; chỉ hạ
> nó khỏi vị trí đường găng.

#### A9.4 — Việc mở ra từ câu trả lời thứ ba

*"Sẽ có query mẫu... hoặc chỉ có GT từ các năm trước"* — **đề và đáp án các mùa
trước có thể xin được.** Đó đúng là thứ §1 của [07_lam_tap_dev.md](07_lam_tap_dev.md)
gọi là "mười phút đáng giá nhất": một tập dev đúng phân bố, đúng văn phong,
đúng độ khó, và **không dính thiên lệch nào của ta**. Hỏi lại BTC xin cụ thể.

---

### A10 — CLIP ViT-B/32 được **0,0000** trên tập dev tiếng Việt

*Đo 16/08 trên **100 câu tập dev, tiếng Việt nguyên văn** — đúng như đề thi ra.*

| Cấu hình | ±2s | ±15s | Kết luận |
| --- | --- | --- | --- |
| **CLIP ViT-B/32** | **0,0000** | 0,0060 | — |
| CLIP + dedup | 0,0000 | 0,0040 | 🟡 |
| **Objects + IDF** | **0,0400** | **0,0660** | ✅ ổn định |
| RRF(CLIP, objects) | 0,0280 | 0,0480 | ✅ ổn định |

**Không phải "yếu" — là không trả về được gì trên 100/100 câu.** Kênh duy nhất
đang chạy là **objects**, và nó chạy được *chính vì* đi qua bảng nhãn Việt–Anh.

#### A10.1 — Phép thử rẻ nhất, và nó dứt điểm

Không cần encode ảnh nào. Nếu model hiểu tiếng Việt thì vector câu tiếng Việt
phải **gần bản dịch tiếng Anh** của nó và **xa** một câu vô quan
(`"a laptop computer on a desk"`):

| Model | cos(việt, anh) | Biên độ so với câu vô quan |
| --- | --- | --- |
| `ViT-B-32-quickgelu` | 0,47 – 0,71 | **−0,0394** ❌ |
| `ViT-B-16-SigLIP2-256` | **0,82 – 0,90** | **+0,2201** ✅ |

Biên độ của CLIP **âm**: câu tiếng Việt còn gần một câu tiếng Anh vô quan hơn
là gần chính bản dịch của nó. Nó không "kém tiếng Việt" — nó **mù**.

#### A10.2 — Hệ quả: bước dịch KHÔNG phải tính năng phụ, nó LÀ kênh 1

Đo trên tập con có bản dịch tay: **0,0000 → 0,5238**. Toàn bộ giá trị của kênh
1 nằm ở bước dịch. Hai đường ra, và đường thứ hai bỏ hẳn được bước dịch:

| Cách | Đánh đổi |
| --- | --- |
| Dịch truy vấn bằng VLM/LLM | thêm phụ thuộc API + quota, và **chậm ngay trong phiên thi** |
| **Đổi sang SigLIP2 đa ngôn ngữ** | không cần dịch. Đúng model kế hoạch GPU đã chốt |

→ **Việc 8 của [kế hoạch GPU](06_ke_hoach_encode_GPU.md) không còn là "thử xem
có đáng không" mà là đường thoát cho kênh 1.**

> **Vì sao phát hiện muộn:** tập dev nhỏ (5 rồi 21 câu) chỉ cho tín hiệu mờ, và
> mọi phép đo trước đều chạy bằng **bản dịch tay sang tiếng Anh** cho tiện. Tập
> dev 100 câu tiếng Việt mới lộ ra. Đây là lý do cụ thể nhất cho nguyên tắc ở
> §3 của [07_lam_tap_dev.md](07_lam_tap_dev.md): **câu hỏi dev phải viết bằng
> đúng ngôn ngữ đề thi**, nếu không ta tự giấu điểm yếu của chính mình.

#### A10.3 — Đã ĐO end-to-end: SigLIP2 thay hẳn được bước dịch

Encode `ViT-B-16-SigLIP2-256` cho **11 video** (3.135 keyframe) rồi so ba cấu
hình trên **cùng một bể ứng viên** (`dense.be_chung`) và **cùng 21 câu**:

| # | Cấu hình | ±2s | ±15s |
| --- | --- | --- | --- |
| **A** | CLIP + tiếng Việt *(hiện tại)* | 0,0095 | 0,0857 |
| **B** | CLIP + **bản dịch tay** sang Anh | 0,8190 | 0,8952 |
| **C** | **SigLIP2 + tiếng Việt** | **0,8571** | **0,9429** |

| So | Hiệu | Thắng–thua–hòa | Kết luận |
| --- | --- | --- | --- |
| C so với A | **+0,8476** | **21–0–0** | ✅ **ổn định** |
| C so với B | +0,0381 / +0,0476 | 8–5–8 | 🟡 yếu |

**C thắng A ở cả 21/21 câu, không thua câu nào.** Và C **ngang hoặc nhỉnh hơn**
B — tức SigLIP2 đọc thẳng tiếng Việt **tốt bằng CLIP đọc bản dịch tay**, mà bỏ
được hẳn khâu dịch.

> ⚠️ **Đừng đọc con số tuyệt đối 0,86–0,94 là năng lực hệ thống.** Bể ứng viên
> ở đây chỉ 3.135 keyframe / 11 video, dễ hơn toàn kho 177.321 rất nhiều — đúng
> hiệu ứng thổi phồng đã đo ở `be_chung()`. **Chỉ phần SO SÁNH là dùng được**,
> vì cả ba cấu hình chạy trên cùng bể.

Kiểm lệch hàng trước khi tin: **16/16 cặp khớp**, trung vị cosine 0,9886.

> **Ghi chú về model.** Đây là `ViT-B-16-SigLIP2-256` — bản **nhỏ nhất** của họ
> SigLIP2, chọn vì chạy được trên CPU (1,9 ảnh/giây). Kế hoạch GPU chốt
> `ViT-SO400M-14-SigLIP2-378` (1152 chiều) mạnh hơn nhiều. Nếu bản nhỏ nhất đã
> lật ngược được kênh 1 thì bản lớn không thể tệ hơn.

---

### A11 — Dedup: **hoãn**, và bài học về việc đo trên một kênh đang hỏng

`scripts/13_do_dedup.py`, 97 câu tập dev + 15 truy vấn tiếng Anh đối chứng.

Phép đo đầu tiên trông như một phát hiện lớn:

| Truy vấn | Dedup bỏ | Vị trí lệch | Top-100 trải |
| --- | ---: | ---: | ---: |
| **Tiếng Việt** (97 câu tập dev) | **58,4/100** | 77,5 | 40 video |
| **Tiếng Anh** (15 câu đối chứng) | **0,5/100** | 2,6 | 63 video |

Chênh nhau **117 lần**. Con số 58,4 **không đo lợi ích của dedup** — nó đo cái
hỏng của kênh 1. CLIP mù tiếng Việt (A10) nên vector truy vấn gần như ngẫu
nhiên, top-100 đổ dồn vào một cảnh tĩnh duy nhất, và dedup dọn đống đó. Số dùng
được là **0,5/100**.

> Nếu chỉ chạy trên tập dev tiếng Việt rồi báo "dedup bỏ được 58% ứng viên
> thừa", ta đã đưa một no-op vào đường ống và tưởng mình vừa tối ưu. Nhóm đối
> chứng là thứ duy nhất chặn được — script tự cảnh báo khi hai nhóm lệch quá 5
> lần.

**Vì sao 0,5 chứ không phải 11,83% như A5.6.** Ba lý do chồng lên nhau:

1. **11,83% của toàn kho là con số đánh lừa.** 18.654 trong 20.975 bản sao
   (**89%**) nằm ở riêng L25. Chín nhóm còn lại đều dưới 2,2%; L21 và L22 chỉ
   0,45% và 0,27%.
2. **Bản sao chỉ tính trong cùng video**, mà top-100 của một truy vấn đọc được
   trải ra 63 video — mỗi video góp một hai frame, hiếm khi hai thành viên cùng
   cụm gặp nhau trong cùng top-100.
3. Ép bể **chỉ còn L25** (49,82% trùng lặp, tệ nhất kho) cũng chỉ bỏ **2,5/100**.

Và **hạng 1 không đổi ở bất kỳ phép đo nào** (0/97 và 0/15). R@1 chiếm 1/5 tổng
điểm; dedup không chạm tới nó.

**Chỗ duy nhất còn đất: khi bật `moi_video`.** Ràng buộc đó giữ tối đa k frame
mỗi video, và nếu k frame ấy là k bản sao thì cả ngân sách của video phí. Đo với
`moi_video=3`: bể L25 lệch **30,5/100** vị trí, toàn kho 2,3. Đúng như docstring
`dedup.py` dự đoán — `gioi_han_moi_video()` **không thay được** dedup, hai cái
bổ sung cho nhau.

**Chưa kết luận được về điểm**, và phải nói rõ vì sao chứ không im lặng: kênh
duy nhất đọc được tiếng Việt là SigLIP2, mà ma trận thử của nó mới encode 11
video L21+L22 — **đúng hai nhóm trùng lặp ít nhất kho**. Đo dedup ở đó là đo chỗ
nó không có gì để làm. Cả hai ma trận đều ra `⚪ KHÔNG ĐỔI GÌ`.

> **Quyết định.** Giữ module, **không bật mặc định**. Đo lại khi có
> `clip_siglip2.npy` toàn kho, trên câu L25, với `moi_video` bật:
> `python scripts/13_do_dedup.py --matrix clip_siglip2.npy --moi-video 3`

**Bẫy phụ tìm ra khi đo.** Sidecar `.json` của `08_encode.py` ghi
`da_encode: 18.635` trong khi ma trận chỉ có **3.135** dòng thật — chênh 6 lần.
Nguyên nhân: dòng chưa tải ảnh cũng được đánh dấu `xong` (để giữ nguyên vị trí
hàng) nhưng vector vẫn là 0. Ai đọc file đó để ước bể ứng viên sẽ ước sai, mà
sai kích thước bể thì điểm lệch tới **+0,2833** (xem `be_chung()`). Đã thêm
trường `co_vector` và in rõ "BỂ ỨNG VIÊN THẬT" khi encode xong.

---

### A12 — Kênh 2 (metadata) chạy: **kênh đầu tiên không được 0 trên tiếng Việt**

`src/bm25.py` + `scripts/15_do_bm25.py`, 97 câu tập dev.

Metadata là **tiếng Việt**, truy vấn là **tiếng Việt**, không có model nào phải
dịch ở giữa — nên kênh này không dính cái bẫy đã giết kênh 1 (A10).

| Thước đo | Kết quả |
| --- | ---: |
| Tìm ra video đúng **ở đâu đó** trong 873 video | **94/97 (97%)** |
| Video đúng trong **top-10** | 22/97 (22,7%) |
| Video đúng ở **hạng 1** | 10/97 (10,3%) |
| Trung vị hạng (khi tìm ra) | 142 / 873 |
| **Điểm BTC** (trên keyframe) | **0,0103** |

**Phải đọc hai con số này tách bạch, và biết con số nào nói gì.** 0,0103 nghe
như thất bại, nhưng nó là hệ quả số học của việc metadata mô tả *cả video*:
trung bình **203 khung/video**, nên dù đoán đúng video ở hạng 1 thì khung đúng
vẫn nằm đâu đó trong 203 khung theo thứ tự thời gian. Kênh này **không biết và
không thể biết** khung nào.

> Giá trị của nó nằm ở hợp nhất: **thu hẹp còn vài video, để kênh ảnh chọn
> khung.** Đó đúng là việc RRF sinh ra để làm. Chưa đo được vì chưa có kênh ảnh
> nào chạy được tiếng Việt — lại chặn ở `clip_siglip2.npy`.

**Nó mạnh ở đâu và yếu ở đâu — không đều chút nào:**

| Nhóm | Video đúng ở top-10 | |
| --- | ---: | --- |
| L24 | 60% | nội dung đa dạng, tiêu đề tả đúng cảnh |
| L27 | 56% | |
| L30 | 44% | |
| L26 | 42% | |
| L29 | 20% | |
| **L21, L22, L23, L25, L28** | **0%** | loạt video **cùng một sê-ri**, metadata gần như giống hệt nhau |

L25 là ví dụ rõ nhất: 88 video *"BÍ QUYẾT ÔN THI THPT 2024 — Môn X — Chuyên đề
Y"*. Metadata phân biệt được **môn học**, không phân biệt được **cảnh trong
video**. Đây là giới hạn bản chất, không phải lỗi cài đặt — và nó chính là lý
do kênh 5 (caption) tồn tại.

**Hai nút đã dò, cả hai đều đáng giữ:**

| Cấu hình | Video ở top-10 | Ở hạng 1 |
| --- | ---: | ---: |
| đầy đủ | **22**/97 | **10**/97 |
| bỏ bigram | 19/97 | 7/97 |
| bỏ title×3 | 20/97 | 9/97 |

*Bigram*: tiếng Việt viết rời từng âm tiết, `"xe máy"` là một từ nhưng hai
token — chỉ unigram thì nó khớp cả `"máy xay"`. Nối thành `"xe_máy"` cho một
token hiếm hơn nhiều, IDF tự lo phần còn lại. **Không cần bộ tách từ**
(`underthesea`/`pyvi`) — thêm phụ thuộc nặng để làm việc IDF đã làm.

*title×3*: BM25 không có khái niệm trường nào quan trọng hơn, mà tiêu đề 62 ký
tự chìm nghỉm cạnh description 954 ký tự. Lặp là cách tăng trọng số trường mà
không sửa công thức.

---

### A13 — Kênh 5 (caption VLM): **vướng hai chỗ, đều nằm ngoài code**

Đã cài xong phần làm được: `scripts/14_sinh_caption.py` (bộ sinh) và
`bm25.KenhVanBan.tu_bang_khung` (phần truy hồi — dùng chung bộ máy với kênh 2,
không viết lại). `tests/test_bm25.py::test_kenh_5_noi_dung_vao_thuoc_do` chốt
cả đường ống caption → BM25 → `Candidate` → `cham()`.

**Vướng 1 — chưa có khóa API.** `GOOGLE_API_KEY` chưa đặt, và việc 12 của PHẦN
H (*trả phí hay chạy local*) vẫn chưa chốt. Bậc miễn phí không làm nổi: ba model
đã chết vì rate-limit trong một đợt test **20 lượt** (D0.3).

**Vướng 2 — máy này chỉ có ảnh của L21, L22, L27.** Đáp án tập dev trải 10 nhóm;
trong 19.832 keyframe của các video chứa đáp án thì chỉ **4.086** có ảnh ở đây.
Bảy nhóm còn lại không thể sinh caption từ máy này — đúng mô hình chia dữ liệu
ở B4.

**Ước lượng** (`--uoc-tinh`, không gọi API lần nào):

| Tập | Số ảnh | Thời gian tường |
| --- | ---: | ---: |
| video chứa đáp án dev *(có ảnh ở máy này)* | 4.086 | 0,4 giờ @ 4 luồng |
| toàn bộ ảnh có trên máy này | 21.810 | 1,1 giờ @ 8 luồng |
| **toàn kho** | 177.321 | **8,6 giờ** @ 8 luồng |

> Đơn giá cố ý **không** chôn trong code — `--gia-1k` là tham số. Giá nhà cung
> cấp đổi, mà một con số bịa nằm trong tài liệu còn tệ hơn không có số.

**Quyết định thiết kế quan trọng nhất của kênh này: caption phải là TIẾNG VIỆT.**
BM25 khớp *mặt chữ*, không hiểu nghĩa — caption tiếng Anh + truy vấn tiếng Việt
khớp đúng **0 token**, không có không gian vector chung để bắc cầu như CLIP. Sinh
caption tiếng Anh là dựng lại nguyên xi lỗi đã cho kênh 1 điểm 0,0000, lần này
tốn thêm tiền API. Có một bài test giữ điều này.

#### A13.1 — Vì sao kênh 5 tìm bằng BM25 chứ không bằng vector

Câu hỏi tự nhiên: caption là văn bản, sao không embed rồi tìm bằng vector cho
hiểu nghĩa? Vì khi đó **caption chỉ còn là một nút cổ chai làm mất thông tin** —
ta vừa dựng một SigLIP2 tệ hơn, tốn thêm một vòng VLM ở giữa. Caption **đáng giá
chính vì nó lexical**: nó bắt được từ hiếm, tên riêng, con số và chữ đọc trên
hình — đúng những thứ tìm bằng vector dở nhất. Hai kênh bù nhau, không thay nhau.

**Cái giá của lexical là từ đồng nghĩa, và kho này có sẵn:**

| Cùng nghĩa "quả dứa" | Số video nhắc tới |
| --- | ---: |
| dứa | 220 |
| thơm | 76 |
| khóm | 3 |

Tiêu đề L27 đúng là *"Trăm Năm làng **Khóm**"*. Truy vấn nói "dứa", caption viết
"khóm" → BM25 cho **0**.

*(Vài nhóm đồng nghĩa khác đo bị nhiễu vì bỏ dấu — "ly" lẫn "lý", "man" lẫn
"màn". Không dùng những con số đó.)*

**Chỗ này caption khác hẳn metadata: metadata cho sẵn nên phải chịu, còn caption
thì TA VIẾT RA.** Nên sửa ngay ở câu nhắc, không tốn thêm lượt gọi nào — bảo VLM
kèm cách gọi vùng miền trong ngoặc: *"quả dứa (thơm, khóm)"*. Đây là *document
expansion*, kỹ thuật cổ điển của IR.

> Không miễn phí: thêm chữ là kéo dài tài liệu, mà **BM25 phạt tài liệu dài** —
> đúng lý do hàm `don()` tồn tại. Bật mặc định (kỹ thuật đã được chứng minh rộng
> rãi) nhưng **chưa đo trên kho này**, nên có nút `--khong-dong-nghia` để A/B
> ngay ở lần chạy thật đầu tiên trên 4.086 ảnh thử.

---

### A14 — RRF thô **làm TỆ ĐI**, và lý do đo được ngay trong cùng phép chạy

`scripts/16_do_rrf.py`, 97 câu tập dev. Lần đầu RRF đo được — nó cần ≥ 2 kênh
chạy được tiếng Việt, mà kênh 1 được 0,0000 (A10). Nay có kênh 2 và kênh 4.

| Cấu hình | ±2s | ±15s | so với kênh 4 |
| --- | ---: | ---: | --- |
| **kênh 4 — objects** | **0,0412** | **0,0701** | *(mốc nền)* |
| kênh 2 — metadata | 0,0000 | 0,0103 | −0,0412 · 0–8–89 |
| **RRF(2, 4)** | 0,0268 | 0,0577 | **−0,0144 · 0–7–90 · ✅ ỔN ĐỊNH** |

**Hợp nhất hai kênh cho kết quả TỆ HƠN kênh mạnh hơn khi đứng một mình.** Thua ở
cả hai mức dung sai, vượt nhiễu, 0 thắng / 7 thua. Không phải nhiễu.

**Vì sao — đo luôn trong cùng phép chạy, không phải suy đoán:**

| Điều kiện để RRF cộng hưởng | Kết quả |
| --- | ---: |
| Hai kênh chung ≥ 1 **KHUNG** | **5/97 câu** |
| Hai kênh chung ≥ 1 **VIDEO** | **79/97 câu** (trung bình 4,3 video) |

RRF cộng `1/(k + hạng)` từ mỗi kênh, nên một ứng viên chỉ **được lợi khi nhiều
kênh cùng đề cử ĐÚNG NÓ** — cùng `row_id`. Ở đây điều đó gần như không xảy ra:
kênh 2 là kênh **cấp video** (trả mọi khung của video khớp), kênh 4 là kênh
**cấp khung**. Hai kênh nói hai độ mịn khác nhau, trùng đúng `row_id` là chuyện
hiếm.

Không cộng hưởng thì RRF chỉ **đan xen** hai danh sách — và đan xen thì mỗi ứng
viên tốt của kênh mạnh bị đẩy lùi một bậc bởi một ứng viên của kênh yếu. Đó
chính xác là −0,0144 đo được.

> **Bài học chung, không riêng cặp kênh này:** *"gom đủ năm kênh rồi RRF"* là
> một giả định, không phải một sự thật. RRF chỉ trả công khi các kênh **đồng ý ở
> cùng độ mịn**. Trước khi thêm kênh nào vào hợp nhất, đo `chồng lấn` trước —
> `16_do_rrf.py` in sẵn.

#### A14.1 — Hợp nhất hai tầng: **đã cài, đã đo, cũng không cứu được**

79/97 câu hai kênh *có* đồng ý ở cấp video, nên hướng sửa hiển nhiên là RRF hai
tầng: chọn video ở tầng 1, xếp khung ở tầng 2 (`rrf.hop_nhat_hai_tang`).

| Cấu hình | ±2s | ±15s | so với kênh 4 |
| --- | ---: | ---: | --- |
| **kênh 4 objects** | **0,0412** | **0,0701** | *(mốc nền)* |
| RRF thô (2, 4) | 0,0268 | 0,0577 | −0,0144 ✅ ổn định |
| 2 tầng, mỗi video 1 | 0,0103 | 0,0412 | −0,0309 🟡 |
| 2 tầng, mỗi video 3 | 0,0206 | 0,0515 | −0,0186 🟡 |
| 2 tầng, mỗi video 10 | 0,0433 | 0,0680 | +0,0021 / −0,0021 ❌ **đảo dấu** |
| *chỉ kênh 4, qua 2 tầng* | 0,0206 | 0,0639 | −0,0206 ✅ ổn định |

**Không cấu hình nào thắng được kênh 4 đứng một mình.** Cái tốt nhất
(`mỗi video 10`) đảo dấu giữa hai mức dung sai — theo kỷ luật đã đặt ở
`bao_cao_do_nhay`, đó là **không kết luận được**, không phải "hơi hơn".

Dòng đối chứng *"chỉ kênh 4, qua 2 tầng"* là dòng quan trọng nhất trong bảng:
chạy **một kênh duy nhất** qua bộ máy hai tầng đã mất **−0,0206**. Tức bản thân
việc ép rải đều theo video đã tốn điểm, trước khi nói tới chuyện hợp nhất. Không
có dòng đối chứng này thì mọi thay đổi đều dễ bị quy nhầm cho "hợp nhất".

**Kết luận đúng phạm vi — và phải nói rõ phạm vi:** hợp nhất không cứu được
**cặp kênh NÀY**, vì kênh 2 được 0,0000 ở ±2s. Một kênh không có thông tin cấp
khung thì không thể đóng góp gì cho việc xếp khung, ở bất kỳ kiến trúc nào.
**Điều này KHÔNG chứng minh hợp nhất vô dụng** khi SigLIP2 — một kênh cấp khung
mạnh — xuất hiện.

> **Quy trình bắt buộc từ nay, thay cho "gom đủ năm kênh rồi RRF":**
> thêm **từng kênh một**, so với **kênh mạnh nhất hiện có**, và **chỉ giữ cái
> nào thắng**. Cộng thêm một kênh yếu vào một kênh mạnh là **pha loãng**, đo
> được, ổn định, và không có kiến trúc hợp nhất nào sửa được.

#### A14.2 — Nguyên nhân thật: **RRF coi mọi kênh đáng tin NHƯ NHAU**

Câu hỏi tự nhiên khi thấy A14/A14.1: *"hợp nhất tệ đi là do chưa đủ kênh à?"*
**Không.** Đo bằng cách hạ trọng số kênh yếu (`hop_nhat` đã có sẵn `trong_so`):

| Cấu hình | ±2s | ±15s | so với kênh 4 |
| --- | ---: | ---: | --- |
| **kênh 4 objects** | **0,0412** | **0,0701** | *(mốc nền)* |
| RRF trọng số **1 : 1** | 0,0268 | 0,0577 | −0,0144 ✅ **hại** |
| RRF trọng số **0,5 : 1** | 0,0412 | 0,0722 | +0,0021 · 1–0–96 · 🟡 |
| RRF trọng số **0,2 : 1** | 0,0412 | 0,0701 | ⚪ **KHÔNG ĐỔI GÌ** |
| RRF trọng số **0,05 : 1** | 0,0412 | 0,0701 | ⚪ **KHÔNG ĐỔI GÌ** |

**Cơ chế:** RRF cộng `1/(k + hạng)` từ *mỗi* kênh, không nhìn kênh đó tốt hay
tệ. Nên **ứng viên hạng 1 của một kênh chết được cộng đúng bằng ứng viên hạng 1
của kênh tốt** — `1/61` cho cả hai. Với `k = 60`, chênh lệch giữa hạng 1 và hạng
100 chỉ là `1/61` so với `1/160`, tức **chưa tới 3 lần**; còn chênh lệch chất
lượng giữa kênh 4 và kênh 2 thì vô hạn (0,0412 so với 0,0000). RRF không có chỗ
nào biểu đạt được điều đó.

**Nhưng phải đọc đúng kết quả:** hạ trọng số **chặn được thiệt hại, KHÔNG tạo ra
lợi ích**. Ở 0,2 và 0,05 điểm hội tụ về đúng kênh 4 đứng một mình — tức kênh yếu
tốt nhất chỉ có thể *tránh đường*. Nghĩa là:

> **Vấn đề không phải thiếu kênh, mà là kênh không có gì để đóng góp.** Thêm ba
> kênh nữa cùng loại sẽ làm tệ hơn chứ không tốt hơn. Hợp nhất chỉ lãi khi các
> kênh **cùng tầm chất lượng** và **đồng ý ở cùng độ mịn** (A14).

Hệ quả thực dụng: trước khi hợp nhất, **đo từng kênh riêng**; kênh nào cách kênh
mạnh nhất quá xa thì để ngoài, hoặc hạ trọng số cho nó khỏi phá.

**Và một con số cần nhớ:** kênh 4 (objects) đang là **kênh mạnh nhất** hiện có
— 0,0412 / 0,0701, gấp bốn lần kênh 2. Trước A10 ta vẫn nghĩ kênh 1 là xương
sống.

> **Đã lỗi thời — A17 (19/08).** SigLIP2 toàn kho đưa kênh 1 lên **0,3258**,
> gấp gần 8 lần kênh 4. Mốc nền cho mọi phép so từ nay là **SigLIP2**, không
> phải objects. Phần *cơ chế* của A14 vẫn đúng nguyên (RRF chỉ trả công khi các
> kênh đồng ý ở cùng độ mịn) — và A17 là lần thứ ba nó lặp lại.

---

### A15 — Dọn rác metadata (NgThanhDat-ne): sửa lỗi thật, **nhưng không đổi điểm**

`scripts/17_compare_tu_metadata_bugs.py`, dò 2×2 trên 873 video / 97 câu dev.

**Lỗi tìm ra là lỗi thật, và là lỗi của tôi.** `tu_metadata` ghép các trường
bằng `" "`, nên `tach()` sinh **bigram bắc cầu qua biên trường**: title lặp 3
lần khiến `"…VIVU TV"` nối với đầu lần lặp sau đẻ ra `tv_món` — một cụm không hề
có trong văn bản gốc, mà lại hiếm nên IDF cao. Sửa bằng cách ghép `". "` là
đúng.

**Tách hai nút ra mới biết nút nào làm việc.** Bản vá đổi *hai* thứ cùng lúc
(dọn rác + đổi cách ghép), nên phải dò riêng:

| Cấu hình | Từ vựng | top-1 | top-10 | top-20 | Trung vị hạng |
| --- | ---: | ---: | ---: | ---: | ---: |
| V0 ghép `' '` · giữ rác | 15.722 | 10 | 22 | 29 | 142 |
| Va ghép `'. '` · giữ rác | 15.168 | 10 | 22 | 29 | 128 |
| **Vb ghép `' '` · dọn rác** | 14.930 | **11** | 21 | 26 | **118** |
| V2 ghép `'. '` · dọn rác | 14.316 | 11 | 21 | 26 | 119 |

**Việc dọn rác là thứ làm việc; cách ghép gần như không.** V2 ≡ Vb ở *mọi*
thước đo hạng. Và điểm nhấn của bản vá — thay rác bằng `'. '` thay vì `' '` để
"triệt tiêu 100% bigram lai" — trên toàn kho tạo ra **đúng 1 bigram khác biệt**
(`của_trên`) trong 14.317 token.

**Điểm BTC: cả bốn cấu hình y hệt nhau.**

    V0 = Va = Vb = V2 = 0,0000 / 0,0103     0–0–97     ⚪ KHÔNG ĐỔI GÌ

Trung vị hạng video tốt lên 24 bậc (142 → 118), từ vựng sạch đi 9% — nhưng
**không câu nào đổi điểm**. Không mâu thuẫn: điểm của kênh 2 do một nhúm câu có
video lọt đủ cao quyết định, mà nhúm đó không xê dịch.

> **Vẫn giữ bản vá.** Bỏ URL/hashtag ra khỏi chỉ mục văn bản là đúng không cần
> bàn, có test, và trung vị hạng cải thiện thật. Chỉ là **đừng ghi nó vào cột
> "tăng điểm"** — nó chưa tăng điểm nào.

**Hai chỗ bản vá bỏ sót, đã vá tiếp:**

1. **`title` không được dọn** — 13/873 tiêu đề có hashtag (`#htvsports #lansurong`),
   mà title lặp 3 lần nên rác ở đây ăn trọng số **gấp ba**. Chỗ nặng nhất lại là
   chỗ bị bỏ qua.
2. **`#\S+` quá tham** — `\S+` chạy tới khoảng trắng gần nhất nên
   `"#amthuc,rau củ"` ăn luôn `rau`. Đo trên kho hiện tại: **0 token mất** (ở đây
   hashtag luôn có khoảng trắng theo sau) — nên đây là siết phòng xa, không phải
   lỗi đang cắn. Nhưng kênh 3 sắp đẩy **chữ OCR** qua đúng hàm này, mà chữ OCR
   bẩn hơn nhiều. Đổi thành `#\w+`.

**Và một lỗi của tôi khi soát:** minh hoạ đầu tiên tôi viết kiểm token `tv_vivu`,
trong khi token bắc cầu đúng là `tv_món`. Kết quả ra `False` ở cả hai cột, làm
một lỗi **có thật** trông như không tồn tại. Đã sửa, và giữ ghi chú tại chỗ.

---

### A16 — Quét toàn hệ thống: bốn lỗi, hai trong số đó chờ sẵn để cắn

Chạy lại toàn bộ: 103 test, 18 script, 11 module, `pyflakes` trên cả ba thư mục.

**1. `cham_diem.py` chấm SAI câu Q&A — lỗi nặng nhất, và nằm đúng chỗ nguy hiểm
nhất.** Bài nộp Q&A là danh sách xếp hạng các bộ `(video_id, frame_idx, answer)`
— **mỗi dòng mang `answer` riêng**, và một dòng ăn điểm khi đúng *cả* khung *lẫn*
chuỗi answer. Bản cũ chấm thứ hạng theo khung rồi mới xóa điểm nếu
**`kq[0]`** sai đáp án. Sai ở **cả hai chiều**:

| Tình huống | Chấm cũ | Đúng ra |
| --- | ---: | ---: |
| Hạng 1 sai đáp án, hạng 4 đúng cả hai | 0 | **0,8** |
| Hạng 1 đúng đáp án nhưng SAI khung, hạng 4 đúng khung nhưng sai đáp án | 0,8 | **0** |

Lệch kiểu này **không đều giữa các cấu hình**, tức nó đảo được thứ hạng — đúng
loại hỏng mà `no_cua_so()` sinh ra để chặn. Hiện chưa kênh nào sinh `answer` nên
**không con số nào đã công bố bị ảnh hưởng**; nó chờ sẵn để cắn đúng ngày kênh 5
chạy. Đã sửa: `_hang()` nhận thêm điều kiện phụ theo từng ứng viên, kèm test cho
từng chiều.

**2. `14_sinh_caption.py` không biên được file sau khi bị ngắt.** `da_xong()` bỏ
qua dòng cụt nên lần chạy tiếp nối được, nhưng `bien()` dùng
`pd.read_json(lines=True)` nên **chết ở cuối** — đúng lúc đã tiêu xong tiền API
thì không lấy ra được `caption.parquet`. Gộp về một hàm đọc khoan dung dùng chung.

**3. `kis-L24-001` bị đặt nhầm nhóm.** `nguon` ghi *"sheet L24_V075"* nhưng
`row_id 175588` nằm ở **L30_V075**. Tra ra: **`L24_V075` không tồn tại** (0
khung), còn `L30_V075` tên là *"Chàng trai tinh thể"* — khớp đúng câu hỏi về mẫu
vật tinh thể. Vậy đáp án đúng, chỉ **tên và chỗ đặt sai**. Đổi thành
`kis-L30-007` và chuyển sang file L30.

> Đây là loại lỗi **không ai kiểm được**: người giữ L24 không có ảnh L30 để mở,
> người giữ L30 không biết câu đó tồn tại. Và nó đã lọt vào **tập test giữ kín**.
> Đã thêm chốt vào `kiem()`: mã nhóm trong `id` phải khớp nhóm của đáp án.

**4. `gop()` để lọt lỗi trùng `id`.** Chốt chống rò chạy **trước** bước soát
trùng, nên một `id` bị nhân đôi mà tình cờ nằm trong tập test sẽ bị `continue` cả
hai lần và **không bao giờ được báo**. Tôi vấp đúng ca này khi tự vá mục 3 (đổi
`c.id` trước khi lọc danh sách cũ, nên bản cũ ở lại dưới tên mới). Dấu hiệu duy
nhất là dòng *"Bỏ 21 câu"* trong khi tập test chỉ có 20 — dễ đọc lướt qua. Đã
đảo thứ tự: soát trùng trước, lọc tập test sau.

**Ghi nhận thêm:** `kf_name` **phụ thuộc máy y như `kf_path`** — trống ở
155.511/177.321 dòng trên máy này (chỉ L21, L22, L27 có ảnh). Đừng dùng
`kf_name.notna()` làm bộ lọc "có keyframe"; nó chỉ có nghĩa "ảnh đã tải **ở
máy này**".

`pyflakes` nay sạch trên `src/`, `scripts/`, `tests/`. Mọi con số đã công bố
(A10–A15) đo lại vẫn khớp.

---

### A17 — SigLIP2 toàn kho: **0,0000 → 0,3258**. Kênh 1 sống lại

Ma trận `clip_siglip2.npy` do máy GPU của Khánh dựng xong (19/08), đặt ở
`aic_data/`. `scripts/18_do_siglip2.py`, 97 câu tập dev.

**Kiểm ma trận trước khi tin — ba bước, không bước nào bỏ được:**

| Kiểm | Kết quả |
| --- | --- |
| Hình dạng | `(177321, 1152)` float16, khớp số dòng bảng cái |
| Độ phủ thật (`co_vector`) | **177.321/177.321 — 100%**, đủ 873 video, 10 nhóm L |
| Chuẩn L2 | trung vị 1,0000 (min 0,9998 / max 1,0002) |

**Lệch hàng — kiểm được mà KHÔNG cần nạp model.** Nếu hai ma trận cùng thứ tự
hàng thì cặp keyframe nào giống nhau ở CLIP cũng giống nhau ở SigLIP2. Đo tương
quan trên 4.000 cặp trong cùng video, kèm nhóm đối chứng là chính SigLIP2 dịch
hàng:

| | Pearson | Spearman |
| --- | ---: | ---: |
| **SigLIP2 nguyên bản** | **0,7065** | **0,6621** |
| dịch 1 hàng *(đối chứng)* | 0,4383 | 0,3916 |
| dịch 7 hàng *(đối chứng)* | 0,2749 | 0,2361 |

Bản gốc cao nhất và tương quan tụt đều theo độ dịch → **hàng thẳng**. Cách này
rẻ hơn `--kiem-lech-hang` nhiều và dùng được cả khi chưa nạp nổi model.

#### Kết quả

| Cấu hình | ±2s | ±15s | so với CLIP |
| --- | ---: | ---: | --- |
| CLIP ViT-B/32 *(mốc nền)* | 0,0000 | 0,0062 | — |
| **SigLIP2 SO400M** | **0,3258** | **0,4412** | **+0,3258 · 46–0–51 · ✅ ỔN ĐỊNH** |
| RRF(CLIP, SigLIP2) | 0,2598 | 0,3546 | +0,2598 · 42–0–55 · ✅ ỔN ĐỊNH |

**SigLIP2 thắng 46 câu, thua 0.** Ở ±15s là 59–1–37. Bể đầy 177.321/177.321 nên
**con số tuyệt đối đọc thẳng được** — không phải đọc dè dặt như A10.3.

Đây là bước nhảy lớn nhất dự án từng đo: kênh 1 từ **vô dụng** thành **kênh mạnh
nhất**, gấp gần 8 lần kênh 4 (objects, 0,0412 — A14).

> **RRF lại làm tệ đi, lần thứ ba.** 0,2598 so với 0,3258 khi SigLIP2 đứng một
> mình — thấp hơn 0,0660. Đúng khuôn mẫu A14/A14.1: cộng một kênh yếu (CLIP đang
> 0,0000) vào kênh mạnh là **pha loãng**.
>
> ⚠️ Nhưng **chưa chạy phép so theo cặp cho riêng cặp này** — mốc nền của lần
> chạy là CLIP, nên chỉ có hiệu số trung bình, chưa có thắng–thua–hòa và ngưỡng
> nhiễu giữa SigLIP2 và RRF. Coi là **dấu hiệu mạnh, chưa phải kết luận**.

#### Hai lỗi lộ ra khi chạy, cả hai chỉ hiện với model lớn

**1. `dense.tim()` nổ RAM với ma trận float16 nhiều chiều.** Bản cũ viết
`np.asarray(self.mat) @ q`: numpy phải **nâng kiểu** để nhân, tức cấp phát một
bản float32 của TOÀN BỘ ma trận — **817 MB cho MỖI truy vấn** với
`(177321, 1152)` float16. Với `clip.npy` (512 chiều, float32) không lộ ra vì
không phải nâng kiểu, nên lỗi này **chờ đúng lúc đổi sang model mạnh hơn mới
cắn**. Đã vá: `_nhan()` nhân theo lô 20.000 dòng, bộ đệm tạm ≤ ~92 MB.

**2. Sidecar ghi đường dẫn máy dựng index, máy khác nạp là chết.**
`clip_siglip2.json` ghi
`pretrained: "D:\Project\AIC_2026\models\...safetensors"`. Cả lý do tồn tại của
sidecar (docstring `dense.py`) là để ma trận **tự mô tả được trên mọi máy**. Đã
vá `08_encode.py`: phát hiện đường dẫn thì đoán tag từ tên file, ghi tag vào
`pretrained` và giữ đường dẫn gốc ở `pretrained_goc`.

#### Chưa đo được — và vì sao

Máy đang dùng chỉ **7,7 GB RAM** và đã crash nhiều lần khi nạp SO400M. Những
phép đo sau **phải để dành cho máy ≥ 16 GB**:

- RRF(SigLIP2, objects) và RRF(SigLIP2, metadata) — câu hỏi *"có gì cộng vào
  SigLIP2 mà lãi không"* vẫn đang mở
- Phép so theo cặp giữa SigLIP2 và RRF (xem cảnh báo ở trên)
- **`dedup` trên SigLIP2 toàn kho** — đúng phép đo A11 hẹn lại, giờ mới làm được
- Các mức `--moi-video`

---

### A18 — `lan_can` và ràng buộc đa dạng: **cả hai đều LÀM TỆ ĐI**

`scripts/21_do_lan_can.py`, 100 câu tập dev, mốc nền **RRF(objects, OCR)** =
0,0640 — cấu hình mạnh nhất chạy được không cần model.

Viết script này sau khi grep toàn repo và thấy: `lan_can.py` **không module nào
import**, `gioi_han_moi_video` **không ai gọi**. Cả hai viết xong rồi để đó.

| Cấu hình | ±2s | ±15s | so với mốc |
| --- | ---: | ---: | --- |
| **RRF(objects, OCR)** | **0,0640** | **0,0940** | *(mốc nền)* |
| + lân cận ±2 | 0,0560 | 0,0660 | −0,0280 ✅ ổn định |
| + lân cận ±5 | 0,0420 | 0,0500 | −0,0440 ✅ ổn định |
| + lân cận ±10 | 0,0260 | 0,0320 | −0,0620 ✅ ổn định |
| + mỗi video ≤ 2 | 0,0420 | 0,0760 | −0,0220 ✅ ổn định |
| + mỗi video ≤ 3 | 0,0460 | 0,0880 | −0,0180 🟡 |
| + mỗi video ≤ 5 | 0,0600 | 0,0900 | −0,0040 🟡 |

#### `lan_can` — A9 đã vô hiệu hoá nó từ trước

A8.7 xếp **"nearby frame" hạng 1 về lợi/công**, lấy từ bài báo AIC'25. Đo ra
**âm, và càng nới càng tệ** (−0,0280 → −0,0620 khi đi từ ±2 lên ±10 bước). Đơn
điệu như vậy là dấu hiệu cơ chế, không phải nhiễu.

**Cơ chế: nó trùng việc với `no_cua_so()`.** BTC chấm theo cửa sổ 4 giây–5 phút
(A9), nên keyframe lân cận **đã được tính đúng sẵn** nếu khung trung tâm đúng.
Chèn thêm chúng vào danh sách không tạo ra cú trúng mới nào, chỉ **đẩy ứng viên
khác ra khỏi top-100**.

> Đây là chỗ hai phát hiện cũ gặp nhau: **A9 (cửa sổ rộng) làm A8.7 #1 mất tác
> dụng.** Kỹ thuật đó có giá trị ở luật chấm hẹp — nơi trúng lân cận vẫn tính là
> trượt. Với luật thật thì không.

#### Ràng buộc đa dạng — **PHẦN C mục 2 nói CỨNG, đo ra là SAI**

PHẦN C mục 2 ghi: *"Ràng buộc cứng: mỗi video ≤ 2 slot trong top-5; top-20 trải
trên ≥ 8 video khác nhau."* Đo: **mv = 2 mất −0,0220 (ổn định)**, và điểm chỉ
hồi về mốc khi nới tới mv = 5, tức khi ràng buộc gần như không còn tác dụng.

Lập luận ban đầu — *"đoán sai video thì phí cả 100 chỗ"* — nghe hợp lý nhưng
không đúng thực tế: top-100 của các kênh **vốn đã trải ra hàng chục video**
(A11 đo được 40–63 video/truy vấn). Ràng buộc không thêm đa dạng, nó chỉ **cắt
mất ứng viên tốt** của video đúng.

> **Sửa PHẦN C mục 2: bỏ chữ "cứng".** Ràng buộc đa dạng là **giả thuyết đã bị
> bác**, không phải luật. Giữ hàm lại để đo với SigLIP2 — bể ứng viên khác thì
> kết luận có thể khác — nhưng **không bật mặc định**.

**Đây là lần thứ tư một kỹ thuật lấy từ bài báo AIC'25 hoặc từ trực giác thiết
kế bị chính phép đo bác** (sau dedup A11, RRF thô A14, hợp nhất hai tầng A14.1).
Danh sách "đã thử mà không hiệu quả" nay dài hơn danh sách "đang dùng".

---

### A20 — Leaderboard 0,8 → 2,6, và tập dev **không giải thích được** phần lớn mức tăng

> 🔴 **SỬA 20/08 23:15 — mục này viết khi tưởng chỉ có HAI lượt nộp. Thực tế
> có BỐN, và lượt tốt nhất không phải lượt nào ở đây.** Xem bảng đầy đủ ở
> [A24](#a24--bốn-lượt-nộp-chứ-không-phải-hai-và-lượt-tốt-nhất-38-là-kênh-ảnh-siglip2).
> Phần phân tích ×3,25 dưới đây vẫn đúng cho cặp #1 → #2, nhưng **đừng đọc 2,6
> là trần của hệ thống**.

Phép đo NGOÀI đầu tiên có hai điểm để so. Đợt 1 nộp bằng kênh objects đơn lẻ;
đợt 2 thêm ba thay đổi: cắt truy vấn theo câu, thêm kênh OCR hợp nhất RRF, bỏ
kênh metadata khỏi hợp nhất.

| | Điểm | |
| --- | ---: | --- |
| Đợt 1 — objects đơn lẻ | **0,8** | hạng 77 |
| Đợt 2 — RRF(objects, OCR) + cắt truy vấn | **2,6** | **×3,25** |
| Tập dev dự đoán | — | **×1,55** |

> ⚠️ **Cả hai con số là điểm trên NỬA bộ đáp án** — bản quy định BTC (C7) ghi
> Public Leaderboard chỉ chấm 50%, xếp hạng cuối chấm 100% ở Private. Tỷ lệ
> ×3,25 vẫn đứng vì cả hai đợt cùng chấm trên 50% đó, nhưng **đừng suy ra thứ
> hạng cuối cùng từ nó**.

**Thực tế gấp hơn hai lần dự đoán của tập dev.** Truy nguyên phần chênh:

| Thay đổi | Tập dev đo được |
| --- | --- |
| Thêm kênh OCR + RRF | +0,0228 (0,0412 → 0,0640) = ×1,55 |
| Bỏ kênh metadata | đã đo ở A14.2 |
| **Cắt truy vấn theo câu** | **⚪ KHÔNG ĐỔI GÌ — 0–0–100** |

Cắt truy vấn cho **0-0-100 trên tập dev**, và không phải vì nó vô dụng mà vì
**tập dev không kích hoạt được nó**:

| | Truy vấn bị tách | Mệnh đề TB |
| --- | ---: | ---: |
| Tập dev | **7/97 (7%)** | 1,1 |
| **Đề thi thật** | **18/21 (86%)** | 2,4 |

> **Đây là lần thứ hai tập dev tỏ ra mù với một lỗi thật.** Lần đầu: 57% truy
> vấn đề thật bị cắt cụt token trong khi dev chỉ 1/40 (A19). Lần này: 86% đề
> thật bị tách trong khi dev chỉ 7%.
>
> Cả hai đều cùng một nguyên nhân — **câu dev tự soạn dài 22 từ, đề thật dài 63
> từ**. Ta đang đo trên một phân bố truy vấn khác với phân bố đi thi.

**Không kết luận được cắt truy vấn đóng góp bao nhiêu trong ×3,25.** Ba thay đổi
đi cùng một lần nộp, mà mỗi gói chỉ có 3 lần nộp nên không A/B được trên
leaderboard. Chỉ biết: phần dev giải thích được là ×1,55, phần còn lại chưa quy
được cho ai.

> **Việc quan trọng nhất cho tập dev từ nay không phải THÊM CÂU, mà là THÊM CÂU
> DÀI NHƯ ĐỀ THẬT** — nhiều câu, nhiều mệnh đề, có bối cảnh trước/sau. Không có
> chúng thì mọi con số ta đo tiếp tục nói về một bài toán dễ hơn bài đang thi.

---

### A22 — Mũi nhọn 1 nối xong: Bước 1 và Bước 2b **đều chưa đủ căn cứ bật**

Bảng trạng thái Giai đoạn 2 đọc ra gần như toàn ❌ trong khi từng module rời rạc
đã viết xong từ lâu — thứ thiếu là **chỗ nối**. `src/mui_nhon_1.py` nối chúng
lại, mỗi bước một cờ, `scripts/22_do_mui_nhon_1.py` đo từng cờ một.

Mốc nền: **RRF(objects, OCR)** — cấu hình mạnh nhất chạy được không cần model,
trên 105 câu tập dev. Máy đo 7,7 GB không nạp nổi SigLIP2.

#### Bước 1 — thu hẹp cấp video bằng metadata

**Đo ĐỘ PHỦ trước khi đo điểm**, vì độ phủ là trần trên của dạng cứng:

| Video đúng nằm trong | Số câu | Tỷ lệ |
| --- | ---: | ---: |
| top-10 metadata | 23/105 | **21,9%** |
| top-50 metadata | 39/105 | **37,1%** |
| top-200 metadata | 58/105 | 55,2% |
| có mặt ở bất kỳ hạng nào | 102/105 | 97,1% |

> ⚠️ **Đọc lại A12 cho đúng.** A12 ghi *"97% tìm ra video đúng"* — con số đó là
> **có mặt ở đâu đó trong xếp hạng**, không phải năng lực thu hẹp. Ở top-50 nó
> chỉ phủ **37,1%**. Ai đọc "97%" rồi thiết kế Bước 1 cắt cứng sẽ mất 63% số câu
> trước khi bước nào phía sau kịp chạy.

Điểm, so theo cặp với mốc nền:

| Cấu hình | ±2s | ±15s | Kết luận |
| --- | ---: | ---: | --- |
| RRF(objects, OCR) — mốc | 0,0781 | 0,1295 | |
| ưu tiên top-10 (mềm) | +0,0057 | +0,0076 | 🟡 YEU |
| **ưu tiên top-50 (mềm)** | **+0,0095** | **+0,0095** | 🟡 YEU — 4-3-98 / 10-8-87 |
| ưu tiên top-200 (mềm) | −0,0076 | −0,0152 | 🟡 YEU (âm) |
| **CỨNG: chỉ top-50** | **−0,0286** | **−0,0476** | 🟡 YEU (âm) |

Ba điều đọc được:

1. **Dạng CỨNG — đúng như PHẦN D viết — là dạng TỆ HƠN**, ở cả hai mức dung sai,
   và cơ chế đã đo được ở bảng trên: cắt ở top-50 là vứt bỏ 63% số câu.
   → **Sửa PHẦN D Bước 1: "BM25 metadata → top-50 video" phải hiểu là XẾP LẠI,
   không phải CẮT.** Cùng nguyên tắc `objects.py`: cho điểm mềm, không lọc cứng.
2. **Dạng mềm có dấu dương ổn định ở N=50 nhưng KHÔNG vượt nhiễu** (ngưỡng
   0,0205 / 0,0317). Chưa đủ căn cứ bật mặc định. Đây đúng là loại kết luận mà
   *"việc quan trọng nhất cho tập dev là thêm câu DÀI như đề thật"* (A20) sẽ
   giải quyết — không phải bằng cách chỉnh tham số.
3. **N=200 quay sang ÂM.** Ưu tiên quá nhiều video thì "ưu tiên" không còn nghĩa
   gì, chỉ còn là xáo thứ tự. Nếu bật thì bật ở N nhỏ.

#### Bước 2b — khử trùng lặp trước khi cắt top-K

| Cấu hình | ±2s | ±15s | Kết luận |
| --- | ---: | ---: | --- |
| dedup ≥ 0,99 | −0,0152 | +0,0057 | ❌ **DAO DAU** |
| dedup ≥ 0,95 | −0,0114 | +0,0095 | ❌ **DAO DAU** |
| dedup ≥ 0,99 + mỗi video ≤ 3 | −0,0324 | −0,0057 | ✅ **ON DINH — TỆ HƠN** |

**Đảo dấu giữa hai mức dung sai = kết luận phụ thuộc vào ẩn số BTC chưa chốt**
(độ rộng cửa sổ, A9), không dùng để quyết. Nhưng dòng thứ ba thì kết luận được,
và nó **bác một dự đoán ghi thẳng trong A11**: ở đó ta viết *"`gioi_han_moi_video()`
không thay được dedup, hai cái BỔ SUNG cho nhau"*. Đo ra: ghép cả hai **tệ hơn
ổn định**, 0 thắng – 7 thua ở ±2s.

Lý do khớp với A18: cả dedup lẫn ràng buộc đa dạng đều **bỏ ứng viên đi**, mà
PHẦN C mục 1 nói không có điểm phạt — dòng thứ 100 vẫn đáng 0,2. Bỏ một ứng viên
để lấy chỗ cho ứng viên khác chỉ có lãi nếu ứng viên bị bỏ **chắc chắn sai**, và
"gần trùng ảnh" không đủ để chắc chắn điều đó khi BTC chấm theo khoảng.

> **Đây là lần thứ năm một kỹ thuật lấy từ bài báo AIC'25 hoặc từ trực giác thiết
> kế bị chính phép đo bác** (sau dedup A11, RRF thô A14, hợp nhất hai tầng A14.1,
> `lan_can` + ràng buộc đa dạng A18).

#### Bước 3b — `lan_can` đã được đặt lại đúng chỗ

A18 đo được chèn khung lân cận vào danh sách ứng viên làm tệ đi. Nguyên nhân là
**cộng dồn hai luật của BTC**: điểm chấm theo khoảng 4 giây–5 phút (A9), nên
khung cách đáp án ±2s **đã được tính đúng rồi** — chèn nó vào là tiêu một trong
100 chỗ để mua lại thứ mình đã có.

Kỹ thuật đó không vô dụng, nó chỉ đứng nhầm chỗ. Chỗ đúng là **Bước 4**: PHẦN D
ghi *"3–5 frame trong cửa sổ ±2s"*, mà cửa sổ ±2s quanh một khung chính là
`lan_can`. Ở đó nó không tiêu chỗ nộp nào. `mui_nhon_1.khung_ngu_canh()` gọi nó
đúng một lần, lọc theo **giây** chứ không theo số bước — mật độ keyframe không
đều (A1), "10 keyframe quanh đây" là 6 giây ở video này và 30 giây ở video kia.

#### Bước 4 — CHƯA ĐO ĐƯỢC, và chặn ở một lệnh

`mui_nhon_1.gan_dap_an()` đã nối `tra_loi.py` vào `run.py` (`--vlm`), có test cho
mọi đường thoát. Nhưng máy đang chạy **không có model nào nhìn được ảnh** —
`ollama list` ra `qwen2.5:3b`, `gemma:2b`, `nomic-embed-text`, cả ba thuần văn
bản. Chặn ở `ollama pull qwen2.5vl:7b`.

Đây vẫn là **việc đáng giá nhất còn lại tính theo điểm trên mỗi đơn vị công**:
42/105 câu dev và 3/24 gói đề mẫu là Q&A, tất cả đang **chắc chắn 0** vì nộp
`answer` là hằng số `"không rõ"`.

#### Ba cảnh báo cho người đọc lại bảng này

1. **Đo trên cấu hình KHÔNG có model.** A14.2 đo được RRF chỉ lãi khi các kênh
   cùng tầm chất lượng; SigLIP2 (0,3258) hơn objects (0,0412) tám lần. Thứ tự
   các cấu hình ở đây **có thể đảo hẳn** khi kênh 1 sống. Đo lại trên máy ≥ 16 GB.
2. **5/105 câu dev là câu ô nhiễm** (A21 — bối cảnh chép từ OCR của chính khung
   đáp án). Chúng nâng mốc nền vì mốc nền có kênh OCR; ảnh hưởng như nhau lên mọi
   cấu hình nên phép so theo cặp vẫn đứng, nhưng con số tuyệt đối thì đọc dè.
3. **Mốc nền nay là 0,0781 chứ không phải 0,0640** — không phải vì cấu hình khá
   lên mà vì tập dev đi từ 100 lên 105 câu. Đừng so hai con số đó với nhau.

---

### A23 — Bản nộp ăn 2,6 điểm **không dựng lại được** bằng lệnh trong sổ tay

Phát hiện lúc dựng lại bộ đề mẫu bằng đúng lệnh ghi ở
[10_nop_bai_toi_nay.md](10_nop_bai_toi_nay.md): ra **19/24 file khác** bản nộp
đã ghi 2,6 điểm. Hạng 1 trùng, nhưng chỉ **50/100 dòng** trùng nhau.

**Truy nguyên, theo thứ tự loại trừ:**

| Giả thuyết | Cách kiểm | Kết quả |
| --- | --- | --- |
| Code đã đổi hành vi (refactor `KenhObjects`) | chạy chính code ở HEAD trong `git worktree` riêng, **cùng một lệnh** | **0/24 khác** — code vô can |
| Bản cũ có kênh 2 (không `--bo-metadata`) | dựng lại, so từng byte | 21/24 khác — không phải |
| Bản cũ bật `--loc-cung` | dựng lại, so từng byte | 21/24 khác — không phải |
| **Bản cũ chạy `--trong-so-phu 1.0`** | dựng lại, so từng byte | **0/24 khác — ĐÚNG** |

Mặc định trong code lúc đó là **0,3**. Tức từ ngày nộp tới nay, chạy bằng lệnh
mặc định là sinh ra **một bài nộp khác bài đã ghi điểm**, mà không có gì báo.

#### Trọng số nào đúng — `scripts/23_do_trong_so_rrf.py`, 105 câu

Mốc nền là **cấu hình đã nộp thật**, không phải mặc định của code:

| Cấu hình | ±2s | ±15s | So với bản đã nộp |
| --- | ---: | ---: | --- |
| **objects : OCR = 1,0 : 1,0** *(đã nộp)* | **0,0781** | **0,1295** | mốc |
| 1,0 : 0,5 | 0,0590 | 0,0971 | 🟡 −0,0190 / −0,0324 |
| 1,0 : 0,3 *(mặc định cũ)* | 0,0476 | 0,0781 | ✅ **−0,0305 / −0,0514 — TỆ ĐI ổn định** |
| 1,0 : 0,1 | 0,0476 | 0,0781 | ✅ TỆ ĐI ổn định |
| chỉ objects | 0,0457 | 0,0762 | ✅ TỆ ĐI ổn định |
| chỉ OCR | 0,0552 | 0,1143 | 🟡 −0,0229 / −0,0152 |

Dìm OCR xuống 0,3 ra **gần đúng bằng objects đứng một mình** (0,0476 so với
0,0457) — tức nó vứt gần hết phần lãi của hợp nhất. Và 0,3 với 0,1 cho **số
giống hệt nhau**: dưới một mức nào đó, kênh phụ chỉ còn là thứ tự phá hòa, không
còn đóng góp gì.

#### Vì sao 0,3 từng đúng, và vì sao ở đây nó sai

Con số 0,3 đến từ **A14.2**, và trong ngữ cảnh đó nó đúng: A14.2 đo cảnh **một
kênh mạnh cộng một kênh yếu** — SigLIP2 0,3258 với objects 0,0412, chênh **8
lần**. Ứng viên hạng 1 của kênh yếu được RRF cộng đúng bằng hạng 1 của kênh
mạnh, nên phải dìm nó xuống.

Cấu hình model-free thì ngược hẳn: **objects 0,0412 và OCR 0,0420 — ngang nhau**.
Dìm một trong hai là vứt bỏ nửa số bằng chứng.

> **Bài học tổng quát hơn cả con số:** trọng số RRF không phải hằng số của hệ
> thống, nó là **hàm của tỷ lệ chất lượng giữa các kênh**. Một mặc định duy nhất
> không thể đúng cho cả hai cảnh. Nay đặt mặc định **1,0** vì đó là cảnh đang
> chạy thật (`--hop-nhat` chỉ dùng được với objects+OCR — với SigLIP2 thì A14.2
> bảo **đừng hợp nhất**, chứ không phải hợp nhất với trọng số nhỏ).

#### Chốt đã dựng để không tái diễn

`run.py` nay ghi `lenh_da_chay.txt` cạnh các file CSV, chứa **nguyên văn lệnh đã
chạy**. `dong_goi()` chỉ nén `*.csv` nên nó không lọt vào bài nộp BTC.

Không có nó thì một điểm số ngoài **không truy nguyên được về một cấu hình**, và
mọi suy luận kiểu "đợt 2 hơn đợt 1 nhờ X" đều là phỏng đoán — đúng loại lỗ hổng
A20 đã vấp khi không quy được ×3,25 cho ai.

---

### A24 — Bốn lượt nộp chứ không phải hai, và lượt tốt nhất (3,8) là **kênh ảnh SigLIP2**

A20 viết khi tưởng chỉ có hai lượt. Bảng "Bài của tôi" của BTC cho thấy **bốn**:

| Lượt | Giờ (20/08) | Điểm public | Là gì |
| --- | --- | ---: | --- |
| #1 | 16:54 | 0,8 | objects đơn lẻ |
| #2 | 20:29 | 2,6 | RRF(objects, OCR), trọng số 1,0 — `firstdance1.zip` |
| **#3** | **21:11** | **3,8** | **kênh ảnh SigLIP2** — xem chuỗi bằng chứng dưới |
| #4 | 23:11 | 2,6 | dựng lại từ repo, **trùng từng byte với #2** |

**#4 trùng byte với #2 và ra đúng điểm của #2.** Đó là xác nhận ngoài rằng đường
ống tái lập chính xác, và cũng là bài học: **lượt cuối mới tính điểm** (C7), nên
nộp lại một cấu hình cũ là tự hạ standing.

#### Vì sao kết luận #3 là SigLIP2, khi không có ai ghi lại lệnh

Không có dấu vết nào của #3 trên máy dựng repo: `submission/` giữ mtime 20:27
(của #2), lịch sử PowerShell không có lệnh `run.py` nào, không zip nào khác
trong khoảng 20:40–21:20. Nên phải vân tay từ chính file nộp:

| Bằng chứng | Đo được | Suy ra |
| --- | --- | --- |
| Chồng lấn với #2 (objects+OCR) | **~0** — 21/24 gói KHÔNG chung dòng nào, hạng 1 khác ở cả 24 | không phải chỉnh trọng số; là **kênh khác hẳn** |
| Cặp `(video, frame)` của dòng KIS/QA | đều là keyframe thật của `master.parquet` | dựng từ **index của ta** |
| Chuỗi TRAKE | `1536,3072,4608,6144` — đúng dấu vết nhánh *rải đều* của `run.dung_trake()` | dựng bằng **`run.py`** |
| Phân bố nhóm L | đủ cả 10 nhóm (L21 456 · L26 355 · L29 194 …) | ma trận phủ **toàn kho**, không phải bể ảnh của một máy |
| Đa dạng | **50 video khác nhau / 100 dòng** (min 17, max 81) | dáng của truy hồi dày đặc, không phải objects |
| Ứng viên còn lại: CLIP B/32 | A10 đo **0,0000 trên 100/100 câu tiếng Việt** | không thể ra 3,8 |

→ `run.py --kenh anh --matrix clip_siglip2.npy` trên máy ≥ 16 GB. Đó là cấu hình
duy nhất còn lại thoả mọi bằng chứng.

**Và nó khớp với tập dev.** Dev đo SigLIP2 **0,3258** so với RRF(objects, OCR)
**0,0781** — hơn 4,2 lần; leaderboard 3,8 so với 2,6. Hướng trùng nhau, độ lớn
thì không so được (leaderboard trộn cả KIS/QA/TRAKE, dev không có TRAKE).

> **Đây là lần đầu tập dev DỰ ĐOÁN ĐÚNG một kết quả ngoài.** A20 ghi tập dev mù
> hai lần (A19, A20). Lần này nó xếp đúng thứ tự hai cấu hình trước khi có điểm
> ngoài. Kênh 1 là kênh mạnh nhất — trên dev, và nay trên cả leaderboard.

#### Ba gói Q&A: **cả bốn lượt đều chắc chắn 0**, và lý do không phải truy hồi

Bản 3,8 nộp `answer` là `5`, `2`, `10` cho ba gói Q&A. Đọc lại đề:

| Gói | Hỏi gì | Đáp án đã nộp |
| --- | --- | --- |
| `query-p1-15-qa` | tên **xã** ở Khánh Hoà nơi CLB FANA trao quà | `5` ❌ |
| `query-p1-19-qa` | **hai câu thơ** ca ngợi Nguyễn Trung Trực | `2` ❌ |
| `query-p1-22-qa` | **tên món ăn** trên công thức có 200g thịt nạc xay | `10` ❌ |

**Không câu nào là câu đếm.** Cả ba đều hỏi **chữ đọc được trên hình**: tên xã
chạy dưới bản tin, câu thơ khắc trong đình, tiêu đề công thức trên tờ giấy.

Hai hệ quả cho Bước 4 của Mũi nhọn 1:

1. **Lời nhắc trong `tra_loi.NHAC` đang lệch hẳn dạng câu hỏi thật.** Nó dạy
   model *"câu hỏi đếm → trả lời bằng chữ số"*, *"màu sắc → tên màu"*, *"tối đa
   4 từ"* — trong khi đề thật hỏi tên riêng và **hai câu thơ** (chắc chắn dài
   hơn 4 từ, và vẫn phải lọt trần 100 ký tự).
2. **Kênh 3 (OCR) có thể trả lời trực tiếp**, không cần VLM: chữ đã nằm sẵn
   trong `ocr_asr.parquet` của đúng khung được truy hồi. Đây là đường rẻ nhất
   tới điểm Q&A đầu tiên của đội — và nó chạy được trên máy yếu.

Ba gói Q&A đang là **3/24 gói ăn 0 chắc chắn** ở cả bốn lượt nộp. Không có phần
nào của hệ thống rẻ hơn phần này tính theo điểm trên mỗi đơn vị công.

#### Chốt đã dựng sau vụ này

`run.py` ghi `lenh_da_chay.txt` cạnh file CSV (A23). Nếu #3 có nó thì toàn bộ
mục này đã không cần viết — và ta suýt mất hẳn cấu hình tốt nhất của mình chỉ vì
không ai ghi lại một dòng lệnh.

**Quy ước từ nay: mỗi lượt nộp giữ lại CẢ file zip LẪN `lenh_da_chay.txt`**, đặt
tên theo lượt (`nop_dot1_luot3.zip`). Windows không phân biệt hoa/thường —
`FirstDance_round1.zip` đã **đè mất** `firstdance_round1.zip` của lượt #2.

---

### A25 — Kênh 3 mạnh gấp 2,8 lần, đảo vai với objects; và leaderboard public **mù** với thay đổi cục bộ

#### Dữ liệu OCR/ASR đầy đủ (21/08): kênh 3 đổi hạng

TV4 nạp `ocr.parquet` + `asr.parquet` + `ocr_asr.parquet` mới vào `index/`:

| | Trước | Sau |
| --- | ---: | ---: |
| OCR có chữ | 47.064 (26,5%) | **165.259 (93,2%)** |
| ASR có chữ | **0** | **137.322 (77,4%)**, 847/873 video |
| `text` hợp nhất | 47.064 | **176.009 (99,3%)** |

Đo trên 115 câu tập dev (`scripts/23_do_trong_so_rrf.py`):

| Cấu hình | ±2s | ±15s |
| --- | ---: | ---: |
| **kênh 3 (OCR+ASR mới) một mình** | **0,1183** | 0,1530 |
| RRF(objects, OCR) 1:1 | 0,1131 | 0,1547 |
| kênh 4 objects | 0,0417 | 0,0696 |
| *kênh 3 OCR cũ, để so* | *0,0420* | *0,0720* |

**Kênh 3 nhảy 2,8 lần và vượt cả cấu hình hợp nhất.** Cộng objects vào nay không
còn lãi (đảo dấu, 16-8-91) — đúng quy luật A14.2, nhưng **vai đã đổi**: objects
giờ là kênh yếu bị dìm, không phải kênh được cứu.

> **Cấu hình model-free tốt nhất không còn là RRF(objects, OCR) mà là kênh 3
> đứng một mình.** A23 vừa đổi mặc định `--trong-so-phu` về 1,0 cho cảnh hai kênh
> ngang nhau; nay cảnh đó không còn tồn tại. Ai chạy máy yếu thì dùng kênh 3.

Và nó đưa RRF(SigLIP2, OCR) vào tầm: khoảng cách hai kênh thu từ **8 lần xuống
2,8 lần** — đúng vùng A14.2 nói hợp nhất bắt đầu có lãi. Đây là ứng viên mạnh
nhất chưa ai thử.

#### Leaderboard public không phản hồi thay đổi ở 3/24 gói

A24 tìm ra ba gói Q&A ăn 0 vì đáp án sai. Đã tìm đáp án bằng OCR/ASR mới rồi
**mở ảnh xác minh tận mắt**, vá vào bài nộp 3,8 (`scripts/24_va_dap_an_qa.py`),
giữ nguyên từng byte 21 gói KIS/TRAKE:

| Gói | Hỏi gì | Đáp án đã xác minh |
| --- | --- | --- |
| `15-qa` | tên xã ở Khánh Hoà nơi CLB FANA trao quà | **Giang Ly** — băng rôn `L30_V072` kf34 |
| `19-qa` | hai câu thơ ca ngợi Nguyễn Trung Trực | **Hỏa hồng Nhật Tảo oanh thiên địa / Kiếm bạt Kiên Giang khấp quỷ thần** — câu đối vàng hai bên tượng, `L27_V010` kf146 |
| `22-qa` | tên món ăn có 200g thịt nạc xay | 🟡 **Bánh ít trần** — chỉ suy từ OCR, máy không có ảnh L26 |

Khung chứa đáp án được đặt **hạng 1**. Kết quả: **3,8 → 3,8. Không đổi.**

Ghép với lượt #3 (đáp án `5`, `2`, `10` — chắc chắn sai) cũng 3,8, ta có một phép
so sạch: **đáp án sai và đáp án đúng cho cùng một điểm public.** Ba khả năng,
xếp theo độ tin:

1. **3 gói Q&A không nằm trong 50% được chấm public** (C7). Chỉ 3/24 gói là Q&A.
2. Chuỗi đáp án không khớp cách BTC ghi — tài liệu tự mâu thuẫn "ngữ nghĩa" vs
   "chuỗi chính xác" (C7), và hai câu thơ có nhiều cách viết.
3. Khung sai — ít khả năng nhất, đã xem ảnh.

> **Hệ quả cho cách tiêu 3 lượt nộp của vòng thật: đừng tiêu một lượt để thử một
> thay đổi cục bộ.** Bảng public là thước đo THÔ — nó đọc được việc đổi cả kênh
> truy hồi (2,6 → 3,8) nhưng mù với việc sửa 3/24 gói. Muốn biết một thay đổi
> nhỏ có lợi không thì phải hỏi tập dev, không phải hỏi leaderboard.
>
> Việc vá đáp án **vẫn giữ**: Private Leaderboard chấm 100% đáp án, nên nếu ba
> gói đó nằm ở nửa private thì `Giang Ly` và câu đối vẫn ăn điểm ở đó. Đáp án đã
> xác minh không thể tệ hơn `5`, `2`, `10`.

#### Ba câu Q&A của đề thật đều là câu ĐỌC CHỮ, không phải câu đếm

Tên xã chạy dưới bản tin, câu thơ khắc trong đình, tiêu đề trên tờ công thức.
Không câu nào hỏi số lượng hay màu sắc.

Nhưng `tra_loi.NHAC` đang dạy model: *"câu hỏi đếm → trả lời bằng chữ số"*,
*"màu sắc → tên màu"*, *"tối đa 4 từ"*. Lời nhắc đó **lệch hẳn phân bố câu hỏi
thật**, và ép 4 từ thì không bao giờ trả lời nổi một câu đối 14 chữ.

→ Việc cho Bước 4: viết lại lời nhắc theo phân bố thật, và **thử đường không cần
VLM trước** — `ocr_asr.parquet` đã chứa sẵn chữ của đúng khung được truy hồi, mà
đó chính là đáp án. Rẻ hơn VLM, chạy được trên máy yếu.

#### Dải cosine 0,72–0,92 của `19_tim_cau_trake.py`: dùng để TÌM thì được, dùng để NGHIỆM THU thì sai

10 câu TRAKE mới (L24, L30) đã soát: cấu trúc hợp lệ 10/10, và ba câu lấy mẫu
khớp mô tả đến từng chi tiết khi mở ảnh. Nhưng chấm bằng chính dải cosine của ta:

| | cosine giữa các sự kiện | Kết luận của ngưỡng |
| --- | --- | --- |
| `trake-L30-004` | 0,714 · **0,582** | 🟡 ngoài dải — *nhưng đã xác minh là câu TỐT* |
| `trake-L24-002` | **0,664** · 0,735 | 🟡 ngoài dải — *cũng đã xác minh là tốt* |
| Tổng | **7/10 câu bị gắn cờ** | |

Ngưỡng loại nhầm 7/10 câu tốt. Nguyên nhân: nó giả định chuỗi sự kiện = cảnh
biến đổi từ từ, nhưng **phóng sự truyền hình đổi CÚ MÁY giữa các sự kiện** —
toàn cảnh → cận cảnh → góc khác — nên cosine tụt sâu trong khi vẫn là một hành
động liên tục.

> Giữ dải này làm **bộ lọc tìm kiếm** (nó thu hẹp hàng nghìn cửa sổ xuống vài
> chục, đó là việc của nó). **Không dùng làm tiêu chí nghiệm thu.** Và cảnh báo
> rộng hơn: mọi kỹ thuật xếp hạng lại theo thời gian giả định "cảnh trôi từ từ"
> đều đang đứng trên một giả định mà dữ liệu thật không thoả.

#### `dense.KenhAnhCache` — gỡ chốt RAM mà không đổi gì khác

Máy 7,7 GB không nạp nổi SigLIP2 nên mọi phép đo dính kênh 1 đều tắc. Nhưng
model chỉ làm **một việc**: biến câu chữ thành vector. Ma trận ảnh đã nằm sẵn
trên đĩa, và tập truy vấn thì **hữu hạn, biết trước** — 24 câu đề + 115 câu dev,
tổng **296 chuỗi** sau khi tính cả các mệnh đề do `tach_truy_van` cắt ra.

    máy khoẻ/Colab, một lần:  python scripts/25_ma_hoa_truy_van.py --de <đề> --tap-dev
    máy yếu, từ đó về sau:    python scripts/22_do_trake.py --cache index/truy_van.npz
                              python src/run.py --de <đề> --cache index/truy_van.npz

File cache vài trăm KB, chép qua chat cũng được.

**`KenhAnhCache.tim` là hàm THỪA KẾ, không phải bản chép** — chỉ `encode_text`
đổi từ "chạy model" thành "tra bảng". Có test chốt đúng điều đó
(`test_cache_dung_lai_tim_cua_kenh_that`), vì hai nhánh lệch nhau âm thầm thì
số đo trên máy yếu không còn so được với số đo trên máy khoẻ.

Truy vấn thiếu trong cache thì **ném lỗi**, không trả vector 0 — vector 0 sẽ cho
ra 100 ứng viên ngẫu nhiên trông hoàn toàn hợp lệ.

⚠️ `25_ma_hoa_truy_van.py` gọi thẳng `open_clip`, **không đi qua `dense.kiem_ram`**,
nên nó tự mang chốt riêng, ngưỡng tra theo số chiều ma trận (SigLIP2 fp16 ~2,9 GB;
CLIP B/32 fp16 ~0,9 GB). Không có chốt đó thì chính công cụ sinh ra để né việc
nạp model lại là thứ treo máy.

---

### A26 — Q&A trả lời từ OCR/ASR: **31% khớp chuỗi chính xác** ở khung đúng

#### Trước hết: `.env` suýt lọt lên GitHub

Khoá Gemini được đặt vào `c:\Code\aic2026\.env`, mà `.gitignore` **không có luật
nào cho `.env`**. Git chưa kịp theo dõi nó nên chưa lọt, nhưng chỉ cần một lần
`git add -A` là xong — đúng kiểu lỗ hổng đã làm lọt 15 ảnh keyframe. Đã vá:
`.env`, `.env.*`, chừa `!.env.mau`.

Hai chi tiết kỹ thuật đáng ghi vì cả hai đều làm mất thời gian:

* Tên biến trong file thật ghi `GEMINI_API_KEY =...` — **có dấu cách trước `=`**.
  Không cắt khoảng trắng quanh TÊN thì tra cứu trượt, mà triệu chứng lại là
  "không tìm thấy khoá" → rất dễ đổ nhầm cho việc chưa đặt khoá. `tra_loi.nap_khoa()`
  cắt cả hai phía.
* Khoá của nhóm dạng **`AQ.A...` 53 ký tự**, không phải `AIza...` 39 ký tự mà
  `14_sinh_caption.py` viết cho, và nó đi qua **header `x-goog-api-key`** chứ
  không phải `?key=` trên URL. Đã thử và xác nhận: hỏi `/v1beta/models` trả về
  50 model, có `gemini-3.1-flash-lite`.

#### Đường OCR → LLM, không cần VLM

Ba câu Q&A của đề mẫu đều hỏi **chữ hiện trên hình** (A25), tức cần **đọc** chứ
không cần **nhìn**. Chữ đó đã nằm trong `ocr_asr.parquet`. Nên `src/tra_loi_ocr.py`
đi đường này, và nó **không cần VLM, không cần ảnh trên đĩa, không đụng chốt RAM**.

Thử trên đúng ba câu đề mẫu, ở khung đã xác minh bằng mắt:

| Gói | Gemini 3.1 flash-lite trả về | Đáp án đã xác minh |
| --- | --- | --- |
| `15-qa` | `Giang Ly` | ✅ đúng |
| `19-qa` | `Hỏa hồng Nhật Tảo oanh thiên địa, kiếm **bạc** Kiên Giang khấp quỷ thần` | 🟡 sai một chữ (`bạt`) |
| `22-qa` | `Bánh ít trần` | ✅ đúng |

Câu thứ ba đáng nói: đó là câu **tôi không xác minh được bằng mắt** vì máy không
có ảnh L26, chỉ suy từ OCR. Gemini độc lập rút ra cùng đáp án — một phép kiểm
chéo, không phải bằng chứng, nhưng làm tăng độ tin.

#### Đo trên 42 câu Q&A của tập dev, tại KHUNG ĐÁP ÁN

`scripts/27_do_tra_loi_qa.py`. Chấm ở khung đúng chứ không phải khung do truy hồi
trả về — đây là **trần trên**, vì điểm Q&A thi thật là tích của hai thứ.

| Mức khớp | Đúng | Tỷ lệ |
| --- | ---: | ---: |
| **chuỗi chính xác** | 13/42 | **31,0%** |
| không phân biệt hoa/dấu câu | 15/42 | 35,7% |
| chứa nhau (một bên nằm trong bên kia) | 24/42 | **57,1%** |

Báo cả ba vì **BTC tự mâu thuẫn** giữa *"ngữ nghĩa"* (tr.2) và *"chuỗi chính
xác"* (tr.8) — C7. Con số thật nằm giữa 31% và 57%, và **khoảng cách đó chính là
giá của việc chưa hỏi BTC**.

Con số này khớp đúng dự đoán của D0.3: *"trần độ đúng ~30–50% ở MỌI model"*.

#### Câu nào trượt — và nó chia làm hai loại rất rạch ròi

    "Có bao nhiêu con mèo được chàng trai cho ăn?"      -> không rõ
    "Những chùm nho có màu gì?"                         -> không rõ
    "Có bao nhiêu gói Blendy xếp chồng trên đĩa?"       -> không rõ

Toàn bộ câu **đếm** và câu **màu sắc** đều trượt, và trượt đúng cách nên trượt:
OCR không chứa thông tin đó. Đây là ranh giới của đường này, và nó **đúng bằng
chỗ VLM phải vào**. Hai đường bổ sung nhau, không thay nhau:

    câu hỏi CHỮ trên hình  -> OCR + LLM văn bản   (rẻ, chạy máy yếu, phủ toàn kho)
    câu hỏi ĐẾM / MÀU      -> VLM nhìn ảnh        (cần model vision + ảnh trên đĩa)

Loại trượt thứ hai tinh vi hơn — đọc được nhưng đọc sai:

    qa-L28-003   đáp án "Cừ Đứt"   -> "Cù lao Cù Dút"   (OCR mất dấu, đoán nhầm)
    qa-L29-006   đáp án "Tân Hôn"  -> "Đôi Mắt"          (đọc nhầm chữ khác trong khung)

#### `qwen2.5:3b` không đủ, và luật thì đủ cho một loại câu

Model 3B local đọc đúng chuỗi OCR `XaGiang Ly.huyen Khanh Vinh.tinh Khanh Hod`
mà vẫn trả *"không rõ"* — nó không gỡ nổi chữ dính mất dấu. Nên `doan_dia_danh()`
rút bằng **luật**: `\s*` chứ không phải `\s+`, vì OCR dính chữ. Chạy đúng trên
dữ liệu thật: `xã → Giang Ly`, `huyện → Khanh Vinh`.

Luật gỡ được vì nó **không cần hiểu**. Giữ nó làm đường lui khi không có mạng
hoặc hết quota.

#### Một lỗi đã đo và đã sửa: ASR chôn mất OCR

Bản đầu dùng thẳng cột `text` (đã gộp `ocr + " . " + asr`). ASR dài trung bình
**463 ký tự và lặp y hệt trên nhiều khung liền nhau**, OCR chỉ ~90 — gộp phẳng
thì phần chữ trên màn hình bị chôn giữa một biển lời thoại. Ba câu ra ba lỗi:

    "xã này tên gì"  -> "không rõ"       (OCR chứa đáp án, bị chôn)
    "hai câu thơ"    -> chỉ ra vế đầu
    "tên món ăn"     -> "Bánh ít trơn"   (nghe theo ASR; OCR ghi TRAN)

Sửa: tách OCR khỏi ASR, **OCR lên trước**, bỏ trùng theo mặt chữ. Sau khi sửa,
Gemini trả đúng cả ba.

---

### A27 — **3,8 → 4,8**: lọc xếp tầng thắng ở đúng chỗ RRF đã thua bốn lần

Lượt #8 nộp `firstdance5.zip`: **4,8 điểm**, tăng 1,0 so với mọi lượt trước đó.
Đây là mức tăng đầu tiên kể từ lượt #3, và là lần đầu một kênh phụ **giúp được**
kênh 1 thay vì kéo nó xuống.

#### Thay đổi là gì, và vì sao nó khác bốn lần thất bại trước

| | RRF (A14, A14.1, A17, A22) | Lọc xếp tầng (đây) |
| --- | --- | --- |
| Kênh yếu được làm gì | **bỏ phiếu ngang hàng** — cộng `1/(k+hạng)` như kênh mạnh | **chỉ xếp lại** trong bể kênh 1 đã chọn |
| Ứng viên có bị mất không | không, nhưng bị đẩy lùi bởi ứng viên kênh yếu | không — chỉ đảo thứ tự trong top-20 |
| Kết quả đo | ❌ tệ đi, cả bốn lần | ✅ **+1,0 điểm trên leaderboard** |

`scripts/28_xep_lai_bang_gemini.py`: lấy 20 ứng viên đầu của mỗi gói KIS trong
bài nộp 3,8, đưa **chữ OCR/ASR của chính các khung đó** cho `gemini-3.1-flash-lite`,
hỏi khung nào có bằng chứng RÕ khớp truy vấn, rồi đẩy chúng lên đầu. Phần còn
lại giữ nguyên thứ tự.

Trên 18 gói KIS: **10 gói được xếp lại**, 8 gói Gemini trả `[]` (văn bản không
giúp gì — hành vi đúng cho truy vấn thuần thị giác). Gói Q&A và TRAKE giữ nguyên
từng byte.

> **Bài học đắt nhất của repo này vừa được sửa lại cho chính xác hơn.** Không
> phải *"kênh yếu luôn làm tệ đi"* — mà là *"kênh yếu không được bỏ phiếu ngang
> hàng"*. Cùng một kênh OCR, cùng một bể ứng viên: hoà RRF thì lỗ, xếp lại trong
> bể thì lãi 1,0 điểm.

#### Và một phát hiện về chính công cụ đo

Bảng public **đọc được** thay đổi ở gói KIS (18/24 gói) nhưng **mù** với gói Q&A
(3/24 — A25/A26). Nghĩa là từ nay ta có một vòng đo NGOÀI thật sự, nhưng **chỉ
cho KIS**:

    doi kenh truy hoi (21 goi)      2,6 -> 3,8    doc duoc
    xep lai top-20 KIS (18 goi)     3,8 -> 4,8    doc duoc
    sua dap an Q&A (3 goi)          3,8 -> 3,8    MU
    pha hong han Q&A (3 goi)        3,8 -> 3,8    MU

Hệ quả cho cách tiêu lượt nộp: **thử nghiệm KIS thì nộp, thử nghiệm Q&A thì đo
trên tập dev** — nộp chỉ tốn lượt mà không trả lời được gì.

#### Chưa biết, và đang dò

`--top` là siêu tham số duy nhất của kỹ thuật này, và **chưa có căn cứ nào cho
con số 20** — nó chỉ là mức thận trọng tôi chọn. Đang dò: `--top 50` để trả lời
"đọc sâu hơn thì lãi thêm hay bắt đầu lỗ".

⚠️ Không đo được trên tập dev vì bể ứng viên do SigLIP2 sinh ra, mà máy dựng
repo không chạy nổi SigLIP2 (A25). Đây là **cấu hình dò bằng leaderboard**, khác
mọi cấu hình khác trong tài liệu này — ghi rõ để người sau không đọc nhầm thành
đã chứng minh trên dev.

---

### A28 — Đường liều của xếp lại: **bão hoà ở top-50**, và lọc cứng OCR **làm tệ đi**

Bốn lượt nộp liên tiếp, mỗi lượt đổi **đúng một thứ** trên cùng một nền
(bài nộp 3,8 của SigLIP2):

| Lượt | Cấu hình | Điểm | So với trước |
| --- | --- | ---: | --- |
| #3,5,6,7 | không xếp lại | 3,8 | mốc |
| #8 | xếp lại **top-20** | **4,8** | **+1,0** |
| #9 | xếp lại **top-50** | **5,4** | **+0,6** |
| #10 | xếp lại **top-100** | 5,4 | ⚪ **+0,0 — bão hoà** |
| #11 | top-50 **+ lọc cứng OCR toàn kho** | 5,0 | ❌ **−0,4** |

#### Bão hoà ở 50, và vì sao con số đó hợp lý

Mức tăng giảm dần rồi tắt: **+1,0 → +0,6 → +0,0**. Lời nhắc buộc Gemini chỉ đẩy
lên khi có **bằng chứng RÕ** trong văn bản, mà những ứng viên như thế nằm tập
trung ở đầu danh sách của SigLIP2 — quét sâu tới 100 chỉ thêm những khung không
có bằng chứng gì, và Gemini bỏ qua chúng đúng như được dặn.

→ **Chốt `--top 50`.** Quét sâu hơn chỉ tốn thêm lượt gọi API.

#### Lọc cứng OCR toàn kho: kỹ thuật thứ **sáu** từ bài báo AIC'25 bị bác

`scripts/29_loc_cung_gemini.py` quét cả 165.259 khung có OCR và chèn khung
SigLIP2 **chưa từng trả về** lên đầu. Đây là A8.5 — cú pháp `/filter all
ocr{hidro}` thắng **3/5 ví dụ thực chiến** của đội AIC'25. Đo được: **−0,4**.

Và nó thua đúng theo cơ chế đã ghi sẵn trong docstring của chính nó:

> *28 chỉ xáo thứ tự nên xấu nhất là hoà; 29 **đẩy khung mới lên hạng 1**, tức
> đánh đổi hạng 1 của SigLIP2 lấy một phỏng đoán dựa trên chữ.*

Ranh giới nay đã rõ và **đo được cả hai phía**:

    XẾP LẠI trong bể kênh 1 đã chọn   -> +1,6 điểm  (3,8 -> 5,4)
    THAY THẾ bằng ứng viên mới         -> -0,4 điểm  (5,4 -> 5,0)

Cùng một kênh OCR, cùng một model Gemini, cùng một truy vấn. Khác nhau **duy
nhất ở chỗ kênh yếu được phép làm gì**.

> **Đây là phát biểu chính xác nhất tới giờ của bài học đắt nhất trong repo,**
> sau khi A27 đã sửa nó một lần:
>
> * ~~"kênh yếu làm tệ đi"~~ — sai, A27 bác
> * ~~"kênh yếu không được bỏ phiếu ngang hàng"~~ — đúng nhưng chưa đủ
> * **"kênh yếu chỉ được XẾP LẠI những gì kênh mạnh đã chọn, không được THÊM
>   hay THAY"** — khớp cả sáu phép đo

Chẩn đoán của chính script cũng nói trước điều này: trong 21 gói KIS/QA, Gemini
nêu được cụm chữ đặc trưng cho 4 gói, và với `query-p1-24-kis` nó nhận `'Team'`
làm token hiếm — một từ tiếng Anh thông thường lọt qua ngưỡng tần suất chỉ vì kho
OCR tiếng Việt ít gặp nó. Đúng kiểu lỗi `run.loc_cung` đã mắc với `Một`, `Trong`,
`Sau`, chỉ là ở tầng khác.

#### Còn dư địa ở đâu

Bão hoà nằm ở **lượng ứng viên được xem**, không nhất thiết ở **lượng bằng chứng
về mỗi ứng viên**. Hiện Gemini chỉ thấy 240 ký tự OCR + 240 ký tự ASR của đúng
một khung. Hướng tiếp: thêm **tiêu đề video** vào bằng chứng
(`28_xep_lai_bang_gemini.py --metadata`).

Vì sao đáng thử dù A12 đo kênh 2 chỉ được 0,0000: tiêu đề mô tả **cả video**, nên
nó vô dụng khi chọn giữa các khung TRONG một video — nhưng ở đây Gemini đang chọn
giữa các khung của **nhiều video khác nhau**, và *"video này nói về cái gì"* đúng
là thứ tiêu đề trả lời được.

**ĐÃ ĐO — KHÔNG ĐỔI GÌ (lượt #12: 5,4).** Thêm tiêu đề video đổi thứ tự ở 7/18
gói KIS mà điểm đứng yên. Nên giả thuyết *"cùng một dữ liệu, đổi vai trò thì đổi
giá trị"* — đúng với kênh OCR — **không mở rộng được sang kênh 2**.

> Ba lần liên tiếp không đổi điểm (top-100, metadata) trong khi hai lần đầu tăng
> mạnh: trần 5,4 **không phá được bằng cách cho Gemini nhiều bằng chứng hơn về
> cùng những ứng viên đó**. Phần còn thiếu nằm ở BỂ ỨNG VIÊN — tức ở kênh 1.

#### Xếp lại bằng ẢNH THẬT: 5,2 — và thủ phạm là **hiện vật của máy**

Cho Gemini nhìn ảnh thay vì đọc chữ (`scripts/30_xep_lai_thi_giac.py`) nhắm đúng
chỗ thủng — 8/18 gói KIS mà bộ xếp lại bằng chữ trả về rỗng đều là truy vấn thuần
thị giác. Kết quả: **5,4 → 5,2**.

Nhưng đừng vội khép hướng này, vì có một nghi phạm đo được:

| | |
| --- | --- |
| Máy dựng repo chỉ có ảnh L21/L22/L24/L27/L30 | **21% toàn kho** |
| Top-50 của 18 gói KIS | 537/900 khung có ảnh (60%), lệch mạnh: gói thì 50/50, gói thì 2/50 |
| Hạng 1 là khung **có ảnh**, sau khi xếp lại bằng ảnh | **13/18** (bản 5,4: 10/18) |

Bộ xếp lại **chỉ đẩy lên được thứ nó nhìn thấy**, nên nó đẩy 21% kho lên trên 79%
còn lại. Nó xếp theo *"máy nào đã tải nhóm nào"*, không theo chất lượng — đúng
họ hàng với bẫy `be_chung` ở A-mục dedup, nơi bể ứng viên hẹp hơn thắng vì lý do
không liên quan tới chất lượng.

→ **Chưa kết luận được.** Đo lại trên máy có đủ keyframe toàn kho — xem
[12_viec_cho_may_manh.md](12_viec_cho_may_manh.md) Việc 6.

---

### A29 — TRAKE đo được lần đầu ở n có ý nghĩa (23 câu, không phải 3), và một chốt "sửa" hoá ra làm tệ đi

Máy này (21/08) có sẵn cả `index/clip_siglip2.npy` toàn kho lẫn
`index/truy_van.npz` (Việc 1 đã xong) — lần đầu đo được TRAKE với ứng viên
kênh 1 thật, không phải kênh 3 (vốn ra 0,0000 cả ba biến thể, xem dưới) hay
n=3 (không đủ để đọc thắng-thua-hoà).

#### 1. `scripts/22_do_trake.py --cache index/truy_van.npz` — n=23

| Biến thể | ±2s | ±15s |
| --- | ---: | ---: |
| A_cu (video_du_chuoi + sorted() cũ, kênh 1 một mình) | 0,2464 | 0,3913 |
| B (xep_video_theo_chuoi + sorted() cũ, kênh 1 một mình) | 0,2493 | 0,3942 |
| C (xep_video_theo_chuoi + sorted() cũ, RRF kênh1+4) | 0,2435 | 0,3652 |
| **D_THAT (`run.dung_trake` thật — DP + xep_video_theo_chuoi)** | **0,2928** | **0,4696** |

`D_THAT` — đúng hàm đang chạy trong bài nộp thật — **thắng cả ba biến thể
giả lập**, xác nhận `run.dung_trake()` không làm tệ đi so với các cách lắp
ráp đơn giản hơn. Và **RRF kênh1+4 (C) lại thua kênh 1 một mình (B)** —
cùng quy luật đã đo ở A28: kênh yếu (objects) trộn ngang hàng vào TRAKE cũng
lỗ, giống hệt lý do RRF thô bị bác ở A14.

#### 2. `scripts/26_do_don_cuc_trake.py --cache index/truy_van.npz` — thêm cờ `--cache` để đo trên kênh 1

Script gốc (13/08) chỉ đo được trên kênh 3 vì máy lúc đó không nạp nổi
SigLIP2 — và ra **0,0000 cả ba biến thể**, không kết luận được gì (kênh 3
không tìm ra sự kiện TRAKE nào). Đã thêm `--cache` để dùng ứng viên kênh 1,
kết quả trên chính 23 câu TRAKE:

| Biến thể | ±2s | ±15s | dòng còn dồn cục |
| --- | ---: | ---: | ---: |
| xét TỔNG độ trải (bản đầu) | 0,2841 | 0,4435 | 62 |
| **xét TỪNG CẶP liền kề (mới)** | **0,2261** | **0,3275** | **0** |
| tắt hẳn việc rải đều | 0,2841 | 0,4435 | 87 |

Hai phát hiện:

* **"xét TỔNG" và "tắt hẳn" ra CÙNG một điểm** — đúng như docstring của
  script đã tiên đoán ("bản đầu không bắt được dồn cục MỘT PHẦN"): chốt cũ
  gần như không bao giờ nổ, tồn tại hay không cũng vậy.
* **"xét TỪNG CẶP" bắt sạch dồn cục (87→0 dòng) nhưng điểm TỆ ĐI** (−0,058 ở
  ±2s, −0,116 ở ±15s). Đây là kỹ thuật thứ **mười hai** bị phép đo bác, và
  cùng họ với mục 5 #5 (ràng buộc đa dạng): ép một ràng buộc "trông hợp lý"
  (sự kiện không nên dồn vào một chỗ) tưởng chỉ lọc rác, hoá ra lọc luôn cả
  những trường hợp ba sự kiện thật sự gần nhau về thời gian — **không thêm
  chốt này vào `run.py`.**

→ **Việc 2 của `12_viec_cho_may_manh.md` coi như xong.** Kết luận: giữ
nguyên `run.dung_trake()` hiện tại (đã là cấu hình tốt nhất trong 4 biến
thể), **không** thêm chốt chống dồn cục "từng cặp", **không** trộn kênh 4
vào TRAKE bằng RRF.

---

### A30 — RRF(SigLIP2, OCR/ASR mới) ở trọng số phụ **0,1**: cải thiện BỂ ỨNG VIÊN đầu tiên qua được ngưỡng ổn định kể từ A17

`scripts/26_do_rrf_siglip2_ocr.py --cache index/truy_van.npz`, 125 câu (97 dev +
28 câu đề mẫu). Khác mọi phép đo RRF trước đó (A14, mục 5 #2/#7 của
`11_tom_tat...md`): dùng OCR/ASR **sau A25** (93,2%/77,4% phủ, không phải
26,5% cũ), và **quét cả dải trọng số phụ** (1,0 → 0,1) thay vì chỉ thử 0,3.

| Cấu hình | ±2s | ±15s | Kết luận |
| --- | ---: | ---: | --- |
| kênh 1 SigLIP2 (mốc, A17) | 0,2757 | 0,4043 | — |
| kênh 3 OCR/ASR mới một mình | 0,1205 | 0,1589 | yếu hơn hẳn kênh 1 |
| RRF trọng số 1,0:1,0 | 0,2976 | 0,4021 | ❌ ĐẢO DẤU |
| RRF trọng số phụ 0,5 / 0,3 / 0,2 | 0,294x / 0,280x / 0,283x | 0,413x / 0,412x / 0,414x | 🟡 YẾU |
| **RRF trọng số phụ 0,1** | **0,2843 (+0,0085)** | **0,4149 (+0,0107)** | **✅ ỔN ĐỊNH** |

Ở trọng số 0,1: **vượt nhiễu ở cả hai mức dung sai** (ngưỡng 0,0084/0,0105),
cùng dấu, 7 thắng–1 thua–117 hoà (±2s). Chồng lấn giữa hai kênh: chỉ
**70/125 câu (56%)** có khung chung — đúng cơ chế RRF cần để cộng hưởng.

**Vì sao khác kết luận cũ ("dìm trọng số kênh phụ" bị bác ở mục 5 #7):** phép
đo cũ dìm trọng số của **hai kênh ngang tầm nhau** (RRF giữa các kênh yếu như
nhau) — dìm không giúp gì vì không kênh nào đủ mạnh để dẫn dắt. Ở đây kênh 1
mạnh hơn kênh 3 tới **2,3 lần** (0,2757 so với 0,1205); dìm cực nhẹ (0,1)
biến kênh 3 thành phiếu "tie-break" chỉ có tiếng nói khi kênh 1 mơ hồ, không
đủ mạnh để kéo sai — khác cơ chế với RRF ngang hàng.

**Hiệu ứng nhỏ nhưng THẬT**, và đáng chú ý vì đây là kỹ thuật **cải thiện bể
ứng viên** (không phải xếp lại) đầu tiên vượt ngưỡng ổn định kể từ khi kênh 1
sống lại (A17) — đúng chỗ ba lần thử trước (A28: top-100, thêm tiêu đề video)
đã dừng lại vì "không phá được trần bằng cách xếp lại nhiều hơn". Nên thử kết
hợp: `RRF(kênh1, kênh3, w=0,1)` làm bể ứng viên đầu vào, rồi Gemini xếp lại
top-50 như cấu hình hiện tại — **CHƯA ĐO** phần kết hợp này, cần Việc 3.

---

### A31 — Việc 3 đo xong: kỹ thuật ăn +1,6 điểm leaderboard chỉ được **+0,01 YẾU** trên dev — tập dev **MÙ LẦN THỨ BA**

`scripts/31_do_xep_lai_tren_dev.py --cache index/truy_van.npz` (script mới,
tái dùng đúng hàm `xep_lai()` của `scripts/28_xep_lai_bang_gemini.py` qua
import động — không đo một bản chép lệch). 60 câu KIS của tập dev, model
`gemini-3.1-flash-lite`, top-50, đúng cấu hình đang chấm 5,4 trên leaderboard.

| Cấu hình | ±2s | ±15s |
| --- | ---: | ---: |
| SigLIP2 top-100 (mốc nền) | 0,4167 | 0,5700 |
| + Gemini xếp lại top-50 | 0,4267 | 0,5833 |
| **hiệu** | **+0,0100** | **+0,0133** |

Cùng dấu ở cả hai mức, nhưng **KHÔNG vượt nhiễu** (ngưỡng 0,0292/0,0354) —
**🟡 YẾU**, không phải ⚪ KHÔNG ĐỔI GÌ hay ✅ ỔN ĐỊNH. 5 thắng–5 thua–50 hoà
(±2s): đúng số câu bị "đẩy lên" (`chon` khác rỗng) ở khoảng 30-35/60 câu,
nhưng phần lớn không đổi thứ hạng đủ để đổi điểm R@k.

**Đối chiếu với leaderboard (A28):** top-50 đưa điểm thật từ 3,8 lên **5,4**
— tức **+1,6 điểm trên thang R@{1,5,20,50,100} trung bình 24 gói**. Trên dev,
cùng kỹ thuật chỉ **+0,01, không đáng tin**. Đây là **lần mù thứ ba** của tập
dev với một cải tiến thật trên leaderboard (sau A19, A20) — nhưng khác hai
lần trước ở chỗ **lần này chênh lệch cực lớn** (160 lần), không phải chỉ
"không thấy gì".

**Ba cách đọc kết quả này, không cách nào loại được bằng phép đo hiện có:**

1. **Tập dev không đại diện cho phân bố đề thi thật** — 60 câu KIS dev do
   chính đội soạn (biết trước đáp án khi viết câu) có thể thiên về loại câu
   mà OCR/ASR ít giúp được, khác đề thi thật nơi Gemini đẩy lên tới 47/100
   dòng có OCR khớp.
2. **+1,6 trên leaderboard công khai (50% đáp án, n nhỏ) có phần là nhiễu** —
   leaderboard không báo sai số chuẩn, và một lượt nộp là một mẫu, không phải
   trung bình nhiều lần.
3. Cả hai đúng một phần.

**KHÔNG khuyến nghị rút kỹ thuật này khỏi bài nộp** — nó vẫn là cấu hình tốt
nhất đã biết trên leaderboard thật, và +0,01 trên dev là **cùng dấu**, không
phải ngược dấu. Nhưng **mọi quyết định TIẾP THEO dựa trên dev cho khâu xếp
lại này cần thêm dè dặt** — dev không đủ nhạy để phân biệt các biến thể nhỏ
ở đây. Ưu tiên đo trên **bể ứng viên** (nơi A17, A30 vẫn cho tín hiệu rõ trên
dev) hơn là tinh chỉnh tiếp khâu xếp lại.

#### Bổ sung cùng ngày: `RRF(1,3,w=0,1)` (A30) cộng dồn với xếp lại — vẫn cùng dấu, vẫn chưa đủ mạnh

`scripts/31_do_xep_lai_tren_dev.py --cache index/truy_van.npz --rrf-w 0.1`,
cùng 60 câu KIS, ~120 lượt gọi Gemini (một lượt cho bể SigLIP2, một lượt
riêng cho bể RRF — hai bể khác ứng viên nên không dùng lại được lượt gọi):

| Cấu hình | ±2s | ±15s |
| --- | ---: | ---: |
| SigLIP2 top-100 (mốc nền) | 0,4167 | 0,5700 |
| + Gemini xếp lại top-50 | 0,4267 | 0,5833 |
| **RRF(1,3,w=0,1) + Gemini xếp lại top-50** | **0,4367** | **0,5833** |

So với mốc nền: +0,0200/+0,0133, vẫn **🟡 YẾU** (ngưỡng nhiễu 0,0280/0,0328)
— nhưng ở ±2s đã nhích gần ngưỡng hơn hẳn so với xếp lại một mình (+0,01).
So trực tiếp hai cấu hình xếp lại: bể RRF hơn bể SigLIP2 thuần **+0,01 ở
±2s, hoà ở ±15s** — cùng hướng với A30 (RRF giúp bể ứng viên), không có dấu
hiệu triệt tiêu lẫn nhau. **Kết luận: hai kỹ thuật KHÔNG loại trừ nhau, có
thể dùng cùng lúc**, nhưng cần nhiều câu hơn 60 để tách khỏi nhiễu — không
phải ưu tiên đo tiếp ngay, ghi nhận làm cấu hình ứng viên cho lượt nộp sau.

---

### A32 — Việc 4 đo xong: phân rã truy vấn bằng LLM làm **TỆ ĐI ổn định** trên dev — kỹ thuật thứ mười ba bị bác, nhưng có nghi phạm là chính tập dev

`scripts/32_do_phan_ra_llm.py --cache index/truy_van.npz --lam 0.15 --fp16`
(script mới). Gemini viết lại mỗi câu KIS dev thành 3 mệnh đề cố định góc
nhìn (cảnh tổng thể / hành động / vật cận cảnh), mã hoá cả ba, gộp điểm
`max + 0,15·tổng`. 60/60 câu phân rã thành công.

| Cấu hình | ±2s | ±15s |
| --- | ---: | ---: |
| `tach_truy_van` hiện tại (mốc nền) | 0,4167 | 0,5700 |
| LLM phân rã 3 mệnh đề, λ=0,15 | 0,3633 | 0,4967 |
| **hiệu** | **−0,0533** | **−0,0733** |

Cùng dấu ở cả hai mức, vượt nhiễu ở ±15s (ngưỡng 0,0616) — **✅ ỔN ĐỊNH, và
theo chiều NGƯỢC kỳ vọng**. 8 thắng–18 thua–34 hoà ở ±2s.

**⚠️ Nhưng đọc kết quả này cùng với cảnh báo đã ghi sẵn trong docstring của
chính script:** câu dev tự soạn dài ~15-20 từ, dưới trần token của SigLIP2
(64 token) — nên `tach_truy_van` trên dev **hầu như không cắt gì**, mốc nền
ở đây gần như nguyên văn câu gốc. Kỹ thuật này nhắm đúng vào vấn đề của **câu
63 từ** (đề thật), thứ tập dev không có mẫu nào đại diện. Vậy phép đo này trả
lời được câu "phân rã có hại cho câu NGẮN không" (có, rõ ràng), nhưng
**không trả lời được câu đang thật sự cần hỏi: phân rã có giúp câu DÀI
không** — đó là lần mù thứ **tư** của tập dev trong một ngày (A31 là lần ba),
và lần này không phải vì thiếu nhạy mà vì **thiếu đúng loại câu hỏi cần đo**.

Cơ chế hại trên câu ngắn cũng hợp lý và đáng ghi: câu gốc đã là một mô tả cụ
thể, sắc nét; 3 mệnh đề LLM viết lại — nhất là mệnh đề "cảnh tổng thể" — có
xu hướng khái quát hoá thành cụm chung chung lặp lại giữa nhiều câu khác hẳn
nhau (`"Không gian bàn ăn..."`, `"Không gian làm việc..."`, `"Không gian
phòng khám..."` — 3/12 câu L26 đều mở đầu giống hệt nhau), pha loãng tín hiệu
sắc của câu gốc bằng phép cộng `λ·tổng`.

> **Khuyến nghị:** không dùng cấu hình này cho câu NGẮN — thêm vào danh sách
> kỹ thuật bị bác. Nhưng **không kết luận đóng hẳn hướng này** như đã làm với
> các kỹ thuật khác — muốn biết thật, phải đo trên chính 24 câu đề mẫu (dài
> đúng kiểu đề thật) chứ không phải tập dev. Đó là việc còn lại nếu muốn theo
> tiếp hướng này, không phải ưu tiên ngay bây giờ.

---

### A33 — Việc 6 đo xong: xếp lại bằng ẢNH THẬT, với đủ 100% keyframe — kết quả **TỐT NHẤT phiên làm việc 21/08**, xác nhận nghi phạm "hiện vật của máy" ở A28 là đúng

`scripts/33_do_xep_lai_thi_giac_tren_dev.py --cache index/truy_van.npz` (script
mới, tái dùng đúng hàm `xep_lai()` của `scripts/30_xep_lai_thi_giac.py`). Máy
này có **177.321/177.321 ảnh (100% kho)** — khác hẳn máy dựng A28 chỉ có 21%.
60 câu KIS, gửi tối đa 20 ảnh/câu cho `gemini-3.1-flash-lite`.

| Cấu hình | ±2s | ±15s |
| --- | ---: | ---: |
| SigLIP2 top-100 (mốc nền) | 0,4167 | 0,5700 |
| **+ Gemini xếp lại (ẢNH) top-50** | **0,4467** | **0,5967** |
| **hiệu** | **+0,0300** | **+0,0267** |

Vượt nhiễu ở ±2s (ngưỡng 0,0229), cùng dấu cả hai mức → **✅ ỔN ĐỊNH**. **9
thắng – 1 thua – 50 hoà** ở ±2s — tỷ lệ thắng/thua tốt nhất trong TOÀN BỘ các
phép so theo cặp đo được hôm nay (so với 5-5 của xếp lại bằng chữ ở A31).

**Đối chiếu trực tiếp với A28:** trên máy chỉ có 21% ảnh, kỹ thuật y hệt này
làm bài nộp **tệ đi** (5,4→5,2). Trên máy đủ 100% ảnh, cùng kỹ thuật là
**cấu hình xếp lại tốt nhất đo được trên dev cho tới nay** — tốt hơn cả xếp
lại bằng chữ (A31: +0,01) lẫn RRF+chữ (A31: +0,02). **Xác nhận dứt điểm nghi
ngờ đã nêu ở A28: vấn đề là do máy thiếu ảnh, không phải do kỹ thuật.** Đây
là câu trả lời rõ ràng nhất — không mù, không mơ hồ — trong cả 4 phép đo trên
dev hôm nay (so với A31, A31-combo, A32 đều yếu hoặc mù).

> **Khuyến nghị mạnh: đưa `30_xep_lai_thi_giac.py` vào bài nộp chính**, chạy
> trên máy có đủ ảnh (máy này, hoặc bất kỳ máy nào sync đủ `index/`/keyframe
> toàn kho qua Drive). Đáng thử kết hợp với A30/A31: `RRF(1,3,w=0,1)` làm bể
> ứng viên → xếp lại bằng ẢNH top-50 (chưa đo phần kết hợp ba tầng này).

---

### A34 — Nộp thật tổ hợp A30+A33 hôm nay: **~5,0, TỆ HƠN mốc 5,4** — cái giá của việc bỏ qua cảnh báo A19/A20

Nộp `RRF(1,3,w=0,1)` (A30) làm bể ứng viên + xếp lại bằng ẢNH top-50 (A33) lên
leaderboard thực hành. Kết quả **~5 điểm — tệ hơn mốc 5,4** đã có từ trước
(A27/A28), dù cả hai kỹ thuật đều đo **✅ ổn định dương** trên tập dev cùng
ngày. Đã lập tức nộp lại bản AN TOÀN (đúng cấu hình A27/A28: SigLIP2 + xếp
lại bằng CHỮ top-50) để phục hồi mốc nền.

**Nguyên nhân khoanh được — lần thứ BA dính đúng lỗi đã cảnh báo ở A19/A20:**
`scripts/26_do_rrf_siglip2_ocr.py` (đo ra A30) gọi `k1.tim(c.cau_hoi, ...)` —
**chuỗi câu NGUYÊN VĂN, không qua `tach_truy_van()`**. Trên tập dev (48/60 câu
KIS < 40 từ) điều này gần như không khác gì so với có tách, vì `tach_truy_van`
là NO-OP ở độ dài đó. Nhưng bài nộp thật chạy qua `run.py --hop-nhat`, nơi
**cả kênh 1 lẫn kênh 3 đều được tách câu trước khi đưa vào RRF** — một đường
tính KHÁC HẲN, chưa từng được đo. Đúng công thức A19/A20 đã ghi: *"câu dev tự
soạn dài 22 từ, đề thật dài 63 từ — ta đang đo trên một phân bố khác với phân
bố đi thi"*.

**Đo bổ sung ngay sau đó, đúng đường tính CÓ tách câu, trên 12 câu KIS dev đủ
dài (≥40 từ, mô phỏng đề thật):**

| Cấu hình | ±2s | ±15s |
| --- | ---: | ---: |
| SigLIP2 (có tách câu, mốc nền) | 0,2833 | 0,4500 |
| RRF(1,3,w=0,1) CÓ tách câu | 0,2833 (hoà 0-0-12) | 0,4667 |
| RRF(1,3,w=0,3) CÓ tách câu | 0,2833 (hoà 0-0-12) | 0,4833 |

Không âm — nhưng **n=12 quá nhỏ để kết luận gì**, và **0-0-12 ở ±2s nghĩa là
RRF không đổi hạng-1 của BẤT KỲ câu nào trong 12 câu** khi có tách câu, khác
hẳn tín hiệu dương rõ đo được ở A30 (n=125, không tách). Chưa loại được RRF
là thủ phạm, cũng chưa kết tội được nó — **n=12 không đủ**.

#### Cập nhật cùng ngày: soạn thêm 20 câu dev DÀI, đo lại với n=32 — **kết luận rõ: RRF trung tính-tới-âm trên câu dài**

Soạn 20 câu KIS dev mới dạng phóng sự/chương trình dài 55-70 từ (paraphrase từ
20 câu ngắn hiện có, GIỮ NGUYÊN `row_id_dung` — khác bộ `-101/102/103` cũ ở
chỗ **không chép chữ OCR của khung đáp án** vào bối cảnh, để không làm giả
tín hiệu kênh 3). Gộp vào `tap_dev.jsonl` qua đúng quy trình (`--gop`,
`--no-cum`, `--kiem`) — tổng 163 câu, 32/80 câu KIS nay ≥ 40 từ (trước đó
12/60). Đo lại RRF trên đúng 32 câu này, cùng đường tính CÓ tách câu:

| Cấu hình | ±2s | ±15s |
| --- | ---: | ---: |
| SigLIP2 (có tách câu, mốc nền) | 0,3063 | 0,4062 |
| RRF(1,3,w=0,05) | 0,3063 (0-0-32) | 0,4062 (0-0-32) |
| RRF(1,3,w=0,1) | 0,3063 (0-0-32) | 0,4062 (1-1-30) |
| RRF(1,3,w=0,3) | 0,2937 (0-2-30) | 0,4000 (2-3-27) |

**⚪ w=0,05 và w=0,1: KHÔNG ĐỔI GÌ** — hoàn toàn trung tính trên câu dài, khác
hẳn tín hiệu dương ổn định đo được trên câu ngắn ở A30 (n=125). **w=0,3: 🟡
YẾU nhưng theo chiều ÂM.** n=32 đủ lớn hơn hẳn để đọc xu hướng: **RRF(1,3)
không giúp gì trên câu dài kiểu đề thật, và có xu hướng hơi lỗ khi trọng số
tăng** — khác hẳn hiệu ứng dương đo được trên câu ngắn.

**Cơ chế hợp lý giải thích khác biệt:** khi câu bị tách thành 2-5 mệnh đề,
kênh 3 (BM25 trên từng mệnh đề rồi lấy max) trở nên nhiễu hơn — mỗi mệnh đề
ngắn hơn cả câu gốc, ít từ hiếm/đặc trưng để BM25 bám vào, nên ứng viên kênh 3
kém tin cậy hơn hẳn so với khi được tính trên nguyên văn câu ngắn dev. RRF vẫn
cộng nó vào bằng trọng số cố định, không biết chất lượng kênh 3 đã tụt.

> **✅ KẾT LUẬN ĐỦ TIN CẬY: không dùng `--hop-nhat` (RRF kênh 1+3) trong bài
> nộp thật cho tới khi có bằng chứng khác** — dev nay đã đủ nhạy (n=32, câu
> đúng độ dài đề thật) để nói nó không lãi, đúng hướng với việc điểm thật tụt
> ở A34. Bản AN TOÀN đã nộp lại (không có RRF) là lựa chọn đúng.

**Nghi phạm thứ hai, chưa đo được:** xếp lại bằng ẢNH (A33) chỉ gửi **tối đa
20/50 ảnh** cho Gemini xem. Nếu RRF(1,3) xếp một ứng viên đúng từ hạng 5 (nằm
trong 20 ảnh được xem) xuống hạng 25 (ngoài 20 ảnh), xếp-lại-bằng-ảnh **không
còn cơ hội thấy nó để đẩy lên** — trong khi A33 đo riêng lẻ chỉ dùng bể
SigLIP2 THUẦN, không hề có bước xáo trộn RRF trước đó. Tổ hợp ba tầng
(RRF → chọn 20 ảnh đầu → xếp lại) **chưa từng được đo trên dev**, đúng như đã
tự cảnh báo lúc đề xuất (cuối mục A33).

**Bài học rút ra, khác các bài học "kỹ thuật bị bác" trước đó:** đây không
phải một kỹ thuật sai — cả A30 lẫn A33 đều là số đo thật, không bịa. Vấn đề
là **quy trình kiểm chứng thiếu một bước**: trước khi ráp nhiều kỹ thuật đã
đo RIÊNG LẺ thành một pipeline rồi nộp thật, phải đo đúng TỔ HỢP đó, và đo
trên **đúng độ dài câu** của đề thật — không phải suy luận "cả hai đều dương
thì cộng lại chắc cũng dương".

> **Quy tắc mới cho mọi lần nộp thật kể từ đây:** không ráp nhiều kỹ thuật
> mới đo riêng lẻ trong CÙNG một lượt nộp. Đổi một thứ, nộp, đối chiếu — đúng
> kỷ luật "chỉ đổi một thứ mỗi lần" mà PHẦN A đã nói từ đầu, nhưng lần này áp
> dụng cho chính **lượt nộp thật**, không chỉ cho phép đo trên dev.

**Việc còn thiếu, đã bị chỉ ra từ A20 nhưng chưa ai làm — ĐÃ LÀM MỘT PHẦN
(xem A35):** tập dev cần thêm câu DÀI như đề thật. Đã soạn thêm 20 câu, nâng
từ 12/60 lên 32/80 câu KIS ≥ 40 từ.

---

### A35 — Sửa `run.py`: RRF chỉ bật cho câu NGẮN, tự tắt cho câu DÀI (`--hop-nhat-chi-cau-ngan`)

Theo đúng đề xuất của người dùng sau sự cố A34: thay vì bật/tắt RRF đồng loạt,
để `run.py` **tự quyết theo từng câu** — câu không bị `tach_truy_van()` cắt
thì bật RRF(1,3,w=0,1) như A30 đã đo có lãi; câu bị cắt (dài, kiểu đề thật)
thì giữ nguyên kênh 1 một mình, đúng như A34 đo được là an toàn hơn.

Đo trên **toàn bộ 80 câu KIS dev** (48 ngắn + 32 dài), gọi thẳng
`quet_anh`/`quet_van_ban` của chính `run.py` — không viết lại logic:

| Cấu hình | ±2s | ±15s |
| --- | ---: | ---: |
| SigLIP2 một mình (mốc nền) | 0,3925 | 0,5225 |
| RRF(1,3,w=0,1) LUÔN LUÔN | 0,4025 (🟡 yếu) | 0,5300 |
| **RRF(1,3,w=0,1) THÍCH NGHI (`--hop-nhat-chi-cau-ngan`)** | 0,4000 (🟡 yếu) | 0,5275 |

Hai cấu hình RRF gần như không khác nhau trên mẫu này — hợp lý, vì ở đúng
w=0,1 (không phải 0,3), hiệu ứng trên câu dài đã đo ở A34 là ⚪ trung tính,
không âm rõ, nên "thích nghi" và "luôn luôn" hội tụ gần nhau ở trọng số này.

**Khuyến nghị dùng bản THÍCH NGHI dù số đo chưa phân biệt được rõ hai bản**:
nó **an toàn hơn về cấu trúc**, không phải vì số đo hôm nay cao hơn. Đề thi
thật có thể dài/nhiều mệnh đề hơn 32 câu dev hiện có, và nếu sau này ai đó
tăng `--trong-so-phu` (đã đo 0,3 là âm ở A34), bản thích nghi tự động không
đụng câu dài nên không thể lỗ theo hướng đó — bản "luôn luôn" thì có thể.

    python src/run.py --de <đề> --hop-nhat --bo-metadata --trong-so-phu 0.1 \
        --hop-nhat-chi-cau-ngan ...

**Chưa nộp thật cấu hình này** — theo đúng quy tắc mới ở A34 (đổi một thứ,
nộp, đối chiếu), đây sẽ là ứng viên cho MỘT lượt nộp riêng, không ráp chung
với xếp lại bằng ảnh (A33) hay bất kỳ thứ gì khác trong cùng lượt.

---

### A36 — Bản AN TOÀN nộp thật: **6,2 — mốc cao nhất từ trước tới giờ**, +0,8 so với 5,4 cũ

Nộp `submission_antoan_vadap.zip` (A34: đúng công thức A27/A28 — SigLIP2 +
Gemini xếp lại CHỮ top-50 — cộng 3 đáp án Q&A đã xác minh bằng mắt: **Giang
Ly**, câu đối Nguyễn Trung Trực, **Bánh ít trần**). Kết quả **6,2**, vượt mốc
5,4 cũ.

**Nguồn tăng điểm khớp đúng dự đoán:** 3 gói Q&A trước đây nộp đáp án đoán bừa
`5`, `2`, `10` — chắc chắn 0 điểm dù khung có đúng hay không (PHẦN C mục 4:
Q&A cần ĐÚNG CẢ khung lẫn `answer`). Vá đáp án đúng bằng mắt là cách tăng
điểm **chắc chắn, không rủi ro** duy nhất đo được hôm nay — khác các kỹ thuật
truy hồi khác vốn cần đo cẩn thận vì có thể phản tác dụng (A34).

**Ghi chú sửa lại A26:** báo cáo cũ (13/08) từng kết luận *"sửa đáp án Q&A
không đổi điểm public vì 3 gói đó không nằm trong 50% được chấm public"* —
kết quả 6,2 hôm nay **mâu thuẫn với kết luận đó**. Có thể do: (a) tập 50%
được chấm không cố định giữa các lượt, hoặc (b) khung được đẩy lên hạng 1 lần
này khác/tốt hơn lần đo cũ. Chưa tách được phần tăng do Q&A khỏi phần tăng do
biến động ngẫu nhiên của khung — nhưng dù nguyên nhân là gì, **kết quả cuối
vẫn tốt hơn**, không cần rút lại.

**Mốc nền mới cho mọi so sánh từ nay: 6,2**, không phải 5,4.

---

### A37 — Xếp lại bằng ẢNH một mình nộp thật: **4,2 — TỆ NHẤT trong ba cấu hình đã nộp hôm nay**, dù trên dev nó là tín hiệu SẠCH NHẤT

Nộp `sub_anh_rieng_vadap.zip` — SigLIP2 một mình + Gemini xếp lại bằng ẢNH
top-50 (không kèm RRF, cô lập đúng MỘT thay đổi so với mốc 6,2: đổi khâu xếp
lại từ CHỮ sang ẢNH). Kết quả: **4,2**.

**Ba điểm thật đã có trong ngày, xếp theo thứ tự:**

| Cấu hình | Điểm thật | Điểm dev (so với mốc SigLIP2 riêng) |
| --- | ---: | --- |
| SigLIP2 + xếp lại CHỮ + QA đã vá (A36) | **6,2** | +0,01 🟡 YẾU (A31) |
| RRF(1,3,w=0,1) + xếp lại ẢNH (A34) | ~5,0 | ảnh: +0,03 ✅; RRF+ảnh: chưa đo riêng |
| SigLIP2 + xếp lại ẢNH một mình (A37) | **4,2** | **+0,03 ✅ ỔN ĐỊNH — tín hiệu SẠCH NHẤT hôm nay** |

**Nghịch lý cần ghi nhận thẳng:** cấu hình có tín hiệu dev đẹp nhất (9 thắng–1
thua–50 hoà, vượt nhiễu rõ) lại là cấu hình **tệ nhất trên leaderboard thật**.
Đây là bằng chứng mạnh nhất từ trước tới giờ cho một mẫu hình đã lặp lại 4 lần
trong ngày (A31 chữ yếu, A34 RRF hại câu dài, A32 phân rã LLM hại, nay A37):

> **Tập dev đáng tin cho quyết định ở tầng BỂ ỨNG VIÊN (A17 SigLIP2, A30 RRF
> câu ngắn) nhưng KHÔNG đáng tin cho quyết định ở tầng XẾP LẠI — bất kể tín
> hiệu dev mạnh hay yếu, dương hay khiến tưởng chắc chắn.** Xếp lại bằng CHỮ
> là kỹ thuật DUY NHẤT ở tầng này có track record thật trên leaderboard (ba
> lần độc lập: A27, A28, A36) — không phải vì đo trên dev tốt nhất, mà vì đã
> được XÁC MINH TRÊN CHÍNH SÂN THẬT nhiều lần.

**Giả thuyết cơ chế (chưa kiểm chứng):** câu đề thật dài, văn phong tường
thuật/phóng sự — Gemini phải khớp một mô tả nhiều câu với ảnh thumbnail
512px, khác hẳn câu dev ngắn-gọn-trực-diện do chính người trong nhóm soạn ra
dựa trên đúng khung đã xem. Khớp CHỮ (OCR/ASR) có thể ổn định hơn khớp Ý qua
ảnh khi mô tả dài và trừu tượng hơn.

> **Khuyến nghị: TẠM DỪNG đầu tư thêm vào xếp lại bằng ảnh** cho tới khi có
> cách đo đáng tin hơn (ví dụ: đo trực tiếp bằng các lượt nộp thật nhỏ, có
> kiểm soát, thay vì dựa vào dev). **Xếp lại bằng CHỮ (script 28, top-50)
> vẫn là lựa chọn mặc định** cho mọi bài nộp từ nay — đã 3/3 lần đúng.

---

### A38 — Ghi chép suy nghĩ của người soát: **đề được viết từ VIDEO, hệ thống lại tìm trên KEYFRAME**

Nhóm soát tay 25 gói đề sơ tuyển đợt 1 trên máy mạnh (SigLIP2) rồi kể lại từng
ca. Đọc hết thì thấy **cùng một nguyên nhân gốc** ở gần như mọi câu, và nó không
phải lỗi của model.

#### Bằng chứng — bảy ca, cùng một dạng

| Gói | Hệ thống trả về | Người phải làm gì thêm |
| --- | --- | --- |
| **p1-4-kis** | kf183: *"đàn sư tử… bảng London Zoo"* — **chỉ mệnh đề 1** | Mệnh đề 2 (*"hai nhân viên áo xanh cân thú"*) nằm ở kf186/187. Mà **mỗi kf chỉ có MỘT nhân viên**; cộng hai kf lại mới đủ "hai người" |
| **p1-6-kis** | kf có mỏ đá quý | Nội dung **có thật trong video nhưng KHÔNG nằm trong keyframe nào**. Lấy dải kf lân cận thì vẫn trúng |
| **p1-21-kis** | 3 kf mới đủ ngữ nghĩa | Ba ổ bánh mì chỉ thoáng hiện ở **một khung chuyển cảnh** |
| **p1-9-qa** | kf có xe lội nước, đúng màu | **Đáp án nằm ở kf SAU đó** — chỉ lúc ấy mới thấy cây cầu và hai biển hiệu |
| **p1-2-kis** | bám *"công trình thuỷ lợi"* + *"bản đồ"* | Chốt đúng là **"con đập"** — có đập thì hiển nhiên có thuỷ điện. Và *"trời mưa"* gần như vô vọng: video chất lượng thấp, mưa nhỏ không thấy hạt |
| **p1-12-kis** | OCR bắt *"mazut"* (từ hiếm) | Nhưng khung **mơ hồ**: có xe ôm công nghệ, mà đông hơn 4 người, và **không ai rẽ trái vào khung** như đề tả |
| **p1-15-qa** | đúng khung cần tìm | **Không đếm nổi** số tâm chấn cấp 4 — Gemini bản web cũng không chốt được con số |

#### Quy luật rút ra

> **Đề thi được viết bằng cách XEM VIDEO, còn hệ thống thì tìm trên KEYFRAME.**
> Người viết đề mô tả một **quãng thời gian**; ta lại đi khớp từng **ảnh tĩnh
> rời rạc**. Một truy vấn 63 từ / 2,4 mệnh đề gần như **không bao giờ** có đủ
> ngữ nghĩa trong một keyframe duy nhất.

Ba hệ quả, và cả ba đều đo được:

**1. Nộp một DẢI luôn tốt hơn nộp một khung.** BTC chấm theo cửa sổ 4 giây–5
phút (A9), nên khung lân cận vẫn được tính đúng. Ca `p1-6-kis` là bằng chứng
sắc nhất: nội dung **không nằm trong keyframe nào cả**, nhưng lấy dải quanh đó
vẫn trúng. Việc này đã làm ở `33_trai_dai_khung.py` và **hai bản nộp đều 11
điểm** — bản trải dài không thua bản gốc.

**2. Nên xếp hạng CỬA SỔ, không xếp hạng khung.** Đây là thứ chưa làm. Hiện
`dense.tim` lấy **điểm cao nhất trên từng keyframe** qua các mệnh đề — tức nó
thưởng cho khung khớp MỘT mệnh đề thật mạnh. Nhưng ca `p1-4` cho thấy đáp án
đúng là khung mà **cả cụm quanh nó** phủ được nhiều mệnh đề:

    điểm(cửa sổ W) = Σ_j  max_{khung f ∈ W}  cos(f, mệnh_đề_j)

Cộng theo mệnh đề, lấy max trong cửa sổ. Khung nào phủ được nhiều mệnh đề khác
nhau **trong vùng lân cận của nó** thì thắng. Đúng cách người soát đã làm bằng
tay: *"cộng hai kf lại thì đúng bằng hai người"*.

**3. Q&A phải ĐI BỘ TIẾP sau khi tìm ra cảnh.** `p1-9-qa` nói thẳng quy trình:
tìm khung có *xe lội nước đúng màu* → rồi **duyệt các khung sau** để tìm *cây
cầu và biển hiệu* — thứ thật sự chứa đáp án. Cảnh và đáp án **không cùng một
khung**.

> ⚠️ Đây KHÔNG mâu thuẫn với A18 (chèn khung lân cận vào danh sách nộp làm tệ
> đi). A18 chèn lân cận **thay chỗ** ứng viên khác trong 100 dòng; ở đây lân cận
> được dùng để **chấm điểm** và **truy tìm đáp án**, không tiêu chỗ nộp nào.
> Cùng một dữ liệu, đổi vai trò thì đổi giá trị — như kênh OCR ở A27/A28.

#### Hai ca KHÔNG thuộc quy luật trên, và cũng đáng ghi

**`p1-2-kis` — hệ thống bám sai danh từ.** Đề nói *"công trình thuỷ lợi"*, người
soát chốt bằng **"con đập"**: có đập thì hiển nhiên có thuỷ điện, việc còn lại
chỉ là tìm bản đồ. Đây là **suy luận bắc cầu bằng tri thức thế giới**, thứ mà
truy hồi vector không làm được nhưng LLM làm được — và là chỗ duy nhất trong cả
25 ca mà "phân rã truy vấn bằng LLM" (A32, đang bị bác) có thể cứu được: không
phải cắt câu, mà **thay danh từ trừu tượng bằng vật thể nhìn thấy được**.

**`p1-15-qa` — trần của việc ĐẾM.** Hệ thống tìm đúng khung, nhưng không đếm nổi
số tâm chấn cấp 4; Gemini bản web cũng không. Ghi lại để đừng đầu tư tiếp: đây
là giới hạn của model hiện tại, không phải lỗi truy hồi. Trùng với A26 (câu đếm
và câu màu sắc trượt 100% qua OCR).

#### Một ghi nhận về chính luồng làm việc

Với `p1-17-qa` (tên đèo), nhóm nhận xét luồng đã chạy là đúng: bám **từ hiếm**
(*"sạt lở" + "đèo"*) → khoanh còn 10 video → đọc lời bình → xác minh bằng ảnh.
Bài học nhóm rút: **chuỗi suy luận NGẮN thì ít lệch**; suy nghĩ dài dòng dễ trôi
khỏi đáp án rồi phải quay lui.

Điều đó nói ngược lại một phần với đề xuất "cho model suy luận nhiều bước hơn":
với truy vấn có **mỏ neo văn bản** (tên riêng, địa danh, từ hiếm), đường ngắn
nhất là bám thẳng mỏ neo đó. Suy luận nhiều bước chỉ đáng dùng khi **không có mỏ
neo nào** — đúng nhóm câu thuần thị giác mà A26 đã khoanh.

### A39 — TRAKE: thuật toán biết THỨ TỰ nhưng không biết KHOẢNG CÁCH, mà khoảng cách thì đo được

Nhóm báo *"câu TRAKE thì hệ thống bên máy mạnh không làm được"*. Đây là chỗ
đáng đổ công sức "mô phỏng luồng suy nghĩ con người" nhất — và cũng là chỗ
**duy nhất** trong hệ thống mà ý tưởng đó không trùng với thứ đã bị đo bác.

Lý do đơn giản: bốn trong năm nhóm kỹ thuật được đề xuất (CoT, ToT, region,
self-correction) đều nằm ở **khâu xếp lại**, tức chọn tốt hơn trong danh sách
đã có. Khâu đó đã đo khá kỹ: xếp lại bằng chữ **+1,6 điểm**, còn mọi nỗ lực làm
nó tinh vi hơn (top-100 A28, tiêu đề video, xếp lại bằng ảnh A33/A37) đều
**0 hoặc âm**. Trần không nằm ở chỗ suy luận sâu hơn về cùng những ứng viên đó.

Còn TRAKE thì **bản chất là bài toán nhiều bước có thứ tự**, và đó đúng dạng
bài mà suy luận từng bước hơn hẳn một phép truy hồi một phát.

#### Quy ý tưởng đó ra số thì nó là cái này

Người soát **không** tìm sự kiện 2 trên cả kho. Họ tìm sự kiện 1, rồi **đi tiếp
về phía trước từ đó một quãng hợp lý**. Bản đang chạy (`run.dong_hang_dp`) chỉ
ràng buộc `khung(i) < khung(i+1)`: nó biết **THỨ TỰ**, nhưng **không biết
KHOẢNG CÁCH**.

Mà khoảng cách đo được, và phân bố rất chặt — đo trên 42 câu TRAKE của tập dev
(106 cặp sự kiện liền kề):

| | trung vị | p25 | p75 | min | max |
| --- | --- | --- | --- | --- | --- |
| độ trải cả chuỗi | 56,6 s | 52,5 | 59,3 | 12,9 | **101,0** |
| khoảng cách 2 sự kiện liền kề | 18,7 s | 12,4 | 28,9 | **5,1** | 55,7 |
| **độ trải / độ dài video** | **0,1** | 0,1 | 0,2 | 0,0 | **0,2** |

Keyframe cách nhau trung vị 2,16 s, nên 18,7 s ≈ **9 keyframe** và 101 s ≈ 47.

#### Ba chỗ sai trong bản đang chạy, cả ba suy thẳng ra từ bảng trên

**1. Chốt chống dồn cục rải N sự kiện đều khắp TOÀN BỘ video.** Mà độ trải thật
chiếm trung vị **10%**, và **không bao giờ quá 20%** độ dài video. Tức khi chốt
nổ, nó đẩy các sự kiện ra xa gấp **5–10 lần** sự thật. Chốt này nổ trên
**47/100 dòng** của `query-p1-18-trake` và 33/100 của `query-p1-4-trake` trong
chính bài nộp thật.

**2. `DON_NHAU = 100` khung nằm DƯỚI mức sàn đo được.** 100 khung là 3,3 s
(30 fps) đến 4,0 s (25 fps), còn khoảng cách nhỏ nhất từng quan sát là
**5,1 s**. Một ngưỡng đặt thấp hơn cả sàn thì gần như không bắt được gì —
giải thích vì sao A14.1 đo ba chính sách dồn cục đều ra 0,0000.

**3. Đếm bằng KHUNG là dính đúng bẫy fps của PHẦN A.** Kho có 4 giá trị fps
(25 / 26,44 / 29,97 / 30) nên `DON_NHAU = 100` mang ý nghĩa khác nhau ở từng
video. `src/chuoi_trake.py` làm việc hoàn toàn trên `pts_time`.

#### Đã cài gì

`src/chuoi_trake.py` — ba thứ, bật/tắt **độc lập** để đo từng cái một:

* `phat_khoang(dt)` — phạt 0 trong vùng đã quan sát `[5,1 s; 55,7 s]`, phạt
  tuyến tính khi ra ngoài, `inf` khi `dt <= 0`. **Phạt mềm chứ không chặn
  cứng**: bóc tách truy vấn có thể sai, chặn cứng thì xoá luôn đáp án đúng
  (cùng nguyên tắc của `objects.py` và `thoi_gian.py`).
* `dong_hang_theo_thoi_gian()` — quy hoạch động có prior + trần độ trải. Cách
  cài phản chiếu đúng luồng suy nghĩ nó mô phỏng: vòng ngoài chọn một **NEO**
  cho sự kiện đầu, vòng trong **đi tiếp về phía trước** trong
  `[t_neo, t_neo + trần]`.
* `rai_deu_hep()` — rải trong cửa sổ 56,6 s quanh neo, thay cho rải khắp video.

Ràng buộc quan trọng nhất, và có test chốt (`test_mac_dinh_giong_het_ban_cu`):
đặt `he_so_phat=0` + `trai_toi_da=inf` phải cho **kết quả trùng khít
`run.dong_hang_dp`**. Đây là bản MỞ RỘNG, không phải bản thay thế — không có
tính chất đó thì mọi phép đo sau không so được với gì.

Bốn tham số mới của `run.dung_trake` (`dong_hang`, `he_so_phat`, `trai_toi_da`,
`rai_hep`) **mặc định giữ nguyên hành vi cũ**. Đo bằng
`scripts/35_do_chuoi_trake.py`.

#### Một chỗ prior này khôn hơn cái test đầu tiên tôi viết cho nó

Test đầu chờ prior sẽ **vứt** khung điểm cao đang dồn cục và chọn bộ ba giãn
đều. DP thật lại trả `[0, 700, 1150]`: nó **giữ** neo điểm cao ở khung 0 mà vẫn
đi ra chuỗi giãn hợp lý. Đúng hơn cái test chờ — prior này phạt **dồn cục**, nó
không có nhiệm vụ vứt bỏ một khung điểm cao. Test đã sửa để chốt theo *tính
chất* (không còn cặp nào dưới sàn 5,1 s) thay vì theo một bộ số cụ thể.

#### Số đo hiện có, và vì sao chưa kết luận được

Chạy với **kênh 3 (OCR+ASR)** trên máy 7,7 GB — kênh duy nhất chạy được không
cần model:

| biến thể | ±2 s | ±15 s | thắng–thua–hoà (±15 s) |
| --- | --- | --- | --- |
| mốc nền (DP ép tăng dần) | 0,0117 | 0,0270 | — |
| + trần độ trải 180 s | 0,0356 | 0,0508 | 1–0–41 |
| + prior khoảng cách | 0,0117 | 0,0349 | 1–0–41 |
| + rải hẹp 56,6 s | 0,0117 | 0,0270 | 0–0–42 |

Không âm ở đâu, nhưng **chỉ 1–2 câu trong 42 nhúc nhích**. Đúng như dự đoán:
A14.1 đã đo kênh 3 một mình cho **0,0000** trên TRAKE, không có gì để lắp ráp
thì khâu lắp ráp không lộ ra được. **Phép đo thật phải chạy với ứng viên kênh 1
(SigLIP2)** — xem `docs/13_lenh_cho_may_manh_dot2.md`.

⚠️ **Cảnh báo về chính tập dev này.** 41/42 câu TRAKE là câu **tự soạn**, chỉ
`trake-DE1-16` do BTC viết — và câu đó ăn 0,0000 ở mọi biến thể. Tập dev tự
soạn đã mù 5 lần (A19/A20/A31/A34/A37). Nhưng phân bố *thời gian đáp án* ở bảng
đầu thì đỡ hơn phân bố *câu hỏi*: nó là tính chất của **video và của cách người
ta chọn một chuỗi sự kiện**, không phải của cách viết câu. Và câu đề thật duy
nhất có độ trải **101,0 s** — mép trên, vẫn trong khoảng. Vì n=1 nên trần trong
code đặt **rộng gấp 1,8 lần** con số đó chứ không bám sát nó.

#### Đã sửa một lỗi trong chính công cụ đo

Hàm in kết luận của `35_do_chuoi_trake.py` bản đầu xét `d[0] > 0 and d[1] > 0`,
nên gán nhãn **`🟡 TỆ HƠN`** cho kết quả `±2s +0,0000 | ±15s +0,0079` — một kết
quả **dương**. Một mức bằng 0 nghĩa là *"ở mức đó không đổi gì"*, không phải
*"xấu đi"*. Cùng họ với lỗi `MOC` bị đè ở PHẦN A: **thước đo sai thì không có
gì báo**, và ở đây suýt kéo theo một kết luận ngược hẳn.
### A40 — `tach_su_kien` vá theo ĐỀ MẪU, đề THẬT đổi format: nộp 4 Frame ID nơi BTC đòi 3

Dựng giao diện truy vấn (`web/`) và chạy thử câu TRAKE thật đầu tiên thì lộ ra:
`query-p1-16-trake` của đề sơ tuyển đợt 1 bị tách thành **4 sự kiện**, trong khi
đề đánh dấu **E1, E2, E3**.

#### Nguyên nhân

```python
_SU_KIEN = re.compile(r"^\s*E\s*\d+\s*[:.]\s*", re.I)   # bản cũ
```

Dấu `[:.]` là **bắt buộc**. Nhưng hai bộ đề viết khác nhau:

| | |
| --- | --- |
| đề MẪU `THUNGHIEM` | `E1: Lân quay vòng trên cột số 4…` |
| đề THẬT `SOTUYEN1` | `E1 Khoảnh khắc đầu tiên xuất hiện…` |

Regex vá theo đề mẫu. Đề thật bỏ dấu hai chấm → **không dòng nào khớp** →
`tach_su_kien` rơi xuống nhánh *"nhiều dòng thì mỗi dòng là một sự kiện"* →
**lời mở đầu (`Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng…`) thành
sự kiện 1**.

Hậu quả: nộp 4 Frame ID cho gói BTC đòi 3. **Sai số Frame ID là sai định dạng,
mất trắng cả gói** — đúng thứ docstring của chính `tach_su_kien` cảnh báo.

#### Vì sao bài nộp 11 điểm không dính

Gói `query-p1-16-trake` được dựng bằng `32_dung_tu_dap_an_tay.py` từ bảng soát
tay `("L24_V031", [1, 14, 25])` — 3 khung, đúng. Nhóm thoát **chỉ vì gói đó
dựng bằng tay**, không phải vì code đúng. Bất kỳ bài nộp nào sinh bằng
`run.py` cho câu này đều đã mất trắng.

#### Bản vá

```python
_SU_KIEN = re.compile(r"^\s*E\s*\d+(?:\s*[:.]\s*|\s+)(?=\S)", re.I)
```

Dấu hai chấm thành tuỳ chọn, nhưng vẫn đòi **một** dấu tách (`:` / `.` /
khoảng trắng) sau số, để `E12abc` không bị nhận nhầm thành sự kiện 12. Bốn test
chốt lại: đề thật và đề mẫu ra cùng số sự kiện; lời mở đầu là bối cảnh chung
ghép vào mọi sự kiện; `E1,E2,E2,E4` vẫn ra 4 (đếm theo DÒNG, không theo số).

#### Một dự đoán của tôi đã SAI, ghi lại vì phần đó mới có giá trị

Tôi cho rằng đây là lý do `trake-DE1-16` ăn **0,0000** ở mọi biến thể trong
phép đo A39 — tách 4 sự kiện thì `diem_trake_bai_nop` so vị trí 1 (lời mở đầu)
với sự kiện 1 thật, lệch hết. Chạy lại `35_do_chuoi_trake.py` sau bản vá:

    trước vá:  đề thật 0,0000 ở mọi biến thể
    sau vá:    đề thật 0,0000 ở mọi biến thể   ← KHÔNG ĐỔI

Toàn bộ bảng 42 câu cũng không đổi một chữ số. Nguyên nhân thật đơn giản hơn:
**kênh 3+4 không tìm ra video múa lân đó**, vì nó gần như không có chữ OCR để
bám. Lỗi tách sự kiện là thật và phải sửa, nhưng nó không phải nguyên nhân tôi
gán cho nó.

#### Hệ quả cho `truy_van.npz`

Cache vector truy vấn chứa **đúng những chuỗi `tach_su_kien` sinh ra**
(`25_ma_hoa_truy_van.py`). Cache sinh trước bản vá mang mệnh đề sai cho mọi câu
TRAKE viết theo kiểu đề thật → phải sinh lại. Đã ghi vào
`docs/13_lenh_cho_may_manh_dot2.md` Việc 0.

#### Bài học chung, và nó lặp lại

Đây là lần thứ **sáu** một thứ được vá theo mẫu tự soạn rồi hụt trên đề thật —
sau A19, A20, A31, A34, A37. Năm lần trước là tập dev mù; lần này là **regex mù**.
Cùng một hình dạng: *cái ta tự tạo ra không phải cái BTC gửi tới.* Mọi chỗ
đọc/tách đề nên được chạy thử trên `dev/SOTUYEN1-bo-de-thi` (đề thật) chứ không
chỉ `dev/THUNGHIEM-bo-de-thi` (đề mẫu).
### A41 — Có cache kênh 1, đo được hai giả thuyết đang treo: **A38 bị bác, A39 không đủ điều kiện bật**

`index/truy_van.npz` sinh trên Kaggle (593 chuỗi, 2,58 MB, xem
`docs/14_kaggle_sinh_truy_van_npz.md`). Nghiệm thu trước khi tin:

| kiểm | kết quả |
| --- | --- |
| cấu trúc | 593 chuỗi × 1152 chiều, chuẩn L2 = 1,00000 |
| model / pretrained | `ViT-SO400M-14-SigLIP2-378` / `webli` ✅ |
| độ phủ | **611/611** chuỗi sẽ được tra, không thiếu cái nào |
| `query-p1-1-kis` → `L30_V046` | video hạng **1**, khung đúng hạng 8 |
| `query-p1-4-kis` → `L22_V021` | video hạng **1**, khung đúng hạng 5 |
| `query-p1-25-kis` → `L30_V003` | video hạng 3, khung đúng hạng 3 |
| mốc nền tự dựng vs `kenh.tim()` | **TRÙNG** |

Trúng video đúng ở hạng 1 trên 873 video hai lần — nếu sai `pretrained` thì đã
là nhiễu. Cache đúng không gian vector.

---

#### A38 — chấm theo CỬA SỔ: **BỊ BÁC, ổn định và âm ở mọi bán kính**

`scripts/36_do_cua_so.py --cache index/truy_van.npz --kiem-moc`, 144 câu KIS/QA.

| bán kính | toàn bộ 144 (±2s / ±15s) | 22 câu ĐỀ THẬT | kết luận |
| --- | --- | --- | --- |
| ±1 | −0,0083 / −0,0625 | +0,0182 / −0,0273 | ❌ ĐẢO DẤU ở đề thật |
| ±2 | −0,0236 / −0,0958 | −0,0182 / −0,0818 | ✅ ổn định, **ÂM** |
| ±3 | −0,0528 / −0,1056 | −0,0364 / −0,1000 | ✅ ổn định, **ÂM** |
| ±5 | −0,0764 / −0,1264 | −0,0364 / −0,1000 | ✅ ổn định, **ÂM** |

Âm ở **cả ba khối** (toàn bộ / đề thật / tự soạn), vượt nhiễu ở hầu hết ô, và
càng nới bán kính càng tệ. Không có góc nào để đọc đây là kết quả tốt.

**Vì sao sai — và đáng ra phải thấy trước khi viết code.** Dòng thứ hai của log
nói hết: **95/144 câu chỉ có MỘT mệnh đề**. Với một mệnh đề thì "cộng qua mệnh
đề" và "max qua mệnh đề" là *cùng một phép tính*, nên phần còn lại của thuật
toán chỉ là **làm nhoè điểm ra ±k khung**. Nhoè thì chỉ mất độ sắc. Hai phần ba
câu rơi vào tình huống đó.

Cơ chế hại **giống hệt A32** đã ghi: *"pha loãng tín hiệu sắc của câu gốc bằng
phép cộng"*. A32 bác `max + λ·tổng`; A38 là **cùng một sai lầm về cấu trúc** —
thêm một phép cộng qua mệnh đề. Sổ đo đã có sẵn câu trả lời từ hôm trước.

Chi tiết xác nhận cơ chế: ở ±15s điểm âm **nặng hơn** ±2s. Vì `no_cua_so` đã nở
đáp án ra ±15 giây — mốc nền vốn đã được tính công cho khung lân cận, nên cửa
sổ chỉ còn đóng góp nhiễu.

> **KHÔNG bác phần quan sát của A38.** Bảy ca soát tay vẫn đúng: `p1-4` cần
> kf183 + kf186/187 mới đủ "hai nhân viên", `p1-6` có nội dung không nằm trong
> keyframe nào. Đề vẫn được viết từ VIDEO còn ta vẫn tìm trên KEYFRAME. Cái sai
> là **cách quy nó ra thuật toán**. Hệ quả thứ nhất của A38 vẫn đứng: *nộp một
> DẢI tốt hơn nộp một khung* — `33_trai_dai_khung.py`, hai bản nộp cùng 11 điểm.

`src/cua_so.py` giữ lại kèm cảnh báo bị bác, không xoá — để người sau không
nghĩ lại ra nó lần nữa.

---

#### A39 — prior khoảng cách TRAKE: mốc nền nhảy **18 lần**, ba đề xuất đều không đủ điều kiện bật

`scripts/35_do_chuoi_trake.py --cache index/truy_van.npz`, 42 câu TRAKE.

**Điều đáng ghi nhất không phải ba biến thể, mà là MỐC NỀN:**

| | kênh 3+4 | kênh 1 (SigLIP2) |
| --- | ---: | ---: |
| ±2s | 0,0117 | **0,2162** |
| ±15s | 0,0270 | **0,3556** |

Gấp ~18 lần. Xác nhận thẳng chẩn đoán ở A39: TRAKE chưa bao giờ tắc vì thuật
toán lắp ráp hay vì RAM — nó tắc vì **thiếu ứng viên kênh 1**.

Ba biến thể, sau khi sửa nhãn cho xét cả nhiễu:

| biến thể | ±2s | ±15s | thắng–thua–hoà (±15s) | kết luận |
| --- | ---: | ---: | --- | --- |
| trần độ trải 180 s | +0,0016 | +0,0095 | 4–2–36 | 🟡 **YẾU** — chưa vượt nhiễu (2·SE 0,0351) |
| prior khoảng cách | −0,0181 | −0,0032 | 6–4–32 | 🟠 tệ hơn |
| rải hẹp 56,6 s | 0,0000 | 0,0000 | 0–0–42 | ⚪ **KHÔNG ĐỔI GÌ** |

**Không cái nào được bật.** Cả ba giữ mặc định TẮT như đã đặt sẵn.

**Một dự đoán đúng, và nó là kết quả hữu ích nhất trong ba.** Trước khi đo, tôi
đã ghi: *"`rai_hep` mà vẫn ⚪ với kênh 1 thì nghĩa là chốt dồn cục hầu như
không nổ nữa khi ứng viên đủ tốt, và cả nhánh đó nên bỏ chứ không phải sửa
tiếp."* Kết quả: **0–0–42 ở cả hai mức dung sai** — chốt không nổ lần nào.

Nghĩa là toàn bộ chẩn đoán "rải khắp video là sai" của A39 tuy **đúng về số
liệu** (độ trải thật chiếm trung vị 10% độ dài video) nhưng **vô nghĩa về hậu
quả**: nhánh đó chỉ chạy khi truy hồi thất bại, mà với kênh 1 nó gần như không
thất bại. Sửa một nhánh chết không đem lại gì.

`query-p1-16-trake` (câu đề thật duy nhất) vẫn **0,0000 kể cả với kênh 1** —
n=1, không kết luận được, nhưng đủ để nói: bản vá A40 không phải nguyên nhân,
và SigLIP2 cũng không tìm ra video múa lân đó.

---

#### Đã sửa một lỗi trong chính công cụ đo — lần thứ hai trong hai ngày

`35_do_chuoi_trake.py` gắn nhãn kết luận **chỉ theo DẤU**, nên `+0,0016` (kém
xa nhiễu, nhúc nhích 2/42 câu) được in là **✅ tốt hơn ở cả hai mức**. Nói quá.

Ngưỡng nó in ra cũng không phải ngưỡng thống kê: `1/(số sự kiện × số câu)` là
**lượng tử nhỏ nhất** điểm có thể đổi — trả lời "có đổi gì không", không trả
lời "đổi có thật không".

Đã đổi sang đúng công thức `cham_diem._hieu` (2·SE của hiệu theo cặp), để hai
script hiểu "vượt nhiễu" giống nhau. Cùng họ với lỗi nhãn `🟡 TỆ HƠN` sửa hôm
trước, và với `MOC` bị đè ở PHẦN A: **thước đo sai thì không có gì báo.**

---

#### Tổng kết phiên: giá trị nằm ở phần ÂM

Hai giả thuyết do chính tôi đề xuất, viết code, viết test, ghi thành luật — cả
hai đều **không sống nổi phép đo đầu tiên có ứng viên thật**. Ghi lại đầy đủ vì
đó mới là phần đáng tiền: ba hướng nữa bị loại khỏi bàn, và mốc nền TRAKE lần
đầu có con số thật để so.

Thứ **thật sự** thay đổi hôm nay không phải thuật toán nào — mà là `truy_van.npz`
về tới máy, biến mọi phép đo dính kênh 1 từ "không chạy được" thành "chạy 10 phút".
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
   *(A22 đo: hoãn bật — đảo dấu, và ghép với ràng buộc mục 2 thì tệ hơn ổn định.)*

### C7 — Bản quy định chính thức của BTC (sotuyenaic.oj.io.vn, 19/08/2026)

Ba điều dưới đây lấy nguyên văn từ *"Hướng dẫn nộp bài sơ tuyển"*, và điều đầu
tiên **đổi cách đọc mọi con số leaderboard**:

| Điều | Nguyên văn | Hệ quả |
| --- | --- | --- |
| **Public leaderboard chỉ chấm 50% đáp án** | *"Kết quả đánh giá trên Public Leaderboard chỉ tính dựa trên 50% đáp án của BTC. Kết quả cuối cùng… tính trên 100% đáp án… tại Private Leaderboard"* | **0,8 và 2,6 ở A20 là điểm trên NỬA bộ đáp án.** So hai đợt với nhau vẫn đứng (cùng 50%), nhưng đừng suy ra thứ hạng cuối |
| 3 lần nộp, **lần CUỐI tính điểm** | *"Kết quả được dùng để xếp hạng là kết quả đội nộp lần cuối cùng"* | Không phải lần tốt nhất. Nộp thử một cấu hình yếu ở lần 3 là **tự hạ điểm** |
| Sai định dạng vẫn tính một lần | *"Khi nộp sai định dạng vẫn tính là 01 lần nộp"* | Lý do `nop_bai.soat()` từ chối ghi file |

**Và tài liệu tự mâu thuẫn về cách chấm `answer`, ngay trong cùng một trang web:**

    trang 2:  "được so sánh chính xác về mặt NGỮ NGHĨA với đáp án"
    trang 8:  "Answer (Q&A) sẽ được so sánh dưới dạng CHUỖI CHÍNH XÁC"

Chưa hỏi được BTC thì `tra_loi.don_dap_an()` chọn dạng **an toàn với cả hai**:
ngắn nhất, chuẩn tắc nhất. `"5"` khớp được nếu chấm chuỗi, và vẫn đúng nếu chấm
ngữ nghĩa; `"Có 5 cái bát trên bàn"` thì chỉ đúng ở vế sau.

> ⚠️ **Bẫy nằm trong chính ví dụ của BTC.** Quy định ghi *"khoảng trắng đầu/cuối
> được giữ nguyên, không tự động trim"*, nhưng ví dụ trang 2 lại viết
> `L01_V028, 3450, "5"` — có khoảng trắng sau dấu phẩy. Đọc bằng parser CSV
> chuẩn, `answer` ra `' "5"'`: khoảng trắng và cả dấu ngoặc kép thành **ký tự
> thật**. Ví dụ CSV chuẩn ở trang 4–5 không có khoảng trắng — đó mới là dạng
> đúng. `_viet_csv()` sinh đúng dạng đó, và `soat_zip()` nay bắt khoảng trắng
> thừa như một lỗi, phòng người sửa tay theo ví dụ trang 2.

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

**(a) Khử trùng lặp trong cùng video** *(A8.8, PHẦN C mục 6)* — ✅ **ĐÃ CÀI, ĐÃ ĐO, HOÃN BẬT**

`src/dedup.py` xong, `scripts/13_do_dedup.py` đo xong. Kết quả đầy đủ ở **A11**.

Tóm tắt: trên truy vấn CLIP **đọc được**, dedup bỏ **0,5/100** ứng viên và
**không đổi hạng 1 câu nào** — gần như no-op. Con số 58,4/100 đo trên tập dev
tiếng Việt là **ảo giác**, nó đo cái mù tiếng Việt của CLIP chứ không đo dedup.
Chỗ duy nhất nó còn tác dụng là khi bật `moi_video` trên bể nhiều bản sao (L25).

**Giữ module, không bật mặc định.** Đo lại khi có `clip_siglip2.npy` toàn kho —
đó mới là lần đo có nghĩa, vì hiện chỉ SigLIP2 đọc được tiếng Việt mà ma trận
thử của nó nằm đúng hai nhóm ít trùng lặp nhất kho.

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

**`dev/label_vi_en.csv` — ĐÃ CÓ.** 135 nhãn, **phủ 98,1% detection**. Kèm
`objects.nhan_tu_truy_van()` biến câu tiếng Việt thành danh sách nhãn.

Không cần dịch hết 473 nhãn. Đo thật ở ngưỡng 0,5 (597.357 detection): top 30
phủ 86,2% — top 50: 91,6% — **top 100: 96,8%** — top 150: 98,6%. Bảng hiện tại
lấy top 100 cộng ~35 nhãn ngoài top nhưng hay xuất hiện trong truy vấn (`Cat`,
`Dog`, `Balloon`, `Kitchen knife`, `Cutting board`, `Bus`...).

Bốn cột `nhan_en, nhan_vi, dong_nghia, cha`:

- **`dong_nghia` là cột quan trọng nhất**, không phải cột dịch. Truy vấn dùng
  cách nói nào không đoán trước được: `Canoe` phải bắt cả *"ghe"*, `Person`
  phải bắt cả *"cán bộ", "công an", "phóng viên"*. Đo thật — chưa thêm từ chỉ
  nghề nghiệp thì *"cán bộ công an ngồi ghi biên bản"* trả về **rỗng**.
- **`cha` xử lý chỗ thứ bậc OpenImages không tự gộp.** `Car`, `Land vehicle`,
  `Vehicle` là ba nhãn riêng biệt — truy vấn *"ô tô"* mà không kéo cha thì bỏ
  sót hai nhãn kia.

⚠️ **Khớp theo CỤM TỪ, không khớp chuỗi con.** Chuỗi con rất nguy hiểm với
tiếng Việt: *"cá"* nằm trong *"cá nhân"*, *"bàn"* nằm trong *"bàn bạc"*.

`tests/test_objects.py` chốt cả bốn thứ: không nhãn chết (gõ nhầm tên nhãn thì
kênh im lặng chết mà không ai biết), cột `cha` trỏ đúng, sáu truy vấn thật
khớp đúng, và kéo cha có hoạt động.

---

### GIAI ĐOẠN 2 — Ba dạng truy vấn

*Giữ nguyên cấu trúc 2 mũi nhọn của v3. Chỉ đổi: objects lên kênh chính,
và Bước 4 Q&A nhấn mạnh ngữ cảnh (theo D0.3).*

#### Mũi nhọn 1 — Textual KIS & Q&A (TV1, TV3, TV5)

> **Trạng thái sau A22:** cả sáu bước đã có chỗ nối trong `src/mui_nhon_1.py`,
> gọi được từ `run.py`. Ba bước phụ **mặc định TẮT** vì chưa bước nào thắng
> được trên tập dev — xem A22. Đo lại từng cờ bằng
> `python scripts/22_do_mui_nhon_1.py`.

**Bước 1 — Thu hẹp cấp video.** BM25 metadata → top-50 video.

> ⚠️ **Sửa ở A22: XẾP LẠI, không phải CẮT.** Cắt cứng ở top-50 đo được là
> **tệ hơn** (−0,0286 / −0,0476), vì metadata chỉ phủ **37,1%** số câu ở top-50
> — đọc "97%" của A12 thành năng lực thu hẹp là đọc sai. Dạng mềm (`uu_tien`)
> dương ở cả hai mức nhưng **dưới ngưỡng nhiễu**, chưa đủ căn cứ bật:
> `run.py --uu-tien-video 50`.

**Bước 2 — Truy hồi đa kênh, hợp nhất bằng Reciprocal Rank Fusion.** Bốn kênh
ở bảng trên. RRF an toàn hơn weighted-sum vì không cần chuẩn hóa thang điểm
giữa cosine (0,2–0,35) và BM25 (không chặn trên).

**Bước 2b — MỚI (4.1): khử trùng lặp rồi mới cắt top-K.** Xem PHẦN C mục 6 và
A8.8. Đặt *sau* RRF, *trước* khi cắt.

> ⚠️ **A22 đo được: ĐẢO DẤU giữa hai mức dung sai** (−0,0152 ở ±2s, +0,0057 ở
> ±15s) — kết luận phụ thuộc vào ẩn số BTC chưa chốt, **không dùng để quyết**.
> Còn ghép với ràng buộc đa dạng thì **tệ hơn ổn định** (−0,0324, 0 thắng–7
> thua), bác đúng câu A11 dự đoán là hai cái bổ sung nhau. `run.py --dedup`.

**Bước 3 — ~~Tinh chỉnh vị trí frame~~ — TỤT ƯU TIÊN.** *(4.1)* Bước này viết
là *"chỉ khi BTC xác nhận cửa sổ hẹp ở 0.a"*. Theo A8.1, luật AIC'25 chấm
`frame_idx` **rơi trong một khoảng**, nên điều kiện đó nhiều khả năng không xảy
ra. **Chưa làm bước này cho tới khi BTC trả lời khác đi.** Công sức chuyển sang
Bước 3b.

**Bước 3b — MỚI: đi bộ theo thời gian ("nearby frame").** *(A8.7 — kỹ thuật
đáng giá nhất trên mỗi đơn vị công sức)*

> ⚠️ **A18 bác dùng nó trong TRUY HỒI; A22 đặt lại đúng chỗ của nó là BƯỚC 4.**
> Chèn khung lân cận vào danh sách nộp là tiêu một trong 100 chỗ để mua lại thứ
> đã có — BTC chấm theo khoảng 4 giây–5 phút (A9) nên khung ±2s vốn đã tính là
> đúng. Trong Bước 4 thì nó không tiêu chỗ nào: `mui_nhon_1.khung_ngu_canh()`.

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

> **Đã nối (A22), chặn ở một lệnh.** `run.py --vlm` gọi
> `mui_nhon_1.gan_dap_an()` → `tra_loi.tra_loi_qa()`, đưa 3 khung trong cửa sổ
> ±2s. Máy đang chạy không có model nào **nhìn được ảnh** →
> `ollama pull qwen2.5vl:7b`. Đây là việc đáng giá nhất còn lại: 42/105 câu dev
> và 3/24 gói đề mẫu là Q&A, **tất cả đang chắc chắn 0**.

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
| 2 | TV2 (trích dày) → câu Q&A dạng **ĐẾM** | không đếm đủ số lượng → `answer` sai → 0 điểm | 🟡 **hạ khỏi đường găng (A9)** — cửa sổ rộng 4s–5 phút nên keyframe có sẵn đã phủ 86–100%; trích dày nay để **NHÌN THẤY NỘI DUNG**, không phải để trúng chỉ số |
| **3** | **Khánh (tập dev) → toàn bộ GĐ3** | rerank thành đoán mò | ⬜ **đường găng số 1** |
| 4 | TV3 (BM25 metadata) → Bước 2 TRAKE | sai video → 0 điểm | ✅ dữ liệu đã đủ (100%) |
| 5 | 0.c (VLM) → Bước 4 Q&A | `answer` sai định dạng → 0 điểm | 🟡 đang test |
| 6 | TV1 (ma trận CLIP) → cả 2 mũi nhọn | cả hai mũi nhọn tắc | ✅ dữ liệu đã đủ (100%) |
| **7** | **MỚI — tải đủ video/keyframe → #2, #3** | **hai phụ thuộc quan trọng nhất đều tắc** | ✅ **100% đã kiểm chứng** |

**Đường găng (sửa theo A9):**
`Bảng cái ✅ → tập dev ⬜ → đo 5 kênh + RRF → GĐ3`

> **Nút thắt đã dịch chuyển HAI LẦN.**
>
> Ở v3, đường găng bắt đầu từ bảng cái — nay đã xong và được chứng minh.
>
> Ở v4, nút thắt là **tải dữ liệu**, vì trích dày cần video và người ta tin
> trích dày quyết định trần điểm TRAKE.
>
> **Ở v4.2, BTC bác bỏ tiền đề đó (A9): cửa sổ rộng 4 giây–5 phút chứ không
> phải 10 frame, nên keyframe có sẵn đã phủ 86–100%.** Trích dày hạ xuống vai
> trò hẹp hơn (nhìn nội dung cho câu đếm). Nút thắt thật bây giờ là **tập
> dev** — không có nó thì không đo được kênh nào đáng giữ, mà chất lượng truy
> hồi mới là thứ quyết định điểm.

---

## PHẦN F — RỦI RO CÒN LẠI

| Rủi ro | Dấu hiệu sớm | Phương án |
| --- | --- | --- |
| **Chia dữ liệu nhiều máy làm đứt đường dẫn tuyệt đối** | `kf_path` trỏ file không tồn tại | Thống nhất đường dẫn dữ liệu giống nhau trên mọi máy, hoặc remap (xem A5.5). Riêng việc **gộp kết quả** thì an toàn: `row_id` như nhau trên mọi máy (đã đo 765/765 trên năm lô) |
| **Không ai chạy verify cho nhóm L mình giữ** | 873/873 video được kiểm chứng — 100% | Bắt buộc mỗi máy chạy `02`+`03` cho phần mình, báo kết quả |
| ~~BTC dùng cửa sổ `[s,e]` hẹp cho cả KIS~~ | — | ✅ **ĐÓNG (A9)** — BTC xác nhận cửa sổ **4 giây–5 phút**. Keyframe có sẵn phủ 86–100%. Rủi ro biến mất |
| **MỚI — câu Q&A dạng ĐẾM không đếm đủ** | tập dev có câu đếm, mọi cấu hình đều sai số lượng | BTC nêu thẳng ca này (A9.3): *"1 em bé được bế bởi 4 người liên tiếp, dựa trên keyframe thì chỉ có 3"*. Cần `trich_day` + VLM nhìn nhiều frame. **Tập dev phải có ít nhất 3–5 câu đếm**, nếu không ta không bao giờ phát hiện điểm yếu này |
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

### G4 — Sửa đổi v4.2 → v4.3 (nguồn: đo trên 100 câu dev, xem A10)

*Lần đầu đo trên tập dev đủ lớn và **đúng ngôn ngữ đề thi**. Kết quả đảo ngược
đánh giá về kênh mạnh nhất.*

| Hạng mục | Trước | **v4.3** |
| --- | --- | --- |
| Kênh 1 (CLIP `ViT-B/32`) | "kênh chính, mạnh nhất" | **0,0000 trên 100/100 câu tiếng Việt** |
| Kênh 4 (objects) | kênh phụ nâng lên chính | **kênh DUY NHẤT đang chạy** (0,0400, ổn định) |
| Bước dịch truy vấn | một gạch đầu dòng trong mục TV1 | **chính là kênh 1** — không có nó thì kênh bằng 0 |
| SigLIP2 | "thử xem có đáng không" | **đường thoát cho kênh 1** |
| Bảng nhãn Việt–Anh | tiện ích nhỏ | **thứ duy nhất giữ hệ thống không đứng im** |

### G3 — Sửa đổi v4.1 → v4.2 (nguồn: BTC trả lời 15/08, xem A9)

*Đây là nguồn **chính thức**, mạnh hơn A8. Nó bác một con số nền.*

| Hạng mục | Trước | **v4.2** |
| --- | --- | --- |
| Độ rộng cửa sổ `[s,e]` | 10 frame *(hiểu nhầm từ ví dụ `[500,510]`)* | **4 giây – 5 phút** |
| Keyframe có sẵn phủ cửa sổ | **14,6%** | **86,1%** (cửa sổ 4s) → **100%** (từ 10s) |
| Trần R-Score TRAKE | ~0,15, chặn bởi mật độ keyframe | **bỏ mốc đó** — chặn bởi chất lượng truy hồi |
| `trich_day` | **đường găng**, để bắn trúng cửa sổ | **hạ khỏi đường găng**; lý do mới là **nhìn thấy nội dung** cho câu ĐẾM |
| Nguồn sinh đáp án | không rõ | **từ video**, keyframe chỉ là giải pháp mẫu |
| Đề mẫu | không có | **sẽ có query mẫu**, và **có thể xin GT các mùa trước** |

> **Cách đọc G2:** mọi dòng đều đến từ *một* bài báo *một* đội *một* mùa. Bài
> không có ablation (A8.2). Nên đây là **giả thuyết có căn cứ**, không phải sự
> thật đã đo — phải qua tập dev của Khánh mới được giữ, y như mọi thứ khác ở
> GIAI ĐOẠN 3.

---

## PHẦN H — VIỆC LÀM NGAY

*Giai đoạn 0 đã đóng (873/873, 0 lệch chỉ số thật). Mọi việc dưới đây thuộc
Giai đoạn 1.*

### H1. Đường găng — ✅ **ĐÃ THÔNG**

**206 câu, đủ cả 10 nhóm L**, đã tách tập test giữ kín:

| | KIS | QA | TRAKE | Tổng |
| --- | ---: | ---: | ---: | ---: |
| `dev/tap_dev.jsonl` | 99 | 45 | 42 | **186** |
| `dev/tap_test.jsonl` 🔒 | 10 | 10 | 0 | **20** |

> **23 câu trong tập dev là ĐỀ THẬT do BTC viết** (đề sơ tuyển đợt 1, nạp bằng
> `scripts/34_de_thanh_tap_dev.py`, đáp án nhóm soát tay). Đây là thứ tập dev
> thiếu suốt từ đầu và là nguyên nhân của cả 5 lần dev mù
> (A19/A20/A31/A34/A37): câu tự soạn ~15-22 từ / 1,1 mệnh đề, đề thật **63 từ /
> 2,4 mệnh đề**. **Mọi phép đo từ nay nên báo RIÊNG cột 23 câu đề thật** —
> `scripts/36_do_cua_so.py` làm sẵn việc đó.
>
> Ngược lại, **câu TRAKE thì 41/42 vẫn là tự soạn** (chỉ `trake-DE1-16` là đề
> thật), nên phân bố *câu hỏi* TRAKE vẫn chưa đáng tin — xem cảnh báo ở A39.

**Còn thiếu: câu đếm.** `scripts/11_tim_cau_dem.py` lọc sẵn ứng viên khung
nhiều vật đếm được. Riêng câu đếm thì A26 và ca `p1-15-qa` cho thấy đó là
**trần của model**, không phải lỗi truy hồi — đừng đầu tư thêm.

Thêm câu mới thì cứ `--gop` bình thường — **câu mới vào tập dev, tập test giữ
nguyên**, `gop()` tự loại. **Không chạy lại `--tach-test`** (nó cũng tự từ
chối). Quy trình đầy đủ: [07_lam_tap_dev.md](07_lam_tap_dev.md).

```powershell
python scripts\10_contact_sheet.py --nhom <L của mình> --thua 10
python scripts\10_contact_sheet.py --tra <row_id...> --mo
# soạn vào dev/tap_dev_thanh_vien/tap_dev_<nhóm>.jsonl
python src\tap_dev.py --gop dev\tap_dev_thanh_vien
python src\tap_dev.py --no-cum
python src\tap_dev.py --kiem
```

### H2. Làm được ngay, không chờ ai

| # | Việc | Ai | Ghi chú |
| --- | --- | --- | --- |
| ~~**1b**~~ | ✅ **XONG — kênh 1 đã sống lại** | Khánh (19/08) | `clip_siglip2.npy` đủ 177.321 dòng, kiểm hàng thẳng. **0,0000 → 0,3258, thắng 46 thua 0** (A17). Kênh 1 nay là **kênh mạnh nhất**, gấp ~8 lần kênh 4 |
| **1c** | 🔴 **Đo tiếp trên máy ≥ 16 GB RAM** | máy GPU / máy khỏe | Máy hiện tại 7,7 GB, crash nhiều lần khi nạp SO400M. Còn nợ: RRF(SigLIP2 + objects/metadata), so theo cặp SigLIP2 vs RRF, **`dedup` trên SigLIP2** (phép đo A11 hẹn lại), các mức `moi_video` |
| 2 | ~~**`src/bm25.py`** — bộ máy văn bản~~ | TV1 | ✅ **XONG.** Dùng chung cho kênh 2, 3, 5. Tự viết, không thêm phụ thuộc. **Kênh 2 chạy: 97% tìm ra video đúng, 22,7% ở top-10** (A12) |
| 2b | **Kênh 3 (OCR/ASR)** — nối vào `KenhVanBan.tu_bang_khung` | TV3 | bộ máy đã sẵn, chỉ cần bảng `(row_id, text)`. Cần **hai chế độ**: lọc cứng cho token hiếm + BM25 hòa RRF (A8.5) |
| 3 | ~~`dev/label_vi_en.csv`~~ — **XONG**, 156 nhãn phủ 98,3% | — | **Kênh 4 nay dùng được từ truy vấn tiếng Việt.** Còn nên: người Việt đọc lại cột `dong_nghia`, thêm cách nói vùng miền |
| 4 | ~~Tối ưu `trich_day`: gộp cả cửa sổ vào MỘT lệnh ffmpeg~~ | TV2 | ✅ **XONG** — `trich_nhieu()` trong `src/trich_day.py`, áp dụng lại cho `scripts/09_trich_day_batch.py`. Đo lại sau khi cài: **28 ms/khung** so với 169–284 ms/khung cũ. Cờ `-vsync 0` đã bị GỠ ở ffmpeg 9.0, đổi sang `-fps_mode passthrough` |
| 5 | **Commit script vá `kf_path`** | Khánh | máy nào tải `index/` từ Drive cũng gặp (A5.5); đừng để mỗi người viết lại |
| 6 | **Gán nhãn 400 mẫu `ocr_v2` + điền `roi_v2.yaml`** | TV4 | đang **0/400**. ROI: ranh giới là **dải chữ chạy cuối cùng**, KHÔNG phải "bỏ nửa dưới" — băng rôn tiêu đề có liên quan tới hình |
| 7 | **`src/run.py`** — đường ống đầu-cuối | TV5 | ~~chỉ đáng viết khi đã có ≥ 2 kênh~~ — **điều kiện đã thoả**: kênh 2 và kênh 4 đều chạy được từ truy vấn tiếng Việt |
| 7b | **Kênh 5 (caption VLM)** — bộ sinh đã xong | — | ⛔ **CHẶN ở việc 12** (khóa API) và ở chỗ máy nào giữ ảnh nhóm nào. Xem A13. Chạy được ngay khi có khóa: `python scripts/14_sinh_caption.py --chon tap-dev` |

### H3. Chờ **ma trận SigLIP2 toàn kho** (không còn chờ tập dev)

| # | Việc | Ai | Trạng thái |
| --- | --- | --- | --- |
| 8 | Đo A/B/C cho SigLIP2 — **nhớ `be_chung()`**, xem [06 §5](06_ke_hoach_encode_GPU.md) | Khánh | đã có câu trả lời sớm trên CPU (A10.3); còn xác nhận trên toàn kho |
| 9 | Đo `dedup.py` — giữ hay bỏ theo số | TV1 | ✅ **ĐÃ ĐO (A11): hoãn bật.** No-op trên truy vấn đọc được (0,5/100, hạng 1 không đổi). Đo lại bằng `13_do_dedup.py --matrix clip_siglip2.npy --moi-video 3` |
| 9b | ~~Đo RRF~~ | TV1 | ✅ **ĐÃ ĐO (A14): RRF thô LÀM TỆ ĐI** (−0,0144, ổn định). Nguyên nhân đo được: chỉ 5/97 câu hai kênh chung một khung. **Việc mới: hợp nhất hai tầng** (RRF cấp video → xếp khung) |
| 9c | Đo `lan_can.py` | TV1 | làm được ngay, kênh 4 đã cho mốc nền khác 0 |
| 10 | Mở rộng bench VLM lên ≥ 50 câu, test với ngữ cảnh thật | TV5 | |

> **Vì sao mọi phép đo đều chặn ở cùng một chỗ.** Kênh 1 chạy CLIP thì được
> **0,0000** trên tập dev tiếng Việt (A10) — không có gì để cải thiện, nên mọi
> cấu hình đo trên nó đều ra `⚪ KHÔNG ĐỔI GÌ`. Đó không phải kết luận về cấu
> hình, đó là kết luận về kênh. **`clip_siglip2.npy` toàn kho mở khóa cả H3.**

### H4. Việc người, không tự động hóa được

| # | Việc | Ai | Ghi chú |
| --- | --- | --- | --- |
| 11 | ~~Gửi BTC câu 0.a~~ — **ĐÃ TRẢ LỜI (A9)**. Còn lại: **xin GT các mùa trước** (BTC nói có), và hỏi **0.e — Chung kết có thi tương tác không** | Khánh | |
| 12 | ✅ **ĐÃ CHỐT (21/08): dùng FREE, không bật billing.** Trái với khuyến nghị trả phí ở báo cáo bench 13/08 (§11.3 của `Test VLM AIC/BAO_CAO_BENCH_VLM_2026-08-13.md`) — nhóm chấp nhận rủi ro rate-limit của free tier thay vì trả phí. Model chính vẫn `gemini-3.1-flash-lite` **free tier**; phương án offline khi hết quota là **`qwen2.5vl:7b` qua Ollama** (không phải bản 3B như ghi trước đây — đã đo lại 13/08: 3B lỗi lặp token 83% trên GPU, 7B mới là bản dùng được, 52-58% đúng, vừa đủ RTX 2060S 8GB) | TV5 | |
| 12b | 🔴 **Xoá khóa Gemini đã dán vào chat**, tạo khóa mới | — | repo công khai; khóa đã lộ thì coi như của chung |
| 13 | Chốt một bảng tên thành viên duy nhất | cả nhóm | |
| 14 | Máy giữ L23+L26+L27 tải lại gói `Keyframes_L21` (thiếu 8 file ảnh) | — | |

> **Kỷ luật cho toàn bộ bản 4.1:** mọi thứ lấy từ bài báo AIC'25 là **một bài
> báo, một đội, một mùa, không ablation** (A8.2). Dựng thì dựng, nhưng **chỉ
> giữ cái nào tăng điểm đo được trên tập dev**. Đó là lý do việc 1 quan trọng
> hơn việc 2–10 cộng lại.
