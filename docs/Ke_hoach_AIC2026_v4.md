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
### A42 — Đo bốn đề xuất "phá lối mòn": ba cái đã chết sẵn, cái thứ tư chết theo cách dạy ta một cơ chế

Bốn hướng do Gemini đề xuất sau khi đọc tới A40. Đối chiếu trước khi bỏ công:

| Gemini đề xuất | thật ra là | |
| --- | --- | --- |
| 1. Sliding Window Pooling | **chính là A38** — công thức trùng khít `cua_so.diem_cua_so` | đã bác (A41) |
| 2. Anchor & Expand | **chưa ai đo** | ⬅ đo ở đây |
| 3. Reverse Caption Alignment | **chính là A32** (`max + λ·tổng`, khác mỗi lời prompt) | đã bác |
| 4. DTW / beam search TRAKE | **chính là A39** `dong_hang_theo_thoi_gian` + trần độ trải | đã đo, 🟡 |

Chỉ hướng 2 đáng dựng. Nhưng trước đó, đóng nốt một cửa còn hé của A38.

---

#### A38 chết cả trên SÂN NHÀ — và điều đó sửa lại lời giải thích ở A41

A41 quy nguyên nhân cho **95/144 câu chỉ có MỘT mệnh đề** (ở đó "cộng qua mệnh
đề" ≡ "max qua mệnh đề", nên thuật toán rút gọn thành làm nhoè). Nếu đúng vậy
thì trên câu NHIỀU mệnh đề nó phải thắng. Thêm hai khối vào
`36_do_cua_so.py` để hỏi thẳng:

| khối | ±1 | ±2 | ±3 |
| --- | --- | --- | --- |
| 49 câu **≥2 mệnh đề** | −0,0082/−0,0653 🟡 | −0,0327/−0,0980 ✅ **âm** | −0,0612/−0,1061 ✅ **âm** |
| 17 câu **đề thật ≥2 mệnh đề** | +0,0353/−0,0235 ❌ đảo | −0,0118/−0,0824 🟡 | −0,0353/−0,1059 ✅ **âm** |

Âm ngay trên tập câu mà A38 sinh ra để phục vụ. **Lời giải thích ở A41 SAI** —
không phải tại câu một mệnh đề. Nguyên nhân thật đơn giản hơn và đúng y như
A32: **phép cộng qua mệnh đề TỰ NÓ pha loãng**, bất kể có mấy mệnh đề.

Ghi lại chỗ này vì nó là bài học về cách sửa sai: lần đầu tôi giải thích thất
bại bằng một giả thuyết *nghe hợp lý*; hỏi thẳng dữ liệu thì giả thuyết đó cũng
sai nốt.

---

#### Hướng 2 — Anchor & Expand: ba biến thể, và một cơ chế đáng giá hơn cả ba

`scripts/37_do_neo_mo_rong.py`. Mỏ neo định nghĩa **đo được**, không phải danh
sách soạn tay: token của truy vấn có mặt trong kho OCR với `1 <= df <= 60`
(có nên tra được, hiếm nên tra ra sắc). Rồi "đi bộ thời gian" `[-15s, +30s]`
quanh mỗi lần khớp, dùng `pts_time` chứ không dùng số khung.

Cơ chế tìm neo hoạt động đúng như người soát tay đã làm bằng tay:

    p1-12-kis  ->  'mazut'   ->  68 khung      (đúng từ nhóm bám khi soát tay)
    p1-17-qa   ->  'nghẽn'   ->  38 khung
    p1-1-kis   ->  không có  ->  rơi về SigLIP2

**38/144 câu có mỏ neo.** Trên chính 38 câu đó:

| biến thể | ±2s | ±15s | thắng–thua–hoà |
| --- | ---: | ---: | --- |
| `chen` — chèn lên đầu (vi phạm A27/A28) | **−0,1474** | **−0,2316** | 5–16–17 ✅ ổn định |
| `xep_lai` — đảo thứ tự trong bể (tuân A27/A28) | −0,0053 | −0,0053 | 0–1–37 |
| `duoi` — chỉ chiếm hạng 81–100 | **+0,0053** | **+0,0105** | **1–0–37 / 2–0–36** |

Trên 10 câu đề thật có mỏ neo, `duoi` cũng **0 thua**: 0–0–10 và 1–0–9.

##### Cơ chế: mỏ neo và SigLIP2 tìm ra những chỗ KHÁC NHAU

Đo trực tiếp phần giao giữa tập khung neo và top-100 của SigLIP2:

> **25/38 câu giao = 0 khung.** Trung vị giao **0**, trung bình 1,8, max 20 —
> trên tập neo trung vị **72 khung**.

Hai tập gần như rời nhau. Từ đó cả ba kết quả trên đều suy ra được:

* `xep_lai` **là no-op** — không có gì để đảo, vì neo không nằm trong bể.
* `chen` **thảm hoạ** — nhét 72 khung rời vào đầu đẩy ứng viên tốt của SigLIP2
  từ hạng 1–100 xuống 73–172, tức **rơi hẳn khỏi bài nộp**. Đây là con số âm
  lớn nhất từng đo trong repo.
* `duoi` **an toàn** — hạng 81–100 chỉ vào `R@100`, mỗi dòng đáng nhiều nhất
  `0,2/5 = 0,04`. Mặt trái bị chặn cứng, mặt phải là từ 0 lên 0,2.

##### Vì sao NGƯỜI làm được mà MÁY không

Người soát tay ở `p1-12` không trộn hai danh sách. Họ **đổi hẳn chiến lược
tìm** (bỏ SigLIP2, quét OCR), rồi **dùng mắt làm bộ xếp lại**. Mắt là bộ xếp
lại hoàn hảo — nó nhìn 68 khung và chọn đúng cái.

Máy không có bộ xếp lại hoàn hảo. Nên khi bắt chước bước một mà thiếu bước hai,
nó chỉ còn cách trộn vào danh sách xếp hạng — và trộn hai tập rời nhau trong
một danh sách **có kích thước cố định** thì bên này lên đúng bằng bên kia xuống.

> **Đây là lời giải thích CƠ CHẾ cho A27/A28**, thứ hai điều luật đó chưa có.
> Không phải "kênh yếu thì có hại". Mà là: **thêm ứng viên rời vào một danh
> sách 100 chỗ là một phép ĐỔI CHỖ, không phải phép BỔ SUNG.** Xếp lại thì
> không đổi chỗ ai — nên nó lãi. Chèn thì đổi chỗ — nên nó lỗ. Xác nhận độc
> lập lần thứ ba, và lần này biết vì sao.

##### Suy ra một thiết kế, và nó là thứ duy nhất dương cả phiên

`duoi` không sinh ra từ trực giác mà **suy ra từ cơ chế**: nếu vấn đề là đổi
chỗ, thì hãy tiêu vào những chỗ rẻ nhất. Hạng 81–100 vốn là vé số ngẫu nhiên
(PHẦN C: không phạt dòng sai), đổi lấy vé số có mỏ neo là phép đổi không mất gì.

Kết quả đúng như dự đoán: **không thua câu nào**, ở cả hai mức dung sai, trên
cả hai khối. Nhưng vẫn 🟡 **YẾU** — chưa vượt nhiễu ở đâu.

> **Khuyến nghị: CHƯA BẬT.** Kỷ luật đo của dự án không cho bật thứ chưa vượt
> nhiễu, và A34/A37 là hai lần trả giá gần nhất cho việc bật sớm. Nhưng đây là
> ứng viên đáng đo lại đầu tiên khi tập dev có thêm câu đề thật — nó là kỹ
> thuật duy nhất trong ngày có **mặt trái bị chặn về mặt toán học**, không phải
> chặn bằng hy vọng.

---

#### Tổng kết: bốn đề xuất, không cái nào phá được trần

Ba cái trùng với thứ đã đo. Cái thứ tư đo mới, và cả ba biến thể của nó đều
không đủ điều kiện bật. Nhưng phiên này không phí:

1. **A38 đóng cửa hẳn**, kể cả trên sân nhà — và lời giải thích ở A41 được sửa.
2. **A27/A28 có cơ chế**, không còn là quy tắc kinh nghiệm.
3. **Một thiết kế mới suy ra từ cơ chế đó** (`duoi`), dương và có mặt trái chặn cứng.

Điều đáng nói nhất về chiến lược: cả bốn đề xuất đều nhắm vào **cách tổng hợp
tín hiệu**, mà ba lần đo hôm nay đều chỉ về cùng một chỗ — trần không nằm ở
cách tổng hợp. `xep_lai` no-op vì neo không nằm trong bể; `chen` lỗ vì bể chỉ
có 100 chỗ. **Cả hai đều nói: bể ứng viên là thứ đang thiếu, không phải phép
gộp.** Đó cũng là điều A27/A28 đã nói từ đầu, nay có thêm số liệu.

---

### A43 — Đắp 35 câu dev soạn từ hình, và cái bẫy rò văn bản hiện ra ở dạng thứ hai

Tập dev đã mù **sáu lần** (A19/A20/A31/A34/A37/A41) vì cùng một lý do: câu tự
soạn **22 từ / 1,39 mệnh đề**, đề thật **60 từ / 2,33**. Đo trên câu ngắn rồi
suy ra cho câu dài là chỗ hỏng lặp lại nhiều nhất trong repo. 23 câu đề thật
nạp ở A34 vá được một phần, nhưng 23 câu thì mọi phép so theo cặp đều nằm dưới
ngưỡng nhiễu.

**Đắp thêm 35 câu**, soạn bằng mắt từ contact sheet, chỉ trong năm nhóm máy này
có ảnh (L21/L22/L24/L27/L30 — 36.506 khung, 215 video, phủ 100% mỗi nhóm).
Công cụ mới: `scripts/39_chon_dai_soan.py` chọn dải, `scripts/38_soat_cau_dev_moi.py`
soát trước khi gộp.

| | n | từ (trung vị) | mệnh đề | ≥2 mệnh đề |
| --- | ---: | ---: | ---: | ---: |
| đề thật (DE1) | 22 | 60 | 2,36 | 77% |
| **mô phỏng (MP) — mới** | **37** | **71** | **2,65** | **100%** |
| tự soạn khác | 122 | 22 | 1,39 | 26% |

Tập dev **186 → 227**. TRAKE 42 → 46, trong đó 4 câu mới viết theo đúng khuôn
`trake-DE1-16` (câu dẫn tả cảnh mở đầu + `E1`/`E2`/`E3` **không dấu hai chấm** —
cố ý, để câu dev cũng đi qua đúng chỗ đã hỏng ở A40).

#### Ba luật rút ra khi chọn dải, đều là thứ đã suýt hỏng

* **Bỏ đầu và cuối video.** Gần như luôn là hình hiệu, MC trong trường quay,
  hoặc chữ chạy — giống hệt nhau giữa hàng trăm bản tin, nên câu viết ra không
  có đáp án duy nhất.
* **Không trùng video đã dùng làm đáp án.** Soạn thêm trên video cũ thì tập dev
  to ra mà độ phủ đứng yên.
* **Tả DẢI, không tả khung lẻ.** Khoảng đúng dài 4 giây – 5 phút (A9); tả một
  khung lẻ là quay lại đúng phân bố câu ngắn đã gây ra sáu lần mù.

#### Rò văn bản có DẠNG THỨ HAI, và cờ cũ không phân biệt được

`38_soat_cau_dev_moi.py` hỏi kênh OCR/ASR xếp đáp án ở hạng nào; hạng ≤ 10 là
cờ. Nó bắt được ba ca, và ba ca đó **không cùng loại**:

| ca | hạng | chẩn đoán | xử lý |
| --- | ---: | --- | --- |
| `kis-MP-02` (bản đầu) | 2 rồi 3 | tôi viết "khung giới hạn chiều cao" — đúng cụm trong bản tin chạy | **bỏ**, thay dải khác |
| `kis-MP-11` (bản đầu) | 5 | mở bằng "một đợt dịch bệnh ở châu Phi" — lấy từ dòng tin chạy | **viết lại** |
| `kis-MP-34` (bản đầu) | **1** | mở bằng "buổi dạy nhạc cụ tre nứa" — trùng gần nguyên cụm ASR | **viết lại** |

Ca thứ ba mới là ca dạy được điều gì. Tôi **không** đọc `ocr_asr.parquet` khi
soạn. Cụm đó không nằm trên khung hình. Vậy vì sao OCR xếp hạng 1?

Vì câu mở đầu ấy là **diễn giải việc đang diễn ra**, không phải tả cái nhìn
thấy. Mà diễn giải thì trùng lời thuyết minh là chuyện đương nhiên — ASR của
đúng khung đáp án nói *"chế tác một số loại nhạc cụ bằng tre nứa"*.

Đây là chỗ phải cẩn thận, vì có **hai thứ khác nhau** cùng làm hạng OCR cao:

* **RÒ** — câu chứa cụm chỉ có trong CHỮ hiển thị trên khung hình. Người soạn
  không nhìn thấy cụm đó, họ *đọc* nó. Câu hỏng, phải sửa hoặc bỏ.
* **TRÙNG TỰ NHIÊN** — câu tả đúng cái nhìn thấy, lời thuyết minh tình cờ gọi
  tên đúng thứ đó. **Đề thật cũng vậy**: người ra đề xem video, phát thanh viên
  đang nói về đúng cảnh ấy. Cắt bỏ loại trùng này là **thiên vị ngược** — làm
  tập dev bất công với kênh 3 và khiến mọi phép đo kênh 3 sau đó thấp giả.

Cờ hạng-≤-10 **không phân biệt được hai thứ đó**, nên script nay in kèm **những
từ trùng giữa câu hỏi và văn bản của chính khung đáp án, sắp theo IDF**:

```
kis-MP-34:
    8.35  tre_nứa      <- IDF cao, cụm dài, khớp nguyên văn ASR
    7.46  nhạc_cụ
    6.95  làm_bằng
    4.29  dạy
```

Quy tắc đọc: IDF cao + danh từ riêng/số/cụm dài → nghiêng về **rò**; IDF thấp +
từ tả cảnh thông thường → nghiêng về **trùng**. Vẫn là **cờ để người soạn đọc
lại**, không phải bằng chứng — nhưng nay người soạn có cái để nhìn.

Cách xử lý an toàn cho ca lưng chừng: **bỏ phần diễn giải, giữ phần tả vật thể**
rồi đo lại. `kis-MP-34` viết lại chỉ còn ống tre, rổ đan, vòng đội đầu kết hạt
đỏ → hạng rơi từ **1** xuống **ngoài top-100**. Chuyện đó tự nó là bằng chứng:
thứ kéo hạng lên là câu diễn giải, không phải vật thể.

Sau ba lần sửa: **35/35 câu sạch cờ**.

#### Hai giới hạn phải nói thẳng

**Người soạn chỉ nhìn được KEYFRAME, không xem được video.** Đề thật do người
*xem video* viết — đó chính là A38. Nên đây là **xấp xỉ** cách đề thật ra đời,
không phải bản sao. Hệ quả cụ thể ở TRAKE: chữ "đầu tiên" trong câu MP chỉ có
nghĩa "keyframe đầu tiên nhìn thấy được", mốc thật có thể sớm hơn vài giây. Mốc
nào không chắc là lần đầu thì đã viết thành **một cấu hình hình ảnh chỉ xuất
hiện một lần** thay vì dùng chữ "đầu tiên".

**Đáp án phải DUY NHẤT, và L24 gần như cố tình phá luật đó.** Cả nhóm L24 là hội
thi lân sư rồng — hàng chục video na ná nhau. Một đoạn múa lân trắng ban đêm ở
`L24_V015` đã bị **bỏ** vì mô tả của nó đúng luôn cho `kis-MP-05` (`L24_V027`,
cũng lân trắng ban đêm, cũng cờ đuôi nheo). Hai câu cùng đúng cho hai video khác
nhau thì đáp án không còn duy nhất, và câu đó đo ra số vô nghĩa. Bốn câu lân sư
rồng còn lại (`kis-MP-17..20`) cố ý đặt cạnh nhau và cố ý tả rõ **điểm phân
biệt** — rồng hay lân, màu lông, ngày hay đêm, trong nhà hay ngoài trời. Với
TRAKE thì **câu dẫn tả cảnh mở đầu** mới là thứ định danh video, không phải các
mốc E.

35 câu chưa đủ để lật kết luận nào — 35 so với 122 câu tự soạn cũ. Muốn dời cán
cân cần cỡ 60–80. Nhưng quy trình nay chạy được đầu-cuối (chọn dải → dựng sheet
→ soi → soát → gộp) nên cả nhóm chia nhau đắp tiếp được, với chính hai script này.


---

### A45 — Đo trên ĐỀ THẬT: kênh 1 tụt một nửa, kênh 3 vượt lên, và RRF thô thắng

Dựng `dev/tap_de_that.jsonl` — **52 câu do BTC viết** (23 đề đợt 1 + 29 đợt 2),
trung vị **62 từ / 2,29 mệnh đề**, đáp án là bài nộp nhóm đã soát. Đo lại đúng
những thứ đã đo trên tập dev tự soạn, trên 49 câu KIS/QA:

| | tập dev cũ (231 câu) | **tập đề thật (49 câu)** |
| --- | ---: | ---: |
| kênh 1 SigLIP2 | 0,3258 | **0,1429** |
| kênh 3 OCR/ASR | 0,1183 | **0,1633** |
| RRF(1,3) trọng số 1:1 | **−0,0144** ❌ | **+0,0694** ✅ |

**Ba thứ đảo ngược cùng lúc.**

**1. Tập dev cũ thổi phồng kênh 1 gấp 2,3 lần.** 0,3258 xuống 0,1429. Con số cũ
đo trên câu 22 từ do người *biết trước đáp án* viết.

**2. Kênh 3 nay MẠNH HƠN kênh 1** (0,1633 so với 0,1429). Tập dev cũ nói kênh 3
chỉ bằng một phần ba. Câu đề thật dài, nhiều mệnh đề, và hay có chữ đọc được
trong khung — đúng chỗ BM25 ăn.

**3. RRF thô, trọng số 1:1, ✅ ỔN ĐỊNH.** `+0,0694 / +0,0735`, vượt nhiễu ở ±2s
(ngưỡng 0,0623), thắng 8 thua 4. Đây chính là thứ A14 đo được là **có hại**
(−0,0144) trên tập dev cũ.

Và xu hướng trọng số **đảo ngược A23**:

| trọng số kênh phụ | 0,1 | 0,2 | 0,3 | 0,5 | **1,0** |
| --- | ---: | ---: | ---: | ---: | ---: |
| hiệu so mốc (±2s) | +0,0041 | +0,0122 | +0,0204 | +0,0327 | **+0,0694** |

Đơn điệu tăng tới 1,0. A23 đo được dìm xuống 0,3 làm tệ đi thì đúng — nhưng
chiều đúng là **nâng lên**, không phải hạ.

#### Nhóm đối chứng: có phải kênh 3 đang được chấm trên đáp án do chính nó sinh?

Ba câu trong tập đo (`kis-DE2-17`, `qa-DE2-19`, `qa-DE2-23`) do tôi **tra bằng
OCR** mà tìm ra. Nếu kênh 3 mạnh chỉ vì thế thì đây là rò ở tầng đáp án, cùng
loại A21. Tách hai nửa:

| tập con | n | kênh 1 | kênh 3 | RRF 1:1 (±2s / ±15s) |
| --- | ---: | ---: | ---: | --- |
| **DE1** — soát xong TRƯỚC khi tôi động vào OCR | 22 | 0,2000 | — | **+0,0909 / +0,0727** 🟡 |
| **DE2 bỏ 3 câu tôi tự tra** | 24 | 0,1000 | 0,1500 | **+0,0417 / +0,0500** 🟡 |

Cùng dấu ở cả hai nửa, cả bốn mức dung sai. Mỗi nửa không vượt nhiễu vì `n` chỉ
còn một nửa — đúng dáng của một hiệu ứng thật bị chia đôi mẫu, không phải dáng
của rò.

#### Hệ quả

* **Bật `--hop-nhat --bo-metadata --trong-so-phu 1.0` làm mặc định.** Không cần
  chờ dense hoá kênh 3, không cần model thứ hai.
* Mọi con số đo trên `dev/tap_dev.jsonl` từ A10 tới A44 đều phải đọc lại với
  giả định "kênh 1 được thổi lên gấp 2,3 lần".
* `dev/tap_de_that.jsonl` là thước đo chính từ nay.

⚠️ Đáp án của tập này là **bài nộp được 11,6 điểm**, không phải đáp án BTC. Có
câu sai trong đó mà không ai biết là câu nào. Mỗi câu mang nhãn `do_chac` trong
`ghi_chu`; phép đo nghiêm ngặt nên lọc còn nhãn `xong` (19/29 câu đợt 2).

⚠️ 3 câu TRAKE chưa đo được — cache truy vấn sinh trước bản vá A44 nên thiếu
chuỗi của `trake-DE2-21`.


---

### A46 — Trần của MỌI cách xếp lại là **63%**, và tác tử VLM chưa vượt được mốc nền

Dựng `src/tac_tu.py`: vòng lặp có công cụ, gửi keyframe cho Gemini xem và hỏi
"khung này khớp câu hỏi tới đâu", rồi xếp lại 100 dòng. Đây là bước định thay
người soát tay — thứ đã kéo bài nộp đợt 2 lên 11,6.

#### Đo TRẦN trước khi đo tác tử

Xếp lại chỉ hoán vị bể có sẵn. Nên câu đầu tiên phải hỏi là: **video đúng có
nằm trong 100 dòng không?** Đo trên 30 gói đợt 2:

| | |
| --- | ---: |
| video đúng CÓ trong top-100 | **19/30 = 63%** |
| trong đó ở hạng 1 | 6 |
| trong top-5 | 7 |
| trong top-20 | 12 |
| hạng trung vị | 8 |

**11/30 gói không cách nào cứu bằng xếp lại** — tác tử giỏi mấy cũng vô ích.
Đây là trần cứng, và nó khớp với điều rút ra từ bài nộp đội 14.800 điểm: chỗ mất
điểm là **truy hồi cấp video**, không phải hậu xử lý.

Số còn lại cũng đáng chú ý: 13/19 gói có video đúng trong bể nhưng **không ở
hạng 1** — rải tới hạng 38, 65, 70, 84, 87, 96. Đó mới là phần xếp lại ăn được.

#### Tác tử: hai lỗi thiết kế, đo mới thấy

Bản đầu chỉ soi **18 dòng đầu** rồi sắp thuần theo điểm VLM. Đo trên 4 gói:

| gói | video đúng | hạng trước | hạng sau |
| --- | --- | ---: | ---: |
| `p2-10` | L26_V120 | 6 | **8** ❌ |
| `p2-12` | L26_V192 | 10 | **23** ❌ |

Tệ đi cả hai gói đo được. Truy nguyên ra hai lỗi:

1. **Chỉ soi 18 dòng đầu** — mà video đúng hay nằm ở hạng 38–96, tức tác tử
   không bao giờ *nhìn thấy* đúng những gói nó có ích nhất. Sửa: dày ở đầu rồi
   rải đều hết 100 dòng.
2. **Ứng viên chưa được soi cũng nhận 0,0** rồi bị đẩy xuống dưới mọi dương tính
   giả. *Chưa soi không phải bằng chứng là sai.* Sửa: chỉ điểm ≥ 0,8 mới được
   nhấc lên, còn lại giữ nguyên thứ tự kênh.

Sau khi sửa cả hai:

| gói | hạng trước | hạng sau |
| --- | ---: | ---: |
| `p2-10` | 6 | 7 ❌ |
| `p2-12` | 10 | 10 ⚪ |

Ca thảm hoạ đã hết (10→23 thành 10→10), nhưng **vẫn không dương**. Nguyên nhân
còn lại: VLM chấm *"trông hợp lý"* chứ không chấm *"đúng cảnh này"* — nó cho
`p2-10` điểm **1,00** cho một video nấu ăn khác cùng chủ đề, đủ để hất ứng viên
đúng ở hạng 6 xuống 7.

#### Cờ `do_chac` đang NÓI DỐI

`p2-1` và `p2-11` có đáp án **ngoài top-100** — không cứu được. Tác tử vẫn báo
`do_chac = chac` cho `p2-1` vì tìm được một khung điểm 0,80. Người soát tin cờ
đó sẽ bỏ qua đúng gói cần mở nhất.

#### Kết luận

* **Không bật tác tử.** `src/tac_tu.py` giữ lại làm công cụ và làm nền để đo
  tiếp, mặc định không nằm trong đường ống.
* Việc đáng làm là **kéo 63% lên**, không phải xếp lại tinh vi hơn. Đúng thứ
  ngày 2–4 của kế hoạch đợt 3 nhắm tới.
* Trước khi tin tác tử lần sau: cờ `do_chac` phải được hiệu chỉnh trên gói **đã
  biết đáp án ngoài bể**, nếu không nó chỉ đo mức tự tin của VLM.

⚠️ `n = 4 gói, 2 gói đo được`. Không đủ để kết luận mạnh — nhưng cũng không có
lý do nào để bật một thứ chưa bao giờ đo được dương.

---

### A47. Ma trận thứ hai `ViT-gopt-16-SigLIP2-384` — thắng đậm, và lật ngược vai trò của SigLIP2

Đo 30/08 trên `dev/tap_de_that.jsonl` (**50/52 câu**; loại `trake-DE2-21` và
`trake-DE2-08` vì thiếu trong cache SigLIP2 — loại khỏi **cả hai** bên).
`clip_gopt.npy` phủ đủ 177.321 dòng, `dense.be_chung()` xác nhận bể chung
**177.321/177.321** nên không có hiệu ứng lệch độ phủ.

| cấu hình | ±2s | ±15s |
| --- | ---: | ---: |
| SigLIP2 một mình | 0,1400 | 0,1800 |
| **gopt một mình** | **0,3160** | **0,3920** |
| RRF(SigLIP2, OCR) — mặc định cũ | 0,2080 | 0,2520 |
| RRF(gopt, SigLIP2) | 0,2880 | 0,3320 |
| RRF(gopt, SigLIP2, OCR) | 0,3240 | 0,3840 |
| **RRF(gopt, OCR)** | **0,3440** | **0,4080** |

**Một mình gopt đã hơn cả cấu hình RRF cũ.** Hơn SigLIP2 một mình **2,3 lần**
— khoảng cách lớn hơn nhiều so với mong đợi từ một model cùng họ.

So theo cặp với mặc định cũ `RRF(SigLIP2, OCR)`:

| | hiệu ±2s | T-B-H | hiệu ±15s | T-B-H | |
| --- | ---: | :---: | ---: | :---: | --- |
| gopt một mình | +0,1080 | 17-8-25 | +0,1400 | 25-8-17 | ✅ ỔN ĐỊNH |
| RRF(gopt, SigLIP2, OCR) | +0,1160 | 15-3-32 | +0,1320 | 19-3-28 | ✅ ỔN ĐỊNH |
| **RRF(gopt, OCR)** | **+0,1360** | 15-2-33 | **+0,1560** | 20-3-27 | ✅ ỔN ĐỊNH |

**SigLIP2 giờ là kênh làm hại, không phải kênh bổ sung.** Đổi mốc nền sang
`RRF(gopt, OCR)` rồi đo lại: thêm SigLIP2 vào cho **−0,0200 / −0,0240**,
thắng-thua-hoà **4-10-36** ở cả hai mức. Cùng dấu, nhất quán, nhưng **🟡 YẾU** —
chưa vượt nhiễu, nên chưa được tuyên bố là chắc. Ghi lại để đo tiếp khi có
thêm câu.

Vai trò đảo hẳn so với A45: ở đó SigLIP2 là kênh chính và OCR là kênh vượt lên
bất ngờ. Giờ **gopt là kênh chính**, OCR vẫn bổ sung có lãi, còn SigLIP2 không
còn chỗ.

**Đã đổi mặc định `src/run.py --matrix` thành `clip_gopt.npy`** (đi kèm
`index/truy_van_gopt.npz`).

> ⚠️ **Vẫn chưa đo trọng số RRF.** Cả bảng trên dùng 1:1. A45 cho thấy trọng số
> đổi kết quả đáng kể, nên đây là việc tiếp theo — và phải đổi MỘT thứ mỗi lần,
> đúng như đã làm ở đây.

> **Điều chưa giải thích được.** Vì sao hai model cùng họ SigLIP2, cùng dữ liệu
> `webli`, lại chênh 2,3 lần? gopt lớn hơn (~1,1 tỷ so với 400 triệu tham số) và
> vào ảnh ở 384px thay vì 378px, nhưng ngần ấy không đủ giải thích. Một khả năng
> đáng nghi: `clip_siglip2.npy` được dựng trên máy khác, và sidecar của nó từng
> ghi `pretrained` là đường dẫn `.safetensors` cục bộ (A17) — nếu lúc dựng đã
> nạp nhầm biến thể thì SigLIP2 yếu vì lý do kỹ thuật chứ không phải vì model.
> Chưa kiểm được. Không ảnh hưởng kết luận chọn gopt, nhưng ảnh hưởng câu
> "SigLIP2 có đáng giữ làm kênh thứ ba không".

### A48. Dò tham số RRF(gopt, OCR) — không tham số nào đủ chắc để đổi

Đo 30/08 trên `dev/tap_de_that.jsonl`, 50/52 câu, cùng bộ câu với A47. Ba thứ
được dò **riêng từng cái**, mốc nền luôn là cấu hình đang chạy.

#### 1. Trọng số gopt : OCR — một chiều bị bác dứt khoát

| tỉ lệ | ±2s | ±15s | hiệu ±2s | T-B-H | |
| --- | ---: | ---: | ---: | :---: | --- |
| 1 : 3 | 0,1720 | 0,2120 | −0,1720 | 4-18-28 | ✅ ỔN ĐỊNH **tệ hơn** |
| 1 : 2 | 0,2320 | 0,2640 | −0,1120 | 4-15-31 | ✅ ỔN ĐỊNH **tệ hơn** |
| 1 : 1,5 | 0,2600 | 0,3160 | −0,0840 | 3-14-33 | ✅ ỔN ĐỊNH **tệ hơn** |
| **1 : 1** | **0,3440** | **0,4080** | — | — | mốc |
| 1 : 0,75 | 0,3560 | 0,4280 | +0,0120 | 6-3-41 | 🟡 YẾU |
| 1 : 0,5 | 0,3520 | 0,4320 | +0,0080 | 10-5-35 | 🟡 YẾU |
| 1 : 0,25 | 0,3440 | 0,4200 | −0,0000 | 12-7-31 | ❌ ĐẢO DẤU |

**Tăng trọng số OCR là hỏng chắc chắn** — ba mức đều ✅ ỔN ĐỊNH theo hướng xấu,
càng tăng càng tệ. Giảm thì *có vẻ* nhỉnh hơn nhưng cả hai mức đều 🟡 YẾU.

Giữ **1 : 1**. `1 : 0,75` là ứng viên duy nhất đáng đo lại khi tập dev lớn hơn —
điểm thô cao nhất bảng, thắng-thua 6-3 và 9-3, nhưng 41/50 câu không đổi gì nên
hiệu tuyệt đối quá nhỏ để vượt nhiễu.

#### 2. SigLIP2 làm "gợi ý phụ" trọng số nhỏ — giả thuyết bị bác

Ý tưởng: A47 thấy SigLIP2 làm hại ở trọng số 1:1, nhưng có thể nó vẫn cứu được
vài ca gopt bỏ sót nếu chỉ cho trọng số nhỏ. Đo trên `RRF(gopt, OCR)` + SigLIP2:

| trọng số SigLIP2 | ±2s | ±15s | T-B-H (±2s) | |
| ---: | ---: | ---: | :---: | --- |
| 1,0 | 0,3240 | 0,3840 | 4-10-36 | 🟡 YẾU, tệ hơn |
| 0,5 | 0,3440 | 0,4080 | 3-2-45 | ⚪ KHÔNG ĐỔI GÌ |
| 0,33 | 0,3440 | 0,4080 | 3-2-45 | 🟡 YẾU |
| 0,25 | 0,3400 | 0,4000 | 2-2-46 | 🟡 YẾU |
| 0,2 | 0,3440 | 0,4040 | 2-2-46 | ❌ ĐẢO DẤU |

**Ở trọng số nhỏ, SigLIP2 không cứu gì cả — nó chỉ không làm gì.** 42–46 trên 50
câu *không đổi một chút nào*. Không có vùng trọng số nào mà nó vừa đủ nhẹ để
không hại vừa đủ nặng để có ích.

Kết luận: **bỏ hẳn SigLIP2 khỏi đường chạy**, không phải hạ trọng số nó.

#### 3. Hằng số `k` của RRF — gần như trơ

| k | ±2s | ±15s | T-B-H (±2s) | |
| ---: | ---: | ---: | :---: | --- |
| 20 | 0,3440 | 0,4120 | 3-3-44 | 🟡 YẾU |
| 30 | 0,3440 | 0,4080 | 1-1-48 | ⚪ KHÔNG ĐỔI GÌ |
| **60** | **0,3440** | **0,4080** | — | mốc |
| 100 | 0,3440 | 0,4080 | **0-0-50** | ⚪ KHÔNG ĐỔI GÌ |
| 120 | 0,3440 | 0,4080 | **0-0-50** | ⚪ KHÔNG ĐỔI GÌ |
| 200 | 0,3480 | 0,4120 | 1-0-49 | 🟡 YẾU |

`k = 100` và `k = 120` cho **0-0-50** — không một câu nào đổi thứ hạng.

Điều đó tự nó nói một chuyện: hai kênh **hiếm khi tranh chấp**. `k` chỉ có việc
làm khi hai kênh đề cử cùng một ứng viên ở hai thứ hạng rất khác nhau; ở đây
OCR đóng góp quá ít ứng viên chen được vào vùng gopt đã xếp. Cũng khớp với việc
`RRF(gopt, OCR)` chỉ hơn `gopt` một mình +0,0280 (🟡 YẾU).

#### Kết luận chung

**Không đổi gì.** Giữ `RRF(gopt, OCR)`, trọng số 1:1, k=60. Ba trục tham số đã
dò hết và không trục nào còn dư địa đáng kể — nghĩa là **muốn tiến tiếp thì phải
thêm TÍN HIỆU MỚI, không phải chỉnh cách trộn tín hiệu cũ**.

Ứng viên tín hiệu mới, theo thứ tự đáng làm:

1. **Kênh 5 (caption)** — kênh văn bản dày, mô tả *quan hệ trong cảnh*, đúng chỗ
   A8.4 nói không kênh nào phủ. Notebook sẵn sàng, chưa ai chạy.
2. **Mở rộng truy vấn bằng LLM** — viết lại câu hỏi thành nhiều biến thể rồi
   hợp nhất. Không cần dữ liệu mới, chỉ cần một lượt gọi mỗi câu.

> ⚠️ Và nhớ A47: trần top-100 của `gopt` trần là **80%**, của `RRF(gopt, OCR)`
> chỉ **74%**. Kênh mới nào cũng nên nhận ứng viên từ gopt trần rồi mới hợp nhất.

### A49. Mã hoá truy vấn: **69 chuỗi/giây** — bước chặn ngày thi đã hết chặn

Đo 31/08 trên Tesla T4, `--lo 64`, cả hai model:

| model | chuỗi | chuỗi/giây | thời gian |
| --- | ---: | ---: | ---: |
| `ViT-gopt-16-SigLIP2-384` | 1.158 | **69** | 17 giây |
| `ViT-SO400M-14-SigLIP2-378` | 1.158 | **68** | 17 giây |

Trước khi vá, `25_ma_hoa_truy_van.py` mã hoá **từng câu một trên CPU** — không có
`.to(device)` nào, và `tok([c])` mỗi vòng lặp. Bật GPU cũng vô ích. Với tháp văn
bản của gopt, 1.158 chuỗi mất hàng chục phút trong khi GPU nằm không.

Viết vậy là đúng lúc script ra đời: nó sinh ra cho máy 7,7 GB **không có GPU**,
nơi lựa chọn duy nhất là CPU. Chỉ khi kho có model lớn hơn và chạy trên Kaggle
thì cái mặc định đó mới thành nút thắt.

#### Vì sao con số này quan trọng hơn nó trông

Lúc thi, **mã hoá đề mới là bước chặn DUY NHẤT** giữa lúc nhận đề và lúc chạy
được kênh 1. Ma trận ảnh, `master.parquet`, chỉ mục BM25 — tất cả đã tính sẵn.

| việc | chuỗi | thời gian |
| --- | ---: | ---: |
| đề sơ tuyển đợt 2 (30 gói) | 71 | **1,0 giây** |
| giả sử đợt 3 gấp đôi | 150 | 2,2 giây |
| cả tập dev 323 câu | 1.158 | 17 giây |

**Mã hoá đề không còn là việc chậm.** Chỗ chậm duy nhất còn lại là **tải 7,49 GB
trọng số** (~40 giây khi mạng tốt), và nó chỉ xảy ra khi phiên Kaggle mới khởi
động.

> **Hệ quả cho quy trình thi:** mở sẵn một phiên Kaggle **trước giờ thi** với
> model đã nạp. Nhận đề → dán → chạy một cell → vài giây có `.npz`. Không phải
> chờ tải model dưới áp lực thời gian.

> ⚠️ Một chi tiết dễ sai khi gộp lô: chuẩn hoá L2 phải theo **từng dòng**
> (`axis=1`). Chuẩn hoá cả khối là chia nhầm chuẩn của cả lô vào từng vector —
> file vẫn hợp lệ, cosine vẫn trong [−1, 1], và **không có gì báo**.

### A50. Hai tập dev nói ngược nhau — và câu tự soạn **không thay được đề thật**, kể cả khi khớp phân bố

Sau khi sinh lại cache cho 323 câu, đo lại A47 và ra kết quả **trái ngược**:

| `RRF(gopt, OCR)` so với `gopt` trần | hiệu ±2s | T-B-H | |
| --- | ---: | :---: | --- |
| trên `tap_de_that` (50 câu) | **+0,0280** | 11-10-29 | 🟡 |
| trên `tap_dev` (323 câu) | **−0,0248** | 49-115-159 | ✅ ỔN ĐỊNH |

Cả hai không thể cùng đúng. Và **không chọn được bằng cách nhìn hai bảng**: chúng
khác nhau ở HAI thứ cùng lúc — cỡ mẫu (50 so với 323) và nguồn câu hỏi. Đúng cái
lỗi "đổi hai thứ rồi quy công cho nhầm cái".

#### Phép phân xử: chia theo NGUỒN, đo riêng (`scripts/54_do_theo_nguon_cau.py`)

Mốc nền `gopt` trần, cùng ba cấu hình, trên từng nhóm:

| nhóm | số câu | `RRF 1:1` | | `RRF 1:0,75` | |
| --- | ---: | ---: | --- | ---: | --- |
| **đề thật** | 52 | +0,0346 | 🟡 | **+0,0471** | **✅ ỔN ĐỊNH** |
| mới sát đề thật | 63 | −0,0175 | 🟡 | −0,0016 | 🟡 |
| tự soạn cũ | 208 | −0,0419 | ✅ **tệ hơn** | −0,0224 | ✅ **tệ hơn** |

**Bảng 323 câu bị 208 câu tự soạn cũ chi phối.** Trên đúng thứ đem đi thi, thêm
kênh 3 vẫn có lãi — và ở trọng số 0,75 thì lãi đó **vượt nhiễu**.

#### Phát hiện đắt hơn: khớp phân bố KHÔNG đủ để thay đề thật

63 câu soạn ngày 30/08 được viết **cố ý khớp phân bố đề thật** — 72 từ / 2,80
mệnh đề, so với đề thật 62 từ / 2,33. Đó chính là điều A43 đặt ra để sửa lỗi
"câu tự soạn quá ngắn".

Nhưng chúng **cư xử như câu tự soạn, không như đề thật**: OCR làm hại (−0,0175),
cùng dấu với nhóm tự soạn cũ và ngược dấu với đề thật.

Nghĩa là thứ làm đề thật khác biệt **không phải độ dài câu**. Giả thuyết đáng
tin nhất: câu tự soạn được viết **trong lúc nhìn keyframe**, nên đương nhiên tả
đúng những gì nhìn thấy — kênh ảnh tìm ra dễ, kênh văn bản thành thừa. Đề thật
do BTC viết với ý đồ khác, và ở đó tín hiệu văn bản độc lập mới có chỗ.

> ⚠️ **Hệ quả cho cách soạn tập dev.** Đếm từ và đếm mệnh đề là thứ *đo được*
> nên dễ nhắm tới, nhưng nó không phải thứ *quan trọng*. Câu tự soạn dùng để đo
> **phủ** (nhóm L nào, loại câu nào) thì tốt; dùng để **quyết định cấu hình** thì
> không thay được đề thật. Mọi quyết định bật/tắt từ nay đọc trên
> `tap_de_that.jsonl`, và ghi rõ số đó ra khi báo cáo.

#### Đã đổi

`src/run.py --trong-so-phu` mặc định **1,0 → 0,75**.

A45 từng đo hiệu tăng **đơn điệu** tới trọng số 1,0 và kết luận "hạ xuống là mất
lãi". Điều đó vẫn đúng — **với kênh 1 cũ**. Đổi kênh 1 sang gopt thì kênh ảnh
mạnh hơn hẳn, và điểm tối ưu của kênh phụ dịch xuống. Một hằng số đúng luôn gắn
với cấu hình đo ra nó.

> **Còn mở:** trên 52 câu, `1:0,75` vượt nhiễu ở ±2s (ngưỡng 0,0467, hiệu 0,0471)
> nhưng sát ngưỡng ở ±15s (0,0487 so với 0,0462). Cùng dấu, một mức vượt — đủ
> theo định nghĩa `ON DINH`, nhưng đây là ca sát ranh nhất từng nhận. Có thêm câu
> đề thật thì đo lại.

### A51. Hợp nhất mệnh đề bằng **RRF hạng**, không phải max cosine — và A47–A50 đo sai cấu hình

#### Trước hết: một lỗi phương pháp của chính các phép đo A47–A50

Tháp văn bản SigLIP2 có `context_length = 64`. Đo bằng chính tokenizer trên đề
thật:

| | token |
| --- | ---: |
| cả câu nguyên | trung vị **64** — chạm trần, **12/20 câu bị cắt cụt** |
| từng mệnh đề | trung vị 34, max 47 |

`run.py` biết điều đó nên gọi `tach_truy_van()` trước (A19/A20). Nhưng các
script đo A47–A50 truyền thẳng `c.cau_hoi` — **đo một cấu hình `run.py` không
dùng**, với truy vấn mất phần đuôi.

    cả câu (cái A47-A50 đo)     0,3125 / 0,3913
    mệnh đề, max cosine (thật)  0,4375 / 0,4981     +0,1250 ✅ ỔN ĐỊNH

So sánh *giữa các cấu hình* vẫn công bằng vì đều cắt như nhau, nên kết luận
tương đối của A47–A50 (gopt thắng SigLIP2; kênh 3 có lãi trên đề thật) vẫn
đứng. Nhưng **mọi điểm tuyệt đối đã ghi đều thấp hơn thực tế**, và trọng số
`0,75` của A50 được chọn dưới điều kiện sai — cần đo lại.

#### Cải tiến thật: RRF hạng giữa các mệnh đề

`KenhAnh.tim` nhận danh sách mệnh đề và lấy **max cosine** trên từng keyframe.
Nghe hợp lý, nhưng cosine của hai mệnh đề KHÁC NHAU không so được với nhau:
*"một người phụ nữ"* khớp mờ với hàng nghìn khung ở cos cao, còn *"biển hiệu
màu tím ghi BỆNH VIỆN"* khớp đúng một khung ở cos thấp hơn. Max cosine để
**mệnh đề dễ nuốt mệnh đề đặc trưng**.

Đây đúng lý do repo hợp nhất KÊNH bằng RRF chứ không cộng điểm (`schema.py`) —
chỉ là chưa ai áp cùng lý lẽ đó cho các mệnh đề trong một truy vấn.

Đo trên 52 câu đề thật, mốc nền là cấu hình `run.py` đang chạy:

| cấu hình | ±2s | ±15s | hiệu ±2s | T-B-H | |
| --- | ---: | ---: | ---: | :---: | --- |
| mệnh đề max cosine — *mốc* | 0,4375 | 0,4981 | — | — | |
| mệnh đề **RRF hạng** | 0,4779 | 0,5712 | +0,0404 | 15-13-24 | 🟡 |
| max cosine + kênh 3 | 0,4644 | 0,5317 | +0,0269 | 9-9-34 | 🟡 |
| **RRF hạng + kênh 3** | **0,5096** | **0,5952** | **+0,0721** | 18-9-25 | **✅ ỔN ĐỊNH** |
| RRF hạng + cả câu + kênh 3 | 0,4875 | 0,5683 | +0,0500 | 15-11-26 | 🟡 |

**Chỉ tổ hợp cả hai mới vượt nhiễu.** Từng thứ riêng lẻ đều 🟡 — đây là ca hiếm
mà hai thay đổi nhỏ cộng lại mới đủ, và nếu đo lần lượt từng cái thì đã bỏ qua
cả hai.

Cũng đáng ghi: **thêm cả câu nguyên vào RRF làm TỆ ĐI** (0,4875 so với 0,5096).
Câu nguyên bị cắt ở token 64 nên nó là một truy vấn *hỏng*; đưa vào hợp nhất
chỉ kéo xuống. Cắt cụt không chỉ vô ích — nó có hại.

#### Đã đổi

`src/run.py` hợp nhất mệnh đề bằng RRF (`--khong-rrf-menh-de` để dựng lại hành
vi cũ). Điểm cấu hình mặc định trên đề thật: **0,4375 → 0,5096**.

> **Việc còn lại:** dò lại trọng số kênh 3 dưới cách đưa truy vấn ĐÚNG. A50
> chọn 0,75 khi truy vấn còn bị cắt cụt, nên con số đó không còn nền.

### A52. Trọng số kênh 3: **0,5**, và tập dev 323 câu trả lời sai câu hỏi này

Việc còn lại của A51. `0,75` (A50) được chọn khi script đo còn truyền cả câu
vào kênh 1 — tức kênh 1 chạy với truy vấn cắt cụt ở token 64. Trọng số là con
số nói "kênh 3 đáng tin bao nhiêu SO VỚI kênh 1"; làm kênh 1 mạnh lên thì nền
của tỉ lệ đó đổi. Đo lại bằng `scripts/57_do_lai_trong_so_kenh3.py`, mốc nền là
cấu hình `run.py` đang chạy sau A51.

#### Hai tập nói ngược nhau — lần thứ hai

| w | đề thật (52) | tập dev (323) |
| ---: | ---: | ---: |
| 0 — bỏ hẳn kênh 3 | 0,4779 | **0,5857** |
| 0,25 | 0,5096 | **0,5899** |
| 0,5 | **0,5173** | 0,5848 |
| 0,75 — *mốc* | 0,5096 | 0,5695 |
| 1,0 | 0,4788 | 0,5338 |
| 1,5 | 0,3394 | 0,3571 |
| 2,0 | 0,2865 | 0,2623 |

Trên đề thật, bỏ kênh 3 làm TỆ đi. Trên tập dev, bỏ kênh 3 làm TỐT lên
(+0,0162 ✅). Ngược dấu, không phải chênh lệch cỡ mẫu.

#### Tách theo nguồn câu (cùng bộ nhớ đệm, chỉ báo cáo là tách)

Hiệu so với mốc 0,75, mức ±2s:

| w | đề thật (52) | mới sát đề (63) | tự soạn cũ (208) |
| ---: | ---: | ---: | ---: |
| 0 | **−0,0317** 🟡 | +0,0135 🟡 | **+0,0290 ✅** |
| 0,25 | −0,0000 ❌ | +0,0111 🟡 | +0,0283 ✅ |
| 0,5 | **+0,0077** 🟡 | +0,0111 🟡 | +0,0185 ✅ |
| 1,0 | −0,0308 ✅ | −0,0270 ✅ | −0,0396 ✅ |
| 1,5 | −0,1702 ✅ | −0,2452 ✅ | −0,2131 ✅ |
| 2,0 | −0,2231 ✅ | −0,3595 ✅ | −0,3124 ✅ |

Mâu thuẫn là **do nguồn câu hỏi**, đúng như A50 đã gặp: 208 câu tự soạn cũ chi
phối bảng 323 câu, và với chúng kênh 3 chỉ là nhiễu — câu tự soạn hầu như không
trích chữ trên màn hình, thứ duy nhất kênh 3 biết đọc.

Đáng ghi thêm: nhóm "mới sát đề thật" — 63 câu cố ý soạn khớp phân bố đề thật —
xử sự giống nhóm **tự soạn**, không giống đề thật. Củng cố A50: khớp phân bố độ
dài KHÔNG làm câu tự soạn thay được câu BTC viết.

#### Chốt: 0,5

Hai điều đúng ở **cả ba nhóm**, không nhóm nào phản đối:

* **w ≥ 1 là có hại** — ✅ ỔN ĐỊNH ở cả ba, và tụt rất nhanh (w=2 mất hơn 0,30
  điểm). Mặc định `1.0` trước A50 nay có bằng chứng chắc là SAI, không chỉ
  "không lợi".
* **0,5 hơn 0,75** ở cả ba nhóm.

Còn 0 hay 0,5 thì hai nguồn câu muốn hai hướng. Đo thẳng cặp đó trên đề thật,
lấy 0,5 làm mốc:

    w = 0 so với w = 0,5     −0,0394 / −0,0385   1-8-43   ✅ ỔN ĐỊNH

Trên đúng loại câu sẽ gặp trong phòng thi, bỏ kênh 3 là **mất điểm chắc chắn**.
`--trong-so-phu` mặc định **0,75 → 0,5**.

> Không kết luận từ bảng 323 câu. Nó không phải "nhiều dữ liệu hơn nên đáng tin
> hơn" — nó là **một câu hỏi khác**, hỏi về loại câu ta không đi thi.

### A53. Kênh 6 — nhúng OCR/ASR bằng tháp văn bản gopt — **KHÔNG CHẠY**. Bỏ.

Ý tưởng: kênh 3 dùng BM25 nên khớp **mặt chữ** — truy vấn "xe cứu thương" mà
bản tin viết "xe cấp cứu" thì điểm bằng 0. Nhúng cùng văn bản đó vào không gian
gopt 1536 chiều thì hai cách gọi nằm gần nhau, lại còn chung không gian với ảnh.

Đã dựng xong: 462.085 đoạn ≤ 60 token từ 176.009 tài liệu (2,63 đoạn/tài liệu),
gộp theo `row_id` bằng max. Đo trên 52 câu đề thật, mốc là cấu hình `run.py`
sau A52:

| cấu hình | ±2s | ±15s | hiệu ±2s | T-B-H | |
| --- | ---: | ---: | ---: | :---: | :---: |
| ảnh + kênh 3 (0,5) — *mốc* | 0,5173 | 0,6096 | — | — | |
| + kênh 6 (0,25) | 0,5163 | 0,6067 | −0,0010 | 1-1-50 | 🟡 |
| + kênh 6 (0,5) | 0,5125 | 0,6067 | −0,0048 | 1-3-48 | 🟡 |
| + kênh 6 (1,0) | 0,4269 | 0,5337 | −0,0904 | 2-24-26 | ✅ |
| kênh 6 THAY kênh 3 | 0,4740 | 0,5702 | −0,0433 | 1-11-40 | ✅ |
| **chỉ kênh 6** *(chẩn đoán)* | **0,0490** | 0,1000 | −0,4683 | 1-36-15 | ✅ |
| chỉ kênh 1 *(chẩn đoán)* | 0,4779 | 0,5712 | −0,0394 | 1-8-43 | ✅ |

#### Dòng quan trọng nhất là dòng chẩn đoán

**0,0490.** Kênh 6 đứng một mình gần như không truy hồi được gì. Mọi con số hợp
nhất phía trên chỉ là kênh 1 đội lốt — thêm kênh 6 vào không "hơi có lãi", nó
đang pha nhiễu vào một danh sách tốt, và trọng số càng lớn càng lộ (w=1 mất
−0,0904 ✅).

Không có dòng "chỉ kênh 6" thì bảng này đọc ra "kênh 6 trung tính, để đó cũng
được" — sai hoàn toàn. **Kênh nào cũng phải có một dòng đứng một mình.**

#### Đã loại trừ khả năng lệch file trước khi kết luận

Kết quả xấu vì ý tưởng sai và kết quả xấu vì ghép nhầm hàng trông y hệt nhau:

    tập row_id khớp đúng tập tài liệu có chữ     ✅
    thứ tự sinh giữ nguyên thứ tự bảng            ✅
    tương quan (số đoạn) vs (độ dài văn bản)      0,9553

File đúng. Kênh sai.

#### Vì sao sai — đã ghi sẵn trong docstring lúc dựng

SigLIP2 huấn luyện để khớp **ảnh ↔ chữ**, không phải **chữ ↔ chữ**. Đem vector
truy vấn so với vector tài liệu là dùng model ngoài phân bố huấn luyện; hai
loại vector nằm hai cụm khác nhau (modality gap), nên khoảng cách giữa chúng
gần như không mang thông tin. Nghi ngờ này đã viết ra TRƯỚC khi đo — và phép đo
xác nhận. Chi phí để biết: một lượt Kaggle ~40 phút.

#### Chốt

`run.py` **không** bật kênh 6. Giữ `src/van_ban_dense.py` và
`index/van_ban_gopt/` (1,32 GB) vì chúng vẫn đúng và có thể dùng lại nếu sau
này có model text–text thật (Vietnamese SBERT chẳng hạn) — lúc đó chỉ cần thay
ma trận, mã truy hồi không phải sửa.

> Muốn sửa cái yếu của BM25 (khớp mặt chữ) thì phải dùng model biết so **chữ
> với chữ**. Dùng tháp văn bản của một model ảnh–chữ là giải sai bài.

### A54. Khoảng trống giữa điểm thật và trần "xếp lại hoàn hảo": **33 điểm phần trăm**

Trước khi đầu tư vào reranker (món đắt: chấm lại ~100 cặp ảnh–truy vấn mỗi
câu), phải biết nó có chỗ để thắng không. Chỗ đó đo được chính xác:

    TRẦN  = đáp án nằm ĐÂU ĐÓ trong bể  ->  xếp lại hoàn hảo cho 1,0
    THẬT  = điểm BTC hiện tại
    TRỐNG = TRẦN − THẬT

`scripts/60_do_khoang_trong_rerank.py`, 52 câu đề thật, cấu hình `run.py` sau
A52, dung sai ±2s:

| bể | R@1 | R@20 | R@100 | THẬT | TRẦN | TRỐNG | câu ngoài bể |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0,2041 | 0,6122 | 0,7551 | 0,5173 | 0,7500 | +0,2327 | 12/49 |
| 300 | 0,1837 | 0,6122 | 0,7755 | 0,5202 | 0,8269 | +0,3067 | 8/49 |
| **1000** | 0,2245 | 0,6122 | 0,7959 | **0,5317** | **0,8654** | **+0,3337** | **6/49** |

Ở ±15s: trần **0,9231**, chỉ 3/49 câu ngoài bể.

**Ba điều đọc ra:**

1. **Xếp lại là trục lãi nhất.** 33 điểm phần trăm nằm sẵn trong bể. Cả A51 +
   A52 cộng lại mới được +0,08 — khoảng trống này gấp bốn lần.
2. **Nới bể nâng TRẦN, nhưng KHÔNG nâng điểm thật** (xem đính chính dưới).
   Số câu vô vọng giảm **12 → 6**. Bài nộp vẫn 100 dòng — giới hạn của BTC là
   số DÒNG NỘP, không phải cỡ bể, nên bể lớn là chỗ hợp lệ cho reranker soi.
3. **R@20 đứng yên 0,6122 ở cả ba cỡ bể.** Nới bể chỉ cứu câu nằm sâu, không
   kéo thêm câu nào vào top-20. Hai việc tách bạch: nới bể lo phần đuôi, xếp
   lại lo phần đầu.

**6 câu vẫn ngoài top-1000/177.321.** Với chúng mọi hậu xử lý đều vô nghĩa —
đó là phần duy nhất một model mới có thể cứu, và nó chỉ đáng tối đa 12% số câu.

#### Đính chính (`scripts/65_do_co_be.py`)

Bảng trên là ba lần chạy RIÊNG, và tôi đã đọc chênh lệch giữa chúng như một
khoản lãi: *"nới bể tự nó có lãi nhỏ, 0,5173 → 0,5317"*. **Sai** — chưa so theo
cặp thì chưa được nói thế. Đo lại đúng cách, mốc là bể 100:

| bể | ±2s | ±15s | hiệu ±2s | T-B-H | |
| ---: | ---: | ---: | ---: | :---: | :---: |
| 100 — *mốc* | 0,5173 | 0,6096 | — | — | |
| 200 | 0,5163 | 0,6029 | −0,0010 | 2-2-48 | 🟡 |
| 300 | 0,5202 | 0,5952 | +0,0029 | 3-2-47 | ❌ |
| 600 | 0,5240 | 0,6029 | +0,0067 | 4-2-46 | ❌ |
| 1000 | 0,5317 | 0,6067 | +0,0144 | 6-2-44 | ❌ |

Ở ±2s bể lớn hơn thật, ở ±15s thì kém đi — **đảo dấu ở mọi cỡ đáng kể**. Cửa sổ
chấm của BTC là ẩn số (4s–5 phút), nên kết luận phụ thuộc nó thì không dùng
được. `run.py` **giữ bể 100**.

Vì sao đảo dấu: bể lớn kéo vào những khung mà cả hai kênh đều xếp rất sâu. Ở
cửa hẹp chúng thỉnh thoảng đúng và ăn điểm; ở cửa rộng thì những khung *gần
đúng* của bể nhỏ vốn đã được tính đúng rồi, nên thứ mới thêm chỉ chen lên trước
chúng. Cùng một thay đổi, hai cửa sổ, hai dấu.

> Bài học lặp lại lần thứ ba trong dự án: **chênh lệch giữa hai bảng chạy riêng
> KHÔNG phải một phép đo.** Chỉ so theo cặp trên cùng bộ câu mới nói được điều
> gì — và ở đây nó lật ngược một câu đã kịp vào tài liệu.

### A55. Ba cách xếp lại **không cần model** — đo cả ba, **bác cả ba**

Nếu 33 điểm phần trăm lấy được bằng tín hiệu sẵn có thì khỏi cần VLM. Đo thử
ba tín hiệu đang bị vứt đi, trên bể 300, mốc là `run.py` sau A52.

#### 1. Gom khung trùng cảnh theo thời gian (`61_`)

| gom trong | ±2s | ±15s | |
| --- | ---: | ---: | :---: |
| *không gom (mốc)* | 0,5202 | 0,5952 | |
| 5s | 0,5077 | 0,5990 | ❌ |
| 15s | 0,4731 | **0,6212** | ❌ |
| 60s | 0,4231 | 0,5798 | ✅ TỆ HƠN |
| mỗi video 1 dòng | 0,3385 | 0,4298 | ✅ TỆ HƠN |

Đảo dấu đúng như dự đoán ghi trước khi chạy: người đại diện được giữ nằm
NGOÀI cửa sổ hẹp còn kẻ bị bỏ nằm TRONG. Cửa sổ BTC là ẩn số (4s–5 phút), nên
kết luận phụ thuộc nó thì không dùng được, dù con số ±15s trông đẹp.

**Chẩn đoán tiền đề — và đây mới là phần đáng nhớ.** Ý này chép từ một nhóm
khác, tiền đề của họ là "top-15 thường có 6–7 khung cùng một cảnh". Đo trên hệ
của ta:

    top-20: 10,2/20 video RIÊNG BIỆT
            1,2 khung trùng cảnh trong 2s | 2,8 trong 5s | 5,9 trong 15s

Bể của ta **không vón cục**. Ý đó chữa một bệnh kênh 1 của ta không mắc — họ
dùng model yếu hơn nên bể của họ dồn cục hơn nhiều. Chép giải pháp mà không
chép chẩn đoán là cách nhanh nhất để chữa nhầm bệnh.

#### 2. Đồng thuận mệnh đề và 3. ủng hộ theo video (`62_`)

| cấu hình | ±2s | hiệu | T-B-H | |
| --- | ---: | ---: | :---: | :---: |
| *mốc* | 0,5202 | — | | |
| đồng thuận mệnh đề w=0,1 | 0,5202 | +0,0000 | 1-1-50 | 🟡 |
| đồng thuận mệnh đề w=0,5 | 0,5038 | −0,0163 | 2-7-43 | ✅ TỆ HƠN |
| **ủng hộ video w=0,25** | **0,5279** | **+0,0077** | 5-2-45 | 🟡 |
| ủng hộ video w=1 | 0,5240 | +0,0038 | 6-3-43 | 🟡 |
| cả hai | 0,5279 | +0,0077 | 5-2-45 | ❌ |

Tốt nhất là +0,0077 với ngưỡng nhiễu 0,0245 — **bằng 2,3% của khoảng trống**.

Đồng thuận mệnh đề còn LÀM HẠI ở trọng số cao. Lý do là mặt trái của chính
A51: ứng viên trúng nhiều mệnh đề thường là khung *chung chung* khớp mờ với
nhiều mệnh đề dễ, không phải khung đặc trưng khớp đúng một mệnh đề khó.

#### Kết luận có giá trị nhất của cả cụm A54–A55

**Khoảng trống 33 điểm KHÔNG lấy được bằng cách sắp xếp lại thông tin đã có.**
Ba tín hiệu miễn phí, sáu mức trọng số, tổng cộng lấy được ~0. Muốn lấp nó thì
phải đưa **thông tin MỚI** vào — tức là thật sự nhìn lại bức ảnh (VLM rerank)
hoặc mô tả nó bằng chữ (caption, kênh 5). Không có đường tắt.

> Giá trị của A55 nằm ở chỗ nó rẻ và nó ĐÓNG một hướng. Ba ý nghe đều hợp lý,
> một ý còn chép từ đội mạnh hơn — và cả ba đều không sống nổi phép đo.

### A56. Chữa **hubness** — hub có thật, nhưng phạt hub làm TỆ ĐI. Bỏ.

A55 đóng ba cách xếp lại bằng tín hiệu sẵn có, nhưng cả ba đều nhìn vào quan hệ
giữa các ứng viên của CÙNG một truy vấn. Hướng này nhìn thứ khác: **hình học của
không gian vector**.

Triệu chứng vẫn là con số A54: **R@20 = 0,6122 nhưng R@1 = 0,2041**. Trong không
gian nhiều chiều, một số điểm thành **"hub"** — gần với *mọi* truy vấn, không
riêng truy vấn nào. Khung chung chung (người nói trước micro, phông studio) nằm
gần tâm đám mây nên cosine với truy vấn nào cũng cao vừa phải, chiếm mất hạng
1–5 và đẩy khung đặc trưng xuống hạng 10–20. Đúng hình dạng R@1 thấp / R@20 cao.

#### Hub CÓ THẬT trong kho này

Quét 177.321 ảnh với bể 1.239 truy vấn đã mã hoá, tính
`log Σ_q exp(sim(q,i)/τ)` cho từng ảnh:

| τ | min | trung vị | max | chênh |
| ---: | ---: | ---: | ---: | ---: |
| 0,01 | 7,78 | 19,36 | 31,52 | **23,74** |
| 0,02 | 6,26 | 11,83 | 16,80 | 10,54 |
| 0,05 | 6,36 | 8,51 | 9,52 | 3,17 |

Có những khung được cả bể truy vấn chấm cao hơn khung khác hàng chục bậc độ
lớn. Chẩn đoán đúng.

#### Nhưng mọi cách chữa đều làm TỆ ĐI

Mốc là `run.py` sau A52, 52 câu đề thật:

| cấu hình | ±2s | hiệu | T-B-H | |
| --- | ---: | ---: | :---: | :---: |
| *mốc* | 0,5173 | — | — | |
| trừ tâm CHỈ phía ảnh | 0,4019 | −0,1154 | 3-17-32 | ✅ TỆ HƠN |
| trừ tâm CẢ HAI phía | 0,4481 | −0,0692 | 6-14-32 | 🟡 |
| QB-Norm τ=0,01 | 0,3692 | −0,1481 | 10-21-21 | ✅ TỆ HƠN |
| QB-Norm τ=0,02 | 0,2202 | −0,2971 | 5-27-20 | ✅ TỆ HƠN |
| QB-Norm τ=0,05 | 0,1663 | −0,3510 | 4-30-18 | ✅ TỆ HƠN |
| QB + trừ tâm | 0,3413 | −0,1760 | 9-17-26 | ✅ TỆ HƠN |

**Càng phạt mạnh càng tệ, đơn điệu.** Đó là dấu hiệu của một tín hiệu bị ĐẢO
CHIỀU, không phải một tham số chưa dò trúng — dò thêm τ là vô ích.

#### Vì sao QB-Norm hỏng ở đây: bể truy vấn KHÔNG trung lập

QB-Norm giả định bể truy vấn là mẫu ĐỘC LẬP với thứ đang tìm. Bể của ta thì
không: 1.239 chuỗi ấy chính là mệnh đề rút từ những câu hỏi ta đang tìm đáp án,
và chúng dồn vào một nhóm nhỏ video bản tin. **Khung được cả bể chấm cao thường
chính là khung đáp án của một câu nào đó trong bể.** QB-Norm phạt đúng thứ cần
thưởng.

Dựng một bể trung lập thì phải mã hoá hàng nghìn câu bịa — mà bịa câu lại vấp
đúng A50: câu tự soạn không thay được câu thật. Đường này cụt ở chỗ ta không có
dữ liệu, không phải ở chỗ ý tưởng sai.

#### Trừ tâm: sai một nửa, sửa lại vẫn thua

Bản đầu chỉ trừ tâm phía ảnh — dịch đám mây ảnh đi mà để đám mây truy vấn đứng
yên, hai bên lệch nhau. Trừ ở CẢ HAI phía (cách chuẩn) kéo được một nửa thiệt
hại (−0,1154 → −0,0692) nhưng vẫn âm ở cả hai mức dung sai. Cứu được nửa đường
không phải là thắng.

#### Kết luận, và giá trị của nó

Đây là hướng tinh vi nhất còn lại trong nhóm "xếp lại mà không cần thông tin
mới", và **không nhóm nào trong 9 repo đối chiếu đụng tới nó**. Nó thua.

Cộng với A55, kết luận giờ đứng vững hơn hẳn vì đã sống qua một phép thử nghiêm
túc hơn nhiều:

> **33 điểm phần trăm của A54 chỉ lấy được bằng THÔNG TIN MỚI** — nhìn lại bức
> ảnh (VLM rerank) hoặc mô tả nó bằng chữ (caption, kênh 5). Không có cách sắp
> xếp lại nào moi được nó ra.

`index/hubness_gopt.npz` giữ lại (thống kê, 4 MB) phòng khi sau này có bể truy
vấn độc lập thật.

### A57. Làm mượt vector theo trục thời gian — thua; và keyframe **trùng nhau nhiều hơn tưởng**

Ý tưởng (một mô hình ngoài gợi ý): một sự kiện diễn ra qua nhiều khung liên
tiếp, nên vector của khung đứng lẻ hay nhiễu vì nhoè chuyển động hoặc góc quay
chuyển tiếp. Cộng thêm vector hàng xóm rồi chuẩn hoá lại:

    v'(t) = chuẩn_hoá( v(t) + α·v(t−1) + α·v(t+1) + α²·v(t±2) … )

Sức hấp dẫn của nó: **chi phí lúc thi bằng KHÔNG**. Ma trận vẫn 177.321 × 1536,
chỉ đổi giá trị — khác hẳn mọi hướng còn lại đang chờ.

Và nó KHÔNG phải thứ A55 đã bác: A55 bỏ bớt khung trùng cảnh khỏi *kết quả*;
cái này không bỏ ai, chỉ cho mỗi vector mang thêm ngữ cảnh. Một cái sửa đầu ra,
một cái sửa đầu vào.

#### Kết quả: thua, đơn điệu theo cường độ

| cấu hình | ±2s | ±15s | hiệu ±2s | T-B-H | |
| --- | ---: | ---: | ---: | :---: | :---: |
| *mốc* | 0,5173 | 0,6096 | — | — | |
| W=1 α=0,2 | 0,4760 | 0,5577 | −0,0413 | 5-12-35 | 🟡 |
| W=1 α=0,35 | 0,4683 | 0,5500 | −0,0490 | 8-13-31 | ✅ TỆ HƠN |
| W=1 α=0,5 | 0,4654 | 0,5500 | −0,0519 | 9-13-30 | 🟡 |
| W=2 α=0,35 | 0,4538 | 0,5346 | −0,0635 | 8-13-31 | ✅ TỆ HƠN |

Càng mượt càng tệ, và **α nhỏ nhất cũng đã âm** — không có vùng nào để dò tiếp.

#### Giả thuyết của tôi SAI, và cái đúng thú vị hơn

Trước khi chạy tôi đoán: keyframe cách nhau trung vị 2,16 giây nên chắc được
trích theo CHUYỂN CẢNH, tức hàng xóm vốn đã là cảnh khác, và làm mượt sẽ bôi
nhoè đúng thứ cần phân biệt.

Đo thật trên 176.448 cặp keyframe liên tiếp cùng video:

| | |
| --- | ---: |
| cosine v(t)·v(t+1) trung vị | **0,9031** |
| cặp có cosine < 0,5 (cảnh khác hẳn) | **0,4%** |
| cặp có cosine > 0,8 (gần như trùng) | **73,1%** |

Ngược hẳn: hàng xóm **rất giống nhau**, không phải khác nhau.

Nhưng chính vì thế mà làm mượt vô ích **và có hại**. Cộng một vector gần trùng
vào thì `v'` gần như trùng `v` — chẳng thêm thông tin gì mới. Thứ nó thật sự
làm là **san phẳng phần dư**, tức những khác biệt nhỏ giữa các khung trong cùng
một cảnh. Mà với cửa sổ chấm ±2s, đúng phần dư đó là thứ phân biệt khung đáp án
với hàng xóm của nó. Làm mượt xoá tín hiệu phân biệt để đổi lấy thông tin đã có
sẵn.

> Cùng một quan sát — "keyframe liên tiếp rất giống nhau" — vừa là lý do ý
> tưởng này nghe hợp lý, vừa là lý do nó hỏng. Chỉ phép đo mới tách được hai
> chuyện đó.

#### Ghi thêm cho các gợi ý cùng đợt

Cùng đợt gợi ý này có hai hướng **đã nằm sẵn trong repo**, nêu ra để người sau
khỏi làm lại:

* **DP cho TRAKE** — `dong_hang_dp()` trong `src/run.py` từ lâu: gom ứng viên
  theo video, quy hoạch động chọn đúng một khung cho mỗi vị trí sự kiện, ép
  tăng dần ngặt, nội suy chỗ thiếu. Đã nằm trong điểm hiện tại.
* **Chỉ mục khái niệm thị giác** — kênh 4 (objects + IDF + bảng nhãn Việt–Anh)
  đã có và đã đo: A25 kết luận kênh 3 mạnh **gấp 2,8 lần** objects. Nâng cấp bộ
  gán nhãn (RAM++/Florence-2) là cải tiến một kênh ĐÃ ĐO ĐƯỢC LÀ YẾU, không
  phải thêm kênh mới.

Hai hướng còn đáng làm trong đợt đó: **nhúng OCR/ASR bằng model text–text thật**
(BGE-M3 / multilingual-e5 — sửa đúng nguyên nhân A53) và **định tuyến mệnh đề**
theo loại (thị giác → kênh 1, chữ trên màn hình → kênh 3).

### A58. Định tuyến mệnh đề vào kênh chuyên trách — thua. Kênh 3 **cần cả câu**.

Ý tưởng: hiện mọi mệnh đề đều đi qua cả kênh 1 lẫn kênh 3. Mệnh đề tả hình ảnh
thuần chạy qua BM25 chỉ sinh điểm rác; mệnh đề nói về CHỮ TRÊN MÀN HÌNH thì
tháp ảnh đọc rất kém. Vậy đưa mỗi mệnh đề vào đúng kênh của nó.

Bảng `ocr_asr.parquet` có sẵn `ocr_text` và `asr_text` riêng nên tách được kênh
3 thành hai. Nhưng "tách kênh" và "định tuyến" là HAI thay đổi, phải đo riêng —
không có dòng 3 thì cải thiện ở dòng 4 sẽ bị quy nhầm cho định tuyến.

#### Có gì để định tuyến

    mệnh đề: 23 có tín hiệu CHỮ | 2 có tín hiệu LỜI | 94 thị giác thuần
    câu    : 15/52 có ít nhất một mệnh đề chữ, 2/52 có mệnh đề lời

#### Kết quả: cả ba biến thể đều âm

| cấu hình | ±2s | ±15s | hiệu ±2s | T-B-H | |
| --- | ---: | ---: | ---: | :---: | :---: |
| *mốc* | 0,5173 | 0,6096 | — | — | |
| định tuyến, kênh 3 gộp | 0,4971 | 0,5865 | −0,0202 | 1-4-47 | 🟡 |
| tách OCR/ASR, KHÔNG tuyến | 0,5096 | 0,5933 | −0,0077 | 1-3-48 | 🟡 |
| tách OCR/ASR + định tuyến | 0,4817 | 0,5750 | −0,0356 | 2-8-42 | ✅ TỆ HƠN |

Hai thay đổi đều âm, và cộng lại thì âm hơn — không có tương tác cứu vãn nào.

#### Vì sao: kênh 3 thực chất là kênh ASR, và ASR nói về CẢ CÂU

OCR trung vị 22 ký tự, ASR trung vị 475 — kênh 3 chủ yếu đang khớp **lời dẫn**.
Mà lời dẫn của một bản tin nói về **toàn bộ chủ đề** của cảnh, không riêng phần
"có chữ". Hạn chế đầu vào của kênh 3 xuống mấy mệnh đề mang từ khoá "biển/chữ/
ghi" là **vứt đi phần khớp rộng đang có lãi**, đổi lấy một phần khớp hẹp mà OCR
22 ký tự không đủ sức đỡ.

Tách OCR thành kênh riêng cũng vậy: 22 ký tự trung vị thì quá thưa để đứng một
mình, mà tách ra là chia đôi trọng số của tín hiệu gộp vốn đang chạy tốt.

> Giả định ngầm của ý tưởng — "mệnh đề nào không nói về chữ thì kênh văn bản
> không giúp gì" — SAI với dữ liệu này, vì kênh văn bản của ta không đọc chữ
> trên màn hình là chính, mà nghe lời dẫn.

#### Trạng thái sau đợt này

Tám hướng "xếp lại / định tuyến mà không cần thông tin mới" đã đo (A55 ba,
A56 hubness, A57 làm mượt, A58 định tuyến, cùng nới bể ở A54 và lọc mệnh đề).
**Không hướng nào lấy quá 2,3% của khoảng trống 33,4 điểm.**

Hai hướng còn sống, cả hai đều đưa THÔNG TIN MỚI vào:

* **kênh 5 — caption** (Qwen2.5-VL, 2,10 s/ảnh; 6,1 giờ cho 47 video tập đề
  thật để đo trước khi tiêu 103 giờ cho cả kho)
* **kênh 6 làm lại bằng BGE-M3** — model text–text thật, sửa đúng nguyên nhân
  A53 (`notebooks/kaggle_bge_m3.md`)

### A59. Hai kênh **thông tin mới** — cùng dương, cùng dưới ngưỡng nhiễu

A55–A58 đóng tám hướng "xếp lại mà không cần thông tin mới", không hướng nào
lấy quá 2,3% của khoảng trống 33,4 điểm (A54). Đây là hai hướng còn lại, và
**cả hai đều cho kết quả dương** — lần đầu sau chín phép đo âm liên tiếp.

#### 1. Kênh 6 làm lại bằng **BGE-M3** — chẩn đoán A53 ĐÚNG

A53 bác kênh 6 vì nhúng OCR/ASR bằng tháp văn bản của SigLIP2: model học khớp
ảnh↔chữ, đem so chữ↔chữ là dùng ngoài phân bố huấn luyện. Thay bằng BGE-M3
(model text–text thật, đa ngôn ngữ, 1024 chiều, nhận 8192 token nên **không
phải chia đoạn**: 176.009 tài liệu = 176.009 vector, thay vì 462.085 đoạn):

| cấu hình | ±2s | ±15s | hiệu ±2s | T-B-H | |
| --- | ---: | ---: | ---: | :---: | :---: |
| *mốc: ảnh + kênh 3 (0,5)* | 0,5173 | 0,6096 | — | — | |
| + kênh 6 (0,25) | 0,5250 | 0,6173 | +0,0077 | 3-2-47 | 🟡 |
| **+ kênh 6 (0,5)** | **0,5279** | **0,6173** | **+0,0106** | 5-5-42 | 🟡 |
| + kênh 6 (1,0) | 0,4394 | 0,5298 | −0,0779 | 5-26-21 | ✅ TỆ HƠN |
| kênh 6 THAY kênh 3 | 0,4962 | 0,5817 | −0,0212 | 3-6-43 | 🟡 |
| **chỉ kênh 6** *(chẩn đoán)* | **0,1462** | 0,1942 | | | |
| chỉ kênh 1 *(chẩn đoán)* | 0,4779 | 0,5712 | | | |

**Dòng chẩn đoán: 0,0490 (A53) → 0,1462, gấp 3 lần.** Đổi sang model text–text
thật chữa được phần lớn cái hỏng. Nhưng vẫn kém xa kênh 1, và lãi khi hợp nhất
chỉ +0,0106 với ngưỡng nhiễu 0,0300.

Chắc chắn: **không thay được kênh 3** (−0,0212), và **trọng số 1,0 có hại**
(−0,0779 ✅) — cùng hình dạng với mọi kênh phụ khác trong repo này.

#### 2. Kênh 5 — caption bằng Qwen2.5-VL

⚠️ **Phải khoá bể ứng viên để đo.** Caption mới phủ 47 video mà tập đề thật
đụng tới (10.488 ảnh = 5,9% kho) — cố ý, để biết có đáng 103 giờ GPU cho cả kho
không. Nhưng dựng BM25 trên đúng ngần ấy thì kênh 5 CHỈ đề xuất được khung từ
chính những video chứa đáp án; con số sẽ đẹp rực rỡ và vô nghĩa, đúng cơ chế
A21 (tăng ẢO 0,400 → 0,840). Nên `71_` ép **mọi kênh** chỉ chạy trong 47 video
đó — tất cả cùng một vũ trụ, so sánh mới công bằng.

**Điểm dưới đây KHÔNG so được với mục khác** (bể chỉ còn 5,9% kho). Chỉ đọc
hiệu giữa các dòng.

| cấu hình | ±2s | ±15s | hiệu ±2s | T-B-H | |
| --- | ---: | ---: | ---: | :---: | :---: |
| *mốc: ảnh + kênh 3, khoá bể* | 0,6606 | 0,7452 | — | — | |
| + kênh 5 (0,25) | 0,6615 | 0,7580 | +0,0010 | 9-8-35 | 🟡 |
| **+ kênh 5 (0,5)** | **0,6712** | **0,7808** | **+0,0106** | 11-11-30 | 🟡 |
| + kênh 5 (1,0) | 0,6500 | 0,7603 | −0,0106 | 10-18-24 | ❌ |
| kênh 5 THAY kênh 3 | 0,6548 | 0,7538 | −0,0058 | 13-15-24 | ❌ |
| **chỉ kênh 5** *(chẩn đoán)* | **0,3904** | 0,5154 | | | |
| chỉ kênh 1 *(chẩn đoán)* | 0,6474 | 0,7426 | | | |

**Dòng chẩn đoán mới là phần đáng giá.** Kênh 5 đứng một mình được **0,3904**
so với 0,6474 của kênh 1 trong CÙNG bể — tức nó là một kênh truy hồi **thật sự
chạy được**, khác hẳn kênh 6 (0,1462 so với 0,4779). Caption mang thông tin
thị giác mà BM25 đọc được.

#### So hai kênh, và điều cần quyết

| | chỉ kênh đó | so với kênh 1 cùng điều kiện | lãi khi hợp nhất |
| --- | ---: | ---: | ---: |
| kênh 6 (BGE-M3) | 0,1462 | 31% | +0,0106 🟡 |
| kênh 5 (caption) | 0,3904 | 60% | +0,0106 🟡 |

Lãi bằng nhau, nhưng **caption là kênh khoẻ hơn hẳn**. Và lãi của caption ở
±15s là +0,0356 (so với +0,0077 của BGE) — nó cứu những câu lệch xa hơn.

Cả hai đều **dưới ngưỡng nhiễu ở 52 câu**, nên chưa bật cái nào mặc định. Điểm
đáng tin hơn các dòng 🟡 trước: cùng dấu ở cả hai mức dung sai VÀ ở nhiều mức
trọng số, không phải thắng nhờ vài câu may.

> **Đây chính là lúc tập dev nhỏ trở thành nút thắt thật sự.** Hai kênh tốn
> hàng chục giờ GPU để dựng, cùng cho +0,0106, và 52 câu không đủ để nói cái
> nào thật. 24 gói `de_thi_thu` (52 → 76 câu) giờ không còn là việc "nên làm"
> mà là **điều kiện để quyết định bất cứ điều gì tiếp theo**.

Việc còn lại: sinh caption cho cả kho (103,4 giờ, chia 12 phần —
`notebooks/kaggle_caption_chay.md` cell B) rồi đo lại KHÔNG khoá bể. Chỉ nên
làm sau khi tập đề thật lên 76 câu, nếu không thì lại ra một con số 🟡 nữa.

### A60. VLM chấm lại top-30 — **có tín hiệu, nhưng không hơn thứ tự sẵn có**

A54 đo khoảng trống 33,4 điểm phần trăm; A55–A58 cho thấy không lấy được bằng
cách sắp xếp lại thông tin đã có. Đây là phép thử của hướng "đưa thông tin mới"
đắt nhất: **cho VLM nhìn lại bức ảnh**.

Cách chấm: Qwen2-VL-2B-Instruct, một lượt forward, lấy **hiệu logit giữa "Có"
và "Không"** cho câu nhắc *"Ảnh này có đúng là cảnh được mô tả không?"*. Không
sinh chữ — điểm liên tục, nhanh, và không phụ thuộc model có chịu trả lời đúng
định dạng hay không. 52 câu × top-30 = 1.445 ứng viên (115 ứng viên L30 không
chấm được vì máy đó thiếu dataset ảnh L30).

| cấu hình | ±2s | ±15s | hiệu ±2s | T-B-H | |
| --- | ---: | ---: | ---: | :---: | :---: |
| *mốc: run.py* | 0,5202 | 0,5952 | — | — | |
| VLM thay hẳn phần đầu | 0,4808 | 0,5692 | −0,0394 | 6-11-35 | 🟡 |
| RRF hạng, w=0,5 | 0,5183 | 0,6067 | −0,0019 | 3-5-44 | ❌ |
| RRF hạng, w=1 | 0,5144 | 0,6029 | −0,0058 | 4-7-41 | ❌ |
| nhân, w=0,5 | 0,5221 | 0,5952 | +0,0019 | 4-5-43 | ❌ |
| nhân, CHỈ đẩy lên (w=1) | 0,5173 | 0,6077 | −0,0029 | 6-7-39 | ❌ |

**Không cấu hình nào vượt nhiễu, và cái tốt nhất là +0,0019** — bằng 0,6% của
khoảng trống.

#### Nhưng VLM KHÔNG mù: nó hơn ngẫu nhiên

Trước khi có điểm thật, cùng bộ đo này đã chạy với **điểm bịa ngẫu nhiên** làm
nhóm đối chứng:

| "thay hẳn phần đầu" | hiệu ±2s |
| --- | ---: |
| điểm ngẫu nhiên | −0,0894 |
| **điểm Qwen2-VL-2B** | **−0,0394** |

VLM hơn ngẫu nhiên khoảng **0,05** — nó thật sự nhìn thấy gì đó. Chỉ là **kém
hơn thứ tự mà kênh 1 + kênh 3 đã cho sẵn**. Không có nhóm đối chứng ngẫu nhiên
thì con số −0,0394 đọc ra "VLM vô dụng", sai; nó là "VLM có tín hiệu nhưng yếu
hơn cái đang có".

Điểm VLM cũng phân biệt được thật: trải trung vị **0,43** giữa ứng viên cao và
thấp nhất trong mỗi câu, chỉ 1/52 câu dưới 0,1.

#### Một lỗi của bộ đo, sửa TRƯỚC khi chạy

Bản đầu gán `-1e9` cho ứng viên không có điểm VLM, tức **đẩy chúng xuống đáy** —
biến "máy chấm thiếu ảnh" thành "ảnh sai". Với 115 ứng viên L30 không chấm
được, nó sẽ lặng lẽ làm hỏng đúng những câu ít điểm nhất (có câu chỉ 4/30).

Đã sửa: **VLM chỉ được đảo thứ tự những gì nó thật sự nhìn**; ứng viên không có
điểm nằm nguyên chỗ cũ. Hai test chốt lại luật đó.

#### Chốt và giới hạn

Không bật rerank VLM. Cũng **không cần dựng hạ tầng ngày thi** (tunnel Kaggle,
đồng bộ ảnh, đường lui) — đó là khoản chi lớn nhất mà A54 từng gợi ý, giờ đã
biết là chưa đáng.

Ba giới hạn của kết luận này, ghi để người sau khỏi đoán:

* **Model 2B là nhỏ.** Qwen2.5-VL-7B có thể khác, nhưng 8 GB VRAM của máy nhóm
  chỉ đủ bản 4-bit và chậm hơn nhiều lần.
* **115/1.560 ứng viên (7%) không được chấm** vì thiếu dataset L30. Kết quả này
  vì thế hơi ĐÁNH GIÁ THẤP VLM — nhưng 7% không lật được +0,0019.
* **Cách nhắc chỉ có một.** Hiệu logit Có/Không là cách rẻ nhất; hỏi VLM mô tả
  rồi so chữ là một thí nghiệm khác, đắt hơn nhiều.

> Cộng với A55–A58: **mười hướng đã đo cho khoảng trống 33,4 điểm**, và thứ duy
> nhất còn sống là caption (A59) — kênh duy nhất đứng một mình đạt 60% sức của
> kênh 1.

### A61. Đo lại ba thứ bị bác dưới mốc nền CŨ — **cả ba vẫn bị bác**, và lần này có nền đúng

A12, A14.2, A18 và A25 đều được đo khi kênh 1 còn là CLIP/SigLIP2 và trước A51
(truy vấn còn cắt cụt ở token 64), phần lớn trên **tập dev tự soạn** — thứ A50
chứng minh là thổi phồng kênh 1 gấp 2,3 lần. Một kênh phụ bị bác vì "pha loãng
kênh 1" rất có thể chỉ bị bác vì kênh 1 được chấm quá cao.

Kênh 1 mạnh lên cắt theo **cả hai chiều**, không đoán được: mạnh hơn thì kênh
phụ càng dễ pha loãng, nhưng RRF cũng có nền tốt hơn nên kênh phụ chỉ cần đúng
ở vài câu kênh 1 trượt là đã có lãi. Nên đo, trên **đề thật**, mốc là `run.py`
sau A52.

| cấu hình | ±2s | ±15s | hiệu ±2s | T-B-H | |
| --- | ---: | ---: | ---: | :---: | :---: |
| *mốc* | 0,5173 | 0,6096 | — | — | |
| + kênh 4 objects (0,25) | 0,5135 | 0,6058 | −0,0038 | 0-1-51 | 🟡 |
| + kênh 4 objects (0,5) | 0,5048 | 0,5913 | −0,0125 | 0-4-48 | ✅ TỆ HƠN |
| + kênh 2 metadata (0,25) | 0,5173 | 0,6096 | +0,0000 | **0-0-52** | ⚪ |
| + kênh 2 metadata (0,5) | 0,5212 | 0,6096 | +0,0038 | 1-0-51 | 🟡 |
| + cả hai (0,25) | 0,5135 | 0,6058 | −0,0038 | 0-1-51 | 🟡 |
| mỗi video tối đa 3 dòng | 0,4346 | 0,5413 | −0,0827 | 3-11-38 | ✅ TỆ HƠN |
| mỗi video tối đa 5 dòng | 0,4538 | 0,5760 | −0,0635 | 1-8-43 | ✅ TỆ HƠN |
| **chỉ kênh 4** *(chẩn đoán)* | **0,0125** | 0,0462 | | | |
| **chỉ kênh 2** *(chẩn đoán)* | **0,0141** | 0,0218 | | | |

#### Hai dòng chẩn đoán giải thích mọi thứ

**0,0125 và 0,0141**, so với 0,4779 của kênh 1. Trên đề THẬT, hai kênh này gần
như không truy hồi được gì — kém hơn cả kênh 6 hỏng của A53 (0,0490). Chúng
không phải "kênh yếu", chúng là **kênh không chạy** với loại truy vấn của BTC.

Vì sao kênh 2 chết: nó là kênh **cấp video**, khớp title + description +
keywords của cả video. Đề thật hỏi một *khoảnh khắc* — "người phụ nữ áo đỏ đứng
cạnh xe máy" không có trong tiêu đề bản tin nào cả.

Vì sao kênh 4 chết: nhãn vật thể là danh từ chung ("person", "car"). Mọi khung
hình đều có người và xe; nhãn không phân biệt được khung NÀO.

Đáng chú ý: kênh 2 ở trọng số 0,25 đổi **đúng 0 câu trên 52** (0-0-52). Ứng
viên của nó không lọt nổi vào top-100 sau hợp nhất — thêm vào cũng như không.

#### `moi_video` thì tệ hơn hẳn kết luận cũ

A18 chỉ nói "làm tệ đi". Đo lại: **−0,0827 ✅ ỔN ĐỊNH**. Lý do rõ khi đọc cùng
A55: top-20 đã trải trên 10,2 video khác nhau, nên ràng buộc đa dạng không còn
gì để đa dạng hoá — nó chỉ còn tác dụng **cắt bỏ** những khung đúng nằm cùng
video với một khung đúng khác. Với TRAKE thì đó là cắt vào chính đáp án.

#### Giá trị của phép đo này

Không đổi mặc định nào. Nhưng ba kết luận vừa chuyển từ *"bị bác dưới điều kiện
không còn đúng"* sang *"bị bác dưới điều kiện hiện tại, trên đề thật"* — và đó
là khác biệt giữa một giả định thừa kế và một sự thật đã kiểm.

> Bài học chung của A61: khi mốc nền đổi lớn, **kết luận cũ không tự động sai,
> nhưng cũng không tự động đúng**. Rẻ nhất là đo lại những cái có sẵn dữ liệu —
> cả ba thứ ở đây chỉ tốn vài phút CPU vì `objects.parquet` và metadata đã nằm
> trong `master.parquet` từ lâu.

### A62. Kênh 4 có **hai lỗi công thức chấm** — sửa xong mạnh gấp 2,5 lần, và vẫn chết

A61 kết luận kênh 4 vô dụng (0,0125 đứng một mình). Nhưng A11 đã dạy: **đo trên
một kênh đang hỏng thì con số nói về cái hỏng, không nói về ý tưởng**. Soi lại
trước khi đóng hồ sơ.

#### Dữ liệu lành, ánh xạ lành, rút nhãn ĐÚNG

| | |
| --- | --- |
| `objects.parquet` | 1.122.384 detection, **95,0%** keyframe có nhãn, 514 nhãn |
| nhãn mỗi keyframe | trung vị 6, p90 12 |
| bảng nhãn Việt–Anh | 156 mục, phủ 30% từ vựng nhưng **97,3% số lần xuất hiện** |
| rút nhãn từ truy vấn | **50/52** câu rút ra ít nhất một nhãn |
| câu nhắc tới nhãn ĐẶC TRƯNG (IDF≥3) có trên khung đáp án | **26/52** |

Và nhãn đặc trưng thì rất chọn lọc: `Umbrella` chỉ ở **355/168.470** keyframe,
`Motorcycle` 905, `Whiteboard` 1.443; trung vị của nhãn IDF≥3 là **54 keyframe**.

Vậy mà `kis-DE1-20` ("2 người cầm dù") rút ra ĐÚNG `Umbrella`, khung đáp án CÓ
`Umbrella`, mà đáp án **không lọt nổi top-100**.

#### Hai lỗi trong `object_score()`

    điểm = Σ (độ tin cậy × IDF) trên MỌI detection khớp

**1. Mỗi DETECTION cộng một lần, không phải mỗi NHÃN.** Khung có 8 người ăn
8 × IDF(Person); khung có đúng một cái ô hiếm ăn 1 × IDF(Umbrella). Công thức
đếm SỐ LƯỢNG vật thể, trong khi thứ cần đếm là SỰ CÓ MẶT.

**2. Cộng dồn nên nhãn phổ biến át nhãn hiếm.** Truy vấn trên rút ra
[Building, Clothing, House, Person, Shirt, Umbrella]; một cảnh phố bất kỳ có đủ
5 nhãn đầu sẽ vượt khung DUY NHẤT có Umbrella — mà toàn bộ sức phân biệt nằm ở
đúng cái nhãn ấy.

#### Sửa: mạnh lên thật, nhưng không đủ

| chỉ kênh 4 | ±2s | ±15s |
| --- | ---: | ---: |
| công thức hiện tại | 0,0125 | 0,0462 |
| **gộp detection** | **0,0317** | 0,0423 |
| lấy MAX thay vì tổng | 0,0231 | 0,0538 |
| chỉ nhãn IDF≥3 | 0,0308 | 0,0346 |
| chỉ nhãn hiếm NHẤT | 0,0269 | 0,0538 |
| *(kênh 1 để so)* | *0,4779* | *0,5712* |

Gấp **2,5 lần** — và vẫn kém kênh 1 **mười lăm lần**. Hợp nhất vào cấu hình
chính: bản tốt nhất được +0,0038 (1-0-51, 🟡), bản còn lại ⚪ không đổi gì.

#### Vì sao vẫn chết, lần này là lý do thật

Nhãn vật thể nói **CÓ GÌ**, không nói **CẢNH NÀO**. Truy vấn cần chọn 1 trong
355 khung có Umbrella, mà objects không có gì để chọn: không màu sắc, không
quan hệ không gian, không hành động. Đếm số lượng thì có, nhưng đúng cái đó lại
là lỗi 1 ở trên.

Đây là giới hạn của **biểu diễn**, không phải của cài đặt — nên sửa cài đặt chỉ
đưa 0,0125 lên 0,0317 rồi dừng.

#### Đã làm gì

Không đổi `object_score()` (kênh 4 tắt mặc định, sửa cũng không cứu được), nhưng
**ghi cả hai lỗi vào docstring của nó** kèm số đo. Người sau định dùng lại kênh
này sẽ đọc được ngay, thay vì tự đo lại từ đầu.

> Giá trị của A62 không nằm ở việc cứu kênh 4 — nó không cứu được. Nó nằm ở chỗ
> A25 và A61 từng kết luận "objects yếu" trong khi thật ra đang đo **một công
> thức sai**. Giờ kết luận vẫn thế, nhưng vì đúng lý do.

### A63. Khâu **lắp ráp TRAKE** mất hơn NỬA số điểm kênh tìm được

Sau A62, soát tiếp xem còn chỗ nào "sai âm thầm" như công thức objects. Tìm
thấy một chỗ, và chính docstring trong repo đã cảnh báo mà chưa ai đo:

> *`diem_trake()` chấm KÊNH … Hàm này chấm BÀI NỘP … Kênh tìm ra đủ ba sự kiện
> nhưng khâu lắp ráp xếp sai vị trí thì `diem_trake()` cho điểm cao còn BTC cho
> 0. Đó chính là tầng mà `run.dung_trake()` làm, và là **tầng chưa ai đo**.*
> — `cham_diem.diem_trake_bai_nop`

`cham_diem.cham()` dùng `diem_trake()`. Nghĩa là **mọi con số TRAKE trong repo**
— kể cả các dòng TRAKE trong A54, A59, A60 — đang đo tầng KÊNH, không phải tầng
NỘP.

#### Đo cả hai tầng trên 3 câu TRAKE của đề thật

| câu | sự kiện | KÊNH | NỘP | mất |
| --- | ---: | ---: | ---: | ---: |
| trake-DE1-16 | 3 | 0,0000 | 0,0000 | 0 |
| trake-DE2-21 | 4 | 0,5000 | **0,0000** | **−0,5000** |
| trake-DE2-08 | 4 | 0,5500 | 0,5000 | −0,0500 |
| **trung bình ±2s** | | **0,3500** | **0,1667** | **−0,1833** |
| trung bình ±15s | | 0,4833 | 0,2500 | −0,2333 |

**Mất 52% ở ±2s, 48% ở ±15s.** Câu `trake-DE2-21` là ca đúng như docstring dự
đoán: kênh tìm ra một nửa số sự kiện (0,5000) mà bài nộp được **0**.

Nguyên nhân nằm ở chỗ BTC chấm **theo vị trí**: khung ở vị trí i chỉ được so
với sự kiện i. Kênh tìm ra khung đúng nhưng `dung_trake()` xếp nó vào vị trí
khác là mất trắng.

#### Đây là con số lớn nhất còn bỏ ngỏ trong repo

So với những thứ đang tranh nhau vài phần nghìn: caption +0,0106, BGE-M3
+0,0106, VLM rerank +0,0019. Còn ở đây là **0,18 điểm cho mỗi câu TRAKE**.

TRAKE chỉ chiếm 3/52 câu đề thật nên ảnh hưởng lên điểm tổng là ~0,01 — nhưng
đó là vì tập đo ít câu TRAKE, không phải vì kỳ thi ít câu TRAKE.

#### Vì sao chưa sửa được ngay: 3 câu là quá ít

A39 đã dựng sẵn **bốn tham số** cho đúng tầng này (`dong_hang`, `he_so_phat`,
`trai_toi_da`, `rai_hep`), mặc định giữ nguyên hành vi cũ vì "chưa thắng trên
tập dev thì chưa được bật". Dò bốn tham số trên 3 câu là vô nghĩa.

`dev/tap_dev_trake.jsonl` có 14 câu nữa, và **câu tự soạn dùng được ở đây**:
thứ tự sự kiện và số Frame ID là ràng buộc HÌNH THỨC, không phụ thuộc câu hỏi
dễ hay khó — khác hẳn việc so kênh, nơi A50/A58 đã cấm dùng câu tự soạn.

Nhưng 14 câu đó **không đo được**: 219/219 chuỗi của chúng chưa có trong
`truy_van_gopt.npz`.

#### Đã sửa cái làm lộ ra chuyện đó

`scripts/25_ma_hoa_truy_van.py` có `--tap-dev` nhưng nó chỉ đọc
`dev/tap_dev.jsonl`; các file câu khác không có đường vào. Thêm `--tap <file>`
(lặp lại được), và gom logic "câu TRAKE phải tách sự kiện RỒI tách mệnh đề" vào
một chỗ — trước đó nó chỉ có ở nhánh `--tap-dev`.

    python scripts/25_ma_hoa_truy_van.py --tap dev/tap_dev_trake.jsonl \
        --matrix clip_gopt.npy --ra index/truy_van_trake.npz

219 chuỗi, ~3 giây GPU. Xong thì gộp bằng `67_gop_cache_truy_van.py` rồi chạy
lại `75_do_lap_rap_trake.py` trên 17 câu — đủ để dò bốn tham số A39.

### A65. Tập đề thật **52 → 68 câu**, và ngưỡng nhiễu bắt đầu co lại

16/24 gói `de_thi_thu` đã tìm được đáp án (`dev/tap_de_thi_thu.jsonl`, soi bằng
`66_soat_de_thi_thu.py`). Soát trước khi dùng:

| | |
| --- | --- |
| số câu | 16 (15 KIS + 1 Q&A) |
| trùng tập test / tập đề thật | **0 / 0** |
| khung đáp án mỗi câu | trung vị 4 (min 1, max 7) |
| Q&A có `dap_an` | ✅ (bỏ trống là 0 điểm dù khung đúng) |
| nhãn độ chắc | `kha` cả 16 |
| cache truy vấn | thiếu **0** chuỗi ở cả gopt lẫn BGE-M3 |

Còn 8 gói: 3 KIS, 2 Q&A, **3 TRAKE**. Ba câu TRAKE đó sẽ đưa tập TRAKE đề thật
từ 3 lên 6 câu — vừa đủ để bắt đầu dò bốn tham số A39 cho khâu lắp ráp (A63,
chỗ đang mất 52% điểm).

#### Trọng số kênh 3 đo lại trên 68 câu

| | 52 câu | 68 câu |
| --- | ---: | ---: |
| hiệu 0,5 so với 0,75 | +0,0077 | **+0,0118** |
| ngưỡng nhiễu ±2s | 0,0290 | **0,0235** |
| thắng–thua–hoà | 7-4-41 | **9-4-55** |
| kết luận | 🟡 | 🟡 |

Cả ba chỉ số đi đúng hướng — hiệu tăng, ngưỡng co, thắng-thua rộng ra — nhưng
vẫn chưa vượt. Mặc định **giữ nguyên 0,5**, không đổi gì.

Và `w = 1,0` giờ ✅ ỔN ĐỊNH có hại (−0,0353, thắng-thua **1-13** ở ±2s và
**0-13** ở ±15s). Kết luận *"kênh ASR/OCR chỉ được làm phụ, không được lấn kênh
1"* đã đủ vững để coi là sự thật của hệ này.

> Đây là lần đầu tiên trong dự án một con số 🟡 được đo lại trên tập lớn hơn và
> **dịch chuyển đúng hướng dự đoán**. Nó xác nhận chẩn đoán của A59: nút thắt
> là CỠ TẬP ĐO, không phải thiếu ý tưởng.

### A64. Ba cách xử lý truy vấn KIS — không cách nào thắng, và **tách mệnh đề được chứng minh là đúng**

Đo lần đầu trên **68 câu đề thật** (52 cũ + 16 câu mới từ `de_thi_thu`, xem
A65). Mốc là `run.py` hiện tại: tách khi câu > 40 từ, kênh 3 trọng số 0,5.

| cấu hình | ±2s | ±15s | hiệu ±2s | T-B-H | |
| --- | ---: | ---: | ---: | :---: | :---: |
| *mốc: ngưỡng 40 từ* | 0,5868 | 0,6721 | — | — | |
| A. ngưỡng 50 từ | 0,5809 | 0,6574 | −0,0059 | 3-4-61 | 🟡 |
| A. ngưỡng 60 từ | 0,5779 | 0,6544 | −0,0088 | 4-6-58 | 🟡 |
| **A. KHÔNG tách** | 0,4956 | 0,5603 | **−0,0912** | 5-18-45 | ✅ TỆ HƠN |
| B. cân mệnh đề — MAX IDF | 0,5985 | 0,6721 | +0,0118 | 6-2-60 | ❌ |
| B. cân mệnh đề — IDF trung bình | 0,5868 | 0,6662 | −0,0000 | 2-2-64 | 🟡 |
| C. cổng kênh 3 — sàn 0 | 0,5743 | 0,6574 | −0,0125 | 2-4-62 | 🟡 |
| C. cổng kênh 3 — sàn 0,1 | 0,5757 | 0,6574 | −0,0110 | 1-3-64 | 🟡 |
| C. cổng kênh 3 — sàn 0,25 | 0,5868 | 0,6743 | +0,0000 | 1-1-66 | 🟡 |
| B+C | 0,5904 | 0,6632 | +0,0037 | 8-5-55 | ❌ |

#### A. Tách mệnh đề: ngưỡng 40 là ĐÚNG, và bỏ tách là mất 0,09

Đây là kết quả ✅ duy nhất của cả bảng, và nó **xác nhận một quyết định cũ**
thay vì đổi nó. Bỏ tách hẳn mất **−0,0912**, thua 5-18. Nâng ngưỡng lên 50 hay
60 từ đều âm, đơn điệu theo cường độ.

Lý do khớp với A51: câu nguyên có trung vị **62 từ ≈ 64 token**, đúng trần tháp
văn bản. Nâng ngưỡng = đẩy thêm câu vào vùng cắt cụt. Lập luận "giữ nguyên câu
để bảo toàn liên kết chủ–vị" nghe hợp lý, nhưng thứ thật sự xảy ra là **mất
phần đuôi câu**.

> Giá trị của dòng này: `TRAN_TOKEN = 40` từ trước tới nay là một con số chọn
> theo lý lẽ, chưa ai dò. Giờ nó có số đo, và số đo nói nó đúng.

#### B. Cân mệnh đề theo độ hiếm từ: đảo dấu, nhưng đáng theo dõi

`max(IDF)` được **+0,0118 ở ±2s với thắng-thua 6-2** — hiệu lớn hơn ngưỡng
nhiễu 0,0165? Không, vẫn dưới. Và ở ±15s thì đúng **−0,0000**. Đảo dấu nên
không dùng để quyết.

Nhưng hình dạng của nó khác hẳn C: 6 thắng 2 thua (C là 1-3, 2-4). Nghĩa là nó
**có tác động thật, đúng chiều, ở cửa sổ hẹp** — chỉ là không đủ mạnh và không
sống ở cửa sổ rộng. Đây là ứng viên đáng đo lại ở 76 câu.

`max(IDF)` hơn `IDF trung bình` (+0,0118 so với −0,0000) đúng như dự đoán: mệnh
đề ngắn chứa một thực thể hiếm bị trung bình cộng kéo tụt oan.

⚠️ Bản đầu của phép đo này chuẩn hoá min–max TRONG TỪNG CÂU, nên một từ gõ sai
(IDF cực đại) tự động thành 1,0 và đẩy mọi mệnh đề khác xuống 0,2. Đã sửa: chia
cho **IDF trung vị của kho** rồi kẹp `[0,2; 1,0]` — mốc cố định, không phụ
thuộc câu.

#### C. Cổng kênh 3: không có gì

Cả ba mức sàn đều ~0 hoặc âm nhẹ. Sàn 0,25 đổi đúng **1 câu trên 68**.

Lý do rõ khi nhìn số: chỉ **24/68 câu (35%)** có từ chỉ thị chữ/lời, nghĩa là
cổng này hạ trọng số ở 65% số câu — và A52 đã đo bỏ kênh 3 làm tệ đi. Kênh 3
thực chất là kênh **ASR**, mà lời dẫn bản tin mô tả cả cảnh chứ không riêng
phần có chữ; không có từ "biển/chữ/nói" trong câu hỏi KHÔNG có nghĩa là ASR vô
dụng cho câu đó.

Đây là lần thứ hai cùng một ý bị bác: A58 bác việc hạn chế ĐẦU VÀO, A64 bác
việc hạ TRỌNG SỐ. Hai cách khác nhau, cùng một giả định sai.

#### Ghi chú phương pháp

Đề xuất gốc khuyên đo TUẦN TỰ: chốt ngưỡng tách, rồi đo cổng trên ngưỡng đã
chốt, rồi đo IDF. Ở đây đo **song song, mỗi cấu hình so với cùng một mốc** —
vì chốt một lựa chọn 🟡 rồi xây tiếp lên nó là đúng cái đã xảy ra ở A50 (trọng
số 0,75 chọn dưới nền sai, kéo theo mọi thứ tới A52 mới gỡ được).

### A66. TRAKE: dồn 100 dòng vào vài video — **lấy lại 80% phần đã mất** ở ±15s

A63 đo được khâu lắp ráp TRAKE mất 52% điểm kênh tìm được. Nguyên nhân soi ra
trong `dung_trake()`: nó xếp **một dòng cho mỗi video**, rồi bù cho đủ 100 bằng
99 video xếp sau.

Nhưng BTC chấm TRAKE **theo vị trí**: chuỗi của video tốt nhất lệch một sự kiện
ra ngoài cửa sổ là cả dòng đó 0 điểm — và 99 dòng còn lại dành cho những video
gần như chắc chắn sai. Với TRAKE, đáp án nằm trong MỘT video, nên 100 dòng nên
là **100 giả thuyết khác nhau về video đó**, không phải 100 video khác nhau.

#### Cách làm

1. Chấm điểm video: `Σ log(điểm cao nhất của video cho từng sự kiện)`. Video
   thiếu hẳn ứng viên cho một sự kiện -> loại.
2. Lấy top-5 video, chia ngân sách 100 dòng theo thứ hạng (50/25/15/7/3).
3. Trong mỗi video, **beam search** (bề rộng 64) sinh K chuỗi khác nhau, đều
   tăng dần ngặt theo thời gian, có phạt trùng thời điểm để chúng trải ra.

#### Kết quả (3 câu TRAKE đề thật)

| dung sai | CŨ 1 dòng/video | K-best 0,5s | 1,5s | 3,0s |
| --- | ---: | ---: | ---: | ---: |
| ±2s | 0,1667 | 0,1667 | 0,1667 | 0,1667 |
| **±15s** | **0,2500** | 0,4167 | 0,4333 | **0,4500** |

Ở ±15s: **+0,2000, tăng 80%**. Khâu lắp ráp từ chỗ mất 48% điểm kênh (0,4833 ->
0,2500) nay chỉ còn mất 7% (-> 0,4500).

Ở ±2s: **không đổi gì**, cả ba mức giãn.

#### Hai điều đọc ra, ngoài con số

**K-best ≈ oracle.** Dòng `oracle` (dồn cả 100 dòng vào ĐÚNG video chứa đáp án)
cho 0,4167 — tức **không hơn** K-best tự chọn video. Nghĩa là khâu CHỌN VIDEO
đã đúng sẵn; toàn bộ mất mát nằm ở khâu LẮP CHUỖI. Không có dòng oracle thì
không tách được hai tầng đó.

**Cửa hẹp không cứu được bằng cách này.** ±2s không nhúc nhích dù giãn 0,5s hay
3,0s. Sinh thêm chuỗi chỉ giúp khi cửa sổ đủ rộng để một biến thể rơi trúng;
ở ±2s thì ứng viên gốc đã không đủ chính xác, và không cách sắp xếp nào tạo ra
độ chính xác chưa có.

#### CHƯA đổi mặc định

**3 câu là quá ít.** Đây là kết quả đúng hướng và có cơ chế rõ ràng, nhưng
ngưỡng nhiễu trên 3 câu thì vô nghĩa. `78_do_kbest_trake.py` giữ nguyên dạng
script đo, `run.py` chưa động tới.

8 gói `de_thi_thu` còn lại có **3 câu TRAKE**; khi có chúng thì tập TRAKE đề
thật lên 6 câu — vẫn ít, nhưng gấp đôi, và đủ để thấy hướng có ổn định không.
Cộng thêm 14 câu `tap_dev_trake.jsonl` (cần 219 chuỗi mã hoá, ~3 giây GPU) thì
thành 20 câu: câu tự soạn dùng được ở đây vì thứ tự sự kiện và số Frame ID là
ràng buộc HÌNH THỨC, không phụ thuộc câu dễ hay khó.

> Đây là hướng duy nhất trong hơn 20 hướng đã thử cho thấy hiệu ứng cỡ **0,2**
> thay vì 0,01. Lý do: nó không cố cải thiện TRUY HỒI (thứ đã bão hoà) mà sửa
> một khâu HẬU XỬ LÝ đang vứt đi thông tin kênh đã tìm ra.

### A67. **Điểm Q&A trong repo là TRẦN TRÊN, không phải điểm thi** — và bài nộp thật đang 0

Đi tìm cách điền `answer` cho từng dòng, lại tìm ra một chuyện lớn hơn.

#### Thước đo đang bỏ qua `answer`

`cham_diem._dung_dap_an()` coi ứng viên **không có** khoá `answer` là HỢP LỆ.
Cố ý — docstring ghi rõ *"mọi kênh hiện tại chưa biết trả lời, và lúc này ta
đang đo TRUY HỒI"*. Nhưng không kênh nào gắn `answer`, nên:

> **Mọi con số Q&A trong repo đều được chấm như thể đáp án luôn đúng.**

BTC cho **0 điểm** nếu `answer` sai hoặc trống. Q&A chiếm **13/68 = 19%** số câu
đề thật.

#### Và bài nộp thật thì tệ hơn nữa

`run.py` chỉ có `--tra-loi`: **một chuỗi dùng chung cho cả 100 dòng**. Không có
gì tự sinh nó. Nên trong một bài nộp thật hôm nay, hoặc người chạy tự gõ đúng
chuỗi đó (mọi dòng có cơ hội), hoặc **cả 100 dòng đều 0 điểm** — bất kể truy
hồi tốt đến đâu.

Mà BTC chấm `answer` **theo TỪNG DÒNG**: mỗi dòng mang `answer` riêng, ăn điểm
khi đúng CẢ khung LẪN chuỗi. Một chuỗi dùng chung là vứt đi đúng cơ chế đó.

#### Đo bốn cách (13 câu Q&A, `79_do_dap_an_qa.py`)

| cách điền `answer` | ±2s | ±15s |
| --- | ---: | ---: |
| bỏ trống — *cách repo đang chấm* | 0,4000 | 0,5231 |
| một chuỗi chung, đào từ khung hạng 1 | **0,0000** | **0,0000** |
| mỗi dòng đào riêng, lấy ứng viên đầu | **0,0000** | **0,0000** |
| mỗi dòng đào riêng, chọn theo từ khoá câu hỏi | 0,0000 | 0,0462 |

Cả ba bộ đào bằng regex đều gần như trắng tay.

#### Trần trên của MỌI cách đào từ văn bản: **7/13**

Đáp án vàng chỉ xuất hiện trong OCR/ASR của khung đúng ở **7/13 câu**. Sáu câu
còn lại (`Cá sòng`, `200g`, `2`…) phải **nhìn ảnh** mới trả lời được — không
lượng văn bản nào cứu được.

Và 7 câu có mặt trong văn bản thì regex vẫn hỏng, vì chọn ĐÚNG con số nào mới
là việc khó: OCR bản tin đầy dấu thời gian (`06:30:11`), số hiệu kênh, ngày
tháng. Lấy "số đầu tiên" gần như luôn trúng chúng. Bản chọn theo khoảng cách
tới từ khoá câu hỏi cứu được đúng **1/13**.

#### Việc phải làm, và nó KHÔNG phải regex

Điền `answer` là bài toán **đọc hiểu**, không phải trích mẫu:

  * VLM đọc ảnh (6 câu bắt buộc phải thế), hoặc
  * LLM đọc `câu hỏi + OCR/ASR của chính khung đó` rồi chọn đoạn (đủ cho 7 câu
    còn lại, và rẻ hơn nhiều).

`mui_nhon_1.gan_dap_an` đã có sẵn đường VLM nhưng sinh **một** đáp án cho cả
gói, không phải mỗi dòng một đáp án.

Chi phí lúc thi cho hướng LLM-đọc-văn-bản: ~24 câu Q&A × top-20 dòng = 480 lượt
gọi. Cần đo trước khi tin.

> **Đây là lỗ hổng lớn nhất về mặt ĐIỂM THI đã tìm ra**, lớn hơn cả khâu lắp
> ráp TRAKE (A63/A66): TRAKE mất một nửa trên 3/68 câu, còn Q&A mất TRẮNG trên
> 13/68 câu. Và nó không lộ ra trong bất kỳ phép đo nào của repo, vì chính
> thước đo đã được thiết kế để bỏ qua nó.

### A68. Hai rào cản **cấu trúc** của việc đào đáp án Q&A từ văn bản

A67 đo được bộ đào regex đúng 1/13 câu. Sửa hai lần (chọn theo khoảng cách tới
từ khoá; loại ứng viên vốn là chữ của chính câu hỏi) — vẫn **1/13, ngay cả khi
đào tại ĐÚNG khung đáp án**. Soi vào dữ liệu thì rõ vì sao, và lý do không nằm
ở regex.

| | |
| --- | --- |
| `ocr_text` có dấu tiếng Việt | **944/3.000 mẫu (31%)** |
| `asr_text` có dấu tiếng Việt | 3.000/3.000 (100%) |
| nhưng ASR viết số bằng CHỮ | *"hai mươi mốt"*, *"ba mươi"* |

**Rào cản 1 — OCR mất dấu.** Đáp án vàng là `Tà Pứa`, `Lý Thường Kiệt`,
`Cá sòng`; OCR cho ra `Ta Pua`, `Soc Trang`, `Khanh Vinh`. So chuỗi chính xác
thì **không bao giờ khớp**, dù đọc đúng chữ trên màn hình.

**Rào cản 2 — ASR viết số bằng chữ.** Đáp án vàng là `46`, `1204`, `200g`; ASR
ghi *"bốn mươi sáu"*. Regex số không bắt được, và ngược lại.

Nghĩa là hai nguồn văn bản **bổ khuyết đúng cái nhau thiếu**: OCR có số nhưng
mất dấu, ASR có dấu nhưng không có số. Câu Q&A hỏi tên thì phải đọc OCR (mất
dấu), hỏi số thì phải đọc ASR (viết chữ).

#### Kết luận cho hướng "đào đáp án bằng regex": **đóng**

Không phải vì mẫu chưa đủ tinh, mà vì **dạng dữ liệu không khớp dạng đáp án**.
Trần 7/13 của A67 còn lạc quan: trong 7 câu "đáp án có mặt trong văn bản", phần
lớn khớp được là nhờ so KHÔNG DẤU — mà bài nộp thì phải đúng chuỗi.

Việc cần làm vẫn là **đọc hiểu**, và giờ có thêm một yêu cầu cụ thể: bộ sinh
đáp án phải **khôi phục dấu** (VLM/LLM làm được, regex thì không).

#### Đã làm gì

`src/dap_an.py` + nối vào `run.py`: mỗi dòng Q&A giờ mang `answer` đào từ
OCR/ASR **của chính khung đó**, thay vì một chuỗi dùng chung. `--khong-dao-dap-an`
để dựng lại hành vi cũ.

Đúng 1/13 — nhưng chốt chặn cũ *bắt buộc* phải có `--tra-loi` mới chạy được,
nghĩa là bài nộp thật hoặc trắng hoặc phụ thuộc người gõ tay. Giờ nó luôn có
chuỗi, và `run.py` in cảnh báo nói rõ đây là bản vá chứ không phải lời giải.
7 test chốt, trong đó một test bắt đúng lỗi "chọn trúng từ khoá của câu hỏi".

### A69. Gom đoạn để tóm tắt bằng LLM: **OCR không gom được, ASR thì được 18 lần**

Ý (học từ một nhóm khác): gom khung liên tiếp cùng video có văn bản giống nhau
thành ĐOẠN, rồi một lượt LLM mỗi đoạn tóm tắt nội dung. Biến 177.321 lượt gọi
thành "số đoạn" lượt.

Chưa ai biết số đoạn là bao nhiêu — mà đó chính là thứ quyết định ý tưởng sống
hay chết. `80_do_gom_doan_ocr.py` chỉ đếm:

| gom theo | ngưỡng Jaccard | số đoạn | khung/đoạn | đoạn 1 khung |
| --- | ---: | ---: | ---: | ---: |
| **OCR** | 0,3 | 62.960 | 2,8 | **54%** |
| OCR | 0,5 | 88.597 | 2,0 | 69% |
| **ASR** | **0,3** | **9.802** | **18,1** | **8%** |
| ASR | 0,5 | 11.093 | 16,0 | 7% |

**Gom theo OCR thất bại**: giảm 2,8 lần, và 54% số đoạn chỉ có MỘT khung — tức
tốn một lượt gọi LLM mà chẳng lan tín hiệu đi đâu. Lý do: OCR trung vị **4
token/khung** và chứa đồng hồ chạy (`06:30:11`), nên hai khung liền kề gần như
không bao giờ giống nhau đủ.

**Gom theo ASR thì được**: 9.802 đoạn, **giảm 18 lần**, 18,1 khung/đoạn, chỉ 8%
đoạn lẻ. Khả thi thật — vài giờ LLM, hoặc một lượt Kaggle với model địa phương.

> Ý tưởng đúng, nhưng gắn nhầm nguồn. Người đề xuất nhắm vào OCR vì họ thấy OCR
> là ticker rời rạc — đúng chẩn đoán, sai lời giải: chính vì rời rạc mà nó
> không gom được. ASR mới là thứ liên tục qua thời gian.

⚠️ Nhưng phải đọc cùng A59: kênh 6 nhúng ASR bằng BGE-M3 (model text–text
thật) đứng một mình chỉ được **0,1462**. Tóm tắt ASR là một biến đổi khác trên
CÙNG nguồn tín hiệu đó, nên trần của nó khó vượt xa. Trước khi tiêu vài giờ
LLM, nên đo trước trên một mẫu nhỏ: tóm tắt 100 đoạn thuộc video của tập đề
thật rồi xem BM25 trên bản tóm tắt có hơn BM25 trên ASR gốc không.

### A70. Gộp vector BGE theo **đoạn ASR**: tệ đi một nửa — và điều đó đóng luôn nhánh "tóm tắt bằng LLM"

A69 đo được gom theo ASR cho 9.802 đoạn (18,1 khung/đoạn, giảm 18 lần) và kết
luận nhánh này *khả thi về chi phí*. Bước tiếp theo lẽ ra là gọi LLM tóm tắt
mỗi đoạn. Nhưng ta đã có sẵn `van_ban_bge.npz` — BGE-M3 nhúng từng khung — nên
**gộp VECTOR theo đoạn** thử được ngay, miễn phí, và nó mô phỏng đúng cấu trúc
mà tóm tắt-bằng-LLM tạo ra: **một biểu diễn duy nhất cho cả đoạn**.

| cấu hình | ±2s | ±15s | hiệu ±2s | T-B-H | |
| --- | ---: | ---: | ---: | :---: | :---: |
| *mốc: run.py* | 0,5868 | 0,6721 | — | — | |
| + kênh 6 GỐC (0,25) | 0,5956 | 0,6838 | +0,0088 | 5-3-60 | 🟡 |
| + kênh 6 GỐC (0,5) | **0,6007** | **0,6868** | **+0,0140** | 7-6-55 | 🟡 |
| + kênh 6 GỘP ĐOẠN (0,25) | 0,5926 | 0,6721 | +0,0059 | 3-1-64 | 🟡 |
| + kênh 6 GỘP ĐOẠN (0,5) | 0,5890 | 0,6721 | +0,0022 | 3-3-62 | 🟡 |
| **chỉ kênh 6 gốc** *(chẩn đoán)* | **0,1853** | 0,2426 | | | |
| **chỉ kênh 6 GỘP ĐOẠN** *(chẩn đoán)* | **0,0794** | 0,1176 | | | |

**Gộp đoạn làm kênh yếu đi hơn một nửa** (0,1853 → 0,0794), và lãi khi hợp nhất
tụt từ +0,0140 xuống +0,0022.

#### Vì sao — và đây là A57 lặp lại

Một đoạn trung bình 18,1 khung × 2,16 s ≈ **39 giây**. Gộp đoạn nghĩa là **mọi
khung trong 39 giây đó nhận CÙNG một vector**, nên kênh không còn phân biệt nổi
khung nào trong đoạn. Mà cửa sổ chấm là ±2s.

Đúng cơ chế đã đo ở A57 (làm mượt vector theo thời gian): gộp không thêm thông
tin, nó **san phẳng phần dư** — đúng thứ dùng để phân biệt khung đáp án với
hàng xóm của nó.

Cái lợi duy nhất có thật là độ phủ: **177.304/177.321 khung (100,0%)** so với
176.009 (99,3%) của kênh gốc. Nhưng 0,7% không đáng gì so với việc mất khả năng
xếp hạng bên trong đoạn.

#### Hệ quả: nhánh "tóm tắt đoạn bằng LLM" nên đóng

Tóm tắt bằng LLM có **cùng cấu trúc** với phép gộp này: một biểu diễn cho cả
đoạn, gán cho mọi khung trong đoạn. Nó hứa hẹn văn bản *tốt hơn* (diễn giải,
thực thể neo) nhưng chịu **cùng một khuyết tật hình học**: 18 khung không phân
biệt được nhau.

Nó chỉ đáng làm nếu vấn đề là "văn bản quá vụn để hiểu", nhưng phép đo nói vấn
đề là "định vị trong 39 giây". Vài giờ LLM để mua đúng thứ phép gộp này vừa cho
thấy là có hại.

> Giá trị của A70: một ý tưởng nghe rất hợp lý bị bác **mà không tốn một giờ
> GPU nào**, nhờ nhận ra rằng thứ có sẵn (vector BGE per-frame) mô phỏng được
> cấu trúc của thứ định làm (tóm tắt per-đoạn). Kiểm cấu trúc trước, mua nội
> dung sau.

#### Ghi thêm: kênh 6 gốc trên 68 câu

Đứng một mình **0,1853** (52 câu: 0,1462), hợp nhất **+0,0140** (52 câu:
+0,0106). Cả hai đi đúng hướng khi tập lớn lên, vẫn 🟡. Cùng dấu ở cả hai mức
dung sai và ở cả hai trọng số.

### A71. Các kênh **gần như không bao giờ đồng ý cùng một khung** — và đó không phải chuyện độ phủ

A59 đo kênh 5 (caption) đứng một mình 0,3904 mà hợp nhất chỉ **+0,0106**. Điểm
số không tách được ba cách giải thích, và chúng dẫn tới ba quyết định khác nhau:
kênh YẾU / kênh TRÙNG kênh 1 / kênh KHÁC nhưng phủ ít. Độ trùng thì tách được.

`scripts/84_do_tuong_quan_kenh.py` đo hai đại lượng, không đo điểm:

* **chồng@k** — bao nhiêu phần trăm top-k của hai kênh là **cùng `row_id`**.
  Đây đúng thứ RRF cần: RRF chỉ cộng hưởng khi hai kênh đề cử cùng `row_id`.
* **Spearman** — tương quan hạng trên phần giao (chỉ tính khi hai kênh chung
  ≥ 5 ứng viên; cột `n` cho biết bao nhiêu câu đủ điều kiện).

| cặp kênh | chồng@10 | chồng@20 | chồng@100 | Spearman | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| *bể 51.088 khung (28,8% kho), 49 câu* | | | | | |
| 1 ảnh × 3 ocr+asr | 4,1% | 4,3% | 6,9% | 0,203 | 27 |
| 1 ảnh × 5 caption | 2,7% | 4,3% | 7,3% | 0,093 | 30 |
| 3 ocr+asr × 5 caption | 3,9% | 2,8% | 3,5% | 0,169 | 6 |
| *bể 134.708 khung (76,0% kho), 65 câu* | | | | | |
| 1 ảnh × 3 ocr+asr | 2,0% | 3,4% | 5,5% | 0,120 | 35 |
| **1 ảnh × 5 caption** | 4,0% | **4,0%** | 4,9% | **0,043** | 37 |
| 3 ocr+asr × 5 caption | 3,4% | 2,9% | 2,5% | 0,190 | 10 |

#### Hai điều đọc ra

**Caption KHÔNG thừa.** Spearman 0,093 rồi 0,043 — gần như độc lập với kênh 1.
Giả thuyết "Qwen2.5-VL nhìn cùng tấm ảnh nên nói cùng thứ với SigLIP2" **bị
bác**. Hai model nhìn cùng ảnh mà xếp hạng gần như không liên quan tới nhau.

**Nhưng chúng cũng không hợp tác.** chồng@20 chỉ 4%: trong 20 ứng viên đầu,
chưa tới một cái trùng. Đây **đúng cơ chế A14** (kênh 2 × kênh 4 chung khung ở
5/97 câu -> RRF thô thua −0,0144). Không cộng hưởng thì RRF chỉ **đan xen**, và
đan xen thì mỗi ứng viên tốt của kênh mạnh bị một ứng viên của kênh yếu đẩy lùi
một bậc. +0,0106 chính là thứ đan xen mua được.

#### Vì sao đo hai lần với hai độ phủ — và vì sao điều đó đóng một quyết định

Lần đầu bể chỉ 5,9% kho, nên "độ chồng thấp vì caption phủ ít" là cách giải
thích cạnh tranh còn sống. Gộp thêm 9 phần (51.088 -> **134.708 ảnh / 663
video**, 76% kho) rồi đo lại: **độ chồng không nhúc nhích**, Spearman còn giảm.

Độ chồng là chuyện **đồng thuận**, không phải chuyện **độ phủ**. Nên 3 phần
caption còn lại (7, 8, 12 — khoảng 26 giờ GPU) mua thêm độ phủ, chứ không mua
thêm khả năng hợp nhất. Quyết định chỉ còn phụ thuộc một số: kênh 5 đóng góp
bao nhiêu khi bể khoá ở 76% (`71_do_kenh5_caption.py`).

> Giá trị của A71: nó tách được "kênh yếu" khỏi "kênh không hợp tác" — hai thứ
> mà điểm hợp nhất trông y hệt nhau, mà cách chữa thì ngược nhau (một bên là
> làm kênh mạnh lên, bên kia là đổi cơ chế hợp nhất).

### A72. Hợp nhất kênh bằng **ĐIỂM chuẩn hoá** thay vì thứ hạng — thua ở **cả 8 biến thể**

A71 chỉ thẳng vào một nghi ngờ: nếu các kênh chỉ đan xen chứ không cộng hưởng,
thì có lẽ vấn đề là RRF **vứt hết biên độ** — một khớp gần như chắc chắn bị đối
xử y hệt một khớp yếu, miễn cùng hạng. `rrf.py` từ chối cộng điểm ngay từ đầu,
nhưng đó là lý do **né**, chưa từng là một phép đo.

`src/hop_diem.py` + `scripts/85_do_hop_nhat_diem.py`. Chỉ đổi **một thứ**: cơ
chế hợp nhất ở tầng KÊNH. Hợp nhất mệnh đề *trong* kênh 1 vẫn là RRF hạng ở mọi
dòng (A51 đã thắng ở đó).

| cấu hình | ±2s | ±15s | hiệu ±2s | T-B-H | |
| --- | ---: | ---: | ---: | :---: | :---: |
| **1. MỐC: RRF hạng** | **0,5809** | **0,6721** | — | — | |
| 2. điểm z-score | 0,5647 | 0,6375 | −0,0162 | 2-7-59 | ✅ |
| 3. điểm min-max | 0,5684 | 0,6456 | −0,0125 | | ✅ |
| 4. sigmoid tau=2 | 0,5735 | 0,6507 | −0,0074 | | ✅ |
| 5. sigmoid tau=8 | 0,5676 | 0,6507 | −0,0133 | | ✅ |
| 6. z-score + log1p BM25 | 0,5647 | 0,6375 | −0,0162 | 2-7-59 | ✅ |
| 7. min-max + log1p BM25 | 0,5713 | 0,6426 | −0,0096 | 2-5-61 | ✅ |
| 8. z-score, **bù 0** | 0,5493 | 0,6147 | −0,0316 | 3-13-52 | ✅ |
| **9. chỉ kênh 1** *(chẩn đoán)* | 0,5537 | 0,6426 | −0,0272 | | ✅ |

68 câu đề thật. Đo trước trên 52 câu, cùng dấu ở cả 8 dòng và hiệu số lớn hơn
(z-score −0,0212 / −0,0490) — tập lớn lên thì hiệu co lại, dấu không đổi.

#### Dòng 9 giải thích toàn bộ

Kênh 1 một mình **0,5537**. RRF hạng đẩy lên **0,5809**, tức kênh 3 đóng góp
**+0,0272**. Hợp nhất theo điểm chỉ tới 0,5647–0,5735: nó **vứt đi khoảng một
nửa đóng góp của kênh 3**.

Lý do nằm ở chỗ `rrf.py` cảnh báo, giờ có số: BM25 lệch nặng — phần lớn ứng
viên gần 0, một nhúm vọt cao — nên ứng viên đầu của kênh 3 có z-score cỡ +8,
còn ứng viên đầu của kênh 1 (cosine phân bố mượt) chỉ cỡ +3. Cộng vào thì BM25
nuốt hết. **Chuẩn hoá không chữa được, vì cái lệch nằm ở HÌNH DẠNG phân phối
chứ không ở thang đo** — mà z-score, min-max và sigmoid đều chỉ đụng tới thang.

Suy luận từ A71 — *"chồng 4% nên RRF chỉ đan xen; cộng điểm giữ được biên độ"* —
đúng phần chẩn đoán, **sai phần kết luận**. Giữ được biên độ đúng là thứ gây hại.

#### Ba thứ đo được thêm

**Bẫy A60 có thật và đắt.** Bù `0` thay vì bù giá trị thấp nhất kênh đó thật sự
trả về làm tệ thêm gần gấp đôi (−0,0316 so với −0,0162). Ứng viên vắng mặt ở
một kênh chỉ nghĩa là *không lọt top-100 của kênh đó*, không phải *kênh đó nói
nó sai*.

**`log1p` không đổi gì với z-score** — dòng 2 và 6 trùng nhau tới bốn chữ số.
"Tham số bị nuốt trên đường truyền" trông y hệt "tham số vô tác dụng", nên đã
thêm test dựng phân phối lệch kiểu BM25 và kiểm rằng `log1p` **thật sự đổi điểm
hợp nhất**. Test qua -> kết luận thật: nó đổi điểm nhưng không đổi thứ tự
top-100 đủ để chạm vào R@k.

**`logit_scale`/`logit_bias` của SigLIP2 không phải thứ đáng đi lấy.**
`σ(s·τ + b)` **đơn điệu** theo cosine, nên nó không đổi được thứ hạng nội bộ
kênh — chỉ đổi biên độ khi cộng. Nếu chỉ cần biên độ thì `τ` **dò được như tham
số thường**, và dò thì bao trùm luôn giá trị của checkpoint. Khỏi mở model, mà
máy thi cũng không mở nổi. Cả `tau=2` lẫn `tau=8` đều thua.

#### Hệ quả

`run.py` giữ RRF. Và **cổng theo độ tự tin của kênh** (dùng ngưỡng trên điểm đã
chuẩn hoá để tắt kênh phụ ở những câu nó không chắc) mất chỗ dựa — nó cần đúng
phép chuẩn hoá vừa bị bác.

#### Hai lỗi repo tìm ra trên đường

* `76_kiem_caption_phan.py --ghep` dùng `Path.rename()` để sao lưu, mà trên
  Windows `rename` ném `FileExistsError` khi đích đã có — và bản sao lưu lần
  ghép trước thì luôn đã có. Chặn hẳn việc ghép caption. -> `.replace()`.
* `25_ma_hoa_truy_van.py` chặn bằng `if not (a.de or a.tap_dev or a.them)`,
  **thiếu `a.tap`** — trong khi câu thông báo lỗi *có* liệt kê `--tap`. Cờ này
  thêm ở A63 mà điều kiện không cập nhật theo, nên `--tap` đứng một mình luôn
  thoát mã 1 với đúng câu "chưa chọn nguồn nào: … --tap …". Nay danh sách nguồn
  và câu thông báo lấy từ **cùng một dict**, không lệch lại được.

### A73. Caption: độ phủ tăng **13 lần** thì đóng góp **biến mất** — và con số cũ là ảo

A59 đo kênh 5 (caption) hợp nhất **+0,0106** với bể khoá 10.488 ảnh (5,9% kho).
A71 cho thấy kênh này độc lập thật với kênh 1 (Spearman 0,043) nhưng chỉ chồng
4% ở top-20. Câu hỏi còn lại là con số +0,0106 có sống nổi khi bể lớn lên không
— vì 3 phần caption còn lại là **~26 giờ GPU**.

Gộp 9/12 phần: **134.708 ảnh / 663 video = 76,0% kho** (trước 51.088 / 249).
`71_do_kenh5_caption.py`, 68 câu đề thật, 66/68 có đáp án trong bể.

| cấu hình | ±2s | ±15s | hiệu ±2s | T-B-H | |
| --- | ---: | ---: | ---: | :---: | :---: |
| **1. mốc: ảnh + kênh 3 (khoá bể)** | **0,5713** | **0,6551** | — | — | |
| 2. **chỉ kênh 5** *(chẩn đoán)* | 0,2625 | 0,3186 | −0,3088 | 7-43-18 | ✅ |
| 3. chỉ kênh 1 *(chẩn đoán)* | 0,5522 | 0,6235 | −0,0191 | 7-10-51 | ✅ |
| 4. + kênh 5 (0,25) | 0,5735 | 0,6522 | **+0,0022** | 5-6-57 | **❌ ĐẢO DẤU** |
| 4. + kênh 5 (0,5) | 0,5684 | 0,6551 | −0,0029 | 9-11-48 | 🟡 |
| 4. + kênh 5 (1,0) | 0,5485 | 0,6265 | −0,0228 | 13-26-29 | 🟡 |
| 5. kênh 5 THAY kênh 3 | 0,5669 | 0,6353 | −0,0044 | 11-10-47 | 🟡 |

**Trọng số tốt nhất giờ ❌ ĐẢO DẤU** (+0,0022 ở ±2s, −0,0029 ở ±15s). Theo đúng
luật của repo, đảo dấu = **không kết luận được**, không phải "hơi hơn". Hai
trọng số còn lại đều âm.

#### Con số cũ đẹp vì bể quá nhỏ, không phải vì kênh tốt

Kênh 5 đứng một mình rơi từ **0,3904 xuống 0,2625**. Bể 5,9% cũ gồm gần như chỉ
những video *có chứa đáp án*, nên kênh 5 chỉ cần chọn đúng khung trong một tập
đã được lọc sẵn hộ. Đây đúng cơ chế **A21** (mức tăng ảo 0,400 -> 0,840), và
điều đáng ghi là **khoá bể chỉ chặn được một phần của nó**: khoá bể giữ cho mọi
kênh cùng nhìn một vũ trụ, nhưng khi vũ trụ đó nhỏ tới 5,9% thì bản thân nó đã
là một gợi ý mạnh, và kênh yếu hưởng lợi nhiều hơn kênh mạnh.

> **Bài học chung:** khoá bể làm phép so *công bằng*, không làm nó *đại diện*.
> Bể càng nhỏ so với kho thật, con số càng nói về bể chứ không về kênh. Lần sau
> đo kênh phủ một phần, ghi kèm tỷ lệ phủ và coi kết quả là **tạm** cho tới khi
> phủ đủ.

#### Quyết định

**Không chạy 3 phần caption còn lại (7, 8, 12).** ~26 giờ GPU để mua thêm 24% độ
phủ cho một kênh mà ở 76% độ phủ đã không còn đóng góp đo được.

Kênh 5 **không bật** trong `run.py`. 9 phần đã sinh thì giữ lại — chúng không
tốn thêm gì, và nếu về sau có cơ chế hợp nhất khai thác được kênh ít chồng lấn
(A71) thì dữ liệu đã sẵn. Nhưng A72 vừa bác cơ chế ứng viên duy nhất cho việc
đó, nên đừng chờ.

#### Ba chẩn đoán chỉ có được nhờ dòng "chỉ kênh 1"

Kênh 1 một mình 0,5522, thêm kênh 3 lên 0,5713 (**+0,0191**, ✅ ổn định ở ±15s).
Đó là mức đóng góp của một kênh *có* tác dụng, để đối chiếu với +0,0022 của
kênh 5. Không có dòng chẩn đoán này thì +0,0022 trông như "nhỏ nhưng dương";
đặt cạnh nhau mới thấy nó nhỏ hơn một bậc độ lớn.

### A74. TRAKE trên **17 câu**: K-best thắng lớn ở ±2s — và A63/A66 (3 câu) đã sai cả hai chiều

219 chuỗi của `dev/tap_dev_trake.jsonl` đã mã hoá (Kaggle, ~3 giây GPU) và gộp
vào cache (1.239 -> 1.458 chuỗi). Tập TRAKE đo được đi từ **3 câu lên 17 câu**.
Câu tự soạn dùng được ở đây vì thứ tự sự kiện và số Frame ID là ràng buộc
**hình thức**, không phụ thuộc câu hỏi dễ hay khó — khác hẳn việc so kênh, nơi
A50/A58 đã cấm dùng câu tự soạn.

#### Khâu lắp ráp mất bao nhiêu (`75_do_lap_rap_trake.py`)

| | KÊNH | NỘP | mất | |
| --- | ---: | ---: | ---: | --- |
| ±2s — 3 câu (A63) | 0,3500 | 0,1667 | −0,1833 | **52%** |
| **±2s — 17 câu** | **0,3982** | **0,2518** | **−0,1465** | **37%** |
| ±15s — 3 câu (A63) | 0,4833 | 0,2500 | −0,2333 | **48%** |
| **±15s — 17 câu** | **0,5400** | **0,5165** | **−0,0235** | **4,3%** |

Ở ±2s con số đứng vững (52% -> 37%). Ở ±15s nó **sụp từ 48% xuống 4,3%** — tức
kết luận "khâu lắp ráp vứt đi gần nửa số điểm ở cả hai mức dung sai" là **hiện
vật của 3 câu**, không phải sự thật của hệ thống.

#### K-best beam search (`78_do_kbest_trake.py`)

| dung sai | CŨ 1 dòng/video | K-best 0,5s | 1,5s | **3,0s** | oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 câu (A66) ±2s | 0,1667 | 0,1667 | 0,1667 | 0,1667 | — |
| 3 câu (A66) ±15s | 0,2500 | 0,4167 | 0,4333 | 0,4500 | 0,4167 |
| **17 câu ±2s** | 0,2518 | 0,3435 | 0,3465 | **0,3547** | 0,4576 |
| **17 câu ±15s** | 0,5165 | 0,5312 | 0,5341 | **0,5488** | 0,6718 |

**±2s: +0,1029 (tăng 41%), thắng–thua–hoà 8-2-7.**
**±15s: +0,0323 (tăng 6%), 7-4-6.**

Cùng dấu ở cả hai mức dung sai, và giãn cách 3,0s tốt nhất ở cả hai — đơn điệu
theo tham số chứ không nhảy loạn.

#### Ba điều A66 kết luận sai, và vì sao

**"±2s không nhúc nhích."** Sai — đó là mức được lợi NHIỀU NHẤT (+0,1029 so với
+0,0323). Trên 3 câu, cả ba câu đều đứng yên ở ±2s nên trông như một quy luật;
thật ra hai trong ba câu đó có điểm 0 tuyệt đối ở mọi cấu hình.

**"K-best ≈ oracle, nên khâu chọn video đã đúng sẵn."** Sai — khoảng cách tới
oracle là **0,1029 ở ±2s** và **0,1230 ở ±15s**, đúng bằng cỡ phần K-best vừa
lấy lại được. Vẫn còn ngần ấy nằm ở khâu CHỌN VIDEO.

**"Khâu lắp ráp mất ~50%."** Chỉ đúng ở ±2s.

> Đây là ca rõ nhất trong repo về việc **n = 3 không phải một phép đo**. Cả hai
> mục A63 và A66 đều đã tự ghi "3 câu là quá ít, chưa đổi mặc định" — kỷ luật
> đó vừa cứu ta khỏi bật một mặc định dựa trên kết luận sai chiều.

#### Vẫn CHƯA đổi mặc định, và lý do khác lần trước

17 câu là đủ để tin dấu, nhưng `78_do_kbest_trake.py` **không in ngưỡng nhiễu**
như `bao_cao_do_nhay()`. Trước khi đổi `run.py` cần:

1. Cho `78_` báo cáo qua `bao_cao_do_nhay()` để có ngưỡng nhiễu và kết luận
   ✅/🟡/❌ như mọi phép đo khác.
2. Soi **hai câu thua**: `trake-L25-004` rơi 0,4800 -> 0,0000 ở CẢ HAI mức, mà
   oracle của nó là 0,8000 — tức K-best chọn nhầm VIDEO ở câu này.
   `trake-L23-008` rơi 0,4000 -> 0,3000.

Hai câu thua đó là 2/17, và mức rơi của chúng lớn hơn mức thắng trung bình —
đúng dạng rủi ro mà điểm trung bình che mất.

### A75. Kênh 3 có **trần hạng do chính công thức RRF** — trần có thật, nhưng phá nó tốn hơn được

RRF cộng `w/(k + hạng)`. Với `k = 60`, `w = 0,5`, một ứng viên **chỉ kênh 3
tìm ra** có điểm cao nhất là `0,5/61 = 0,008197`, đúng bằng `1,0/122` — tức
hạng 62 của kênh 1. Suy ra công thức trần:

    h = (k+1)/w − k      k=60 -> 62    k=20 -> 22    k=10 -> 12
                         k=30 -> 32    k=15 -> 17    k= 5 ->  7

`h` là hạng thấp nhất của kênh 1 mà ứng viên chỉ-kênh-3 có thể vượt. Điểm BTC
là trung bình R@{1,5,20,50,100}, nên chừng nào `h > 20` thì **kênh 3 không thể
chạm vào ba mốc đầu**, bất kể nó đúng tới đâu.

#### Chẩn đoán: trần là THẬT (`86_do_dinh_muc_kenh3.py`, 68 câu)

| top-N kênh 3 | hạng trung vị sau hợp nhất | vào top-20 | vào top-100 | chỉ kênh 3 |
| --- | ---: | ---: | ---: | ---: |
| top-1 | **64** | **10%** | 100% | 74% |
| top-3 | 68 | 10% | 99% | 75% |
| top-5 | 70 | 10% | 99% | 74% |

Công thức dự đoán 62, đo được trung vị **64**. Và **74%** ứng viên tốt của kênh
3 là loại "chỉ kênh 3" — khớp A71 (chồng@20 chỉ 3,4%).

#### Cách chữa 1 — định mức chỗ dành riêng: THUA

Vì "dòng sai không bị phạt", thử cắt M dòng cuối dành cho top-M của riêng kênh 3:

| cấu hình | ±2s | ±15s | hiệu ±2s | T-B-H | |
| --- | ---: | ---: | ---: | :---: | :---: |
| *RRF thuần* | 0,5919 | 0,6669 | — | — | |
| dành 10 dòng | 0,5890 | 0,6640 | −0,0029 | 0-1-67 | 🟡 |
| dành 20 dòng | 0,5860 | 0,6640 | −0,0059 | 0-2-66 | 🟡 |
| dành 30 dòng | 0,5801 | 0,6640 | −0,0118 | 0-4-64 | ✅ tệ hơn |

**Không một câu nào thắng**, và càng dành nhiều càng tệ. Lý do nằm ngay trong
bảng chẩn đoán: **100% top-5 của kênh 3 đã nằm trong top-100 rồi**, nên định
mức ở đuôi chỉ đổi chỗ thứ vốn có mặt, mà đánh đổi bằng ứng viên kênh 1 hạng
71–100. Cảnh báo này đã ghi vào docstring TRƯỚC khi chạy.

#### Cách chữa 2 — hạ `k` để phá trần: THUA, đơn điệu (`87_do_hang_so_k.py`)

| k | trần hạng | ±2s | ±15s | hiệu ±2s | T-B-H | |
| ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| **60** | 62 | **0,5809** | **0,6721** | — | — | mốc |
| 30 | 32 | 0,5809 | 0,6654 | +0,0000 | 2-1-65 | 🟡 |
| 20 | 22 | 0,5765 | 0,6544 | −0,0044 | 2-3-63 | 🟡 |
| 15 | 17 | 0,5735 | 0,6551 | −0,0074 | 2-4-62 | 🟡 |
| 10 | 12 | 0,5647 | 0,6441 | −0,0162 | 2-6-60 | ✅ tệ hơn |
| 5 | 7 | 0,5537 | 0,6353 | −0,0272 | 2-10-56 | ✅ tệ hơn |
| *chỉ kênh 1* | | 0,5537 | 0,6426 | −0,0272 | 2-9-57 | ✅ |

A48 từng dò `k` và kết luận "gần như trơ", nhưng dò ở trọng số 1:1 dưới mốc nền
cũ 0,3440 và **dừng đúng ở k = 20** — tức `h = 22`, sát ngay bên ngoài top-20.
Nay bước qua rồi: **càng hạ càng tệ, đơn điệu**, và `k = 5` tụt đúng về mức
kênh 1 một mình (0,5537).

Cơ chế, đã ghi vào docstring trước khi chạy: `k` nhỏ **không chỉ** nâng ứng viên
chỉ-kênh-3, nó còn làm top-1 của riêng kênh 1 áp đảo mạnh hơn. Nó thưởng cho sự
tự tin của **cả hai** kênh, và kênh 1 mạnh hơn nên hưởng nhiều hơn.

> **Kết luận dùng được:** trần hạng là thật và **đáng sống chung**. Kênh 3 đáng
> giá đúng bằng phần nó đóng ở R@50 và R@100. Ba cách khai thác nó mạnh hơn —
> định mức chỗ (A75), hạ `k` (A75), cộng điểm chuẩn hoá (A72) — đều đã đo và
> đều thua. Muốn kênh 3 chạm được R@1/R@5 thì phải làm nó **đúng hơn**, không
> phải trộn nó **khác đi**.

### A76. VietOCR: **gộp** với OCR cũ thắng cả hai phía, và ngưỡng đặt trước đã sai

`vietnamese-news-video-ocr` của một nhóm khác dùng PaddleOCR dò vùng + VietOCR
đọc chữ. Ba lượt Kaggle chết trong chuỗi phụ thuộc của paddle (`.post120`,
`pyclipper`, `imgaug`, rồi `imgaug` gọi `np.sctypes` đã bị bỏ ở NumPy 2). Bỏ
paddle, dùng **EasyOCR dò vùng (CRAFT, chạy trên torch)** — cùng ngăn xếp với
VietOCR, không thêm runtime thứ hai. 251 khung, T4.

| | CŨ | MỚI (VietOCR) | **GỘP** |
| --- | ---: | ---: | ---: |
| có dấu — mọi khung | 8% | 48% | 48% |
| có dấu — riêng khung đáp án | 20% | **82%** | **82%** |
| khớp đúng dấu | 4/13 | 5/13 | **6/13** |
| khớp khi bỏ dấu | 6/13 | 5/13 | 6/13 |
| **CHỈ đáp án CÓ dấu** | **0/5** | **2/5** | **2/5** |

Tốc độ **0,74 s/ảnh** -> cả kho 36,6 giờ.

#### Ngưỡng đặt trước là SAI, và chỗ sai đáng ghi lại

Ngưỡng ghi trước khi đo: *"khớp đúng dấu phải ≥ 4/13"*, dựa trên câu của A68 là
"hiện 0/13". Nhưng mốc cũ đo được **4/13** — tức ngưỡng chính là nguyên trạng,
một cái bar mà không làm gì cũng đạt.

Nguyên nhân: **8/13 đáp án không chứa dấu nào** (`46`, `2,15`, `7`, `20`, `2`,
`200g`, `1204`, `Giang Ly`). Với chúng, "khớp đúng dấu" và "khớp khi bỏ dấu" là
**cùng một phép thử** — chúng không nói gì về dấu, chỉ làm loãng con số.

> Ghi ngưỡng trước khi xem số vẫn đúng và vẫn nên làm. Nhưng ngưỡng phải đặt
> trên **đại lượng thật sự đo được thứ cần đo**. Ở đây đại lượng đúng là "khớp
> đúng dấu **trên các đáp án CÓ dấu**": **0/5 -> 2/5**. `83_do_vietocr.py` nay
> in thẳng dòng đó.

#### GỘP chứ không THAY — phép đo quyết định

VietOCR **làm mất một con số** OCR cũ đọc được: `qa-DE1-15`, đáp án `46`, đi từ
✅ về —. Hai bộ hỏng ở chỗ khác nhau: cũ mất dấu, mới mất số.

Văn bản gộp `OCR cũ + VietOCR` giữ **cả hai**: 2/5 đáp án có dấu *và* lấy lại
`46`. **Không câu nào tệ hơn cả CŨ lẫn MỚI** — hợp đúng nghĩa, không đánh đổi.
BTC không phạt dòng sai và BM25 chỉ lợi khi có thêm từ, nên gộp không có mặt trái.

#### Kèm theo: đào đáp án ưu tiên bản CÓ DẤU

`run.py` ghép `OCR + " " + ASR` cho mỗi khung, mà ASR 100% có dấu — nên cùng
một thực thể hay xuất hiện hai lần: `Ta Pua` rồi `Tà Pứa`. Chọn theo khoảng
cách không phân biệt, mà OCR đứng trước nên hay thắng. `dap_an.uu_tien_co_dau()`
chọn bản có dấu khi **cả hai cùng có mặt** — không đoán dấu, không cần model.
Với văn bản gộp, tình huống này càng hay xảy ra.

#### Chia việc: 7/12 phần KHÔNG cần L26

L26 chiếm **44,9% kho** (498 video, 79.590 khung), và `chia_caption` trải nó đều
ra cả 12 phần (39–43 video L26 mỗi phần) nên ai chưa tải được L26 thì không chạy
nổi phần nào. `chia_ocr/` tách hai nhóm, cân theo **số khung** chứ không theo số
video (L23: 25 video/2.326 khung; L25: 88 video/37.445 khung):

    A1-A7 : 53-54 video, ~13.960 khung, 2,9 giờ  -> KHÔNG cần L26
    B1-B5 : 99-100 video, ~15.920 khung, 3,3 giờ -> chỉ L26

Chia mới ở đây hợp lệ vì OCR chưa ai chạy phần nào; bài học "đừng chia lại" nói
về bản chia **đang có người chạy dở**. `88_chia_viec_ocr.py` tự từ chối ghi đè.

### A77. ASR viết số bằng chữ — **nhưng đáp án số không có trong ASR ở dạng nào cả**. Đóng.

A68 nêu hai rào cản của việc đào đáp án Q&A từ văn bản. A76 vá rào cản 1 (OCR
không dấu). Rào cản 2 là *"asr_text 100% có dấu nhưng viết SỐ bằng CHỮ"*, và
**7/13 đáp án Q&A là số** (`46`, `2,15`, `7`, `20`, `2`, `200g`, `1204`). Đề
xuất: dùng `vietnam-number` chuyển chữ -> số trước khi đưa vào BM25/đào đáp án.

#### Đo chiều NGƯỢC trước khi cài gì (`90_do_so_bang_chu.py`)

Thư viện làm chiều *chữ -> số*, nhưng để biết có ĐÁNG cài thì chiều ngược trả
lời rẻ hơn và chắc hơn: sinh cách đọc tiếng Việt của đáp án rồi tìm trong ASR.
Viết số thành chữ dễ hơn và không mơ hồ; và nếu dạng chữ **không** có trong ASR
thì không thư viện nào cứu được.

Sinh **nhiều biến thể** để không đếm thiếu — phép đo chỉ có nghĩa nếu nó rộng
lượng với hướng đang xét: `mười lăm`/`mười nhăm`, `hai mươi tư`/`hai mươi bốn`,
`linh`/`lẻ`, `nghìn`/`ngàn`, và cả cách đọc rời từng chữ số.

| | |
| --- | ---: |
| đáp án có chứa số | **7/13** |
| khớp bằng **chữ số** trong ASR | **0/7** |
| tìm thấy **dạng chữ** trong ASR | **1/7** |

1/7 đó là `7` -> `"bảy"`, một từ quá phổ biến để tin là trùng thật.

#### Nhưng chúng CÓ trong OCR — nên rào cản này không tồn tại

Cột CŨ của `83_do_vietocr.py` cho thấy `46`, `200g`, `1204` đều khớp trong OCR.
Đáp án số đến từ **chữ trên màn hình**, không phải lời nói. Chuyển số-bằng-chữ
trong ASR sẽ không cứu được câu nào.

> Giá trị của A77: một rào cản có thật ở mức thống kê (ASR đúng là viết số bằng
> chữ) hoá ra **không chặn thứ ta cần**, vì thứ ta cần nằm ở kênh khác. Đo
> "rào cản có tồn tại không" và đo "rào cản có chặn ĐƯỜNG ĐI CỦA TA không" là
> hai câu hỏi khác nhau.

### A78. TRAKE: cách chấm video **trơ**, và dòng `oracle` đã bị tôi đọc sai

A74 đo được K-best lấy lại +0,1029 ở ±2s nhưng vẫn cách oracle đúng 0,1029, và
tôi kết luận *"vẫn còn ngần ấy nằm ở khâu CHỌN VIDEO"*. Hai phép đo dưới đây
cho thấy kết luận đó sai.

#### 1. Bốn cách chấm điểm video: gần như trơ hoàn toàn (`89_do_chon_video_trake.py`)

Giả thuyết: video đúng có TẤT CẢ sự kiện khớp ở mức chấp nhận được; video nhiễu
chỉ có 1–2 sự kiện khớp tình cờ rất mạnh. Nên càng phạt nặng **mắt xích yếu**
càng chọn đúng.

Vì mọi video có cùng số sự kiện, `Σ log(max)` **tương đương đơn điệu với trung
bình NHÂN**. Bốn cách xếp thành họ đơn điệu theo độ khắt khe, nên kết quả đọc
được dù đi hướng nào:

| cách chấm | hạng 1 | top-5 | top-20 | ngoài bể |
| --- | ---: | ---: | ---: | ---: |
| tổng (trung bình cộng) | 10/17 | 15/17 | 16/17 | 0/17 |
| **tổng-log (trung bình nhân) ← đang dùng** | **10/17** | **15/17** | **16/17** | 0/17 |
| điều hoà | 10/17 | 15/17 | 16/17 | 0/17 |
| min (mắt xích yếu nhất) | 9/17 | 15/17 | 16/17 | 0/17 |

Ba cách đầu **giống hệt nhau tới từng câu**. Không phải hình chuông quanh trung
bình nhân — **phẳng**. `min` còn tệ hơn: nó cũng phạt cả video ĐÚNG có một sự
kiện khó. Giả thuyết bị bác.

Và **15/17 video đúng đã nằm trong top-5**, mà thuật toán lấy đúng top-5. Khâu
chọn video **không phải nút thắt**.

#### 2. Ngân sách dòng: dồn hết là TỆ NHẤT (`91_do_ngan_sach_trake.py`)

Hạng của video đúng: **h1:10, h2:3, h3:1, h5:1, h8:1, h23:1**

| ngân sách 100 dòng | ±2s | ±15s |
| --- | ---: | ---: |
| **dồn hết hạng 1** | **0,2976** | **0,4382** |
| 70/30 | 0,3318 | 0,5106 |
| 50/30/20 | 0,3494 | 0,5341 |
| *50/25/15/7/3 — mốc* | *0,3465* | *0,5341* |
| **40/25/15/12/8** | **0,3571** (+0,0106) | **0,5476** (+0,0135) |
| trải đều 5 | 0,3435 (−0,0029) | 0,5476 (+0,0135) |
| trải đều 10 | 0,3206 (−0,0259) | 0,5641 (+0,0300) |
| *oracle* | *0,4606* | *0,6747* |

#### Đọc lại `oracle`: nó KHÔNG phải "dư địa với tới được"

Oracle dồn 100 dòng vào đúng video **vì nó biết trước video nào đúng**. Khi
không biết, làm y hệt thao tác đó — "dồn hết hạng 1" — cho kết quả **tệ nhất
bảng**: 0,2976 so với 0,3465.

> **Bài học chung, không riêng TRAKE:** một dòng `oracle` đo *"nếu biết trước
> thì được bao nhiêu"*, **không** đo *"còn lấy được bao nhiêu"*. Hai thứ chỉ
> trùng nhau khi cái oracle biết là thứ có thể suy ra được. Ở đây nó không
> phải, và trải ngân sách chính là **cái giá bắt buộc phải trả cho việc không
> biết**. Câu "vẫn còn 0,1029 nằm ở khâu chọn video" (A74) nói quá.

#### Ứng viên còn sống, và vì sao vẫn chưa bật

`40/25/15/12/8` — trải phẳng hơn mốc một chút — dương ở **cả hai** mức dung sai
(+0,0106 / +0,0135). `trải đều 10` cao hơn ở ±15s nhưng **âm ở ±2s**: đảo dấu,
không dùng được.

Nhưng `91_` chưa in ngưỡng nhiễu, và trên 17 câu thì hiệu 0,01 rất dễ là nhiễu.
**Chưa đổi mặc định** — cùng lý do đã giữ K-best chưa bật ở A74. Việc cần làm
trước: cho `78_`, `89_`, `91_` báo cáo qua `bao_cao_do_nhay()` như mọi phép đo
khác, để có ngưỡng nhiễu và kết luận ✅/🟡/❌.

### A79. TRAKE: **thứ đầu tiên qua ngưỡng kể từ A52** — và nó đã BẬT trong `run.py`

Tập TRAKE đo được lên **20 câu** (6 đề thật + 14 tự soạn) sau khi 3 câu TRAKE
mới của `de_thi_thu` được soi xong. Soát trước khi dùng: cả ba cùng MỘT video,
mốc thời gian TĂNG DẦN, số sự kiện tách ra khớp số đáp án.

#### Trước tiên: hai ứng viên 🟡 riêng lẻ

| trên 20 câu | ±2s | ngưỡng | T-B-H | |
| --- | ---: | ---: | :---: | :---: |
| K-best 3s (ngân sách CŨ) | +0,0900 | 0,0937 | 9-2-9 | 🟡 thiếu 4% |
| ngân sách 40/25/15/12/8 | +0,0090 | 0,0101 | 3-0-17 | 🟡 thiếu 11% |

Ba câu mới **không phản đối**: 1 thắng, 0 thua, 2 hoà. Hiệu K-best giảm từ
0,1029 (17 câu) xuống 0,0900 chỉ vì chúng là câu điểm thấp, delta gần 0 — kéo
trung bình về 0 mà không giảm phương sai. **Thêm câu DỄ không giúp vượt ngưỡng.**

#### Chỗ thật sự chặn: K-best đổi BỀ RỘNG lấy CHIỀU SÂU

Bảng từng câu chỉ ra một câu ăn một phần tư hiệu ứng:

    trake-L25-004   CŨ 0,4800  ->  K-best 0,0000   (oracle 0,8000)

Hạng của video ĐÚNG trên 20 câu: **14 câu hạng 1, 4 câu hạng 2–5, 2 câu NGOÀI
top-5** — `trake-L25-002` hạng 8 và `trake-L25-004` **hạng 23**.

K-best chỉ xét top-5 video nên với hai câu đó nó sinh **không một giả thuyết
nào** -> đúng 0 điểm, trong khi cách cũ rải 1 dòng cho mỗi trong 100 video vẫn
chạm tới hạng 23. Một câu đó vừa ăn mất hiệu ứng vừa **thổi phồng phương sai**,
tức tự đẩy ngưỡng nhiễu lên chống lại chính nó.

#### Lưới an toàn, và cơ chế TỰ XÁC NHẬN (`92_do_lai_ghep_trake.py`)

Dành `n` dòng cuối cho **1 chuỗi tốt nhất mỗi video từ hạng 6 trở đi**:

| | ±2s | ±15s | T-B-H ±2s | |
| --- | ---: | ---: | :---: | :---: |
| *K-best thuần — mốc* | 0,3355 | 0,5555 | — | |
| **CŨ 1 dòng/video** | 0,2365 | 0,5090 | **2-11-7** | **✅ TỆ HƠN −0,0990** |
| + 10 dòng đuôi | 0,3305 | 0,5555 | 0-1-19 | 🟡 −0,0050 |
| **+ 20 dòng đuôi** | **0,3490** | **0,5740** | 2-1-17 | 🟡 +0,0135 |
| + 30 dòng đuôi | 0,3490 | 0,5740 | 2-1-17 | 🟡 +0,0135 |
| + 50 dòng đuôi | 0,3495 | 0,5870 | 2-4-14 | 🟡 +0,0140 |

`n = 10` **âm**, `n = 20` dương. Chỗ nhảy tính ra được: top-5 cộng `n` dòng đuôi
phủ tới hạng `5+n`. `n=10` phủ tới hạng 15 — **chưa chạm hạng 23**, nên trả giá
mà không được gì. `n=20` phủ tới hạng 25, chạm được, và lãi hiện ra.

> Hiệu ứng xuất hiện **đúng chỗ cơ chế nói nó phải xuất hiện**. Đó là loại xác
> nhận đáng tin hơn con số, vì nó không thể đến từ nhiễu — nhiễu không biết
> hạng 23 nằm ở đâu.

`n = 50` không hơn `n = 20` mà lại 2-4-14: cắt quá nhiều chiều sâu của top-5.

#### Hai cải tiến 🟡 cộng lại thành một cải tiến ✅

Với ngân sách 40/25/15/12/8, K-best thuần lên 0,3355 (so với 0,3265 ở ngân sách
cũ), và hiệu so với cách CŨ thành **−0,0990 với ngưỡng 0,0932 -> ✅ ỔN ĐỊNH**,
2 thắng / 11 thua.

⚠️ Điều này chạm vào luật *"chỉ đổi MỘT thứ mỗi lần"*, nên phải nói rõ: ✅ ở đây
là của **tổ hợp**, còn từng thành phần đứng riêng đều 🟡 — và ta biết vậy vì đã
đo riêng từng cái (A74, A78). **Quy công không bị nhầm**, chỉ là thứ đem bật là
cả cặp. Luật đó tồn tại để chống quy công nhầm, không phải để cấm ghép.

#### ĐÃ BẬT — `src/kbest_trake.py`, mặc định trong `run.py`

    cach_nhau = 3,0s          A74: 0,5 / 1,5 / 3,0 đều dương, 3,0 tốt nhất
    ty_le = 40/25/15/12/8     A78: dồn hết hạng 1 là TỆ NHẤT (✅), trải đều hẳn
                              thì ❌ đảo dấu; tối ưu sát ngay cạnh cách cũ
    n_duoi = 20               A79: cơ chế trên

`--trake-cu` quay lại cách cũ để dựng lại bài nộp cũ khi cần đối chiếu.

**`run.dung_trake()` KHÔNG bị đổi** — nó là mốc nền "CŨ" mà `75_`, `78_`, `92_`
đang so với. Đổi nó là làm mốc nền trôi theo và mọi phép đo TRAKE cũ thành
không so lại được. Chỉ CHỖ GỌI trong `run.py` đổi.

`beam_video` cũng được dời từ `scripts/78_` sang `src/` — hai bản song song là
hai bản sẽ trôi khỏi nhau, đúng bài học vừa gặp với `diem_bai_nop`.

#### Hai điều phải nhớ khi đọc lại kết luận này

* ✅ dựa trên **±2s**; ở ±15s hiệu cùng dấu (−0,0465) nhưng **chưa vượt nhiễu**
  (0,0900). Kết luận mạnh ở cửa hẹp, yếu ở cửa rộng.
* **14/20 câu TRAKE là câu tự soạn.** A63 lập luận dùng được ở đây vì thứ tự sự
  kiện và số Frame ID là ràng buộc **hình thức**, không phụ thuộc câu dễ hay
  khó — lập luận đó vẫn đứng, nhưng nó là lập luận, không phải phép đo.

### A80. Phạt **mềm** theo khoảng cách thời gian — bác, và phân bố thật giải thích vì sao

Hướng TRAKE cuối cùng trong danh sách chưa đo. Đề xuất gốc: thay *"ép tăng dần
NGẶT + nội suy chỗ thiếu"* bằng `DP[i,t] = S[i,t] + max_τ(DP[i-1,τ] − λ(t−τ))`.

#### Nửa đề xuất đã tự hết hiệu lực

A79 bật K-best, mà K-best **vốn không nội suy**: beam sinh chuỗi tăng dần thật,
không chèn khung đoán vào chỗ thiếu. Nửa "bỏ nội suy" của đề xuất **đã có sẵn**
trước khi đo. Phần còn áp dụng được là phạt theo độ dài khoảng cách khi nối:

    điểm chuỗi = Σ điểm ứng viên − λ · Σ (khoảng cách giây tới sự kiện trước)

#### Chọn dải λ bằng ĐƠN VỊ, và tôi vẫn ước sai

Đề xuất gợi ý λ ∈ [0,001; 0,01]. Dải đó phụ thuộc thang điểm của hệ thống khác,
nên tôi tự tính lại: điểm ở đây là RRF (**0,008–0,03**), khoảng cách giữa hai
sự kiện tôi đoán ~50s, vậy λ ≈ 0,0002 để hai vế sánh nhau.

Script in luôn phân bố **thật** để kiểm chính giả định đó, và nó sai:

    khoảng cách giữa hai sự kiện liền kề (n=63):
        trung vị 12,0s   |   min 1,5s   |   max 259,3s

Trung vị 12s chứ không phải 50s. Nên ngay ở λ = 0,0002 phép phạt đã bằng
**24%** một điểm RRF điển hình — đủ để lấn át.

#### Kết quả: hại đơn điệu theo cường độ (20 câu TRAKE)

| λ | ±2s | ±15s | hiệu ±2s | T-B-H | |
| ---: | ---: | ---: | ---: | :---: | :---: |
| **0 (TẮT) — mốc** | **0,3490** | **0,5740** | — | — | |
| 5e-05 | 0,3495 | 0,5705 | +0,0005 | 2-2-16 | ❌ ĐẢO DẤU |
| 0,0002 | 0,3390 | 0,5605 | −0,0100 | 2-5-13 | 🟡 |
| 0,001 | 0,3285 | 0,5230 | −0,0205 | 3-9-8 | ✅ TỆ HƠN |
| 0,005 | 0,2575 | 0,5090 | −0,0915 | 2-12-6 | ✅ TỆ HƠN |

#### Vì sao — và câu trả lời nằm ngay trong phân bố vừa in

Khoảng cách thật trải từ **1,5s tới 259,3s**, rộng gấp **170 lần**. Phạt theo
khoảng cách giả định các sự kiện nằm gần nhau, nhưng dữ liệu nói khoảng cách
thật cực kỳ tản mạn — nên phép phạt **trừng phạt đúng những chuỗi ĐÚNG mà có
khoảng cách dài thật**. Nó không phân biệt được *"nhảy cóc vì đoán bừa"* với
*"hai sự kiện thật sự cách nhau 4 phút"*.

> Giá trị của A80: script in **phân bố thật của đại lượng bị phạt** trước khi
> in kết quả. Nhờ vậy con số âm không dừng ở "không hiệu quả" mà nói được vì
> sao — và nó cũng bắt luôn chỗ tôi ước sai gấp bốn lần khi chọn dải tham số.

Mốc nền là cấu hình `run.py` vừa bật ở A79, tức mốc mạnh nhất hiện có. Tham số
`phat_giay` giữ lại trong `kbest_trake.beam_video`, mặc định 0,0 — để lần sau
ai nghĩ ra ý này thì thấy nó đã được đo.

#### Trạng thái các hướng TRAKE sau A80

| hướng | kết cục |
| --- | --- |
| K-best beam search thay 1-dòng/video | ✅ **ĐÃ BẬT** (A79) |
| ngân sách dòng 40/25/15/12/8 | ✅ **ĐÃ BẬT** cùng A79 |
| lưới an toàn 20 dòng đuôi | ✅ **ĐÃ BẬT** cùng A79 |
| cách chấm điểm video (4 biến thể) | nút **trơ** (A78) |
| phạt mềm theo khoảng cách | **bác** (A80) |
| dồn hết ngân sách cho hạng 1 | **bác**, ✅ tệ hơn (A78) |

Danh sách hướng TRAKE **đã cạn**. Thứ chặn tiếp theo không phải thiếu ý tưởng
mà là **số câu TRAKE**: 20 câu, trong đó 14 tự soạn, và 8–9 câu không đổi gì
giữa các cấu hình nên n hiệu dụng chỉ khoảng 11.

### A81. "Cluster flooding" **không có**, và 6 câu ngoài bể **không có tín hiệu chung**

Hai chẩn đoán trên cùng một lượt chạy (`94_soi_cau_that_bai.py`, 66 câu KIS/QA,
bể 1.000).

#### 1. Top-20 KHÔNG bị khung lân cận chiếm chỗ

Giả thuyết: R@20 đứng yên ở 0,6122 qua mọi cỡ bể (A54) vì top-20 bị các
keyframe lân cận của cùng một shot SAI chiếm hết; lọc phi cực đại theo thời
gian sẽ giải phóng 8–12 chỗ.

**Lập luận đó không đứng vững trước khi đo.** R@20 không đổi khi bể đi từ 100
lên 1.000 chỉ nói **ứng viên mới đều xếp dưới hạng 20** — nó không nói gì về
thứ ĐANG chiếm top-20. Đo thẳng:

| trong top-20 | trung vị | min | max |
| --- | ---: | ---: | ---: |
| video khác nhau | **10/20** | 1 | 20 |
| hàng xóm thời gian ≤4s | **2/20** | 0 | 10 |
| *câu có ≥10 hàng xóm* | **1/66** | | |

Top-20 điển hình gồm **10 video khác nhau** và chỉ **2** ứng viên là khung lân
cận. Lọc phi cực đại sẽ giải phóng khoảng 2 chỗ, không phải 8–12.

> Và nó giải thích luôn **vì sao A18 thất bại**: ràng buộc đa dạng (tối đa 2
> ứng viên mỗi video) làm tệ đi vì **không có dư thừa để dọn** — cắt theo video
> chỉ xoá mất ứng viên tốt. Hai phép đo cách nhau nhiều tuần, cùng một nguyên
> nhân.

#### 2. Sáu câu ngoài top-1.000 — và một kết luận suýt bị công bố

| câu | chữ ở khung đúng | có dấu | dài câu |
| --- | ---: | :---: | ---: |
| kis-DE1-11 | 168 ký tự | có | 57 |
| kis-DE1-02 | 1.318 ký tự | có | 45 |
| kis-DE1-23 | 5.002 ký tự | có | 112 |
| qa-DE2-27 | 147 ký tự | — | 62 |
| qa-DE2-28 | 499 ký tự | có | 72 |
| qa-DE2-30 | 33 ký tự | — | 74 |

**6/6 đều là câu dài >40 từ** — trông như một phát hiện. **Tỷ lệ nền giết nó:**
55/66 câu (83%) của cả tập đã dài >40 từ, nên xác suất cả 6 đều dài khi chọn
ngẫu nhiên là **0,335**. Trung vị 67 từ so với 62 của toàn tập.

> Suýt nữa tôi ghi "câu dài là nguyên nhân" vào tài liệu. Thứ chặn lại là một
> phép tính hai dòng: **luôn tính tỷ lệ nền trước khi gọi một tương quan là
> phát hiện.** Với n = 6 thì gần như mọi thuộc tính phổ biến đều xuất hiện ở
> cả 6.

Thứ nói được, và chỉ thế thôi:

* **Cả 6 câu đều CÓ chữ ở khung đúng** (4/6 có dấu). Nên chúng không thất bại
  vì thiếu dữ liệu văn bản, và **A76 (VietOCR) sẽ không cứu được câu nào trong
  số này**. Đó là câu trả lời cho câu hỏi "6 câu này có cùng lý do với Q&A
  không": **không**.
* Kênh 3 có văn bản trong tay mà vẫn không đưa chúng lên -> chữ ở khung đúng
  không khớp cách diễn đạt của truy vấn. Khoảng cách **từ vựng**, không phải
  khoảng cách dữ liệu.
* **n = 6 quá nhỏ để nói thêm gì** (A74: "n=3 không phải một phép đo").

### A82. Khuếch tán điểm kênh 3 theo thời gian — cơ chế bị bác **bởi chính phép đo cơ chế**

A71 đo chồng@20 giữa kênh 1 và kênh 3 chỉ 3,4%. Có một cách đọc rất thuyết
phục: **chữ và hình không xuất hiện cùng một mili-giây** — người nói nhắc chủ
đề ở giây 10, hình minh hoạ hiện ở giây 14, biển hiệu lướt qua ở giây 8. RRF
cộng theo `row_id` nên ba sự kiện đó không bao giờ gặp nhau.

Cách chữa: trước khi hợp nhất, cho điểm BM25 lan sang keyframe cùng video theo
Gauss thời gian `S'(t) = Σ_k S(k)·exp(−(t_k−t)²/2τ²)`, giữ nguyên vector ảnh
sắc nét. Dự đoán kèm theo: **chồng@20 tăng từ 4% lên 25–35%**.

#### Dự đoán đó là thứ làm phép đo này dứt khoát

`95_do_khuech_tan_thoi_gian.py` đo **hai** thứ: điểm cuối, và chồng@20. Ghi rõ
trong docstring trước khi chạy: *nếu điểm tăng mà chồng KHÔNG tăng thì cơ chế
giả thuyết sai dù kết quả đúng, và không được ghi cơ chế đó vào tài liệu.*

| | chồng@20 |
| --- | ---: |
| gốc (τ = 0) | **2,8%** |
| τ = 2s | **1,5%** |
| τ = 4s | **1,5%** |
| τ = 6s | 1,7% |

**Chồng GIẢM gần một nửa**, không tăng gấp tám. Dự đoán sai cả về hướng.

Lý do: khuếch tán đổi **thứ hạng** của kênh 3. Sau khi làm mềm, top-20 của kênh
3 bị các vùng dày chữ chiếm — nơi nhiều khung có điểm nằm sát nhau nên cộng dồn
lên cao — mà đó không phải chỗ top-20 của kênh 1 đang đứng. Hai kênh **đồng ý
ít hơn**, không phải nhiều hơn.

#### Điểm cuối (72 câu, mốc là kênh 3 gốc)

| cấu hình | ±2s | ±15s | hiệu ±2s | T-B-H | |
| --- | ---: | ---: | ---: | :---: | :---: |
| **MỐC: kênh 3 gốc** | **0,5611** | **0,6514** | — | — | |
| chỉ kênh 1 *(chẩn đoán)* | 0,5326 | 0,6264 | −0,0285 | 3-10-59 | ✅ |
| khuếch tán τ=2s | 0,5583 | 0,6514 | −0,0028 | 2-3-67 | ❌ ĐẢO DẤU |
| khuếch tán τ=4s | 0,5528 | 0,6597 | −0,0083 | 1-4-67 | ❌ ĐẢO DẤU |
| khuếch tán τ=6s | 0,5528 | 0,6542 | −0,0083 | 1-4-67 | ❌ ĐẢO DẤU |

Cả ba mức ❌ đảo dấu — không dùng để quyết được, và cơ chế thì đã đổ.

#### Kèm theo: đóng góp của kênh 3 nay ✅ ỔN ĐỊNH

Dòng chẩn đoán trên 72 câu: bỏ kênh 3 mất **−0,0285 với ngưỡng 0,0260, 3 thắng
/ 10 thua -> ✅**. Trước đây con số này luôn 🟡. Tập lớn lên đúng như A65 dự
đoán, và **giá trị của kênh 3 giờ là kết luận vững** chứ không còn là ứng viên.

#### Vì sao A82 khác A57 và A70 — và vì sao vẫn thua

Ba thứ cùng họ "làm mềm theo thời gian", đã bác cả ba, nhưng mỗi cái một cơ chế:

* **A57** làm mượt *vector kênh 1* -> san phẳng phần dư dùng để phân biệt khung
  đáp án với hàng xóm.
* **A70** gộp vector theo *đoạn ASR* (~39 giây) -> mọi khung trong đoạn nhận
  cùng một biểu diễn, mất khả năng xếp hạng bên trong đoạn.
* **A82** chỉ làm mềm *trường điểm kênh 3*, τ = 2–6 giây, không đụng kênh 1 —
  tránh được cả hai khuyết tật trên, và **vẫn thua**, vì nó phá thứ khác: thứ
  hạng nội bộ của chính kênh 3.

> Ba lần cùng một họ ý tưởng, ba cơ chế hỏng khác nhau. Kết luận dùng được:
> **trục thời gian của kho này không có dư thừa để khai thác** — A81 đo được
> top-20 chỉ có 2/20 hàng xóm, tức các khung gần nhau vốn đã không chen chỗ
> nhau. Làm mềm theo thời gian không có gì để dọn, chỉ có thứ để phá.

### A83. Hai cách khai thác **luật chấm**: phạt bậc hết hại nhưng vô ích; rải biến thể đáp án **gấp đôi điểm Q&A**

#### 1. Phạt khoảng cách dạng BẬC — chẩn đoán đúng, cách chữa không đủ

A80 bác phạt **tỷ lệ thuận** với khoảng cách, và chỉ ra lý do: khoảng cách thật
có trung vị 12,0s nhưng trải từ 1,5s tới 259,3s, nên phạt đều tay trừng phạt cả
những chuỗi ĐÚNG có khoảng cách dài thật.

Hàm bậc né đúng chỗ đó: phạt nặng khi Δt < 1s, **không phạt trong [1s, 60s]**,
phạt tuyến tính khi Δt > 60s. A80 không bác được nó, nên đo riêng (`93_`):

| hàm phạt | ±2s | ±15s | |
| --- | ---: | ---: | :---: |
| **λ = 0 (TẮT) — mốc** | **0,3490** | **0,5740** | |
| tỷ lệ thuận λ=0,001 | −0,0205 | −0,0510 | ✅ TỆ HƠN |
| tỷ lệ thuận λ=0,005 | −0,0915 | −0,0650 | ✅ TỆ HƠN |
| bậc `<1s:1 >60s:0,0005` | −0,0035 | −0,0110 | 🟡 |
| bậc `<1s:1 >60s:0,002` | +0,0040 | −0,0110 | ❌ ĐẢO DẤU |
| bậc `<1,5s:1 >60s:0,0005` | +0,0055 | −0,0085 | ❌ ĐẢO DẤU |
| bậc `<1s:0,02 >60s:0,0002` | −0,0035 | −0,0110 | 🟡 |
| bậc `<1s:1 >30s:0,0005` | +0,0030 | −0,0160 | ❌ ĐẢO DẤU |

**Chẩn đoán đúng, và sửa đúng chỗ:** phạt tỷ lệ thuận ✅ ổn định có hại, còn
phạt bậc thì **hết hại** — không cấu hình nào vượt nhiễu theo hướng xấu, ba
trong năm còn nhỉnh lên ở ±2s.

**Nhưng bỏ được cái hại không tạo ra cái lợi.** Ba cấu hình ❌ đảo dấu, và cả
năm đều âm ở ±15s.

> Kết luận dùng được: giả thuyết *"chuỗi có khoảng cách đều và mạch lạc thì
> đáng tin hơn"* **không mang tín hiệu** trên kho này. Không phải hàm phạt sai
> hình dạng — hình dạng đã sửa đúng — mà là **bản thân khoảng cách thời gian
> không nói gì về việc chuỗi đúng hay sai**. Khác biệt giữa *"đo sai"* và
> *"không có gì để đo"*.

#### 2. Rải nhiều biến thể `answer` qua nhiều dòng — 🟡 mạnh nhất của Q&A

BTC cho 100 dòng và **không phạt dòng sai**, nhưng `dap_an.dao()` chỉ trả về
MỘT chuỗi cho mỗi khung — chuỗi đó sai thì cả 100 dòng cùng sai. `dao_nhieu()`
trả về nhiều biến thể; `96_do_rai_bien_the.py` rải chúng ra nhiều dòng.

13 câu Q&A có đáp án vàng:

| cấu hình | điểm | hiệu | T-B-H | ngưỡng |
| --- | ---: | ---: | :---: | ---: |
| **MỐC: 1 biến thể/dòng** | **0,0462** | — | — | — |
| **rải 2 biến thể, 10 khung đầu** | **0,1077** | **+0,0615** | **1-0-12** | 0,1231 |
| rải 3 biến thể, 10 khung đầu | 0,1077 | +0,0615 | 1-0-12 | 0,1231 |
| rải 2 biến thể, 30 khung đầu | 0,1077 | +0,0615 | 1-0-12 | 0,1231 |
| rải 3 biến thể, 30 khung đầu | 0,1077 | +0,0615 | 1-0-12 | 0,1231 |
| **TRẦN: mọi biến thể** *(chẩn đoán)* | **0,1231** | +0,0769 | 1-0-12 | 0,1538 |

**Gấp 2,3 lần điểm mốc, 0 câu thua.** Nhưng số học nói thẳng: `+0,0615 = 0,8/13`
— **đúng MỘT câu** đi từ 0 lên 0,8. Trần `+0,0769 = 1,0/13` cũng là câu đó lên
hạng 1. **Toàn bộ hiệu ứng là một câu trên mười ba**, nên không kết luận được
dù không câu nào thua.

Cả bốn cấu hình cho kết quả **giống hệt nhau** — 2 hay 3 biến thể, 10 hay 30
khung đều như nhau. Nếu bật thì `2 biến thể / 10 khung đầu` là đủ.

#### Dòng TRẦN trả lời câu quan trọng hơn cả kết quả

Rải biến thể đạt **0,1077 trên trần 0,1231 — lấy được 87% khoảng cách**. Cơ chế
rải hoạt động gần như hoàn hảo; thứ chặn là **khâu ĐÀO**, không phải khâu rải.

Nên hướng đáng đầu tư tiếp không phải chia dòng khéo hơn, mà là **đào ra đúng
chuỗi hơn** — tức bảng tra thực thể cấp video từ ASR (chưa làm).

#### Kèm theo: một hạn chế của A76 lộ ra khi viết `dao_nhieu`

    văn bản  : "… Ta Pua vừa hoàn thành … Tại Tà Pứa hôm nay."
    dao()    : 'Ta Pua'          <- KHÔNG đổi sang bản có dấu

`uu_tien_co_dau()` so hai chuỗi sau khi bỏ dấu, mà mẫu tên riêng bắt tới **3 từ
viết hoa liên tiếp** nên bản có dấu bị bắt thành `"Tại Tà Pứa"` —
`bỏ_dấu("Tại Tà Pứa") ≠ bỏ_dấu("Ta Pua")`, không khớp, phép ưu tiên không nổ.

Không làm sai kết quả A76 (0/5 -> 2/5 vẫn đúng), nhưng nó nói phép ưu tiên
**yếu hơn tôi tưởng**: chỉ bắt được khi hai bản có **cùng số từ**. Đây đúng là
chỗ bảng tra n-gram 1–4 từ từ ASR mạnh hơn hẳn.

### A84. Bảng tra dấu từ ASR: **0/5 ở mọi phạm vi** — và lý do gốc không phải chuyện dấu

A68: `ocr_text` chỉ **31% có dấu**, `asr_text` **100%**. Nên ASR của chính kho
này là **một cuốn từ điển có dấu của chính nó** — không cần model phục hồi dấu,
chỉ cần tra. A83 còn chỉ ra chỗ này là nút thắt: rải biến thể đã lấy 87% khoảng
cách tới trần, phần chặn nằm ở **khâu ĐÀO**.

`dap_an.bang_tra_ngram()` dựng `{dạng bỏ dấu: dạng có dấu}` cho mọi n-gram 1–4
từ. Nó khắc phục đúng hạn chế A83 phát hiện ở `uu_tien_co_dau()` (chỉ bắt được
khi hai bản **cùng số từ**): `"tà pứa"` là một 2-gram riêng nên khớp được
`"Ta Pua"` dù ASR viết `"tại Tà Pứa"`.

    cả kho : 1.803.158 n-gram có dấu
    video  : 847 video có ASR, trung bình 4.021 n-gram/video

#### Ba phạm vi, và cả ba đều 0/5 (`97_do_bang_tra_asr.py`)

| phạm vi bảng tra | khớp đúng chuỗi |
| --- | ---: |
| không bảng tra (A76) | **0/5** |
| cùng KHUNG | **0/5** |
| cùng VIDEO | **0/5** |
| cả KHO | **0/5** |

Mẫu số là **5 câu có đáp án CHỨA DẤU** — 8/13 câu còn lại không có dấu nào nên
chúng không nói gì về phép phục hồi dấu.

#### Chẩn đoán: hỏng nằm TRƯỚC khâu dấu

| câu | đáp án | có trong văn bản (bỏ dấu)? | bộ đào ra |
| --- | --- | :---: | --- |
| qa-DE2-28 | Cá lóc | **có** | `['Nam', 'Nuoc', 'Gao']` |
| qa-DE2-09 | Cá sòng | **có** | `['Online']` |
| qa-DE1-17 | Tà Pứa | **có** | `['Binh Thuan', 'Som']` |

**3/5 câu có sẵn chuỗi đúng trong văn bản mà bộ đào không bao giờ chọn nó.**
Bảng tra không cứu được vì nó **không bao giờ được đưa cho ứng viên đúng**.

#### Lý do gốc: bộ đào chỉ nhìn cụm bắt đầu bằng chữ HOA

    TEN = [HOA][thường]+ (\s+[HOA][thường]+){0,2}

Nguyên văn ở khung đúng:

    qa-DE2-28  'NGUYEN LIEU Thit ca loc 300g Gao deo 100g …'   <- toàn chữ thường
    qa-DE2-09  '…để làm lần lượt cho nó hết cá sòng nướng…'      <- thường, CÓ dấu

Đáp án Q&A ở đây là **danh từ thường nằm giữa câu**, không phải tên riêng. Điều
kiện đầu tiên của mẫu là chữ hoa, nên chúng **về mặt cấu trúc không bao giờ đào
ra được**.

Đã thử nới: `TEN_RONG` (1 từ hoa + tối đa 2 từ thường) cộng phát ra mọi **đoạn
con** 1–3 từ. Vẫn **0/5** — vì điều kiện chữ hoa ở từ ĐẦU vẫn còn.

> Nới regex thêm nữa là đi tới *"mọi cụm 1–3 từ bất kỳ"*, tức nổ tung tập ứng
> viên và biến bài toán thành **xếp hạng cụm**, không phải trích cụm. Đó là
> thiết kế khác, không phải một bản vá. Dừng ở đây và ghi lại, thay vì đẩy một
> nửa lời giải vào đường chạy.

#### Ba thứ được giải thích cùng lúc

* **A83** trần thấp — vì mọi biến thể đều đến từ cùng một mẫu hỏng.
* **A76** được 2/5 — vì VietOCR viết hoa đầu cụm, tình cờ lọt qua mẫu.
* **A77** đáp án số nằm trong OCR chứ không trong ASR — cùng một chuyện: đáp án
  Q&A của kho này phần lớn là **chữ trên màn hình dạng danh sách nguyên liệu**,
  không phải tên riêng trong lời nói.

#### Việc cần làm tiếp, và nó không phải regex

Bài toán đúng là: *cho câu hỏi + văn bản của khung, chọn cụm nào là đáp án*.
Đó là **đọc hiểu**, và hai đường khả dĩ đều cần model — VLM đọc ảnh (A60 đã đo
là không hơn thứ tự sẵn có) hoặc LLM đọc `câu hỏi + văn bản`. Chưa đo đường thứ
hai.

Trong lúc chờ, `--rai-bien-the 2` (A83) là thứ rẻ nhất còn dùng được: gấp 2,3
lần điểm Q&A, 0 câu thua, và BTC không phạt dòng sai.

### A85. Soát toàn bộ đường ống: **`run.py` chết trên 19/25 gói đề thật** — và 328 test không bắt được

Rà lại toàn bộ mã chính (soát ngày 03/09). Bốn lỗi thật, hai phép đo mới.

#### 1. `run.py` ném `NameError` trên mọi truy vấn dài hơn 40 từ — MẤT TRẮNG CẢ BÀI NỘP

`quet_anh.hoi()` gọi `hop_nhat(...)` khi truy vấn bị `tach_truy_van` cắt thành
hơn một mệnh đề — nhánh A51 bật mặc định. Nhưng `run.py` **không import
`hop_nhat` ở tầng module**: lời gọi `from rrf import hop_nhat` duy nhất nằm
*bên trong* `main()`, tức là một tên cục bộ của `main`, không phải biến toàn
cục. Lỗi vào repo cùng commit A51 (`0b04a7b`).

Hậu quả không phải "một câu hỏng" mà là **không có file nào được ghi**:
`quet_anh` chạy trước vòng lặp gói, nên gói đầu tiên vượt trần đã giết cả lượt.

| bộ đề | gói làm `run.py` chết |
| --- | ---: |
| `De_Thi_Chinh_Thuc` | **19/25** |
| `de_thi_thu` | **18/24** |

Dựng lại được bằng một lệnh, không phải suy luận:

```
.venv\Scripts\python.exe src\run.py --de De_Thi_Chinh_Thuc --ra out --cache index\truy_van_gopt.npz
  File "src\run.py", line 251, in hoi
    return hop_nhat([kenh.tim(m, k=sl) for m in md])[:sl]
NameError: name 'hop_nhat' is not defined
```

**Vì sao 328 test không thấy.** `tests/test_run.py` chỉ chạm các hàm THUẦN —
`tach_su_kien`, `dong_hang_dp`, `dung_trake`. **Không test nào gọi
`quet_anh`**, tức hàm chạy kênh 1 và là chỗ duy nhất nhánh RRF mệnh đề đi qua.
Đã bổ sung ba test (`test_quet_anh_*`) dựng kênh giả, không cần model.

> Bài học chung: test phủ được từng viên gạch không có nghĩa là phủ được chỗ
> ghép. Chỗ ghép là nơi `run.py` hỏng, và cũng là nơi ba lỗi còn lại ở dưới nằm.

#### 2. `--rai-bien-the` là **VÔ HIỆU** trong `run.py` — A83 đo ở nơi khác

`nop_bai.tu_ung_vien()` bỏ trùng theo khoá `(video_id, frame_idx)`. Nhưng dòng
nộp Q&A là **bộ ba** `(video, frame, answer)`, và `dap_an.rai_bien_the()` phát
`k` biến thể `answer` cho CÙNG một khung — nên mọi biến thể trừ cái đầu bị vứt
lặng lẽ ngay tại đó. Chính `nop_bai.soat()` thì bỏ trùng theo cả ba ô, tức hai
hàm trong cùng một file bất đồng ý về "thế nào là hai dòng khác nhau".

A83 đo được rải biến thể **gấp 2,3 lần điểm Q&A** (0,0462 → 0,1077) — nhưng đo
trong `96_do_rai_bien_the.py`, **không đi qua `tu_ung_vien`**. Cờ chưa bao giờ
làm gì trên bài nộp thật.

Đã sửa: khoá bỏ trùng của Q&A nay gồm cả `answer`; KIS giữ nguyên
`(video, frame)`.

#### 3. K-best TRAKE nộp được **hai sự kiện cùng một Frame ID**

`beam_video` ép tăng dần theo `pts_time`, `lap_trake` lại nộp `frame_idx` — và
A5.7 đo được **614 cặp** cùng video có `pts_time` tăng nhưng `frame_idx` BẰNG
NHAU (0 cặp giảm). Dựng lại được: ba sự kiện ra `[0, 0, 519]`.

`nop_bai.soat` **không bắt** vì nó so với `sorted()`, mà `[0, 0, 519]` đã
sorted. `run.dung_trake` (đường CŨ) có chốt "phải TĂNG THẬT, không bằng nhau";
K-best — đường MẶC ĐỊNH từ A79 — thì không. Hai sự kiện là hai khoảnh khắc
khác nhau, nộp trùng ID là chắc chắn phí một. Đã thêm chốt vào `lap_trake`.

#### 4. Tác tử và `--vlm` **mù 45% kho**, im lặng

`src/tac_tu.py` và `mui_nhon_1.khung_ngu_canh()` đọc thẳng cột `kf_path` để
tìm ảnh. `kf_path` nghĩa là *"ảnh GỐC có ở máy này"* (A5.5) — nó rỗng ở **cả
79.590 dòng của L26**, vì không máy nào giữ 12,13 GB ảnh gốc đó.

Nhưng `anh.thong_ke()` trên chính máy này trả về **97.731 gốc + 79.590 bản thu
nhỏ = 177.321/177.321, tức 100%**. Ảnh có đủ; hai chỗ trên chỉ không hỏi đúng
cửa. Đúng lỗi `anh.ban_do_co_anh` đã vá cho `web/server.py` — hai chỗ này còn
sót. Đã cho cả hai đi qua `anh.tim()`, và nhãn nói rõ khung nào là bản nhỏ (ảnh
nhỏ đủ để NHẬN RA CẢNH, không đủ để ĐỌC CHỮ).

### A86. Hai phép đo mới: bù dòng TRAKE **vô ích**, và kênh 3 **không được gộp mệnh đề bằng RRF**

#### 1. TRAKE bỏ trống trung bình 40/100 dòng — và bù vào **không đổi một câu nào**

`run.py` trên `de_thi_thu` in ra `58`, `37`, **`11`** dòng cho ba gói TRAKE.
`kbest_trake.cham_video()` loại mọi video thiếu ứng viên cho *bất kỳ* sự kiện
nào, nên số video sinh được chuỗi có thể rất nhỏ; cách CŨ có nhánh rải cho tròn
100, K-best bỏ mất lưới đó. Theo PHẦN C mục 1 thì đó là 89 cơ hội vứt đi.

`scripts/98_do_bu_dong_trake.py`, 17 câu TRAKE:

| cấu hình | số dòng TB | ±2s | ±15s | thắng-thua-hoà | kết luận |
| --- | ---: | ---: | ---: | :---: | --- |
| K-best ← MỐC | 60,3 | 0,3812 | 0,5753 | — | — |
| + bù MỀM (video thiếu sự kiện, nội suy) | 95,4 | 0,3812 | 0,5753 | 0-0-17 | ⚪ KHÔNG ĐỔI GÌ |
| + bù MỀM + RẢI | 100,0 | 0,3812 | 0,5753 | 0-0-17 | ⚪ KHÔNG ĐỔI GÌ |

Không một câu nào đổi điểm, ở cả hai mức dung sai. Cơ chế giải thích được:
điểm là `max R-Score trong top-k`, mà dòng bù toàn là video kênh đã xếp **dưới
hạng 25**, với vị trí nội suy. Muốn ăn điểm TRAKE thì phải trúng *nhiều vị trí
trong cùng một dòng* — xác suất đó ở video hạng 40 là gần 0, khác hẳn KIS nơi
một dòng chỉ cần trúng một khung.

> **Chỗ này ngược với trực giác "không phạt thì cứ điền cho đủ"**, và ngược có
> lý do: luật "dòng thứ 100 vẫn đáng 0,2" đúng cho KIS/QA, nơi mỗi dòng là một
> phỏng đoán ĐỘC LẬP. TRAKE bắt trúng N vị trí cùng lúc nên đuôi danh sách
> gần như vô giá trị. **Đừng sửa `lap_trake` để bù dòng** — nó chỉ làm chậm và
> làm bài nộp khó soi hơn. Con số `11/100` trông đáng sợ nhưng vô hại.

#### 2. Ba đường chạy đưa truy vấn vào kênh 3 theo **ba cách khác nhau**

| nơi | cách gộp mệnh đề cho kênh 3 |
| --- | --- |
| `src/run.py` | `k3.tim(tach_truy_van(nd))` → **MAX điểm** qua mệnh đề |
| `scripts/57_, 77_, 86_` | `k3.tim(c.cau_hoi)` → **CẢ CÂU** |
| `web/server.py` | `hop_nhat([k3.tim(m) …])` → **RRF HẠNG** |

Nghĩa là trọng số 0,5 chốt ở A52 và kết luận A58 (*"kênh 3 cần cả câu"*) đều
được đo trên cấu hình `run.py` **không chạy**, còn giao diện soát tay thì vẽ ra
một bể ứng viên thứ ba. Đúng loại lệch A23 đã cắn.

`scripts/99_do_menh_de_kenh3.py`, 49 câu đề thật (80% bị tách >1 mệnh đề), mốc
nền là `run.py` như nó đang chạy:

| cấu hình | ±2s | ±15s | thắng-thua-hoà (±2s / ±15s) | kết luận |
| --- | ---: | ---: | :---: | --- |
| MỐC — run.py (max mệnh đề) | 0,5184 | 0,6082 | — | — |
| A. cả câu (script đo cũ) | 0,5224 | 0,6122 | 1-0-48 / 2-1-46 | 🟡 +0,0041 |
| B. RRF hạng (web/server) | 0,5102 | 0,5918 | 0-2-47 / 1-4-44 | 🟡 **−0,0082 / −0,0163** |
| C. kênh 1 một mình | 0,4939 | 0,5796 | 1-6-42 / 1-6-42 | 🟡 −0,0245 / −0,0286 |

**Hai kết luận dùng được:**

* **Lệch thì có thật nhưng NHỎ** (≤ 0,016). A52 và A58 không bị lật — đó là tin
  tốt, và nay có số để nói thế thay vì phải tin.
* **Lập luận A51 KHÔNG chuyển sang được cho kênh 3.** A51 thắng vì *cosine của
  hai mệnh đề khác nhau không so được với nhau*. Điểm BM25 thì **cùng thang** —
  cùng công thức, cùng kho — nên tiền đề biến mất, và đo ra RRF hạng là cấu
  hình **tệ nhất trong ba**, cùng dấu âm ở cả hai mức. Đã sửa `web/server.py`:
  RRF hạng cho kênh vector (1, 6), MAX điểm cho kênh BM25 (3, 5) — giao diện
  nay thấy đúng bể ứng viên của bài nộp.

**KHÔNG đổi `run.py`.** "Cả câu" hơn +0,0041 nhưng dưới ngưỡng nhiễu 0,0082 ở
±2s — chưa đủ căn cứ, và đổi mốc nền thì mọi phép đo cũ hết so được.

### A87. Hệ thống hiện tại so với bản SigLIP2-1152: **gấp 2,4 lần**, và trần còn 0,35

Đo cả hai ma trận trong **một lượt chạy, trên cùng bộ câu** — tôn trọng đính
chính A54, nơi tôi từng đọc chênh lệch giữa ba lượt riêng như một hiệu ứng thật.

#### Khoá tập câu trước, vì hai cache phủ khác nhau

    gopt-1536     72/72 câu đủ chuỗi
    siglip2-1152  52/72 câu đủ chuỗi   <- cache cũ chỉ phủ `tap_de_that`
    TẬP KHOÁ      52 câu

Đo gopt trên 72 câu rồi so với SigLIP2 trên 52 câu là so **hai bộ đề khác
nhau**. Không cần khoá bể ứng viên: cả hai ma trận đều phủ trọn 177.321 dòng,
nên bẫy `dense.be_chung` của A17 (+0,2833 vì bể nhỏ hơn thắng) không áp dụng.

#### Điểm thật (52 câu)

| cấu hình | ±2s | ±15s | hiệu ±2s | T-B-H | |
| --- | ---: | ---: | ---: | :---: | :---: |
| **gopt + kênh 3 — ĐANG CHẠY** | **0,5317** | **0,6067** | — | — | |
| gopt một mình | 0,4904 | 0,5683 | −0,0413 | 2-9-41 | ✅ |
| SigLIP2-1152 + kênh 3 | 0,2221 | 0,3077 | **−0,3096** | 5-31-16 | ✅ |
| SigLIP2-1152 một mình | 0,1760 | 0,2462 | −0,3558 | 4-35-13 | ✅ |

**Gấp 2,4 lần** (0,5317 so với 0,2221), ✅ ổn định ở cả hai mức dung sai,
31 thắng / 5 thua. Đây là khoảng cách lớn nhất giữa hai cấu hình từng đo trong
repo, và nó xác nhận A47 ở quy mô toàn hệ thống chứ không chỉ riêng kênh 1.

Dòng chẩn đoán cũng cho một con số đáng nhớ: **kênh 3 đóng góp +0,0413** trên
nền gopt, ✅ ổn định ở cả hai mức. Trước đây con số này luôn 🟡.

#### Trần — điểm cao nhất hệ thống có thể đạt

Trần = *đáp án nằm đâu đó trong bể thì coi như xếp lại hoàn hảo cho 1,0*.
Bể 1.000, tính trên **49 câu KIS/QA** (TRAKE chấm theo vị trí nên "có trong bể"
không cùng nghĩa):

| cấu hình | trần ±2s | trần ±15s |
| --- | ---: | ---: |
| **gopt + kênh 3** | **0,8776** | **0,9592** |
| SigLIP2-1152 + kênh 3 | 0,6531 | 0,8163 |

Khớp A54 (trần 0,8654 ở cùng cấu hình, chênh do A51 đã đổi cách hợp nhất mệnh
đề sau đó). **Khoảng trống còn ~0,35** — và hơn 20 hướng xếp lại đã thử, chưa
hướng nào lấy quá 2,3% của nó (A55–A62, A72, A81, A82).

⚠️ **Trần tính trên 49 câu KIS/QA, điểm thật tính trên 52 câu gồm 3 câu TRAKE**
(chấm ở tầng KÊNH, xem cảnh báo tự động). Hai con số **không trừ thẳng cho
nhau** được — dùng A54 nếu cần con số trống chính xác trên cùng mẫu số.

#### Đọc ra ba điều

1. **Đổi model là thay đổi lớn nhất từng đo**, gấp nhiều lần mọi tinh chỉnh
   hậu xử lý cộng lại. Cả A51 + A52 + A79 gộp lại được ~0,09; đổi model được
   +0,3096.
2. **Trần của bản cũ (0,6531) còn thấp hơn ĐIỂM THẬT của bản mới (0,5317) chưa
   nhiều** — nghĩa là mọi công sức xếp lại trên nền SigLIP2-1152 có trần thấp
   hơn hẳn thứ ta đang có sẵn mà không cần xếp lại gì.
3. **Trần ±15s là 0,9592.** Gần như mọi câu đều có đáp án trong bể khi cửa sổ
   rộng. Vấn đề của hệ thống này chưa bao giờ là *tìm không ra*, mà là *xếp
   không lên*.

### A88. VietOCR cả kho: nâng **TRẦN** Q&A 50%, nhưng không phép đào nào với tới — và IDF làm mọi thứ tệ hơn

12/12 phần VietOCR về đủ: **177.321/177.321 keyframe**, 166.605 khung có chữ.
Tỷ lệ có dấu toàn kho **7% -> 45%**, riêng khung đáp án **20% -> 82%** — tái
lập chính xác bản thử 251 khung của A76.

Một khác biệt hệ thống giữa hai nhóm phần, đã truy ra nguyên nhân:

| | rỗng | có dấu | s/ảnh |
| --- | ---: | ---: | ---: |
| A1–A7 | 9–14% | 52–59% | 0,77–1,08 |
| B1–B5 (L26) | **0%** | **31–33%** | **0,44–0,46** |

Không phải chạy sai cấu hình: **43% khung L26 chỉ có watermark `HTV Online`**
(15.685 dòng, cộng 3.971 dòng `HIV Online` do VietOCR đọc nhầm `T` thành `I`).
Watermark luôn có nên không khung nào rỗng; watermark không dấu nên tỷ lệ có
dấu thấp; ít vùng chữ nên nhanh gấp đôi. **Ba dấu hiệu lệch cùng lúc, một
nguyên nhân.**

#### 1. Kênh 3 trên văn bản GỘP: ❌ đảo dấu ở mọi α (`102_`, 72 câu)

| cấu hình | ±2s | ±15s | |
| --- | ---: | ---: | :---: |
| **MỐC: OCR cũ (α=0,5)** | **0,5611** | **0,6514** | |
| GỘP + VietOCR, α=0,5 | 0,5688 | 0,6486 | ❌ ĐẢO DẤU |
| GỘP, α=0,6 | 0,5660 | 0,6486 | ❌ |
| GỘP, α=0,7 | 0,5660 | 0,6486 | ❌ |
| GỘP, α=0,8 | 0,5632 | 0,6486 | ❌ |

Lý do nằm ngay ở số khung: gộp chỉ thêm **192 khung** có chữ (176.009 ->
176.201). Phần còn lại là **từ trùng** đổ vào khung vốn đã có chữ — TF tăng
(bão hoà theo `k1`) nhưng `dl` cũng tăng, mà BM25 **phạt độ dài** qua `b`. Hai
hiệu ứng ngược chiều, và độ dài trung vị 489 -> 510 ký tự đủ để `b` cắn.

Ngưỡng ghi trước ở `kaggle_vietocr.md` đã báo đúng: *"`bm25.py` đã có nhánh
không dấu nên lợi ích ở đó nhỏ"*. Truy hồi không phải chỗ VietOCR giúp.

`alpha` (tỷ trọng nhánh có dấu) cũng trơ: nâng α chỉ làm ±2s tụt dần, ±15s
không đổi. Nhánh không dấu tồn tại để cứu truy vấn gõ thiếu dấu **và OCR đọc
sai dấu** — mà VietOCR vẫn đọc `HTV` thành `HIV` ở 8,3% khung L26.

#### 2. Q&A: TRẦN tăng 50%, mà mọi phép đào đều XA HƠN (`103_`)

Trần = có cụm 1–4 từ nào trong văn bản **bằng đúng đáp án** không:

| văn bản | trần |
| --- | ---: |
| CŨ (ocr + asr) | 4/13 |
| **GỘP (ocr + VietOCR + asr)** | **6/13** |

VietOCR đọc ra `'Thịt cá lóc 300g'` **đúng dấu** — đáp án có mặt thật. Nhưng:

| phép đào | khớp đúng chuỗi |
| --- | ---: |
| CŨ · regex chữ hoa | **3/13** |
| GỘP · regex chữ hoa | 2/13 |
| GỘP · cụm + IDF | **0/13** |
| *TRẦN (gộp)* | *6/13* |

**Cả ba đều đi xa trần hơn, không gần hơn.** Gộp văn bản còn làm regex tụt
3 -> 2 (mất `46`): thêm chữ làm đổi cụm gần từ khoá nhất.

#### Vì sao IDF hỏng — và đây là bài học chung, không riêng câu này

Ý tưởng: bỏ điều kiện chữ hoa, sinh mọi cụm 1–4 từ rồi xếp theo `max(IDF)`, vì
đáp án là **thực thể hiếm**. Đúng với nhãn vật thể (A62). Sai ở đây, và số liệu
nói thẳng — năm cụm điểm cao nhất ở khung chứa `Cá lóc`:

    ['Gao deo 100g Bapnep', 'deo 100g Bapnep', 'deo 100g Bapnep 2', …]

    IDF('ca')  = 1,18      <- từ THẬT nên phổ biến
    IDF('loc') = 3,87
    IDF('Bapnep') = cực đại — nó là LỖI OCR dính chữ ("Bắp nếp"), xuất hiện
                    ĐÚNG MỘT LẦN trong cả kho

> **OCR sinh ra rác DUY NHẤT.** Mỗi lần đọc sai một ký tự là một token hapax,
> tức IDF cực đại. Xếp theo IDF trên văn bản OCR là xếp **rác lên đầu**. IDF đo
> "hiếm thì đáng chú ý" — đúng khi từ vựng đóng và sạch, sai khi hiếm nghĩa là
> **SAI**.

`dap_an.dao_cum()` giữ lại kèm kết quả 0/13 trong docstring, để lần sau ai nghĩ
ra ý này thì thấy nó đã được thử.

#### Kết luận cho cả đợt

**Không có cải thiện nào bật được.** Điểm cao nhất dùng được vẫn là cấu hình
hiện tại: **0,5611 ở ±2s / 0,6514 ở ±15s** trên 72 câu (0,5317 / 0,6067 trên
52 câu của tập so A87).

Nhưng đợt này không vô ích: nó **dịch chuyển trần Q&A từ 4/13 lên 6/13** và
chứng minh phần chặn nằm hoàn toàn ở **khâu đào**, không ở dữ liệu. Ba phép đào
đã thử (regex chữ hoa, regex nới, cụm + IDF) đều là **phép chọn theo hình thức
bề mặt**, và cả ba đều thua vì bài toán thật là **đọc hiểu**: *cho câu hỏi và
văn bản của khung, cụm nào là đáp án*. Đường còn lại là LLM đọc
`câu hỏi + văn bản khung` — chưa đo.

Trong lúc chờ, `--rai-bien-the 2` (A83) vẫn là thứ rẻ nhất còn dùng được.

### A89. Bảng điểm THẬT của BTC kiểm chính **thước đo** — và bắt được một vòng lặp khép kín

Lần đầu repo có **nhãn vàng thật**: bảng điểm từng câu của bài nộp Sơ tuyển 1
(**14,5/25**). Mọi con số 88 mục trước đo trên đáp án **tự soi**; nhãn của BTC
là thứ duy nhất kiểm được cách soi đó có đúng không.

#### Đối chiếu (`104_doi_chieu_diem_that.py`, 20/25 câu có nhãn, bể 1.000)

| gói | BTC | hạng đáp án ta soi |
| --- | ---: | ---: |
| p1-12 | **0** | **1** |
| p1-13 | **0** | **2** |
| p1-18 | **0** | **2** |
| p1-11 | **0** | 6 |
| p1-23 | **0** | 6 |
| p1-16 | 0 | 131 |
| **p1-6** | **1** | **152** |

**Không một câu nào có đáp án ngoài bể**; 18/20 nằm trong top-20. Nhưng thứ
hạng của ta **không phân biệt được** câu BTC cho 1 với câu BTC cho 0 — thậm chí
ngược dấu ở hai đầu bảng.

#### Nguyên nhân: nhãn được hái từ chính đầu ra của hệ thống

`66_soat_de_thi_thu.py` tìm đáp án bằng cách cho người soi **top ứng viên của
chính hệ thống** rồi bấm chọn khung trông đúng. Docstring của nó đã tự cảnh báo
*"ĐÁP ÁN TÌM THẤY LÀ NIỀM TIN, KHÔNG PHẢI SỰ THẬT"*, và **cả 20 câu** dừng ở
nhãn `do_chac: kha — CHUA doi chieu anh goc`.

Khi đáp án thật không nằm trong top-20, người soi chọn một khung *trông hợp lý*
nhưng sai. Nên hạng 1 của p1-12 **không đo gì cả**: hệ thống được chấm bằng
chính lựa chọn của nó.

    20/72 câu (28% tập đo) mang nhãn hái từ hệ thống
    6/20 trong số đó bị BTC bác thẳng

#### Thiên vị CÓ CHIỀU, và đo được nó đổi kết luận

Nhãn hái từ top-20 của cấu hình **hiện tại** thì bênh đúng cấu hình đó, nên mọi
cấu hình MỚI bị đẩy xuống. Chạy lại A88 (kênh 3 trên văn bản gộp VietOCR) chỉ
trên 52 câu `tap_de_that` (nhãn sạch):

| tập đo | ±2s | ±15s | T-B-H | kết luận |
| --- | ---: | ---: | :---: | :---: |
| 72 câu (có nhiễm) | +0,0076 | −0,0028 | 4-2-66 | ❌ ĐẢO DẤU |
| **52 câu sạch** | **+0,0144** | **+0,0000** | **4-1-47** | **🟡 YẾU** |

**Hiệu ở ±2s tăng gấp đôi, ±15s hết âm, kết luận đi từ "không dùng được" sang
"ứng viên sống".** Vẫn 🟡 (+0,0144 so với ngưỡng 0,0151, thiếu 5%) nên chưa
bật — nhưng nó chứng minh nhiễm nhãn **không phải rủi ro lý thuyết**.

#### Ba điều phải làm, và một điều đã làm

* **Đã làm:** `105_danh_dau_nhan_sai.py` hạ 6 nhãn bị BTC bác xuống
  `do_chac: sai`. **Không xoá câu** — soi lại từ ảnh gốc thì dùng lại được.
* 14 câu còn lại vẫn ở `do_chac: kha`: **chưa có bằng chứng phản bác không
  phải là bằng chứng đúng**. Chúng vẫn hái từ cùng một quy trình.
* Mọi kết luận 🟡/❌ đo trên 68–72 câu (A64, A71, A72, A82, A88) **cần chạy lại
  trên 52 câu sạch** trước khi tin.
* **Đổi quy trình soi**: đáp án phải soi từ **ẢNH GỐC theo mô tả**, không phải
  chọn từ danh sách hệ thống trả về. Bắt buộc dùng danh sách thì phải lấy từ
  **cấu hình KHÁC** với cấu hình sắp đo.

#### Điều KHÔNG bị ảnh hưởng

**A87 dùng đúng 52 câu `tap_de_that`** (tập khoá theo độ phủ hai cache), tức
nhãn sạch. Kết luận lớn nhất của repo — gopt gấp **2,4 lần** SigLIP2-1152
(+0,3096, ✅ ổn định, 31 thắng / 5 thua) và trần **0,8776 / 0,9592** — đứng
nguyên.

> **Bài học lớn nhất của cả dự án, và nó không nói về mô hình.** Repo này dựng
> cả một bộ máy để chống tự lừa mình: hai mức dung sai, so theo cặp, ngưỡng
> nhiễu, mốc nền mạnh nhất, chỉ đổi một thứ, nhóm đối chứng. Bộ máy đó kiểm
> **kết luận**, nhưng **không kiểm được NHÃN** — và một nhãn hái từ hệ thống
> làm mọi tầng phía trên trở thành trang trí. 88 mục đo không phát hiện ra;
> một bảng điểm 25 dòng của BTC phát hiện ra trong mười phút.

### A90. Caption phủ **100% kho**: A73 không lật — và dãy ba điểm đo là bằng chứng sạch nhất về bẫy bể nhỏ

12/12 phần caption về đủ. Soát bằng `76_kiem_caption_phan.py`: mọi phần đúng
phần được giao, **0 caption rỗng**, số dòng lệch **0,2%**, độ dài trung vị
280–289 ký tự (lệch 3% -> cùng cấu hình sinh). Gộp: **177.321 ảnh / 873 video
= 100,0% kho**.

Đo lại trên **52 câu `tap_de_that`** — nhãn sạch, sau khi A89 phát hiện 20 câu
`de_thi_thu` mang nhãn hái từ chính hệ thống.

#### Vì sao phải đo lại chứ không tin A73

A73 kết luận ❌ đảo dấu, nhưng phép đo đó có **hai** điều kiện nay đã đổi: độ
phủ 76% (chưa đầy đủ) và tập 68 câu (có nhiễm nhãn). A89 vừa chứng minh nhiễm
nhãn đè được một kết quả dương xuống thành ❌. Nên A73 đúng là loại kết luận
phải chạy lại.

#### Kết quả: không lật, và mạnh hơn

| cấu hình | ±2s | ±15s | hiệu ±2s | T-B-H | |
| --- | ---: | ---: | ---: | :---: | :---: |
| **mốc: ảnh + kênh 3** | **0,5173** | **0,6096** | — | — | |
| + kênh 5 (0,25) | 0,5048 | 0,6019 | −0,0125 | 2-6-44 | 🟡 |
| + kênh 5 (0,5) | 0,5125 | 0,6173 | −0,0048 | 5-8-39 | ❌ ĐẢO DẤU |
| + kênh 5 (1,0) | 0,4577 | 0,5635 | **−0,0596** | 7-24-21 | **✅ TỆ HƠN** |
| kênh 5 THAY kênh 3 | 0,4846 | 0,5779 | −0,0327 | 5-10-37 | 🟡 |

Ở độ phủ đầy đủ, `w = 1,0` đi từ 🟡 sang **✅ ổn định TỆ HƠN** — bằng chứng
chống caption **mạnh lên**, không yếu đi.

#### Dãy ba điểm đo: bằng chứng sạch nhất về bẫy bể nhỏ trong cả repo

| độ phủ caption | kênh 5 đứng một mình (±2s) |
| ---: | ---: |
| 5,9% kho (A59) | **0,3904** |
| 76,0% kho (A73) | **0,2625** |
| **100% kho (A90)** | **0,1615** |

**Càng phủ đủ càng thấp, đơn điệu, ba trên ba.** Đây là cơ chế A21 ở dạng
thuần khiết: bể càng nhỏ so với kho thật, con số càng nói về **BỂ** chứ không
về **KÊNH**. Ở 5,9% thì bể gần như chỉ gồm video *có chứa đáp án*, nên kênh 5
chỉ phải chọn khung trong một tập đã được lọc sẵn hộ.

> Nếu chỉ có điểm đo đầu tiên (0,3904 ở độ phủ 5,9%), kênh caption trông như
> kênh mạnh thứ hai của hệ thống. Ba điểm đo cho thấy nó là **hiện vật của
> phép đo**. Đây là lý do A73 ghi *"khoá bể làm phép so CÔNG BẰNG chứ không
> làm nó ĐẠI DIỆN"* — và giờ có ba điểm để chứng minh câu đó.

#### Kết luận: đóng kênh 5, giữ dữ liệu

**103 giờ GPU, và kênh không vào được bài nộp.** Nói thẳng vậy thì đúng hơn là
tìm cách bào chữa. Nhưng dữ liệu giữ lại: nó không tốn thêm gì, A71 đo được
caption **độc lập thật** với kênh ảnh (Spearman 0,043), nên nếu về sau có cơ
chế hợp nhất khai thác được kênh ít chồng lấn thì nó đã sẵn sàng. A72 vừa bác
ứng viên duy nhất cho việc đó, nên đừng chờ.

`71_do_kenh5_caption.py` bỏ dòng cảnh báo "bể bị khoá" khi độ phủ đạt 100% —
một cảnh báo luôn hiện là một cảnh báo không ai còn đọc.

### A91. Quét lại TOÀN BỘ kết luận cũ trên nhãn sạch — và hai ứng viên 🟡 thì giẫm lên nhau chứ không cộng

A89 để lại một việc: mọi kết luận 🟡/❌ đo trên 68–72 câu đều có nhãn nhiễm,
phải chạy lại trên 52 câu `tap_de_that`. Đã quét xong.

| mục | trên tập NHIỄM | trên 52 câu SẠCH | đổi |
| --- | :---: | :---: | --- |
| A88 kênh 3 văn bản gộp | ❌ đảo dấu | **🟡** +0,0144 | lật, hiệu gấp đôi |
| A73 kênh 5 caption | ❌ đảo dấu | **❌/✅ TỆ HƠN** | không lật, mạnh hơn (A90) |
| A72 hợp nhất bằng điểm | ✅ tệ hơn | **✅ tệ hơn, cả 8/8 biến thể** | không lật, dứt khoát hơn |
| A82 khuếch tán τ=2s | ❌ đảo dấu | **🟡** +0,0038 | lật, nhưng bé xíu |

**Nhiễm nhãn không lật mọi thứ — nó lật đúng những thứ ở sát ngưỡng.** Ba mục
kết luận mạnh (A72, A73, A90) đứng nguyên hoặc mạnh lên; hai mục sát ngưỡng
(A88, A82) đều lật sang dương. Đúng như cơ chế A89 mô tả: nhãn hái từ cấu hình
cũ tạo một lực đẩy CÓ CHIỀU chống cấu hình mới, và lực đó chỉ đủ đổi kết luận ở
vùng biên.

#### Hai ứng viên 🟡 cùng nằm trên kênh 3 — cộng lại thì sao? (`106_`, lưới 2×2)

A88 đổi **văn bản đầu vào** của kênh 3; A82 đổi **cách điểm lan ra khung lân
cận**. Hai khâu khác nhau, không cái nào bao cái nào, nên hiệu *có thể* cộng —
và cộng lại thì `+0,0144 + 0,0038 = +0,0182` sẽ **vượt ngưỡng 0,0151**.

| cấu hình | ±2s | ±15s | hiệu ±2s | T-B-H | |
| --- | ---: | ---: | ---: | :---: | :---: |
| **1. MỐC: ocr cũ, không khuếch tán** | **0,5173** | **0,6096** | — | — | |
| 2. + văn bản GỘP (A88) | **0,5317** | 0,6096 | +0,0144 | 4-1-47 | 🟡 |
| 3. + khuếch tán τ=2s (A82) | 0,5212 | **0,6135** | +0,0038 | 2-1-49 | 🟡 |
| **4. CẢ HAI** | 0,5288 | 0,6058 | +0,0115 | 3-1-48 | **❌ ĐẢO DẤU** |

**Không cộng — trừ.** Ô "cả hai" thấp hơn ô "chỉ A88" ở **cả hai** mức dung
sai, và ±15s đi xuống dưới mốc nền. Đây là lý do lưới 2×2 phải có đủ bốn ô:
nếu chỉ đo "cả hai" so với mốc thì thấy +0,0115 và tưởng hai cải tiến đang
cộng vào nhau, trong khi thật ra chúng đang huỷ nhau.

Cơ chế hợp lý nhất (chưa đo riêng, nên ghi là giả thuyết): khuếch tán làm mềm
trường điểm để một đỉnh nhọn lan sang khung lân cận. Văn bản gộp làm kênh 3
**bắn ra nhiều khung hơn và nhiều token hơn mỗi khung** — nên khuếch tán không
còn một đỉnh để trải, nó trộn nhiều nguồn gần nhau thành một mảng phẳng. Đó là
A70 (gộp vector theo đoạn ASR, đã bị bác) ở quy mô nhỏ hơn.

> **Dự đoán ghi TRƯỚC khi chạy** (nằm trong docstring `106_`): *"tôi cho rằng
> hiệu sẽ KHÔNG cộng đủ để qua ngưỡng"*. Đúng chiều. Ghi dự đoán trước là cách
> rẻ nhất để không tự chấm điểm mình sau khi đã biết kết quả — và lần này nó
> cũng chặn được cám dỗ đọc `+0,0115` của dòng 4 thành "gần thắng rồi".

#### Kết luận đợt quét

**Vẫn không có gì bật được.** Cấu hình mạnh nhất không đổi: ảnh (gopt) + kênh 3
OCR cũ, w=0,5, RRF hạng k=60, K-best TRAKE — **0,5173 / 0,6096** trên 52 câu
nhãn sạch. A88 giữ nguyên trạng thái 🟡 (thiếu 5% để qua ngưỡng) và giờ đã biết
thêm: **đừng chờ A82 đẩy nó qua**.

### A92. Điểm đang mất nằm Ở ĐÂU — phân rã đầu tiên sau 91 mục đo

91 mục trước đều trả lời *"cấu hình A có hơn cấu hình B không"*. Không mục nào
trả lời **"trong 0,4769 điểm đang mất, phần nào nằm ở đâu"** — mà đó mới là thứ
quyết định nên đầu tư vào chỗ nào. `107_phan_ra_diem.py`, 52 câu nhãn sạch, ±2s:

| ô | câu | điểm | **mất** | R@1 | R@20 | R@100 | trượt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KIS | 37 | 0,5838 | **0,2962** | 0,24 | 0,68 | 0,81 | 7 |
| QA | 12 | 0,3500 | **0,1500** | 0,08 | 0,42 | 0,58 | 5 |
| TRAKE | 3 | 0,4667 | 0,0308 | 0,00 | 0,67 | 0,67 | 1 |

Cột `mất` đã nhân với số câu trong ô, nên nó xếp đúng thứ tự nên đầu tư — ô có
tỷ lệ tệ nhất chưa chắc là ô đáng chữa nhất.

#### Ba điều bảng này nói mà không mục nào trước nói

**1. Chỗ mất là R@1, không phải R@100.** KIS có đáp án trong top-100 ở **81%**
số câu nhưng chỉ ở hạng 1 ở **24%**. Tức phần lớn điểm mất không phải vì không
tìm ra, mà vì **không xếp đúng thứ tự**. Thêm kênh truy hồi thứ sáu không chữa
được điều đó; chỉ **xếp lại hạng trong top-100** mới chữa được. Đây là lý do
mọi kênh mới đo từ A59 tới A90 đều thất bại — chúng giải sai bài toán.

**2. Q&A hỏng ở khâu TÌM KHUNG, không chỉ ở khâu đọc đáp án.** Con số 0,3500
trên **chưa xét `answer` đúng hay sai** — nó thuần truy hồi, và vẫn tệ hơn KIS ở
mọi mốc. Điểm nộp thật của Q&A còn thấp hơn nữa. A83/A88 dồn hết sức vào khâu
đào đáp án là đúng nhưng mới một nửa bài toán.

**3. Độ dài truy vấn KHÔNG phải thủ phạm — và suýt nữa thì tin là có.**
`94_soi_cau_that_bai.py` báo *"6/49 câu có đáp án ngoài top-1000, cả sáu đều dài
>40 từ"*, nghe như một phát hiện. Tỷ lệ nền bác nó ngay: **42/52 câu vốn đã dài
>40 từ**. Sáu trên sáu là điều gần như chắc chắn xảy ra kể cả khi độ dài chẳng
liên quan gì.

> Bài học lặp lại A21 ở dạng khác: **một tỷ lệ không có mẫu số thì không phải
> số liệu.** `107_` từ nay in tỷ lệ nền cạnh mọi ô.

### A93. Mệnh đề HỎI của câu Q&A làm nhiễu kênh 1 — dương ở mọi lát cắt, nhưng 🟡 ở mọi lát cắt

A92 chỉ ra Q&A hỏng ở khâu tìm khung. Nhìn vào chính truy vấn thì thấy ngay
nguyên nhân khả dĩ: `tach_truy_van` cắt câu Q&A ra, và mệnh đề cuối bao giờ
cũng là câu HỎI:

    | Đoạn video mô tả quá trình làm bánh, bánh có màu tím...   <- tả cảnh
    | Mỗi lần khuôn này làm được bao nhiêu cái bánh?            <- HỎI

Mệnh đề thứ hai nói về **thứ cần trả lời**, không nói **cảnh trông thế nào**.
Đem đi tìm ảnh thì nó kéo về bất cứ gì — mà RRF hạng (A51) cho nó **tiếng nói
ngang** mệnh đề tốt.

#### Bộ nhận diện tách sạch, nên nhóm đối chứng có sẵn

| loại | câu có mệnh đề hỏi tách riêng |
| --- | ---: |
| Q&A | **11/12** |
| KIS | **0/37** |
| TRAKE | **0/3** |

0/37 câu KIS bị đụng tới — không phải dựng thêm nhóm đối chứng, nó có sẵn.

#### Kết quả (`108_do_menh_de_hoi.py`) — bốn lát cắt, bảy phép so, không lát nào âm

| lát cắt | n bị ảnh hưởng | hiệu ±2s | hiệu ±15s | T-B-H | ngưỡng |
| --- | ---: | ---: | ---: | :---: | ---: |
| 52 câu đề thật | 11 | +0,0154 | +0,0154 | 4-1-47 | 0,0287 |
| 74 câu Q&A tập dev | 28 | +0,0189 | +0,0189 | 9-3-62 | 0,0245 |
| **chỉ 11 câu bị ảnh hưởng** | 11 | **+0,0727** | **+0,0727** | 4-1-6 | 0,1351 |

Trên riêng 11 câu đó: **0,3455 -> 0,4182 ở ±2s, 0,4909 -> 0,5636 ở ±15s.**

`w = 0,25` và `bỏ hẳn` cho kết quả y hệt trên đề thật; trên tập dev thì `0,25`
nhỉnh hơn (0,5243 so với 0,5216). Nếu bật thì bật `0,25`, không bật `0`.

#### Vì sao vẫn KHÔNG bật, dù dương ở cả bảy phép so

Thu hẹp về 11 câu bị ảnh hưởng là **phép thử đúng** — câu không bị bắn thì hoà
tuyệt đối, không mang thông tin về hiệu nhưng vẫn phình mẫu số. Nhưng nó không
cứu được kết luận: **hiệu tăng 4,7 lần thì ngưỡng cũng nở 4,7 lần**. 11 câu thì
không đủ để nói gì, dù đúng chiều.

⚠️ **Và `tap_de_that` là TẬP CON của `tap_dev`** (52/52 câu trùng) — nên hai
dòng đầu bảng **không phải hai phép nhân bản độc lập**, mà là một tập và tập lớn
hơn chứa nó. Suýt viết nhầm thành "nhân bản độc lập"; kiểm giao mới thấy.

**Đã cài `--trong-so-hoi`, mặc định `1,0` = không đổi gì.** Có test chốt rằng
mặc định phải là 1,0, kèm lý do — để lần sau ai muốn bật thì phải sửa test, tức
phải đối diện với việc nó mới 🟡.

**Thứ sẽ giải quyết: thêm câu Q&A đề thật có nhãn sạch.** 8 gói `de_thi_thu`
chưa gán nhãn là nguồn gần nhất (A89 việc 4). Đây là lần thứ hai trong ba mục
liên tiếp mà **thiếu nhãn, chứ không thiếu ý tưởng**, là thứ chặn kết luận.

#### Hạn chế đã biết của bộ nhận diện

Khớp danh sách từ khoá **có dấu** (cộng dấu `?`). Câu vừa không dấu vừa không có
`?` thì lọt. Cố ý giữ vậy: đổi sang khớp bản bỏ dấu sẽ đổi tập câu bị ảnh hưởng
và làm A93 không tái lập được. Có test ghi rõ hạn chế này.

### A94. TRAKE thiếu **VIDEO ứng viên**, không thiếu cách xếp hạng — hiệu lớn nhất kể từ A79

Xuất phát từ một đề xuất bên ngoài: *xếp hạng video TRAKE bằng điểm chuỗi hợp
lệ tốt nhất thay vì tổng-log-max*. Đã đo (`109_`), và phép đo đó **không tìm ra
thứ nó nhắm tới, nhưng chẩn đoán kèm theo lại tìm ra thứ lớn hơn nhiều**.

#### Chẩn đoán hạn ngạch dòng — con số làm đổi hướng cả đợt

| | trung vị | min | max |
| --- | ---: | ---: | ---: |
| video có đủ ứng viên cho MỌI sự kiện | **11** | 4 | 40 |
| trong 25 video được chia dòng, số video KHÔNG có chuỗi hợp lệ | 6 | 0 | 25 |
| **dòng thực sự nộp được** | **46/100** | 0 | 81 |

Hạn ngạch `40/25/15/12/8 + 20 dòng đuôi` được thiết kế cho **25 video**. Thực tế
trung vị chỉ có **11**. **Hạn ngạch và bể chưa bao giờ khớp nhau** — suốt thời
gian qua nó chia 100 dòng cho một danh sách 11 mục.

#### Vì sao bể nhỏ

Mỗi sự kiện lấy `--k = 100` ứng viên; một video chỉ vào danh sách khi có ứng
viên cho **TẤT CẢ** N sự kiện. Giao của N tập nhỏ đi theo cấp số nhân, mà phân
bố số sự kiện là 3-5 (trung vị **4**).

> `--k` có **hai vai trò khác hẳn nhau** ở hai loại câu. Với KIS/Q&A nó là "số
> dòng nộp", nới ra vô ích vì chỉ nộp được 100. Với TRAKE nó là **bể để GIAO**,
> và cái nộp đi là chuỗi ghép từ giao đó. Cùng một tham số, và sự nhập nhằng ấy
> đã ẩn nút thắt này suốt 93 mục đo.

#### Nới bể (`110_`, 18 câu TRAKE, chấm ở tầng NỘP)

| `--be` | video đủ ứng viên | dòng nộp được | ±2s | ±15s | T-B-H | |
| ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| **100** (đang chạy) | 11 | 46/100 | 0,3578 | 0,5950 | — | |
| 150 | 15 | 56/100 | 0,3561 | 0,5867 | 2-3-13 | 🟡 âm |
| 200 | 18 | 62/100 | 0,3617 | 0,5894 | 2-2-14 | ❌ |
| **300** | **25** | 67/100 | **0,4317** | **0,6483** | **7-3-8** | **🟡** |
| 500 | 33 | 77/100 | 0,4250 | 0,6400 | 8-3-7 | 🟡 |
| 1000 | 59 | 92/100 | 0,4011 | 0,5922 | 6-3-9 | ❌ |

**+0,0739 ở ±2s / +0,0533 ở ±15s** — hiệu lớn nhất kể từ A79, gấp gần 5 lần
phát hiện mệnh đề hỏi (A93). Vẫn 🟡: ngưỡng 0,0876, đạt 84%.

Đường cong **không đơn điệu**: phẳng ở 150-200, nhảy ở 300, tụt lại ở 500, đảo
dấu ở 1000. Có đỉnh thật, không phải "càng nhiều càng tốt" — bể lớn thả video
nhiễu vào tranh hạn ngạch.

Ở bể 300, số video vừa đúng **25** — đúng con số hạn ngạch được thiết kế cho.

### A95. Hiệu ứng dồn vào hai câu, và một dự đoán của A79 được xác nhận tới từng số

Trung bình +0,0739 che mất phân bố. Từng câu, ±2s:

| câu | bể 100 | bể 300 | hiệu |
| --- | ---: | ---: | ---: |
| trake-L21-002 | 0,9500 | 0,7500 | **−0,2000** |
| trake-L22-004 | 0,4500 | 0,3000 | −0,1500 |
| trake-L21-004 | 0,6800 | 0,6000 | −0,0800 |
| trake-L25-003 | 0,1600 | 0,2400 | +0,0800 |
| trake-L22-003 · L22-002 · **DE2-08** | | | +0,1500 mỗi câu |
| trake-L23-008 | 0,3000 | 0,5500 | +0,2500 |
| **trake-L25-004** | **0,0000** | **0,4800** | **+0,4800** |
| **trake-DE2-21** | **0,0000** | **0,5000** | **+0,5000** |

**Hai câu chiếm 74% tổng hiệu**, và ba câu bị hại thật. Trung bình là thật
nhưng mong manh.

Nhưng một trong hai câu đó là bằng chứng chứ không phải cảnh báo. A79 đã chỉ
đích danh: *"`trake-L25-004` rơi 0,4800 -> 0,0000 ở CẢ HAI mức dung sai dù
oracle của nó là 0,8000"*. Nới bể đưa nó về **đúng 0,4800**. Một dự đoán ghi
trước từ 15 mục trước, được xác nhận tới từng chữ số — và nó xác nhận đúng cơ
chế: câu đó mất điểm ở khâu **video không có mặt trong danh sách**.

#### Lưới 2×2 tách đầu/đuôi — và nó BÁC giả thuyết (`111_`)

Giả thuyết: được là nhờ **đuôi** (video đúng lọt vào hạng 6-25), mất là do
**đầu** (video nhiễu tranh hạn ngạch top-5). Nếu đúng thì nới riêng đuôi lấy
được phần thắng mà không trả phần thua.

| | ±2s | ±15s | T-B-H |
| --- | ---: | ---: | :---: |
| 1. đầu 100 / đuôi 100 | 0,3578 | 0,5950 | — |
| 2. đầu 300 / đuôi 300 | **0,4317** | **0,6483** | 7-3-8 |
| 3. chỉ nới ĐUÔI | 0,3844 | 0,6217 | **1-0-17** |
| 4. chỉ nới ĐẦU | 0,4050 | 0,6217 | 6-3-9 |

**Sai, và ngược hẳn.** Phần thắng nằm ở ĐẦU (6 câu đổi) chứ không ở đuôi (1
câu). Và hai phần **cộng đúng khít**: `0,0267 + 0,0472 = 0,0739`, khớp tới bốn
chữ số.

> Đây là lần ĐẦU trong repo hai thay đổi cộng dồn chính xác. A91 đo hai cải
> tiến 🟡 trên kênh 3 và chúng **huỷ nhau**; ở đây chúng cộng. Không suy được
> cái nào sẽ xảy ra — phải đo, và đó chính là lý do lưới 2×2 phải có đủ bốn ô.

Cơ chế thật: bể lớn cho mỗi sự kiện nhiều ứng viên hơn -> `max_e` chính xác
hơn -> **thứ tự tổng-log-max tốt hơn**, và cái đó ăn vào top-5. Không có mẹo
tách nào; muốn phần thắng thì nới cả hai.

### A96. Ba đính chính cho bản rà soát ngoài — và một con số bị gán sai việc

Bản rà soát đề xuất sáu thí nghiệm, ưu tiên 1 là *"xếp hạng video bằng best
valid beam score, pool ≥100, giữ beam=64"*. Ba khẳng định trong đó kiểm được:

**1. "Nên dùng 40 chuỗi beam khác nhau thay vì lặp 1 chuỗi 40 lần" — đã làm
sẵn.** `chuoi(v, k)` gọi `beam_video(..., k_chuoi=k)`, trả về tối đa `k` chuỗi
**khác nhau**, có bộ lọc đa dạng `cach_nhau = 3,0s` ở cuối. Không có chỗ nào
spam một đáp án.

**2. "37% ở ±2s" bị gán sai việc.** Đó là A63 — khoảng cách giữa chấm ở tầng
KÊNH và tầng NỘP, một hiện vật của cách đo, và A79 đã cho thấy nó **sụp từ 48%
xuống 4,3% ở ±15s**. Con số đúng cho khâu chọn video nằm ở A79 và nó **mạnh
hơn** vì không sụp: khoảng cách tới oracle **0,1029 ở ±2s / 0,1230 ở ±15s**.

**3. "Top-30 để rerank là quá hẹp, cần pool ≥100" — ngược hoàn toàn.** Bể sơ
tuyển chưa bao giờ là ràng buộc: trung vị chỉ có **11 video tồn tại**. Đo thẳng:
bể 150 và bể TOÀN BỘ cho kết quả **y hệt nhau**.

#### Kết quả của chính ưu tiên 1 (`109_`)

| | ±2s | ±15s | T-B-H | |
| --- | ---: | ---: | :---: | :---: |
| MỐC: tổng-log-max | 0,3578 | 0,5950 | — | |
| chỉ LỌC video không có chuỗi hợp lệ | 0,3578 | 0,5950 | **0-0-18** | ⚪ |
| XẾP LẠI theo điểm chuỗi (bể 150) | 0,3772 | 0,6228 | 1-0-17 | 🟡 |
| XẾP LẠI theo điểm chuỗi (bể TOÀN BỘ) | 0,3772 | 0,6228 | 1-0-17 | 🟡 |

Đúng chiều ở cả hai mức nhưng **một câu đổi**. Bản rẻ nhất — chỉ lọc video
không có chuỗi hợp lệ, giữ nguyên thứ tự — **không đổi một câu nào**.

Điểm chuỗi tốt nhất tính bằng **quy hoạch động chính xác**, không bằng beam:
`D_i[j] = m_i[j] + max{D_{i−1}[k] : t_k < t_j}`. Với N ≤ 5 và 20 ứng viên mỗi
sự kiện thì O(N·C²) = 2.000 phép — rẻ hơn beam và không có sai số xấp xỉ.

#### Vì sao đây KHÔNG phải thứ A78 đã bác

A78 dò bốn cách hợp điểm (tổng / tổng-log / điều hoà / min) và thấy nút này
**trơ**. Nhưng cả bốn đều là hàm hợp của `max_e`, tức đều **mù với ràng buộc
thời gian**. Điểm chuỗi hợp lệ không nằm trong họ đó. Nên phép đo là chính
đáng — nó chỉ không thắng.

#### Trạng thái: cài cờ, KHÔNG bật

`--be-trake`, mặc định `None` = dùng `--k` = không đổi gì. Có ba test chốt:
mặc định phải là None; bể riêng không được lọt sang nhánh KIS/Q&A; và tham số
phải nối qua **cả hai** đường gọi `quet_anh` (quên một đường là cờ im lặng vô
hiệu — đúng loại lỗi commit `8a27e29` đã sửa bốn lần).

⚠️ **Cảnh báo cỡ mẫu, phải đọc kèm mọi con số trên: 15/18 câu TRAKE là TỰ
SOẠN**, chỉ 3 câu là đề thật — đúng chỗ A39 ghi là yếu nhất của tập dev. Tín
hiệu thuận duy nhất: hai câu đề thật có đổi (`DE2-08` +0,15 và `DE2-21` +0,50)
**đều thắng**, và **cả ba câu bị hại đều là câu tự soạn**. n = 3 thì không kết
luận được, nhưng nó không đi ngược.

**Thứ sẽ giải quyết: câu TRAKE ĐỀ THẬT có nhãn sạch.** Đây là lần thứ ba liên
tiếp (A88, A93, A96) mà thứ chặn kết luận là **thiếu nhãn, không thiếu ý tưởng**.

### A97. CSLS (phạt hub) — cơ chế SAI, không phải bị pha loãng

Ý: không gian nhúng nhiều chiều có **hub** — vài điểm nằm gần mọi thứ và được
trả về cho mọi truy vấn. `s(q,d) = 2·cos(q,d) − λ·r_K(d)`.

`112_tinh_hubness.py` tính `r_K` cho cả 177.321 vector (**849 giây**, ghi ra
693 KB; online chỉ là một phép trừ). Hai sai lệch đã chặn trước khi đo:

* **Loại toàn bộ láng giềng CÙNG VIDEO**, không chỉ loại chính nó. A5.6 đo được
  11,83% keyframe có bản sao cùng video ở cos ≥ 0,99 (L25: 49,82%) — không loại
  thì khung đáp án có năm bản sao sẽ bị phạt nặng nhất, tức hỏng ngược.
* **Gọi đúng tên: đây là hub ẢNH–ẢNH, không phải CSLS gốc** (vốn đo hub xuyên
  miền văn bản↔ảnh). Bản xuyên miền cần một tập truy vấn, mà tập duy nhất đang
  có là chính 52 câu dùng để chấm — dựng chỉ mục từ đầu vào kiểm thử là **rò
  rỉ**, không phải kỹ thuật.

`r_K`: trung vị **0,9313**, độ lệch chuẩn **0,0576** — bằng ~38% biên độ cosine
(0,25–0,40), tức KHÔNG phải hằng số. Nên tiền đề "có đủ biến thiên để đổi thứ
hạng" là đúng; điều sai nằm ở chỗ khác.

| λ | ±2s | ±15s | T-B-H | |
| ---: | ---: | ---: | :---: | :---: |
| **0 (mốc)** | **0,5173** | **0,6096** | — | |
| 0,1 | 0,5212 | 0,6135 | 2-2-48 | 🟡 |
| 0,25 | 0,5173 | 0,6135 | 3-4-45 | ❌ |
| 0,5 | 0,5173 | 0,5904 | 4-6-42 | 🟡 |
| 1,0 | 0,5096 | 0,5981 | 5-11-36 | 🟡 âm |

**Hại đơn điệu theo λ.** Và hai dòng chẩn đoán trả lời câu quan trọng hơn: chỉ
kênh 1, không CSLS = **0,4779 / 0,5712**; chỉ kênh 1, λ=1 = **0,4808 / 0,5644**
— tức +0,0029 ở ±2s nhưng −0,0068 ở ±15s, **đảo dấu ngay cả khi không bị RRF và
kênh 3 pha loãng**.

> Đó là lý do phải có dòng chẩn đoán "chỉ kênh 1". Không có nó thì kết luận
> đúng nhất có thể nói là *"có thể tốt nhưng bị pha loãng"* — một câu an ủi
> không kiểm được. Có nó thì biết **cơ chế sai**, và đóng hướng lại được.

Giữ `index/hubness_clip_gopt.npy` (693 KB) vì nó rẻ và là dữ liệu chẩn đoán
thật: `r_K` trung vị 0,93 cho biết keyframe tin tức **giống nhau ở mức rất cao**
trên toàn kho — một sự thật về dữ liệu, độc lập với việc CSLS thất bại.

### A98. α-Query Expansion — bác dứt khoát, và dự đoán của tôi SAI ngược

`q' = chuẩn_hoá(q + Σ cos(q,d_i)^α · v_{d_i})`, áp trên **từng mệnh đề trước
RRF** (áp sau RRF là vô nghĩa: A51 hợp nhất bằng HẠNG, điểm gốc không còn).

| k | α | ±2s | ±15s | T-B-H | |
| ---: | ---: | ---: | ---: | :---: | :---: |
| **mốc** | | **0,5173** | **0,6096** | — | |
| 2 | 1 | 0,4394 | 0,5144 | 7-16-29 | **✅ TỆ HƠN** |
| 2 | 3 | 0,4981 | 0,5904 | 1-2-49 | 🟡 âm |
| 3 | 1 | 0,4125 | 0,4798 | 8-17-27 | **✅ TỆ HƠN** |
| 3 | 3 | 0,4865 | 0,5712 | 1-4-47 | 🟡 âm |
| 5 | 3 | 0,4750 | 0,5635 | 1-5-46 | 🟡 âm |

**−0,1048 / −0,1298 ở cấu hình tệ nhất** — thiệt hại lớn nhất repo từng đo được
từ một tính năng. Cơ chế đúng như ghi trước: α-QE là **phản hồi giả định**, nó
tin top-k đúng, mà A92 đo `R@1 = 0,24` — hạng 1 **sai ở 76% số câu**. Cộng
vector của ứng viên sai vào truy vấn là kéo nó đi xa hơn khỏi đích.

#### Dự đoán ghi trước của tôi sai ngược, và lý do đáng ghi hơn kết quả

Tôi ghi trước: *"α lớn sẽ TỆ hơn α nhỏ, vì α lớn = tin hạng 1 nhiều hơn"*. Số
liệu đi **ngược**: α=3 hại ít hơn α=1 ở mọi k.

Lý do là số học, và nó chỉ đúng ở ĐÂY:

    cosine của kênh 1 nằm khoảng 0,25-0,40
    cos = 0,30 -> α=1 cho trọng số 0,3000
                  α=3 cho trọng số 0,0270      (bằng 9% của α=1)

Với cosine **nhỏ hơn 1**, nâng lên mũ làm **mọi** trọng số co về 0. Nên α lớn
không phải "tin hạng 1 hơn" mà là **mở rộng ÍT hơn**. Trong bài báo gốc
(Radenović 2018) cosine của ảnh khớp gần 1, nên mũ chỉ làm *sắc* tương quan —
ở đây nó làm *tắt* cả cơ chế.

> **Đọc lại được toàn bộ bảng theo một trục duy nhất: mở rộng càng nhiều càng
> hại, đơn điệu.** α=3 ít hại nhất vì nó gần như không mở rộng. Không có mức
> nào có lợi.
>
> Bài học: **một siêu tham số mượn từ bài báo khác mang theo giả định về THANG
> ĐO của bài báo đó.** Ở đây giả định "cosine gần 1" bị vi phạm, và tham số đổi
> luôn ý nghĩa mà không có gì báo.

### A99. n-gram ký tự cho kênh 3 — và bảng Jaccard bác tiền đề TRƯỚC khi chạy

Ý: OCR sai một ký tự sinh ra token hapax (`HTV` -> `HIV`, 3.971 dòng ở L26), mà
BM25 theo TỪ coi đó là từ hoàn toàn khác. n-gram ký tự thì chia sẻ được phần
chung.

**Đo Jaccard giữa hai tập n-gram trước khi viết phép đo** — và nó bác tiền đề:

| ca | n=2 | n=3 | n=4 |
| --- | ---: | ---: | ---: |
| `HTV` vs `HIV` — sai 1 ký tự giữa từ NGẮN | 0,33 | **0,00** | **0,00** |
| `Bapnep` vs `bap nep` — dính chữ, từ DÀI | 0,75 | 0,50 | 0,29 |
| `Tà Pứa` vs `Ta Pua` — mất dấu | 1,00 | 1,00 | 1,00 |

**Ca thúc đẩy cả ý tưởng lại là ca KHÔNG được cứu.** Ký tự sai nằm giữa một từ
ba chữ nên với n ≥ 3 không n-gram nào sống sót. Ca ba thì n-gram thắng tuyệt
đối — nhưng **nhánh bỏ dấu đã trị xong ca đó**. Còn lại đúng một ca có thật.

Chạy `n = 3` (giữ 0,50 ở ca hai, n=4 chỉ còn 0,29):

| | ±2s | ±15s | T-B-H |
| --- | ---: | ---: | :---: |
| **MỐC: từ, hai nhánh α=0,5** | **0,5173** | **0,6096** | — |
| chỉ 3-gram ký tự | 0,5029 | 0,5904 | 3-7-42 |
| từ + 3-gram (RRF) | 0,5096 | 0,5837 | 5-8-39 |
| **chẩn đoán: chỉ nhánh CÓ DẤU** | 0,5096 | 0,6058 | **0-2-50** |

Dòng chẩn đoán là câu trả lời: bỏ hẳn nhánh không dấu chỉ đổi **2/52 câu**
(−0,0077). **Nhánh bỏ dấu gần như không gánh gì** — nên chỗ "chịu lỗi chính tả"
không có gì để lấy thêm, và điều đó giải thích luôn vì sao `alpha` trơ ở A88.

### A100. ĐIỂM CAO NHẤT hệ thống đạt được, và trần còn cách bao xa

`116_do_diem_cao_nhat.py`, 52 câu đề thật nhãn sạch (37 KIS / 12 QA / 3 TRAKE).

| cấu hình | ±2s | ±15s | hiệu ±2s | T-B-H | |
| --- | ---: | ---: | ---: | :---: | :---: |
| **1. ĐANG CHẠY (mặc định)** | **0,5173** | **0,6096** | — | — | |
| 2. + mệnh đề hỏi w=0,25 (A93) | 0,5327 | 0,6250 | +0,0154 | 4-1-47 | 🟡 |
| 3. + văn bản gộp VietOCR (A88) | 0,5317 | 0,6096 | +0,0144 | 4-1-47 | 🟡 |
| **4. CAO NHẤT: cả hai** | **0,5433** | **0,6173** | **+0,0260** | **7-2-43** | **✅** |

**Hai thay đổi 🟡 riêng lẻ, gộp lại thì VƯỢT ngưỡng** (0,0260 so với 0,0248) —
✅ ỔN ĐỊNH theo đúng luật của repo (cùng dấu ở cả hai mức, vượt 2×SE ở ít nhất
một mức), cùng luật đã dùng cho A79 và A87.

Cộng gần khít: `0,0154 + 0,0144 = 0,0298` so với `+0,0260` đo được — hơi dưới
tổng vì cả hai cùng chạm nhóm câu Q&A.

Riêng TRAKE thì `--be-trake 300` (A94) được **+0,0739/+0,0533** trên 18 câu,
nhưng tập 52 câu này chỉ có 3 câu TRAKE nên nó gần như không hiện ở bảng trên.

#### ⚠️ Ba lý do KHÔNG bật ngay, dù ✅

1. **Nhiều phép so.** Riêng đợt này đã chạy ~20 cấu hình. Ngưỡng 2×SE là
   khoảng tin cậy cho MỘT phép so; chạy hai chục phép rồi báo cái vượt là bài
   toán so sánh bội. Mà nó chỉ vượt **5%** (0,0260 so với 0,0248).
2. **Không cái nào tự thắng.** Đây sẽ là lần đầu bật một thứ mà từng thành phần
   đều 🟡. Nếu hiệu ứng thật thì thành phần phải thắng khi có đủ câu.
3. **BTC tính LẦN NỘP CUỐI, không phải lần tốt nhất** (C7). Đổi cấu hình ở lần
   nộp cuối bằng một thứ vượt ngưỡng 5% là đánh cược ở đúng chỗ không nên.

**Nên: bật ở lần nộp 1 hoặc 2 để lấy số thật từ BTC, giữ cấu hình mặc định cho
lần cuối** trừ khi số thật xác nhận.

#### TRẦN — và nó nói bài toán còn lại là gì

| dung sai | trần | đang đạt | **còn thiếu** |
| --- | ---: | ---: | ---: |
| ±2s | 0,7885 | 0,5231 | **0,2654** |
| ±15s | 0,8654 | 0,6115 | **0,2538** |

`trần` = tỷ lệ câu có đáp án nằm **đâu đó** trong 100 dòng nộp. Xếp hạng hoàn
hảo thì mỗi câu đó được 1,0.

**Hơn một phần tư tổng điểm đang nằm trong bể mà xếp sai chỗ.** Cộng với A92
(KIS: R@100 = 0,81 nhưng R@1 = 0,24), kết luận không còn chỗ để tranh cãi:

> **Bài toán còn lại là XẾP LẠI HẠNG, không phải TÌM KIẾM.** Sáu kênh truy hồi
> đã thử từ A59 tới A99 và không cái nào bật được — vì cả sáu đều đi tìm thêm
> ứng viên, trong khi ứng viên đã có sẵn ở 79% số câu.
>
> Và **đó cũng là hướng duy nhất chưa có cách làm được trên máy 7,7 GB không
> GPU.** Đây là câu hỏi mở thật sự của dự án, không phải việc đang chờ ai làm.

#### Đề xuất số 6 (dịch vi→en) — CHẶN bởi phần cứng, không phải bởi lựa chọn

Cần nạp model dịch **và** nạp lại SigLIP2 để mã hoá câu đã dịch. Cả pipeline
hiện chạy được trên máy này chính là nhờ `KenhAnhCache` — vector truy vấn mã
hoá sẵn ở nơi khác. Không mã hoá được câu mới thì không đo được, và ràng buộc
"đừng mở model" là ràng buộc cứng.

Tiền đề của nó cũng chưa có gì đỡ: **kênh 6 BGE-M3 — một mô hình đa ngữ mạnh —
đã thử và thua** (A59, 🟡, đang TẮT). "Thêm sức mạnh đa ngữ" không tự động
thắng trong môi trường đo này.

Muốn làm thì phải: dịch 52 câu trên máy khác -> mã hoá bằng SigLIP2 trên
Kaggle -> đổ vào `truy_van_gopt.npz` -> đo trên máy này. Ba bước, không bước
nào chạy được ở đây.

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

### H0. TRẠNG THÁI HÔM NAY (04/09) — đọc mục này trước, H1–H4 bên dưới là LỊCH SỬ

`CLAUDE.md` bảo đọc PHẦN H trước khi sửa gì. Nhưng H1–H4 được viết ở Giai đoạn 1
khi kênh 1 vừa sống lại; phần lớn việc trong đó đã xong hoặc đã bị chính phép đo
bác bỏ. **Giữ nguyên chúng làm lịch sử, nhưng đừng lấy làm việc phải làm.**

#### Cấu hình đang chạy (và mọi thứ trong đó đều đã thắng trên tập dev)

| thành phần | giá trị | căn cứ |
| --- | --- | --- |
| kênh 1 — ảnh | `clip_gopt.npy`, `ViT-gopt-16-SigLIP2-384` (1536 chiều) | A87: gấp **2,4 lần** SigLIP2-1152 |
| kênh 3 — OCR/ASR | `ocr_asr.parquet`, w = 0,5 | A52 |
| hợp nhất kênh | RRF hạng, k = 60 | A72 (8/8 biến thể bằng điểm đều tệ hơn) |
| hợp nhất mệnh đề | RRF hạng cho kênh 1, MAX cho kênh BM25 | A51, A86 |
| TRAKE | K-best, `cách_nhau` 3,0s, ngân sách 40/25/15/12/8, 20 dòng đuôi | A79 |
| kênh 2 / 4 / 5 / 6 | **TẮT** | A14.2 / A62 / A73+A90 / A59 |

**Điểm trên 52 câu đề thật nhãn sạch: 0,5173 ở ±2s, 0,6096 ở ±15s** (A91).
Trần của bể hiện tại: 0,8776 / 0,9592 (A87).

#### Điểm đang mất nằm ở đâu (A92 — `107_phan_ra_diem.py`)

| ô | câu | điểm | **mất** | R@1 |
| --- | ---: | ---: | ---: | ---: |
| KIS | 37 | 0,5838 | **0,2962** | 0,24 |
| QA | 12 | 0,3500 | **0,1500** | 0,08 |
| TRAKE | 3 | 0,4667 | 0,0308 | 0,00 |

Hai điều bảng này nói mà 91 mục đo trước KHÔNG nói:

* **R@1 mới là chỗ mất, không phải R@100.** KIS có đáp án trong top-100 ở 81%
  số câu nhưng chỉ xếp hạng 1 ở 24%. Kênh truy hồi thứ sáu không chữa được điều
  đó — **xếp lại hạng trong top-100 mới chữa được.**
* **Q&A hỏng ở khâu TÌM KHUNG, không chỉ ở khâu đọc đáp án.** Con số 0,3500
  trên chưa xét `answer` đúng hay sai; điểm nộp thật còn thấp hơn. A83/A88 tập
  trung vào khâu đào đáp án là đúng nhưng chưa đủ.

#### Việc còn mở, xếp theo `mất` chứ không theo độ thú vị

| # | việc | vì sao |
| --- | --- | --- |
| 0a | **`--trong-so-hoi 0.25` + văn bản gộp** (A100) | gộp hai thứ 🟡 thì **✅ vượt ngưỡng**: 0,5433/0,6173 (+0,0260, 7-2-43). Nhưng chỉ vượt 5% sau ~20 phép so — bật ở lần nộp 1-2 để lấy số THẬT, đừng bật ở lần cuối |
| 0b | **`--be-trake 300`** (A94) | hiệu lớn nhất đang có: +0,0739/+0,0533, 🟡 ở 84% ngưỡng. Cần câu TRAKE **đề thật** để chốt — 15/18 câu hiện tại là tự soạn |
| 1 | **Xếp lại hạng top-100** | A100 đo trần: **0,7885/0,8654**, đang đạt 0,5231/0,6115 -> **hơn 1/4 tổng điểm nằm trong bể mà xếp sai chỗ**. Chưa có cách nào chạy được trên máy 7,7 GB không GPU — đây là câu hỏi mở thật sự, không phải việc chờ làm |
| 2 | **Soi lại 14 nhãn `de_thi_thu` từ ẢNH GỐC** | A89: chưa có bằng chứng phản bác ≠ bằng chứng đúng. Phải soi theo mô tả, **không** chọn từ danh sách hệ thống trả về |
| 3 | **LLM đọc `câu hỏi + văn bản khung` để ra `answer`** | A88: trần Q&A là 6/13, mọi phép chọn theo hình thức bề mặt đều ra xa hơn |
| 4 | 8 gói `de_thi_thu` chưa có nhãn (`p1-3`, `p1-19`, `p1-21`, `p1-22`…) | thêm câu nhãn sạch là cách rẻ nhất để 🟡 thành ✅ |

#### Thứ ĐÃ THỬ VÀ BỊ BÁC — đừng thử lại nếu không có cơ chế mới

dedup (A11) · RRF thô (A14) · hợp nhất hai tầng (A14.1) · kênh 2 (A14.2) ·
kênh 4 (A62) · làm mượt vector (A57) · gộp theo đoạn ASR (A70) · hợp nhất bằng
điểm, 8 biến thể (A72) · caption ở mọi độ phủ (A73, A90) · NMS thời gian (A81) ·
khuếch tán điểm (A82) · phạt bậc TRAKE (A83) · bảng tra ASR (A84) · đào cụm
theo IDF (A88) · gộp văn bản VietOCR (A88, 🟡) · khuếch tán + gộp văn bản cùng
lúc (A91, giẫm lên nhau).


*Giai đoạn 0 đã đóng (873/873, 0 lệch chỉ số thật). Mọi việc dưới đây thuộc
Giai đoạn 1.*

### H1. Đường găng — ✅ **ĐÃ THÔNG**

**247 câu, đủ cả 10 nhóm L**, đã tách tập test giữ kín:

| | KIS | QA | TRAKE | Tổng |
| --- | ---: | ---: | ---: | ---: |
| `dev/tap_dev.jsonl` | 136 | 45 | 46 | **227** |
| `dev/tap_test.jsonl` 🔒 | 10 | 10 | 0 | **20** |

> **23 câu trong tập dev là ĐỀ THẬT do BTC viết** (đề sơ tuyển đợt 1, nạp bằng
> `scripts/34_de_thanh_tap_dev.py`, đáp án nhóm soát tay). Đây là thứ tập dev
> thiếu suốt từ đầu và là nguyên nhân của cả 5 lần dev mù
> (A19/A20/A31/A34/A37): câu tự soạn ~15-22 từ / 1,1 mệnh đề, đề thật **63 từ /
> 2,4 mệnh đề**. **Mọi phép đo từ nay nên báo RIÊNG cột 23 câu đề thật** —
> `scripts/36_do_cua_so.py` làm sẵn việc đó.
>
> **Thêm 35 câu MÔ PHỎNG phân bố đề thật** (A43, tiền tố `MP`): 71 từ / 2,65
> mệnh đề, 100% có ≥2 mệnh đề, soạn bằng mắt từ contact sheet trong
> L21/L22/L24/L27/L30. Cộng lại **59/227 câu** nay đúng phân bố đang đi thi.
>
> Ngược lại, **câu TRAKE thì 41/46 vẫn là tự soạn** (1 đề thật `trake-DE1-16` +
> 4 câu mô phỏng `trake-MP-01..04`), nên phân bố *câu hỏi* TRAKE vẫn là chỗ yếu
> nhất — xem cảnh báo ở A39.
>
> **Soạn câu mới thì dùng `scripts/39_chon_dai_soan.py` để chọn dải và BẮT BUỘC
> chạy `scripts/38_soat_cau_dev_moi.py` trước khi gộp** — nó bắt rò văn bản
> (A21), lệch phân bố, và đáp án không tra được. Ba câu đã bị nó chặn (A43).

**Còn thiếu: câu đếm.** `scripts/11_tim_cau_dem.py` lọc sẵn ứng viên khung
nhiều vật đếm được. Riêng câu đếm thì A26 và ca `p1-15-qa` cho thấy đó là
**trần của model**, không phải lỗi truy hồi — đừng đầu tư thêm.

Thêm câu mới thì cứ `--gop` bình thường — **câu mới vào tập dev, tập test giữ
nguyên**, `gop()` tự loại. **Không chạy lại `--tach-test`** (nó cũng tự từ
chối). Quy trình đầy đủ: [07_lam_tap_dev.md](07_lam_tap_dev.md).

```powershell
# chọn 30 dải chưa ai dùng, dựng sẵn contact sheet (bỏ đầu/cuối video)
python scripts\39_chon_dai_soan.py --nhom <L của mình> --so 30
# soạn vào dev/tap_dev_thanh_vien/tap_dev_<nhóm>.jsonl — TẢ CẢ DẢI, ~60 từ
python scripts\38_soat_cau_dev_moi.py dev\tap_dev_thanh_vien\tap_dev_<nhóm>.jsonl
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
