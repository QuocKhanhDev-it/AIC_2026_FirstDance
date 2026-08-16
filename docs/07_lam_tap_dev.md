# Hướng dẫn làm tập dev — từng bước

*Soạn 2026-08-12. Dành cho người soạn tập dev (theo kế hoạch là Khánh).*

Tập dev là **đường găng thật** của bản 4.1: bản đó thêm sáu giả thuyết lấy từ
bài báo AIC'25, và **không giả thuyết nào được giữ nếu không đo được**. Không
có tập dev thì cả PHẦN A8 chỉ là đọc truyện.

---

## §0. Cái bẫy sẽ hủy hoại tất cả — đọc trước khi làm gì

**ĐỪNG dùng `06_tim.py` (hoặc bất kỳ kênh nào của ta) để tìm đáp án.**

Cách làm có vẻ tự nhiên nhất — gõ truy vấn, thấy kết quả đúng thì ghi lại làm
đáp án — tạo ra một tập dev **chỉ chứa những khoảnh khắc mà CLIP vốn đã tìm
được**. Đo trên nó thì:

- CLIP luôn trông rất tốt
- Mọi cải tiến (RRF, dedup, SigLIP2, caption) đều trông vô dụng
- Và ta kết luận sai rằng bản 4.1 không đáng làm

Đáp án phải tìm bằng **mắt duyệt nội dung**. Đó là lý do có `10_contact_sheet.py`.

---

## §1. Mười phút đáng giá nhất: tìm đề thi AIC'25

Trước khi tự soạn câu nào, tìm xem BTC có công bố **bộ truy vấn mùa 2025**
không (trang chính thức, kỷ yếu SOICT 2025, kho GitHub của các đội).

Nếu có thì đó là tập dev tốt hơn mọi thứ ta tự viết: **đúng phân bố, đúng văn
phong, đúng độ khó, và không dính thiên lệch nào của ta**. Ta chỉ còn phải dò
đáp án trong kho của mình.

Không tìm thấy thì mới sang §2.

---

## §2. Có vật liệu để duyệt

### Vấn đề

```
Keyframe có ảnh trên máy này:  16.896 / 177.321   —  chỉ L21 + L22
```

Mà L21 và L22 đều là *"60 Giây Sáng/Chiều — HTV Tin Tức"*: **cùng một đài,
cùng một chương trình**. Soạn tập dev từ đó là vi phạm A2 (phải phân tầng theo
nhóm L) lẫn A7 (kho lệch về ẩm thực, không phải tin tức).

### Cách gỡ: contact sheet, không phải 30 GB ảnh

```powershell
# duyệt nhanh cả một nhóm L, mỗi video một file
python scripts\10_contact_sheet.py --nhom L21 --thua 10
```

Mỗi ô có ghi sẵn **`row_id`** — đó là con số duy nhất cần chép lại.

Đo thật: một video 24 ô ra file **279 KB**. Cả 873 video ≈ **240 MB** — đẩy
Drive được thoải mái, so với 30,5 GB nếu gửi ảnh gốc.

> **Việc cần nhắn cả nhóm ngay:** mỗi máy chạy `10_contact_sheet.py --nhom <L
> của mình> --thua 10` rồi đẩy thư mục `dev/sheets/` lên Drive. Sau đó ai cũng
> duyệt được toàn kho mà không phải tải gì nặng.

Chọn được video ưng ý thì xem dày quanh vùng quan tâm:

```powershell
python scripts\10_contact_sheet.py --video L25_V012 --thua 1 --tu 120 --den 180
```

---

## §3. Chốt quy ước TRƯỚC khi viết câu nào

Sửa quy ước giữa chừng là phải soát lại toàn bộ câu đã viết. Chốt bốn điều
sau, ghi vào đầu file:

| Điều | Chốt gì | Vì sao |
| --- | --- | --- |
| **Ngôn ngữ câu hỏi** | **Tiếng Việt** | Đề thi ra tiếng Việt. Viết dev bằng tiếng Anh là giấu đi một điểm yếu thật — xem §7 |
| **Chuẩn tắc `dap_an` của QA** | vd: số viết bằng **chữ số** ("2" không phải "hai"), tối đa 4 từ, không kèm chữ định tính | PHẦN C mục 4: `answer` sai định dạng -> **0 điểm dù frame đúng** |
| **Độ cụ thể** | mỗi câu phải **chỉ có một đáp án** trong kho | xem §5 |
| **Tỷ lệ loại câu** | vd 30 KIS / 20 QA / 10 TRAKE | KIS dễ soạn nhất, làm trước |

---

## §4. Duyệt và đánh dấu

1. Mở file sheet, lướt tìm khoảnh khắc **đáng hỏi** — cảnh có nội dung riêng
   biệt, không phải cảnh dẫn chương trình lặp đi lặp lại.
2. Chép `row_id` ghi trên ô đó.
3. **MỞ ẢNH GỐC KIỂM LẠI.** Không bỏ bước này.

> **Vì sao bước 3 bắt buộc — chuyện có thật khi soạn tập mẫu:** nhìn ô
> thumbnail `row_id 713`, người soạn ghi *"xe container va chạm với xe máy"*.
> Mở ảnh gốc ra thì đó chỉ là cảnh camera hành trình **dừng chờ đèn đỏ**,
> không có va chạm nào. Ảnh 320 px đánh lừa được mắt.
>
> Đáp án sai trong tập dev còn tệ hơn không có tập dev: nó làm mọi cấu hình bị
> chấm sai theo cùng một hướng mà không ai biết.

Tra `row_id` đọc trên sheet ra đường dẫn ảnh gốc — nhận nhiều số một lần,
thêm `--mo` để mở luôn bằng trình xem ảnh mặc định:

```powershell
python scripts\10_contact_sheet.py --tra 593 605 713 809 --mo
```

```text
  row_id  video_id      kf  frame_idx      giây  đường dẫn ảnh gốc
     593  L21_V003      25       2220     88.80  C:\Code\aic_data\...\L21_V003\025.jpg
     605  L21_V003      37       3620    144.80  C:\Code\aic_data\...\L21_V003\037.jpg
```

Nó in kèm **khung JSONL điền sẵn** `row_id_dung` và `nguon` — chỉ còn phải gõ
`cau_hoi`, và gõ **sau khi** đã xem ảnh gốc. Nhớ sửa `000` trong `id` thành số
thứ tự thật.

### Bẫy riêng của kho này: BA vùng chữ, chỉ hai vùng dùng được

Bản tin có **ba** loại chữ trên màn hình, và chúng khác nhau hoàn toàn về mức
đáng tin. Phân biệt sai là viết câu hỏi trật:

| Vùng | Ví dụ đo thật | Có liên quan tới hình? |
| --- | --- | --- |
| **Chữ trong cảnh** — biển hiệu, chữ trên công trình, biển số | `THỦY ĐIỆN HÒA BÌNH` trên thân đập; `CHÙA LIÊN HOA` trên cổng chùa; biển số `63-B3 366.32` | ✅ **có** — dùng thoải mái |
| **Băng rôn tiêu đề** — dải đỏ lớn kèm logo `60 giây` | `BẮC NINH: CHÁY NHÀ 5 TẦNG TRONG ĐÊM RẰM THÁNG 7` trên đúng cảnh cháy nhà | ✅ **có** — nhưng xem cảnh báo dưới |
| **Dòng chữ chạy ở đáy** | Ảnh đập Hòa Bình đi kèm *"cao tốc TP.HCM - Mộc Bài"*; ảnh con ngựa con đi kèm *"Đà Nẵng: Ô tô 'điên' tông văng 6 xe máy"*; ảnh dung nham đi kèm *"Tòa án Indonesia yêu cầu hãng dược phẩm bồi thường"* | ❌ **KHÔNG** — tin chạy độc lập |

**Tuyệt đối không viết câu hỏi dựa vào dòng chữ chạy.** Nó là luồng tin riêng,
chạy song song và không dính gì tới cảnh đang chiếu.

**Băng rôn tiêu đề thì có liên quan, nhưng đừng chép nguyên văn vào câu hỏi.**
Chép nguyên si thì kênh OCR trúng ngay mà chẳng đo được gì về thị giác — câu
hỏi thành bài kiểm tra chép chữ. Hãy tả **cảnh**, và nếu muốn đo OCR thì làm
hẳn một câu `QA` riêng lấy đáp án từ chữ đó.

> **Hệ quả cho kênh OCR (kênh 3):** cắt ROI **không phải** là "bỏ nửa dưới màn
> hình". Băng rôn tiêu đề và dòng chữ chạy nằm sát nhau ở đáy nhưng một cái
> vàng một cái độc. Ranh giới đúng là **dải chữ chạy cuối cùng**, không phải
> toàn bộ vùng đáy.

---

## §5. Viết câu hỏi cho đúng

### Quy tắc một đáp án

Sau khi viết xong, tự hỏi: **"câu này có mô tả trúng 10 khoảnh khắc khác trong
kho không?"**

*"người dẫn chương trình ngồi ở bàn tin"* — kho có hàng trăm cảnh như vậy. Ta
đánh dấu một cái, hệ thống trả về một cái khác **cũng đúng**, mà thước đo lại
chấm sai. Kết quả: mọi cấu hình đều bị hạ điểm một cách ngẫu nhiên, và nhiễu
nuốt hết tín hiệu.

Trả lời "có" thì thêm chi tiết phân biệt cho tới khi thành "không".

### Quy tắc không nhìn tiêu đề

Viết câu hỏi **chỉ từ những gì nhìn thấy trong hình**, đừng đọc cột `title` /
`description` trước. Nếu chép chữ từ metadata thì câu đó kênh BM25 trúng ngay
mà chẳng đo được gì về thị giác.

### Quy tắc "viết lại sau khi quên"

Viết xong, đọc lại và tự hỏi: *"nếu chưa từng thấy tấm ảnh này, tôi có viết
câu như vậy không?"*

Người soạn hay tả **pixel** (*"người phụ nữ áo đỏ ở góc trái khung hình"*)
trong khi đề thi thật tả **sự việc** (*"một người đang chuẩn bị món tráng
miệng từ trái cây nhiệt đới"*). Tả pixel làm tập dev dễ giả tạo.

### Định dạng file

`dev/tap_dev.jsonl`, mỗi dòng một câu. Xem mẫu đầy đủ ở
[dev/tap_dev_mau.jsonl](../dev/tap_dev_mau.jsonl):

```json
{"id": "kis-001", "loai": "KIS",
 "cau_hoi": "đập thủy điện đang xả nước, bọt trắng xóa cuộn lên dưới chân đập",
 "row_id_dung": [593],
 "nguon": "duyệt sheet L21_V003, đã mở ảnh gốc 025.jpg kiểm lại",
 "ghi_chu": "trong hình có chữ THỦY ĐIỆN HÒA BÌNH"}
```

| Trường | Ghi chú |
| --- | --- |
| `loai` | `KIS` / `QA` / `TRAKE` |
| `row_id_dung` | **danh sách**. TRAKE thì là danh sách của danh sách, mỗi sự kiện một cái |
| `dap_an` | chỉ QA. Theo đúng quy ước chuẩn tắc đã chốt ở §3 |
| `nguon` | tìm ra bằng cách nào — để sau này soát lại được |

Lưu `row_id` chứ không lưu `frame_idx`: từ `row_id` suy ra được `video_id`,
`frame_idx`, `pts_time`, `fps` — ngược lại thì không.

---

## §6. Nở cụm trùng lặp rồi kiểm

```powershell
python src\tap_dev.py --file dev\tap_dev.jsonl --no-cum
python src\tap_dev.py --file dev\tap_dev.jsonl --kiem
```

`--no-cum` nở mỗi `row_id` đã đánh dấu ra **cả cụm bản sao cùng video**
(cosine ≥ 0,99). Bắt buộc, vì A5.6 đo được **11,83% keyframe có bản sao cùng
video** — riêng L25 là 49,82%, có cặp cosine đúng 1,0000. Không nở cụm thì hệ
thống trả về bản sao của đúng khoảnh khắc lại bị chấm là sai.

`--kiem` soát: id trùng, `row_id` ngoài khoảng, câu QA thiếu `dap_an`, đáp án
nằm ở nhiều video, và **in bảng phân bố theo nhóm L**. Nhìn bảng đó để giữ cân
bằng — A2: L26 chiếm 57% số video, lấy ngẫu nhiên sẽ ra tập lệch hẳn.

---

## §7. Cỡ mẫu — chỗ ta đã bị bỏng hai lần

Kế hoạch ghi 30–50 câu. Nói thẳng: **so theo cách thông thường thì 30–50 câu
gần như không phân biệt được gì.**

Điểm mỗi câu nằm trong {0; 0,2; 0,4; 0,6; 0,8; 1,0}, độ lệch chuẩn cỡ 0,35.
Với 40 câu, sai số chuẩn của trung bình là **0,055** — chênh lệch dưới ~0,11
là nhiễu. Đúng cái bẫy đã làm hỏng kết luận của bench VLM (20 mẫu) và bench
OCR (13 mẫu).

**Cách cứu: luôn so THEO CẶP trên cùng bộ câu hỏi.** Phần lớn câu cho điểm y
hệt dưới cả hai cấu hình và triệt tiêu nhau, nên độ lệch của *hiệu* nhỏ hơn
hẳn. `src/cham_diem.py` làm sẵn việc này:

```
tiếng Việt = 0.0000    tiếng Anh = 0.4400    hiệu = +0.4400
tiếng Anh thắng 4 câu / thua 0 / hòa 1   (n = 5)
sai số chuẩn của hiệu = 0.1939  ->  |hiệu| cần > 0.3878 mới đáng tin
```

Dòng **thắng–thua–hòa** nói nhiều hơn hiệu trung bình. `11 thắng / 4 thua`
trên 40 câu là tín hiệu thật; `+0,04` kèm `6 thắng / 5 thua` chỉ là nhiễu.

### Một kết quả đo được ngay trên tập mẫu 5 câu

Chạy tập mẫu bằng câu hỏi **nguyên văn tiếng Việt** và bằng **bản dịch tay
sang tiếng Anh**:

| Cách hỏi | Điểm | Chi tiết |
| --- | --- | --- |
| Tiếng Việt | **0,00** | trượt cả 5/5 câu, không câu nào lọt top-100 |
| Tiếng Anh | **0,44** | 1 câu hạng 1, 1 câu hạng 5, 2 câu hạng 86 |

CLIP ViT-B/32 **không phải "yếu" với tiếng Việt — nó mù hẳn**. Và hiệu +0,44
vượt ngưỡng 2 sai số chuẩn ngay cả với n = 5.

Đây là lý do câu hỏi dev **phải viết bằng tiếng Việt**: nếu hệ thống cần một
bước dịch thì bước dịch đó **là một phần của hệ thống** và phải bị đem ra đo.
Viết dev bằng tiếng Anh là tự giấu đi một điểm yếu thật. Đây cũng là bằng
chứng độc lập cho khuyến nghị chọn **SigLIP2 đa ngôn ngữ** ở kế hoạch GPU.

---

## §7b. Chấm theo CỬA SỔ, đừng chấm theo `row_id` chính xác

BTC xác nhận (A9) đáp án là **cửa sổ `[s,e]` rộng 4 giây đến 5 phút**. Nếu ta
so `row_id` chính xác thì một keyframe cách đáp án 2 giây bị tính **sai**,
trong khi BTC tính **đúng**.

Đo thật trên 21 câu có bản dịch tiếng Anh:

| Chấm ở | Điểm CLIP |
| --- | --- |
| `row_id` chính xác | **0,5048** |
| ±2s *(cửa sổ 4s — hẹp nhất BTC nêu)* | 0,5238 |
| ±5s | 0,5619 |
| ±15s | 0,5905 |
| ±90s *(cửa sổ 3 phút)* | **0,6095** |

**Chấm chặt hạ điểm mất 0,10.** Để so sánh: toàn bộ lợi ích đội AIC'25 thu
được khi thêm SigLIP2 chỉ là **+0,07**. Sai lệch do chấm chặt **lớn hơn thứ
đang cần đo**, và nó hạ **không đều** giữa các cấu hình — cấu hình nào hay trả
về keyframe lân cận bị phạt nặng hơn, nên thứ hạng có thể **đảo ngược**.

> ⚠️ **Một lỗi đã xảy ra thật khi viết mục này, đáng ghi lại.** Hằng số mới
> `MOC = (2.0, 15.0)` (hai mức dung sai) được đặt trùng tên với
> `MOC = (1, 5, 20, 50, 100)` — **các mốc R@k của chính công thức chấm BTC** —
> nên nó **đè mất**, và `diem_cau()` lặng lẽ tính trung bình R@2 với R@15 thay
> vì R@{1,5,20,50,100}.
>
> Kết quả: bảng đo tụt từ 0,5238 xuống 0,3810, và tôi **suýt sửa tài liệu theo
> con số hỏng đó**. Không có gì crash, không có gì cảnh báo — chỉ là điểm sai.
>
> `tests/test_cham_diem.py::test_bac_thang_dung_cong_thuc_btc` là thứ bắt được,
> vì nó chốt cứng `diem_cau(2) == 0,8`. **Đó chính là lý do phải có test cho
> thước đo**, dù thước đo "chỉ là mấy phép trung bình".

**Đừng gọi `cham()` tay cho từng mức.** Dùng hàm báo cáo — nó chấm ở cả hai mức
và tự kết luận độ ổn định:

```python
from cham_diem import bao_cao_do_nhay
print(bao_cao_do_nhay(dev, {
    "CLIP (mốc nền)": chay_clip,
    "CLIP + dedup":   chay_dedup,
    "RRF(CLIP, objects)": chay_rrf,
}, master=kenh.master))
```

| Kết luận nó in ra | Nghĩa |
| --- | --- |
| `✅ ON DINH` | cùng dấu ở cả hai mức **và** vượt 2 sai số chuẩn ở ít nhất một mức |
| `🟡 YEU` | cùng dấu nhưng chưa vượt nhiễu — cần thêm câu hỏi, **chưa quyết được** |
| `❌ DAO DAU` | **đổi dấu giữa hai mức** — kết luận phụ thuộc ẩn số BTC chưa chốt, **không dùng để quyết** |
| `⚪ KHÔNG ĐỔI GÌ` | cấu hình không tác động trên tập dev này |

## §7c. Ba cấu hình đã đo — chưa cấu hình nào kết luận được

Chạy `bao_cao_do_nhay` trên 21 câu, mốc nền là CLIP đơn thuần:

| Cấu hình | ±2s | ±15s | Hiệu (±2s / ±15s) | Kết luận |
| --- | --- | --- | --- | --- |
| CLIP *(mốc nền)* | 0,5238 | 0,5905 | — | — |
| CLIP + `dedup` | 0,5238 | 0,5905 | 0 / 0 | ⚪ **không đổi gì** (0-0-21) |
| CLIP + ràng buộc đa dạng | 0,4381 | 0,5810 | −0,0857 / −0,0095 | 🟡 yếu |
| RRF(CLIP, objects) | 0,5429 | 0,5905 | **+0,0190 / −0,0000** | ❌ **ĐẢO DẤU** |

**Chưa cấu hình nào kết luận được.** Với 21 câu, ngưỡng nhiễu là 0,06–0,12 mà
mọi hiệu đều nằm dưới — đúng điều §7 cảnh báo.

Đáng chú ý nhất là dòng cuối: **RRF(CLIP, objects) đổi dấu giữa hai mức dung
sai**. Ở ±2s nó có vẻ giúp (+0,019), ở ±15s nó bằng không. Nếu chỉ chấm ở một
mức thì ta đã kết luận "objects có ích" — và kết luận đó **phụ thuộc hoàn toàn
vào một con số BTC chưa chốt**. Đây chính là thứ hàm báo cáo sinh ra để bắt.

Việc cần làm không phải chỉnh thuật toán, mà là **thêm câu hỏi**.

### `dedup` chưa kiểm được — vì tập dev sai chỗ

Đo `dedup` ở mọi mức dung sai: **0 thắng – 0 thua – 21 hòa.** Không đổi một câu
nào.

Nhưng đó **không phải bằng chứng dedup vô dụng** — mà là bằng chứng **tập dev
sai chỗ để đo nó**. Keyframe trùng lặp dồn hết vào L25 (**49,82%**), còn các
nhóm khác chỉ 0,27–2,16%. Tập dev hiện có L21, L22, L23, L26, L27 — **không có
câu L25 nào**, tức đang đo dedup ở nơi gần như không có trùng lặp.

→ **Muốn kết luận về `dedup` thì phải có câu L25.** Đây là một lý do cụ thể để
ưu tiên nhóm L25 khi phân công soạn tiếp.

## §7d. Thiếu hẳn một dạng câu: ĐẾM SỐ LƯỢNG

BTC nêu thẳng dạng này khi trả lời (A9):

> *"Các trường hợp thường gặp là về bài toán **đếm số lượng**, 1 cái frame có
> thể không đếm được hết... 1 em bé được bế bởi 4 người liên tiếp, dựa trên
> keyframe thì chỉ có 3."*

Tập dev hiện **không có câu đếm nào**, nên ta sẽ không bao giờ phát hiện được
điểm yếu này. Cần **ít nhất 3–5 câu**, và đáp án phải lấy từ **video**, không
phải từ một keyframe.

Đây cũng là dạng câu duy nhất mà `trich_day` thật sự cần thiết — không có câu
đếm thì không đo được nó có đáng giữ hay không.

## §8. Cất tập test

Giai đoạn 3 yêu cầu kiểm cuối trên tập **chưa từng nhìn**. Với 50 câu mà chia
đôi thì hỏng cả hai.

Đề nghị: soạn **60 câu**, cất **15 câu** sang `dev/tap_test.jsonl`, **không mở
ra cho tới lượt đo cuối cùng**. Không cần cơ chế gì phức tạp — chỉ cần kỷ luật
không mở file đó.

---

## §9. Đo baseline

```python
import sys; sys.path.insert(0, "src")
import tap_dev
from cham_diem import cham, tom_tat, so_sanh_cap
from dense import KenhAnh

dev  = tap_dev.doc("dev/tap_dev.jsonl")
kenh = KenhAnh("./index")

goc = cham(dev, lambda c: kenh.tim(c.cau_hoi, k=100))
print(tom_tat(goc).to_string())
```

Có con số đó rồi thì mọi giả thuyết của bản 4.1 mới bắt đầu trả lời được:

| Câu hỏi treo | So cặp cấu hình nào |
| --- | --- |
| Dedup có lợi không? | CLIP vs CLIP + `gom_ban_sao` |
| RRF nhiều kênh có lợi không? | CLIP vs RRF(CLIP, BM25, objects) |
| SigLIP2 có đáng encode toàn kho không? | ViT-B/32 vs SigLIP2 vs RRF cả hai |
| Đi bộ theo thời gian có cứu được Q&A không? | có vs không `lan_can` |

---

## §10. Sáu người cùng soạn thì chia thế nào

### Nguyên tắc chia: ai giữ nhóm L nào thì soạn câu cho nhóm đó

Không phải để chia đều cho công bằng, mà vì **bước 3 của §4 bắt buộc mở ảnh
gốc kiểm lại** — và chỉ máy đang giữ gói `Keyframes_*` mới mở được ảnh gốc ở
độ phân giải đầy đủ. Người khác chỉ có contact sheet 320 px, mà §4 đã cho thấy
320 px đánh lừa được mắt.

| Việc | Ai | Gửi gì |
| --- | --- | --- |
| Dựng contact sheet nhóm L mình giữ | mỗi máy | `dev/sheets/` lên Drive (~240 MB cả kho) |
| Soạn câu cho nhóm L mình giữ | mỗi máy | `dev/tap_dev_<nhóm>.jsonl` — **vài KB, gửi qua chat được** |
| Gộp + soát + cất tập test | một người | `dev/tap_dev.jsonl` |

Contact sheet lên Drive vẫn có ích cho **mọi người** — để nhìn ra kho có gì,
và để người gộp soát chéo. Chỉ việc *viết câu* mới cần ảnh gốc.

### Quy ước `id`: gắn nhóm L vào

```
kis-L21-001    qa-L25-003    trake-L26-002
```

Sáu người cùng đánh `kis-001` là trùng ngay. Gắn nhóm L thì vừa hết trùng vừa
**nhìn ra phân bố ngay trên `id`** — đếm nhanh xem nhóm nào đang thiếu câu.

### Mỗi người soạn bao nhiêu câu

Mục tiêu 60 câu / 10 nhóm L = **6 câu mỗi nhóm**. Ai giữ 2 nhóm thì soạn 12
câu, giữ 3 nhóm thì 18. Chia theo nhóm L chứ đừng chia theo đầu người — A2:
L26 có 498 video còn L21 chỉ 30, nhưng **tập dev cần cân bằng theo nhóm**, chứ
không theo kích thước nhóm.

### Gộp lại

```powershell
python src\tap_dev.py --gop dev\tap_dev_thanh_vien --file dev\tap_dev.jsonl
python src\tap_dev.py --file dev\tap_dev.jsonl --no-cum
python src\tap_dev.py --file dev\tap_dev.jsonl --kiem
```

Truyền thẳng **thư mục** — đừng gõ `dev\tap_dev_thanh_vien\*.jsonl`.
**PowerShell không bung dấu sao cho chương trình ngoài** (khác bash), nên
Python nhận đúng chuỗi `*.jsonl`, đọc không ra file nào, và **gộp ra 0 câu mà
không báo lỗi gì**. Đã vấp thật một lần.

`--gop` **báo lỗi khi trùng `id` chứ không tự đổi tên**: đổi ngầm thì sau này
không truy được câu đó của ai. Bảng phân bố in ra ở `--kiem` là chỗ nhìn để
biết còn thiếu nhóm nào.

---

## Bảng việc

| # | Việc | Thời gian | Chặn bởi |
| --- | --- | --- | --- |
| 1 | Tìm đề AIC'25 đã công bố | 10 phút | — |
| 2 | Nhắn cả nhóm dựng contact sheet, đẩy Drive (~240 MB) | 1 giờ | — |
| 3 | Chốt 4 quy ước ở §3 | 15 phút | — |
| 4 | Duyệt sheet, soạn 30 câu KIS | ~1 ngày | 2, 3 |
| 5 | Thêm 20 QA + 10 TRAKE | ~1 ngày | 4 |
| 6 | `--no-cum` rồi `--kiem` | 5 phút | 5 |
| 7 | Cất 15 câu sang `tap_test.jsonl` | 5 phút | 6 |
| 8 | Đo baseline | 15 phút | 7 |

Việc **1 và 2 làm được ngay hôm nay** và cả hai đều gỡ bí cho người khác.
