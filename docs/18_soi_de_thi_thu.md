# 18 — Soi 24 gói `de_thi_thu/` để tìm đáp án

**Người làm:** một người, không cần GPU, không cần hiểu mã nguồn.
**Thời gian:** ước 2–4 giờ cho 24 gói, làm nhiều lần cũng được.
**Kết quả cần nộp:** một file `dev/tap_de_thi_thu.jsonl`.

---

## Vì sao việc này đáng làm trước mọi thứ khác

Tập đề thật của nhóm có **52 câu**. Ở cỡ đó, ngưỡng nhiễu của phép đo là
**±0,02–0,03** — nghĩa là mọi cải tiến nhỏ hơn ngần ấy đều **không kết luận
được**, dù nó có thật hay không.

Ba kết quả gần đây rơi đúng vào vùng chết đó:

| ý tưởng | hiệu | kết luận |
| --- | ---: | --- |
| ủng hộ theo video (A55) | +0,0077 | 🟡 chưa vượt nhiễu |
| giữ 2 mệnh đề dài nhất (A55) | +0,0154 | 🟡 chưa vượt nhiễu |
| trọng số kênh 3 = 0,5 (A52) | +0,0077 | 🟡 chưa vượt nhiễu |

Cả ba **có thể đều đúng** — tập dev quá nhỏ để biết. 24 gói này đưa 52 → **76
câu (+46%)**, kéo ngưỡng nhiễu xuống, và tự phân định những câu 🟡 đó mà không
cần nghĩ thêm ý tưởng nào.

Nói cách khác: đây không phải việc nhập liệu. Nó là việc **làm cho thước đo
chính xác hơn**, mà mọi quyết định kỹ thuật về sau đều dựa vào thước đó.

---

## Trước khi bắt đầu — một lượt Kaggle 1 giây

Máy thường không nạp nổi model, nên kênh ảnh chạy bằng vector mã hoá sẵn trong
`index/truy_van_gopt.npz`. **63 chuỗi truy vấn của 24 gói này chưa có trong
đó**, nên chưa mã hoá thì trang soi sẽ trống.

Làm theo `notebooks/kaggle_ma_hoa_de_thi_thu.md` — mất khoảng **1 giây GPU**
(A49: 69 chuỗi/giây), phần còn lại là thời gian tải model. Xong thì ghi đè
`index/truy_van_gopt.npz`.

⚠️ Kiểm số chuỗi sau khi ghi đè: phải **tăng** từ 1.158 lên khoảng 1.221. Nếu
**giảm** thì `--gop` đã không ăn và bạn vừa xoá cache của cả tập dev — lúc đó
mọi script đo đều tắc.

---

## Bước 1 — sinh trang soi

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe scripts\66_soat_de_thi_thu.py
```

Ra `dev/soat_de_thi_thu.html`. Mở bằng trình duyệt (bấm đúp là được).

Muốn nhiều ứng viên hơn cho những câu khó: `--dau 60`.
Muốn làm từng loại một: `--chi kis`, `--chi qa`, `--chi trake`.

## Bước 2 — soi

Mỗi câu là một lưới **40 ảnh thu nhỏ**, kèm `video_id`, số keyframe và mốc
thời gian. Bấm vào ô đúng để chọn, bấm lần nữa để bỏ.

**Chọn được nhiều ô cho một câu, và nên chọn nhiều.** Đáp án là một *danh
sách*: cùng một cảnh thường trải trên vài keyframe liên tiếp, chọn hết những
khung thật sự đúng thì phép chấm mới phản ánh đúng.

Lựa chọn được lưu trong trình duyệt (`localStorage`), nên đóng tab không mất.
Làm 5 câu rồi nghỉ cũng được.

### Ba loại câu, làm khác nhau

**KIS** — tìm cảnh được tả. Một câu, một lưới.

**Q&A** — cũng tìm cảnh, nhưng câu hỏi có đáp án chữ (biển số, tên, con số).
Trang này **không** điền `dap_an`; cứ chọn khung đúng, phần đáp án chữ điền
sau (xem Bước 4).

**TRAKE** — mỗi **sự kiện** một lưới riêng, phải chọn cho **tất cả** các sự
kiện. Thiếu một sự kiện thì trang sẽ **không xuất câu đó** (nó ghi ra một dòng
`// ... THIEU ...`). Đó là cố ý: số Frame ID phải khớp số sự kiện, thiếu một
vị trí là sai định dạng và **mất trắng cả câu**.

### Bốn quy tắc, đọc kỹ

**1. Ô xám (không có ảnh) thì đừng chọn.** Máy này không có ảnh của video đó.
Bấm chọn là ghi vào thước đo một đáp án chưa ai nhìn thấy.

**2. Không chắc thì bỏ qua cả câu.** Câu không chọn gì sẽ không được xuất, và
như thế là đúng. Một đáp án sai nằm trong thước đo làm lệch mọi phép đo sau đó
**mà không có gì báo** — nó không crash, không cảnh báo, chỉ âm thầm cho ra số
sai. Bỏ 5 câu còn hơn đoán bừa 2 câu.

**3. Nhãn `nhỏ` nghĩa là bạn đang nhìn ảnh thu nhỏ 256px.** Đủ để **nhận ra
cảnh**, không đủ để **đọc chữ nhỏ** trên biển hiệu. Câu Q&A đọc chữ thì phải mở
ảnh gốc mới kết luận được.

**4. Đáp án chỉ là niềm tin.** Trang tự gắn nhãn `do_chac: kha`. Ai đã mở ảnh
gốc soát kỹ thì sửa tay thành `xong` trong `ghi_chu`. Phép đo nghiêm túc chỉ
nên chạy trên nhãn `xong`.

## Bước 3 — xuất

Bấm **Xuất JSONL** ở đầu trang. Chép toàn bộ nội dung ô văn bản vào file mới:

```
dev/tap_de_thi_thu.jsonl
```

Những dòng bắt đầu bằng `//` là câu TRAKE còn thiếu sự kiện — **xoá chúng đi**
hoặc quay lại soi nốt, đừng để lẫn trong file.

## Bước 4 — kiểm trước khi giao

```powershell
.venv\Scripts\python.exe src\tap_dev.py --kiem --file dev\tap_de_thi_thu.jsonl
```

Phải thấy `✅ Tập dev hợp lệ.` Nếu báo lỗi thì đọc dòng lỗi — nó nói rõ câu nào
và sai gì.

Với câu **Q&A**, mở file bằng trình soạn thảo và điền trường `dap_an` (đáp án
chữ). Bỏ trống là câu đó chấm sai định dạng, **0 điểm**, dù khung có đúng.

Xong thì báo lại, đừng tự `--gop` vào `tap_dev.jsonl`.

---

## Những cái bẫy đã cắn thật

**Đừng sửa file bằng `Get-Content | Set-Content` trên PowerShell.** PS 5.1 đọc
UTF-8 không BOM thành ANSI và làm hỏng hết tiếng Việt. Dùng VS Code hoặc trình
soạn thảo bình thường.

**Đừng chạy `src/tap_dev.py --gop` hay `--tach-test`.** `--gop` từng làm tập
dev tụt 260 → 63 câu; `--tach-test` làm rò tập test kín.

**Đừng chép đáp án từ chữ hiện trên màn hình vào câu hỏi.** Không áp dụng ở đây
(đề do BTC viết, ta chỉ tìm đáp án), nhưng đó là lỗi đã làm hỏng 6/14 câu TRAKE
tự soạn — kênh OCR "tìm ra" đáp án mà không hề nhìn thấy gì (A21).

---

## Nếu bí

* Trang trống trơn / không ảnh nào → chưa chạy lượt mã hoá Kaggle ở trên.
* Ảnh không hiện → thiếu `index/anh_nho/` (chạy `scripts/49_sinh_anh_nho.py`
  hoặc xin bản 1,34 GB từ nhóm).
* `UnicodeEncodeError` khi chạy script → thiếu
  `$env:PYTHONIOENCODING = "utf-8"`.
* Không câu nào có đáp án trong 40 ứng viên → tăng `--dau 100`. Nếu vẫn không
  thấy, ghi lại mã câu đó và bỏ qua: **đó là thông tin có ích**, nó nói truy hồi
  đang trượt hẳn câu này.
