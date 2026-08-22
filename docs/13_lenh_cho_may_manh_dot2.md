# Lệnh cho máy mạnh — đợt 2

*Soạn 22/08/2026. Gửi người giữ máy ≥ 16 GB RAM (hoặc Colab/Kaggle).*

Đợt 1 (`12_viec_cho_may_manh.md`) đã xong cả 6 việc — kết quả ở A29–A37. Đợt
này chỉ có **hai phép đo**, nhưng khác đợt 1 ở một điểm quan trọng:

> Lần đầu tiên ta có **thước đo đúng phân bố đề thi**. Tập dev vừa nạp thêm
> **23 câu do BTC viết** từ đề sơ tuyển đợt 1 (`34_de_thanh_tap_dev.py`), trong
> đó 42 câu TRAKE. Suốt từ đầu, tập dev tự soạn đã **mù 5 lần**
> (A19/A20/A31/A34/A37) và mỗi lần truy nguyên đều ra cùng một nguyên nhân:
> câu dev ~15-22 từ / 1,1 mệnh đề, còn đề thật 63 từ / 2,4 mệnh đề.

Hai phép đo dưới đây là hai thứ **duy nhất** hiện có thuật toán xong, test
xong, mà chưa có số. Cả hai đều tắc ở đúng một chỗ: máy 7,7 GB không nạp nổi
`ViT-SO400M-14-SigLIP2-378`.

---

## VIỆC 0 — Làm mới cache vector truy vấn *(bắt buộc, ~10 phút)*

`index/truy_van.npz` hiện có **thiếu 23 câu đề mới**. Cả hai phép đo dưới đây
đều dừng ngay và in ra chuỗi còn thiếu nếu cache chưa đủ, nên phải làm cái này
trước.

```powershell
git pull
$env:PYTHONIOENCODING = "utf-8"

.venv\Scripts\python.exe scripts\25_ma_hoa_truy_van.py --tap-dev --gop
```

⚠️ **`--gop`** chứ không phải chạy trần: chạy trần **dựng lại** file từ đầu và
mất các chuỗi đã mã hoá trước đó. Cùng cái bẫy đã làm tập dev tụt 105 → 24 câu.

Xong thì `index/truy_van.npz` cần **gửi ngược về cho cả nhóm** — nó vài trăm
KB, và nó là thứ mở khoá kênh 1 cho mọi máy yếu.

Kiểm nhanh cache đã đủ chưa:

```powershell
.venv\Scripts\python.exe scripts\36_do_cua_so.py --cache index\truy_van.npz --kiem-moc
```

Dòng `[kiểm mốc] ... TRÙNG kenh.tim()` phải hiện ra. **LỆCH thì dừng lại** —
mốc nền sai thì mọi so sánh sau đó vô nghĩa.

---

## VIỆC 1 — Đo cách chấm theo CỬA SỔ (A38)

**Đang hỏi gì.** `dense.tim` lấy `max` qua các mệnh đề *trên cùng một khung* —
thưởng cho khung khớp MỘT mệnh đề thật mạnh. A38 nói cách đó lệch với cách đề
được viết: người ra đề **xem video** rồi tả một quãng thời gian.

Bằng chứng là chính ca `query-p1-4-kis` nhóm đã soát tay: kf183 phủ mệnh đề 1,
kf186/187 phủ mệnh đề 2, **không khung nào phủ cả hai** — cộng lại mới ra "hai
nhân viên" như đề tả. `cua_so.diem_cua_so` đảo hai phép toán: **cộng** qua mệnh
đề, lấy `max` qua khung lân cận.

```powershell
.venv\Scripts\python.exe scripts\36_do_cua_so.py --cache index\truy_van.npz
```

Chạy ~5-10 phút (144 câu KIS/QA × 4 bán kính × 2 mức dung sai).

### Đọc kết quả

Script in **ba khối**: toàn bộ / chỉ câu đề thật / chỉ câu tự soạn.

> **Khối đáng tin là `chỉ N câu ĐỀ THẬT`.** Câu tự soạn phần lớn chỉ có MỘT
> mệnh đề, mà với một mệnh đề thì "cộng qua mệnh đề" và "max qua mệnh đề" là
> **cùng một phép tính** — A38 về nguyên tắc **không thể lộ ra** ở đó. Script
> in sẵn số câu một-mệnh-đề ở đầu.
>
> Nên nếu khối tự soạn ra ⚪ mà khối đề thật ra dương, đó **đúng là điều A38 dự
> đoán**, không phải mâu thuẫn. Ngược lại, đề thật ⚪ hoặc âm thì A38 sai ở
> khâu thuật toán và **đừng bật** — ghi lại là một mục "đã thử, không ăn thua",
> phần đó của tài liệu có giá trị ngang phần thắng.

Bán kính nào thắng cũng là một thông tin: keyframe cách nhau trung vị **2,16 s**
nên `±3` là khoảng **±6,5 giây**. Thắng ở bán kính lớn nghĩa là nên nới; thắng
ở `±1` rồi tụt dần nghĩa là tín hiệu thật nhưng hẹp.

---

## VIỆC 2 — Đo prior KHOẢNG CÁCH cho TRAKE (A39)

**Đang hỏi gì.** Bạn báo *"câu TRAKE thì hệ thống bên máy mạnh không làm được"*.
Truy nguyên vào khâu lắp ráp thì thấy: `run.dong_hang_dp` chỉ ràng buộc
`khung(i) < khung(i+1)` — nó biết **thứ tự** nhưng **không biết khoảng cách**.

Mà khoảng cách đo được, và phân bố rất chặt (42 câu TRAKE của tập dev):

| | trung vị | p25 | p75 | min | max |
| --- | --- | --- | --- | --- | --- |
| độ trải cả chuỗi | 56,6 s | 52,5 | 59,3 | 12,9 | **101,0** |
| khoảng cách 2 sự kiện liền kề | 18,7 s | 12,4 | 28,9 | **5,1** | 55,7 |
| độ trải / độ dài video | 0,1 | 0,1 | 0,2 | 0,0 | **0,2** |

Ba chỗ sai trong bản đang chạy, cả ba đều suy ra thẳng từ bảng này:

1. **Chốt chống dồn cục rải N sự kiện đều khắp TOÀN BỘ video** — mà độ trải
   thật chiếm trung vị 10%, không bao giờ quá 20%. Tức nó đẩy các sự kiện ra xa
   gấp **5–10 lần**. Chốt này nổ trên **47/100 dòng** của `query-p1-18-trake`
   trong bài nộp thật.
2. **`DON_NHAU = 100` khung nằm DƯỚI mức sàn đo được.** 100 khung là 3,3 s
   (30 fps) đến 4,0 s (25 fps); khoảng cách nhỏ nhất từng thấy là 5,1 s.
   Ngưỡng thấp hơn cả sàn thì gần như không bắt được gì.
3. **Đếm bằng KHUNG là dính đúng bẫy fps** — kho có 4 giá trị fps nên
   `DON_NHAU = 100` mang ý nghĩa khác nhau ở từng video.

```powershell
.venv\Scripts\python.exe scripts\35_do_chuoi_trake.py --cache index\truy_van.npz
```

### Đọc kết quả

Bốn biến thể, mỗi cái so **riêng** với mốc nền `cu` (không cộng dồn — cộng dồn
rồi quy công cho nhầm cái là lỗi đã vấp nhiều lần):

| | đổi đúng một thứ |
| --- | --- |
| `tran` | + trần độ trải 180 s |
| `phat` | + prior khoảng cách (phạt dồn cục / giãn quá) |
| `rai_hep` | + rải hẹp 56,6 s thay vì rải khắp video |

`tat_ca` in ở cuối **chỉ để biết trần trên** — đừng dùng nó để quyết.

> ⚠️ **41/42 câu TRAKE của tập dev là câu TỰ SOẠN**, chỉ `trake-DE1-16` do BTC
> viết. Nên phân bố *câu hỏi* không đáng tin. Nhưng phân bố *thời gian đáp án*
> ở bảng trên thì đỡ hơn nhiều — nó là tính chất của **video và của cách người
> ta chọn một chuỗi sự kiện**, không phải của cách viết câu hỏi. Và câu đề thật
> duy nhất có độ trải **101,0 s**, nằm ở mép trên nhưng vẫn trong khoảng; vì
> n=1 nên trần trong code đặt **rộng gấp 1,8 lần** con số đó chứ không bám sát.

---

## Gửi về những gì

1. `index/truy_van.npz` đã làm mới (VIỆC 0) — thứ này mở khoá cho cả nhóm.
2. **Toàn văn stdout** của hai lệnh đo, dán nguyên. Đừng tóm tắt thành "cái
   này tốt hơn": dòng thắng–thua–hoà và ngưỡng nhiễu nói nhiều hơn điểm trung
   bình, và dòng kết luận `✅ ON DINH` / `🟡 YEU` / `❌ DAO DAU` / `⚪ KHÔNG ĐỔI
   GÌ` là thứ quyết định có bật hay không.

**Cả hai thứ mặc định đang TẮT trong code.** `dung_trake(dong_hang="cu",
he_so_phat=0.0, rai_hep=False)` cho lại đúng hành vi đã đẻ ra bài nộp 11 điểm,
và `test_mac_dinh_giong_het_ban_cu` chốt điều đó. Không thắng trên dev thì
không bật — kỷ luật đo của dự án, và A34/A37 là hai lần trả giá gần nhất cho
việc bật sớm.

---

## Ngoài lề: một chỗ tôi nghĩ ngược lại đề xuất "cho model suy luận nhiều bước"

Ghi chép của nhóm về `p1-17-qa` nhận xét: **chuỗi suy luận NGẮN thì ít lệch**.
Tôi thấy đúng, và nó giới hạn phạm vi của ý tưởng CoT/ToT:

- Truy vấn có **mỏ neo văn bản** (tên riêng, địa danh, từ hiếm như *mazut*):
  đường ngắn nhất là bám thẳng mỏ neo. Suy luận nhiều bước chỉ làm trôi.
- Truy vấn **thuần thị giác, không mỏ neo**: mới đáng suy luận nhiều bước.

Nên nếu có thì giờ làm thêm, thứ đáng làm là **định tuyến theo việc có mỏ neo
hay không**, chứ không cho mọi câu đi qua cùng một chuỗi suy luận dài. Chưa
viết code, chưa đo — ghi ra để khỏi quên.
