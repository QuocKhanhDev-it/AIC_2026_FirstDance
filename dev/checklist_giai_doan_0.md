# Bảng kiểm tra Giai đoạn 0 — cập nhật 2026-08-05

Theo PHẦN 6 của `HUONG_DAN_GIAI_DOAN_0.md`. Đủ 8 gạch mới mở Giai đoạn 1.

| | Mục | Trạng thái |
| --- | --- | --- |
| 1 | `paths.parquet` phủ ≥ 99% video ở `csv`, `clip`, `keyframe_dir` | ⚠️ **một phần** |
| 2 | `master.parquet` dựng xong, không còn lỗi `lech_so_vector` | ✅ **đạt** |
| 3 | `clip.npy` shape `(N, 512)`, `N` = số dòng `master` | ✅ **đạt** |
| 4 | `verify_report.csv` cho ≥ 95% `KHOP` | ✅ **đạt — 203/203** |
| 5 | Ghi lại 4 con số vào tài liệu nhóm | ✅ **đạt** |
| 6 | `index/` đã lên Drive, cả 6 người tải được | ⬜ **chưa** |
| 7 | Câu hỏi về cửa sổ `[s,e]` đã gửi BTC | ⬜ **chưa** |
| 8 | Chốt một bảng tên thành viên duy nhất | ⬜ **chưa** |

## Chi tiết

**1 — một phần.** `csv` và `clip` phủ **100%** (873/873). `keyframe_dir` trên
máy này mới **6,9%** (60/873) vì mới tải `Keyframes_L21` + `Keyframes_L22`.
Không phải lỗi, chỉ là chưa tải xong. Gạch này sẽ tự đạt khi tải hết các gói
Keyframes.

Tính cả kết quả thành viên khác gửi về thì **248/873 video (28,4%) đã được
kiểm chứng**, thuộc 7/10 nhóm L (thiếu L23, L26, L27). Chạy `python scripts/07_gop_kiem_chung.py`
để xem bảng độ phủ mới nhất.

**2 — đạt.** `problems.csv` có 844 dòng nhưng **toàn bộ là `lech_so_keyframe`**
(chưa tải ảnh). **Không có dòng `lech_so_vector` nào** — đây mới là lỗi
nghiêm trọng mà hướng dẫn cảnh báo.

**3 — đạt.** `clip.npy` = `(177321, 512)` float32 chuẩn L2. `master.parquet`
= 177.321 dòng. Khớp chính xác.

**4 — đạt.** 203/203 mẫu đạt trên 7 nhóm L (202 `KHOP` + 1 `KHOP_YEU`).
L29 và L24+L30 do máy khác chạy rồi gửi về. Mẫu đáng giá nhất là
`L24_V044/003.jpg` — video 26,44 fps duy nhất của kho, keyframe ứng với
`frame_idx=5` tức 0,19 giây, vẫn đạt corr 0,9997 và phân biệt được với hai
dòng kề. **Cả hai loại fps lạ nay đều đã kiểm và đạt** — 26,44 (1/1) và 29,97 (17/30).
Rủi ro fps đã đóng; chỗ nguy hiểm nhất còn lại là keyframe trùng lặp (A5.6).

`KHOP_YEU` là phán quyết mới: tương quan pixel thấp nhưng vượt xa dòng kề.
Hiệu chuẩn trên L22 — mẫu tệ nhất `L22_V013/116.jpg` chỉ đạt corr 0,675
nhưng là đoạn đồ họa chuyển cảnh, và đồng hồ trên hình đọc cùng `18:36:43`
với frame trích từ video, tức ghép đúng giây. Tương quan pixel tuyệt đối
không chịu được cảnh động; biên độ so với dòng kề mới là dấu hiệu đáng tin.

**5 — đạt.** Xem [so_lieu_giai_doan_0.md](so_lieu_giai_doan_0.md).

**6, 7, 8 — việc của người, không phải của máy.** Ba gạch này không tự động
hóa được:

- **6** — `index/` hiện 395 MB. Đẩy Drive. *Nhưng* nên đợi tải xong hết
  keyframes rồi dựng lại một lần, tránh 6 người tải về bản dở dang.
- **7** — Khánh gửi email BTC hỏi độ rộng cửa sổ `[s,e]` của KIS/Q&A.
- **8** — chốt bảng tên TV1..TV5 + Khánh, xóa chỗ ghi lệch trong kế hoạch v1.

## Việc kiểm tra chéo bằng mắt

Hướng dẫn yêu cầu mỗi người mở `verify_report.csv`, chọn 3 dòng `KHOP`, mở
ảnh keyframe và trích frame tương ứng bằng tay để nhìn tận mắt. Đây là lần
**duy nhất** kiểm tra bằng mắt — để tin vào công cụ. Từ đó về sau tin con số.

Lệnh trích frame thủ công:

```powershell
# ví dụ dòng L21_V030 / 004.jpg / pts_time=11.7
ffmpeg -ss 11.7 -i "C:\Code\aic_data\Videos_L21_a\video\L21_V030.mp4" -frames:v 1 kiemtra.png -y
start kiemtra.png
start "C:\Code\aic_data\Keyframes_L21\keyframes\L21_V030\004.jpg"
```

## Sau khi tải thêm dữ liệu

Chạy lại đủ 3 bước (không dùng `--objects-only`, vì bảng cái phải dựng lại
để nhận ảnh keyframe mới), rồi `02_verify.py` sẽ tự lấy mẫu trên các nhóm L
mới có video.
