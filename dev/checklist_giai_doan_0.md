# Bảng kiểm tra Giai đoạn 0 — cập nhật 2026-08-05

Theo PHẦN 6 của `HUONG_DAN_GIAI_DOAN_0.md`. Đủ 8 gạch mới mở Giai đoạn 1.

| | Mục | Trạng thái |
| --- | --- | --- |
| 1 | `paths.parquet` phủ ≥ 99% video ở `csv`, `clip`, `keyframe_dir` | ⚠️ **một phần** |
| 2 | `master.parquet` dựng xong, không còn lỗi `lech_so_vector` | ✅ **đạt** |
| 3 | `clip.npy` shape `(N, 512)`, `N` = số dòng `master` | ✅ **đạt** |
| 4 | `verify_report.csv` cho ≥ 95% `KHOP` | ✅ **đạt — 100%** |
| 5 | Ghi lại 4 con số vào tài liệu nhóm | ✅ **đạt** |
| 6 | `index/` đã lên Drive, cả 6 người tải được | ⬜ **chưa** |
| 7 | Câu hỏi về cửa sổ `[s,e]` đã gửi BTC | ⬜ **chưa** |
| 8 | Chốt một bảng tên thành viên duy nhất | ⬜ **chưa** |

## Chi tiết

**1 — một phần.** `csv` và `clip` phủ **100%** (873/873). `keyframe_dir` mới
**3,3%** (29/873) vì mới tải `Keyframes_L21`. Không phải lỗi, chỉ là chưa tải
xong. Gạch này sẽ tự đạt khi tải hết các gói Keyframes.

**2 — đạt.** `problems.csv` có 844 dòng nhưng **toàn bộ là `lech_so_keyframe`**
(chưa tải ảnh). **Không có dòng `lech_so_vector` nào** — đây mới là lỗi
nghiêm trọng mà hướng dẫn cảnh báo.

**3 — đạt.** `clip.npy` = `(177321, 512)` float32 chuẩn L2. `master.parquet`
= 177.321 dòng. Khớp chính xác.

**4 — đạt.** 29/29 mẫu `KHOP`, tương quan **0,990–1,000**. Ngưỡng chỉ cần
0,95 nên biên an toàn rất rộng. Lưu ý chỉ chạy được 29 mẫu (không phải 60)
vì chỉ 29 video có cả ảnh keyframe lẫn video gốc.

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
