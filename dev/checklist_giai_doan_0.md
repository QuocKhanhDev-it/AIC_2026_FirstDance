# Bảng kiểm tra Giai đoạn 0 — **ĐÃ ĐÓNG** 2026-08-12

> **Giai đoạn 0 khép lại: 873/873 video (100%), 0 lệch chỉ số thật.**
> Tài liệu này giữ làm hồ sơ. Việc đang mở theo dõi ở
> [PHẦN H của kế hoạch](../docs/Ke_hoach_AIC2026_v4.md#phần-h--việc-làm-ngay),
> đừng theo dõi ở hai chỗ.

| | Mục | Trạng thái |
| --- | --- | --- |
| 1 | `paths.parquet` phủ ≥ 99% video ở `csv`, `clip`, `keyframe_dir` | ✅ **đạt** — `csv`/`clip` 100%; `keyframe_dir` phân tán theo máy, đúng thiết kế |
| 2 | `master.parquet` dựng xong, không còn lỗi `lech_so_vector` | ✅ **đạt** |
| 3 | `clip.npy` shape `(N, 512)`, `N` = số dòng `master` | ✅ **đạt** |
| 4 | `verify_report.csv` cho ≥ 95% `KHOP` | ✅ **đạt — 871/873 (99,8%)**, script 03 863/873 (98,9%) |
| 5 | Ghi lại 4 con số vào tài liệu nhóm | ✅ **đạt** |
| 6 | `index/` đã lên Drive, cả 6 người tải được | ✅ **đạt** — máy Khánh đã đồng bộ về và chạy được |
| 7 | Câu hỏi về cửa sổ `[s,e]` đã gửi BTC | 🟢 **có đáp án tạm** — luật AIC'25 chấm theo **khoảng** (A8.1). Còn chờ BTC xác nhận → PHẦN H mục 11 |
| 8 | Chốt một bảng tên thành viên duy nhất | ⬜ → PHẦN H mục 13 |

## Chi tiết

**1 — một phần.** `csv` và `clip` phủ **100%** (873/873). `keyframe_dir` trên
máy này mới **6,9%** (60/873) vì mới tải `Keyframes_L21` + `Keyframes_L22`.
Không phải lỗi, chỉ là chưa tải xong. Gạch này sẽ tự đạt khi tải hết các gói
Keyframes.

Tính cả kết quả thành viên khác gửi về thì **873/873 video — 100% — đã được
kiểm chứng**, đủ cả 10/10 nhóm L. Chạy `python scripts/07_gop_kiem_chung.py`
để xem bảng độ phủ mới nhất.

**2 — đạt.** `problems.csv` có 844 dòng nhưng **toàn bộ là `lech_so_keyframe`**
(chưa tải ảnh). **Không có dòng `lech_so_vector` nào** — đây mới là lỗi
nghiêm trọng mà hướng dẫn cảnh báo.

**3 — đạt.** `clip.npy` = `(177321, 512)` float32 chuẩn L2. `master.parquet`
= 177.321 dòng. Khớp chính xác.

**4 — đạt.** 871/873 mẫu đạt ở script 02 và 863/873 ở script 03.
**Không mẫu nào là lệch chỉ số thật** — mọi cảnh báo đều truy ngược được về
keyframe trùng lặp (A5.6) hoặc cụm frame liên tiếp (A5.7).

Hai phán quyết mới sinh ra từ quá trình này:

- `KHOP_YEU` (script 02) — tương quan pixel thấp nhưng vượt xa dòng kề. Hiệu
  chuẩn trên `L22_V013/116.jpg`: corr chỉ 0,675 nhưng đồng hồ trên hình đọc
  cùng `18:36:43` với frame trích từ video, tức ghép đúng giây.
- `KHOP_TRUNG_LAP` (script 03) — cosine ≥ 0,95 nhưng không đúng hạng 1 vì
  dòng thắng là bản sao của chính nó.

Cả hai loại fps lạ đã kiểm và đạt: 26,44 (1/1) và 29,97 (30/30).

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
