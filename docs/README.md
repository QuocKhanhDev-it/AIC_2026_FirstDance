# Tài liệu cho thành viên nhóm

Đọc theo thứ tự. Ai mới vào chỉ cần file 01 là chạy được.

| File | Dành cho ai | Nội dung |
| --- | --- | --- |
| [Ke_hoach_AIC2026_v4.md](Ke_hoach_AIC2026_v4.md) | **cả nhóm** | **Kế hoạch hiện hành.** Thay thế v3 — PHẦN A đo lại trên toàn bộ dữ liệu |
| [01_cai_dat.md](01_cai_dat.md) | **mọi người, đọc trước tiên** | Từ `git clone` tới chạy được lệnh đầu tiên |
| [02_bang_cai.md](02_bang_cai.md) | ai đụng tới dữ liệu | Bảng cái có gì, truy vấn thế nào, cột nào nghĩa gì |
| [03_quy_uoc_git.md](03_quy_uoc_git.md) | mọi người | Nhánh, commit, PR — để 6 người không giẫm chân nhau |
| [04_loi_hay_gap.md](04_loi_hay_gap.md) | khi có gì đó hỏng | Các lỗi đã gặp thật và cách sửa |

## Bạn có cần tải dữ liệu không?

Câu hỏi này quyết định bạn tốn 5 phút hay 3 tiếng.

| Bạn làm gì | Cần tải |
| --- | --- |
| Truy vấn bảng cái, thử ý tưởng, viết code xếp hạng | **chỉ `index/`** (~400 MB) |
| Cần nhìn ảnh keyframe | `index/` + Keyframes (~15–30 GB) |
| Làm module trích dày, cắt video, kiểm chứng | thêm Videos (**200–400 GB**) |

**Chỉ MỘT người trong nhóm cần tải video.** Nói rõ điều này trước khi cả
nhóm mỗi người tải 300 GB.

## Bạn cần cài thư viện gì?

| Bạn làm gì | Cài gì | Nặng |
| --- | --- | --- |
| Truy vấn bảng cái, code xếp hạng, phân tích | `requirements.txt` | ~50 MB |
| Chạy `02_verify.py` | thêm ffmpeg | ~150 MB |
| Chạy `03_verify_CLIP.py` | thêm `requirements-clip.txt` | **~2,5 GB** |

Lệnh cụ thể ở [01_cai_dat.md](01_cai_dat.md). Đừng cài gói CLIP nếu không
chạy tới script đó.

## Trạng thái hiện tại

Giai đoạn 0 đã xong: bảng cái dựng từ 873 video, 177.321 keyframe, và đã
được `02_verify.py` chứng minh là ghép đúng (29/29 KHỚP).

**Nhưng mới tải keyframes + video của L21.** L22–L30 hiện chỉ có
csv/clip/objects/media-info. Xem [../dev/checklist_giai_doan_0.md](../dev/checklist_giai_doan_0.md)
để biết còn thiếu gạch nào.

Bốn con số phải nhớ: [../dev/so_lieu_giai_doan_0.md](../dev/so_lieu_giai_doan_0.md)
