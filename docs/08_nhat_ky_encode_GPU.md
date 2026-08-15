# Nhật ký encode GPU — máy Khánh

*Ghi 2026-08-14. Máy: RTX 2060 Super, Windows 10. Đọc kèm:
[06_ke_hoach_encode_GPU.md](06_ke_hoach_encode_GPU.md).*

Tài liệu này ghi lại **đã làm gì, kết quả gì, còn vướng gì** — để không phải
dò lại từ đầu nếu quay lại việc này sau vài ngày, và để người khác trong
nhóm biết máy này đang ở đâu.

---

## 1. Trạng thái theo bảng kế hoạch chính

| # | Việc | Trạng thái | Ghi chú |
| --- | --- | --- | --- |
| 1 | Viết `scripts/08_encode.py` | ✅ Xong (từ trước) | Có sửa thêm 2 chỗ nhỏ, xem §4 |
| 2 | Cài `torch` bản CUDA | ✅ Xong | `torch 2.6.0+cu124`, nhận đúng RTX 2060 SUPER |
| 3 | Loại `AIC_data` khỏi antivirus quét real-time | ✅ Xong | Không phải Defender mà là **Avast** — xem §3 |
| 4 | Tải keyframe 100 video phân tầng | 🟡 **Một phần** | Máy này có ảnh thật cho **123 video** (L26: 99/498, L28: 24/24) — 2/10 nhóm L, không phải phân tầng đủ 10 nhóm |
| 5 | Chạy thử 100 video | ✅ Xong | 23,2 ảnh/giây thật — xem §5 |
| 6 | Kiểm lệch hàng `row_id` | ✅ Xong | 134/134 cặp khớp (100%) — HÀNG KHỚP |
| 7 | Soạn tập dev trên 100 video | 🟡 **Chuẩn bị xong, chưa soạn câu** | 123 contact sheet đã tạo (`dev/sheets/`), `dev/tap_dev.jsonl` còn trống |
| 8 | Đo 3 cấu hình A/B/C | ⛔ Chưa làm | Chặn bởi việc 7 |
| 9 | Tải nốt 30,5 GB keyframe | ⛔ Chưa làm | Luật cứng: không bắt đầu trước khi việc 8 thắng |
| 10 | Chạy toàn kho | ⛔ Chưa làm | Chặn bởi việc 9 |
| 11 | Kiểm lệch hàng lần cuối + đẩy Drive | ⛔ Chưa làm | Chặn bởi việc 10 |

---

## 2. Dữ liệu trên máy này

`index/` đã đồng bộ từ Google Drive (đầy đủ `master.parquet`, `clip.npy`,
`objects.parquet`, `trung_lap.parquet`...), nhưng cột `kf_path` trong
`master.parquet` lúc tải về trỏ vào `C:\Code\aic_data\...` — đường dẫn của
máy khác, không tồn tại ở đây.

**Đã vá lại** bằng script riêng (không sửa `01_build_index.py`, đúng luật ở
§6.2 của kế hoạch): quét `D:\Project\AIC_2026\AIC_data`, tìm được ảnh thật
cho 123 video (L26_a: 99 video, L28: 24 video — toàn bộ L28), gán lại
`kf_path` cho đúng 24.918 dòng tương ứng, giữ nguyên `row_id`/`kf_n`/mọi cột
khác. 0 video bị lệch số ảnh so với số dòng CSV.

`AIC_data` hiện chỉ có `Keyframes_L26_a` (một phần L26) và `Keyframes_L28`
(đủ 24/24) — chưa có video, objects, media-info, map-keyframes CSV, hay
clip-features cho video nào (những thứ này đã có sẵn trong `index/` đồng bộ
từ Drive, không cần tải lại).

---

## 3. Sự cố môi trường đã gặp và cách sửa

| Sự cố | Nguyên nhân | Cách sửa |
| --- | --- | --- |
| `torch` cài xong nhưng `cuda.is_available()` → `False` | `requirements-clip.txt` cài bản PyPI mặc định, là bản CPU-only | Gỡ, cài lại từ `https://download.pytorch.org/whl/cu124` |
| `Add-MpPreference` lỗi `0x800106ba` | **Tamper Protection** của Windows chặn đổi exclusion qua PowerShell/CLI | Không dùng lệnh — phải thêm qua giao diện |
| Không thấy "Manage settings" trong Windows Security | Máy có **Avast Antivirus** cài song song → Defender lùi về chế độ chờ, Avast mới là cái quét thật | Thêm exception trong **Avast** (Settings → General → Exceptions), không phải Defender |
| Tải file lớn (torch 2,5 GB) bị `ConnectionResetError` giữa chừng | Mạng chập chờn thật, không phải lỗi cấu hình | Chuyển sang `curl -C -` (tự nối lại chỗ đứt) thay vì `pip install` (tải lại từ đầu mỗi lần lỗi) |
| `curl` báo `CRYPT_E_NO_REVOCATION_CHECK` | Avast can thiệp HTTPS, Windows không kiểm tra được revocation của cert | Thêm cờ `curl --ssl-no-revoke` |
| Python báo `CERTIFICATE_VERIFY_FAILED` khi tải qua `huggingface_hub` | Cùng nguyên nhân Avast — cert gốc của Avast không nằm trong danh sách tin cậy mặc định của Python (`certifi`) | Cài `pip install pip-system-certs` — Python chuyển sang dùng kho chứng chỉ của Windows |
| `huggingface_hub` treo ở 0 byte hàng chục phút, không báo lỗi | Không rõ — có thể do cách giữ kết nối không hợp mạng này | Bỏ qua `huggingface_hub`, tải thẳng file `.safetensors` bằng `curl -C -` |
| `UnicodeEncodeError` khi in tiếng Việt ra terminal | Terminal Windows mặc định cp1252 (bug đã biết, ghi trong README) | Set `PYTHONIOENCODING=utf-8` trước khi chạy |

---

## 4. Sửa trong `scripts/08_encode.py`

Khi nạp model từ **file cục bộ** (`--pretrained <đường dẫn file>` thay vì tag
như `webli`), `open_clip` không tự lấy được cấu hình tiền xử lý ảnh đi kèm
tag đó (kích cỡ ảnh 378×378) — sinh ra 2 lỗi kích cỡ ảnh. Đã sửa:

1. Thêm cờ `--image-size` để ép đúng kích cỡ khi cần (mặc định `None`, không
   ảnh hưởng cách chạy bằng tag như trong kế hoạch gốc).
2. Chỗ dò chiều vector (khi `model.visual` không có `output_dim`) từng
   hardcode ảnh giả `224×224` để thăm dò — đổi sang dùng đúng kích cỡ thật
   của model (`cx`), nếu không sẽ nổ ngay với model đòi kích cỡ khác 224.

Cả hai chỗ sửa đều không đụng tới CHỐT 1/2/3 (lệch hàng, sai checkpoint,
quên chuẩn hóa L2) đã có sẵn trong file.

---

## 5. Kết quả việc 5 — chạy thử 100 video

Lệnh chạy thật (khác bản gốc ở chỗ trỏ `--pretrained` vào file cục bộ vì
`huggingface_hub` bị treo — xem §3):

```powershell
python scripts/08_encode.py `
  --model ViT-SO400M-14-SigLIP2-378 `
  --pretrained "D:\Project\AIC_2026\models\ViT-SO400M-14-SigLIP2-378_webli_open_clip_model.safetensors" `
  --image-size 378 `
  --videos 100 --batch 16 --out index/clip_siglip2_thu.npy
```

| Chỉ số | Giá trị |
| --- | --- |
| Chiều vector | 1152 |
| Video chọn (phân tầng 10 nhóm L) | 100 |
| Dòng "chưa tải ảnh" (không có ảnh trên máy này) | 21.161 |
| Ảnh thật đã encode | 5.919 (từ 20 video nằm trong L26/L28) |
| **Tốc độ thật** | **23,2 ảnh/giây** (kế hoạch ước ~26 ảnh/s — sát) |
| Ước thời gian toàn kho | ~2,1 giờ (kế hoạch ước ~1h55 — sát) |
| File ra | `index/clip_siglip2_thu.npy` — `(177321, 1152)` float16, 390 MB |
| File ghi chú | `index/clip_siglip2_thu.json` |

## 6. Kết quả việc 6 — kiểm lệch hàng

```powershell
python scripts/08_encode.py --kiem-lech-hang index/clip_siglip2_thu.npy
```

**134/134 cặp keyframe trùng lặp khớp (100%)**, trung vị cosine mới = 0,9977.
→ **✅ HÀNG KHỚP. Ma trận dùng được.**

---

## 7. Chuẩn bị việc 7

Đã tạo 123 contact sheet (ảnh ghép để duyệt bằng mắt, đúng cách làm ở
`docs/07_lam_tap_dev.md` §2):

```powershell
python scripts/10_contact_sheet.py --nhom L26 --thua 10   # 99 file
python scripts/10_contact_sheet.py --nhom L28 --thua 10   # 24 file
```

Nằm ở `dev/sheets/*.jpg` (không đẩy git, xem `.gitignore`). Đã tìm thử xem
BTC có công bố bộ câu hỏi AIC'25 kèm đáp án không (việc 1 của
`07_lam_tap_dev.md`) — **không có**, chỉ có "website thử nghiệm" để làm quen
cách mô tả câu, không tải được ground truth. Vẫn phải tự soạn theo §3-§5 của
tài liệu đó.

**Việc còn lại của việc 7** (chưa làm): duyệt 123 sheet, chốt 4 quy ước ở §3,
soạn 30–50 câu vào `dev/tap_dev.jsonl` theo mẫu `dev/tap_dev_mau.jsonl`.

---

## 8. File model — lưu ý vị trí

Trọng số model (`open_clip_model.safetensors`, 4,4 GB) đã tải bằng `curl`
**không nằm trong cache mặc định của `huggingface_hub`** (do phải né chỗ bị
treo — xem §3), nên `--pretrained webli` sẽ **không tự thấy** nó ở lần chạy
sau. Đã copy sang vị trí cố định, ngoài mọi thư mục tạm:

```
D:\Project\AIC_2026\models\ViT-SO400M-14-SigLIP2-378_webli_open_clip_model.safetensors
```

Lần chạy **việc 10** (toàn kho) sau này: dùng lại đúng 3 cờ
`--pretrained <đường dẫn trên> --image-size 378`, không cần tải lại.

---

## 9. Việc tiếp theo, đúng thứ tự

1. **Việc 7** — duyệt `dev/sheets/`, soạn `dev/tap_dev.jsonl` (việc của
   Khánh, cần mắt người, xem §4-§5 của `07_lam_tap_dev.md`)
2. Việc 8 — đo A/B/C, cần việc 7 xong
3. Việc 9 — tải nốt 30,5 GB, **chỉ sau khi việc 8 thắng**
4. Việc 10 — chạy toàn kho (dùng lại model đã lưu ở §8)
5. Việc 11 — kiểm lệch hàng lần cuối + đẩy Drive
