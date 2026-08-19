# Nhật ký encode GPU — máy Khánh

*Ghi 2026-08-14, cập nhật 2026-08-17, 2026-08-19. Máy: RTX 2060 Super, Windows
10. Đọc kèm: [06_ke_hoach_encode_GPU.md](06_ke_hoach_encode_GPU.md) và mục
H2/H3 của [Ke_hoach_AIC2026_v4.md](Ke_hoach_AIC2026_v4.md).*

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
| 7 | Soạn tập dev trên 100 video | ✅ **Xong (cả team)** | **117 câu, đủ 10 nhóm L** — 97 vào `dev/tap_dev.jsonl`, 20 giữ kín ở `dev/tap_test.jsonl`. Xem §7 |
| 8 | Đo 3 cấu hình A/B/C | 🟢 **Đã có câu trả lời sớm** | A10.3: SigLIP2 + tiếng Việt **thắng CLIP 21/21 câu** (đo trên CPU, subset video theo tập dev) — **B thắng A đã CHỨNG MINH**, không còn là giả thuyết. Còn thiếu: xác nhận trên ma trận toàn kho |
| 9 | Tải nốt 30,5 GB keyframe | ✅ **873/873 video** | Xong — `L26_d` (100 video cuối) tải xong 2026-08-19 |
| 10 | Chạy toàn kho | ✅ **Xong 2026-08-19** | Chạy 2 lượt: 773 video trước (160.821 vector) + vá riêng `L26_d` sau (16.500 vector) — xem §9 và §11 |
| 11 | Kiểm lệch hàng lần cuối | ✅ **Xong — HÀNG KHỚP** | **177.321/177.321 dòng đều có vector thật.** Lần kiểm cuối: 198/200 (99,0%), trung vị cosine 0,9993 |
| — | Đẩy `clip_siglip2.npy` lên Drive | ⛔ Chưa làm | Xem §12 |

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

**Cập nhật 17/08:** team đã viết `scripts/12_va_duong_dan.py` — bản chính
thức thay cho script vá tay ở trên, sửa đúng lỗi này (đặt tên A5.5, đã cản
trở ba lần). Có sao lưu tự động (`master.parquet.truoc_khi_va`) và tự kiểm
`row_id`/số dòng/mọi cột khác không đổi trước khi cho ghi. Đã chạy lại bằng
bản này (`--ghi`): xác nhận 24.918 dòng ảnh khớp y hệt bản vá tay, đồng thời
dọn sạch 16.896 dòng `video_path` rác (trỏ vào `.mp4` không tồn tại trên máy
này, tồn dư từ máy khác). **Dùng script này cho mọi lần vá đường dẫn sau
này**, không dùng lại cách vá tay thủ công nữa.

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

**Cập nhật 17/08:** cả team đã duyệt xong — 117 câu, đủ 10 nhóm L (không chỉ
riêng L26/L28 ở trên), tách sẵn 20 câu giữ kín. Việc 7 coi như ✅ xong.

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

## 9. Việc 10-11 — ĐÃ XONG (2026-08-19)

**Chạy khi còn thiếu đúng 1 nhóm `L26_d`** (~100/873 video, ~16.500 keyframe)
— không đợi đủ 873/873, vì chốt an toàn "thiếu ảnh → vector 0" của
`08_encode.py` làm việc này an toàn (xem §10 tại sao).

| Chỉ số | Giá trị |
| --- | --- |
| Model | `ViT-SO400M-14-SigLIP2-378` / `webli` |
| Chiều | 1152 |
| Tổng dòng | 177.321 |
| Vector thật (có ảnh) | **160.821** (90,7%) |
| Vector 0 (thiếu `L26_d`) | 16.500 (9,3%) |
| Tốc độ thật | **23,9 ảnh/giây** (ổn định dần từ 23,1 lúc đầu) |
| Thời gian chạy thật | ~1 giờ 55 phút |
| File ra | `index/clip_siglip2.npy` — 390 MB |
| **Kiểm lệch hàng (việc 11)** | **198/200 cặp khớp (99,0%)**, trung vị cosine 0,9994 → **✅ HÀNG KHỚP** |

Sự cố gặp trong lượt chạy này: ổ C gần đầy giữa chừng (186 MB trống rồi tụt
xuống 186 MB → dọn `scratchpad/whl` cũ (torch wheel + bản sao model, 6,6 GB,
dư thừa vì đã cài/copy xong) giải phóng được ~6,8 GB. Sau đó ổ C lại gần đầy
lần nữa — nguyên nhân thật là **`pagefile.sys` giãn lên 26,9 GB** do RAM bị
dồn ép lúc nạp model + 8 luồng đọc ảnh cùng lúc (máy 16 GB RAM). Không nguy
hiểm (RAM vẫn còn ~5 GB trống), không đụng vào vì sửa cần quyền admin + khởi
động lại, có thể làm gián đoạn tiến trình đang chạy.

**Việc còn lại:** khi tải được `L26_d`, xem §11 — vá riêng phần đó vào ma
trận, không chạy lại từ đầu.

---

## 9b. Lệnh gốc đã dùng (tham khảo cho lần chạy lại từ đầu, ví dụ trên máy khác)

Sau khi keyframe 873/873 video đã tải đủ vào `AIC_data`, chạy theo đúng thứ
tự sau (đừng bỏ bước 1 — kf_path cần vá lại cho các nhóm L vừa tải thêm):

```powershell
# 1. Vá lại kf_path cho các nhóm L mới tải (bắt buộc, xem §2)
python scripts\12_va_duong_dan.py --ghi

# 2. Encode toàn kho — không giới hạn --videos, ước ~2 giờ
python scripts\08_encode.py `
  --model ViT-SO400M-14-SigLIP2-378 `
  --pretrained "D:\Project\AIC_2026\models\ViT-SO400M-14-SigLIP2-378_webli_open_clip_model.safetensors" `
  --image-size 378 `
  --batch 16 --out index\clip_siglip2.npy

# 3. Việc 11 — kiểm lệch hàng lần cuối (BẮT BUỘC trước khi tin kết quả)
python scripts\08_encode.py --kiem-lech-hang index\clip_siglip2.npy
```

Có checkpoint mỗi ~10.000 dòng (`08_encode.py`) — mất điện giữa chừng thì
chạy lại đúng lệnh trên, tự tiếp tục chứ không làm lại từ đầu.

Sau khi việc 11 báo ✅ HÀNG KHỚP: đẩy `index/clip_siglip2.npy` +
`index/clip_siglip2.json` lên Drive (xem bảng bàn giao ở
[06_ke_hoach_encode_GPU.md §7](06_ke_hoach_encode_GPU.md#7-bảng-bàn-giao)),
rồi báo cả nhóm để chạy việc 8 (đo A/B/C xác nhận trên toàn kho) và mở khóa
H3 của kế hoạch v4.3.

---

## 10. Vì sao chạy trước khi đủ 873/873 là an toàn

`08_encode.py` có CHỐT 1: video/keyframe chưa có ảnh trên máy này thì ghi
**vector 0** vào đúng vị trí dòng, không bỏ dòng, không dịch hàng. Vector 0
nghĩa là cosine 0 với mọi truy vấn → không bao giờ được truy hồi, nhưng cũng
không sai lệch gì. Nên chạy sớm khi thiếu 1 nhóm L (`L26_d`, 9,3%) chỉ đánh
đổi **độ phủ tạm thời thấp hơn**, không đổi lấy rủi ro đúng/sai.

## 11. Vá `L26_d` — ĐÃ XONG (2026-08-19)

Không chạy lại nguyên lệnh việc 10 — `08_encode.py` đánh dấu dòng thiếu ảnh là
**`xong = True`** (vector 0) ngay từ lượt chạy trước, nên chạy lại đúng lệnh
cũ trên cùng file `--out` **sẽ không encode lại** các dòng đó dù ảnh đã có.
Thay vào đó, làm 3 bước, cả 3 đều đã chạy thật và đã dùng được:

**1. Vá `kf_path`** (script cũ, không đổi):

```powershell
python scripts\12_va_duong_dan.py --ghi
```

**2. Encode RIÊNG chỉ các video mới** — thêm cờ mới `--chi-video` vào
`08_encode.py` (nhận file danh sách `video_id`, mỗi dòng một cái; hàm
`chon_theo_danh_sach()`), ra file tạm, **không đụng `clip_siglip2.npy`**:

```powershell
python scripts\08_encode.py `
  --model ViT-SO400M-14-SigLIP2-378 `
  --pretrained "D:\Project\AIC_2026\models\ViT-SO400M-14-SigLIP2-378_webli_open_clip_model.safetensors" `
  --image-size 378 --batch 16 `
  --chi-video danh_sach_video_moi.txt `
  --out index\clip_siglip2_L26d_va.npy
```

Kết quả thật: 16.500/16.500 dòng có vector khác 0, ~22,6 ảnh/giây, ~12 phút.

**3. Ghép vào ma trận chính** — script mới `scripts/18_ghep_ma_tran.py`.
Nguyên lý: dòng nào trong ma trận vá có vector khác 0 thì ghi đè vào ma trận
chính; dòng khác giữ nguyên tuyệt đối (script tự kiểm bằng
`np.array_equal`, giống tinh thần `12_va_duong_dan.py`). Mặc định chỉ xem
trước, thêm `--ghi` để ghi thật, tự sao lưu bản cũ và tự cập nhật `.json` đi
kèm.

```powershell
python scripts\18_ghep_ma_tran.py --chinh index\clip_siglip2.npy `
  --va index\clip_siglip2_L26d_va.npy --ghi
```

Kết quả thật: **+16.500 dòng mới, 0 dòng bị ghi đè nhầm** → tổng
**177.321/177.321 dòng có vector thật**. Kiểm lệch hàng lại lần cuối
(**bắt buộc** sau mọi lần ghép) — **✅ HÀNG KHỚP**, 198/200 (99,0%), trung vị
cosine 0,9993. Đã dọn file tạm (`_L26d_va.*`, `.truoc_khi_ghep`).

`scripts/18_ghep_ma_tran.py` dùng lại được cho bất kỳ lô vá nào sau này
(không riêng `L26_d`) — chỉ cần encode riêng lô mới bằng `--chi-video` rồi
ghép, không phải bao giờ cũng chạy lại 2 giờ từ đầu.

---

## 12. Việc tiếp theo, đúng thứ tự

1. ~~Việc 7~~ — ✅ xong (cả team, 117 câu)
2. ~~Việc 8~~ — 🟢 đã có câu trả lời sớm (A10.3), còn xác nhận trên toàn kho
3. ~~Việc 9~~ — ✅ 873/873 video
4. ~~Việc 10~~ — ✅ xong 2026-08-19, `index/clip_siglip2.npy` (177321, 1152)
5. ~~Việc 11~~ — ✅ HÀNG KHỚP, **177.321/177.321 dòng có vector thật**
6. ~~Vá `L26_d`~~ — ✅ xong, xem §11
7. **Đẩy `clip_siglip2.npy` + `.json` lên Drive** — chưa làm, xem §7 của
   [06_ke_hoach_encode_GPU.md](06_ke_hoach_encode_GPU.md#7-bảng-bàn-giao)
8. Báo nhóm chạy việc 8 (đo A/B/C xác nhận trên toàn kho, mở khóa H3 của
   `Ke_hoach_AIC2026_v4.md`)
