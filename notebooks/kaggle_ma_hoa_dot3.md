# Kaggle — mã hoá đề Sơ tuyển đợt 3 (ngày thi)

**Mục tiêu: từ lúc có đề tới lúc có file `.npz` tải về, dưới 5 phút.**

Notebook Kaggle mới → **Settings → Accelerator: None** (mã hoá văn bản không cần
GPU, và chọn GPU thì hàng đợi lâu hơn). Internet: **On**.

Hai cell. **Cell 1 chạy TRƯỚC khi có đề** — nó cài đặt *và tự thử mã hoá một
câu*, nên mọi lỗi môi trường lộ ra lúc còn thời gian. Cell 2 sinh sẵn từ thư
mục đề, chỉ copy–dán.

---

## Cell 1 — CHẠY TRƯỚC khi có đề (~3 phút)

```python
# ⚠️ GHIM `transformers < 5`. Kaggle nay ship transformers 5.x, mà bản đó BỎ
# `batch_encode_plus` — hàm open_clip 2.32.0 gọi để tokenize. Không ghim thì
# cell 2 chết đúng lúc đã có đề, với lỗi
#     AttributeError: TokenizersBackend has no attribute batch_encode_plus
# Đã cắn thật ở đợt 3.
!pip -q install "open_clip_torch==2.32.0" "transformers<5" 2>&1 | tail -3

!git clone -q --depth 1 -b giai-doan-0 https://github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git /kaggle/working/aic
%cd /kaggle/working/aic

# Cạnh ma trận phải có file .json để encoder biết dùng ĐÚNG model. Thiếu nó thì
# nó rơi về model mặc định và sinh vector SAI SỐ CHIỀU — `KenhAnhCache` sẽ từ
# chối, nhưng chỉ phát hiện được khi đã về tới máy nhà.
import json, pathlib
pathlib.Path("index").mkdir(exist_ok=True)
pathlib.Path("index/clip_gopt.json").write_text(json.dumps({
    "model": "ViT-gopt-16-SigLIP2-384",
    "pretrained": "webli",
    "chieu": 1536,
}), encoding="utf-8")

# ------------------------------------------------------------------
# THỬ MÃ HOÁ THẬT MỘT CÂU — đây mới là phần đáng giá của cell này.
#
# Cài xong không có nghĩa là chạy được: lỗi transformers ở trên xảy ra ở
# TOKENIZER, tức sau khi `import open_clip` đã thành công. Chỉ có gọi thật
# `tok(...)` rồi `encode_text(...)` mới biết môi trường lành.
# ------------------------------------------------------------------
import transformers, open_clip, torch
print("transformers", transformers.__version__, "| open_clip", open_clip.__version__)

model, _, _ = open_clip.create_model_and_transforms(
    "ViT-gopt-16-SigLIP2-384", pretrained="webli")
tok = open_clip.get_tokenizer("ViT-gopt-16-SigLIP2-384")
model.eval()

with torch.no_grad():
    v = model.encode_text(tok(["Một người phụ nữ đang thái cà chua trên thớt gỗ"]))
assert v.shape[-1] == 1536, f"SAI SỐ CHIỀU: {v.shape} — sai cặp model/ma trận"
print(f"✅ SẴN SÀNG — mã hoá thử OK, vector {tuple(v.shape)}")
del model, tok
```

Phải thấy dòng `✅ SẴN SÀNG`. Thấy rồi thì môi trường lành và trọng số đã nằm
trong cache đĩa — cell 2 nạp lại tức thì, không tải mạng nữa.

**Nếu cell 1 lỗi**, xem mục *Sự cố* ở cuối.

---

## Cell 2 — SINH SẴN, không gõ gì

Ở máy nhà, sau khi giải nén đề vào `dev\SOTUYEN3-bo-de-thi\`:

```powershell
.venv\Scripts\python.exe scripts\122_sinh_cell_kaggle.py --de dev\SOTUYEN3-bo-de-thi
```

Ra `dev\SOTUYEN3-bo-de-thi\_cell_kaggle.py`. **Mở file đó, copy toàn bộ, dán
vào cell 2, chạy.**

Cell tự ghi đề ra đĩa → mã hoá → **tiền kiểm** → đóng gói `dot3.zip`. Nó
`assert` sau mỗi bước nên hỏng là dừng, không im lặng chạy tiếp.

> **Vì sao không dán tay từng câu.** 36 gói tiếng Việt có dấu; mỗi lần dán là
> một lần có thể lệch một ký tự — mà lệch một ký tự là **một chuỗi khác**, là
> trượt cache, là đúng lỗi làm mất trắng `p2-22` ở Sơ tuyển 2. Script nhúng nội
> dung dưới dạng JSON và tự kiểm: 36/36 gói khớp từng byte với file gốc.

Chỉ tải `dot3.zip` về khi thấy `✅ ĐỦ — kênh 1 sẽ chạy cho MỌI truy vấn`.

---

## Về tới máy nhà — ba lệnh

```powershell
Expand-Archive dot3.zip -DestinationPath C:\Code\aic2026 -Force

.venv\Scripts\python.exe scripts\120_gop_cache.py `
    index\truy_van_gopt.npz index\truy_van_dot3.npz

.venv\Scripts\python.exe scripts\119_kiem_truy_van.py --de dev\SOTUYEN3-bo-de-thi
```

Lệnh cuối in `✅ ĐỦ` thì mới chạy bài nộp. Chốt này chạy thử trên đề Sơ tuyển 2
đã **bắt đúng** `query-p2-22-kis` — câu duy nhất đợt đó mất điểm vì vận hành.

---

## Sự cố

| lỗi | nguyên nhân | xử lý |
| --- | --- | --- |
| `TokenizersBackend has no attribute batch_encode_plus` | `transformers` 5.x, open_clip 2.32 cần 4.x | đã ghim `"transformers<5"` ở cell 1. Nếu vẫn lỗi: chạy lại cell 1 rồi **Run → Restart & Clear Cell Outputs**, chạy lại từ đầu |
| `SAI SỐ CHIỀU` ở cell 1 | thiếu/sai `index/clip_gopt.json` | chạy lại cell 1 nguyên vẹn, đừng bỏ đoạn ghi json |
| Cell 1 lâu ở bước tải trọng số | mạng Kaggle | bình thường ~90 giây; chờ, đừng ngắt giữa chừng |
| `❌ THIẾU` ở bước tiền kiểm | có gói hụt chuỗi | đọc tên gói nó in ra, kiểm file `.txt` tương ứng rồi sinh lại cell 2 |
| Kaggle chết giữa chừng | — | mở notebook mới, chạy lại cell 1 + cell 2. **Đừng gõ lại đề bằng tay** |

Không có Kaggle thì Colab chạy được cùng hai cell (đổi `/kaggle/working` thành
`/content`).
