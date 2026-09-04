# Kaggle — mã hoá đề Sơ tuyển đợt 3 (ngày thi)

**Mục tiêu: từ lúc có đề tới lúc có file `.npz` tải về, dưới 5 phút.**

Notebook Kaggle mới → **Settings → Accelerator: None** (mã hoá văn bản không cần
GPU, và chọn GPU thì hàng đợi lâu hơn). Internet: **On**.

Chỉ có **hai cell**. Cell 1 chạy trước khi có đề (chuẩn bị sẵn ~2 phút), cell 2
chạy khi đã có đề.

---

## Cell 1 — CHẠY TRƯỚC khi có đề (cài sẵn, ~2 phút)

```python
!pip -q install open_clip_torch==2.32.0 2>&1 | tail -2
!git clone -q --depth 1 -b giai-doan-0 https://github.com/QuocKhanhDev-it/AIC_2026_FirstDance.git /kaggle/working/aic
%cd /kaggle/working/aic

# Cạnh ma trận phải có file .json để encoder biết dùng ĐÚNG model.
# Không có nó thì nó rơi về MODEL_MAC_DINH và sinh vector SAI SỐ CHIỀU —
# `KenhAnhCache` sẽ từ chối, nhưng chỉ phát hiện được lúc về tới máy nhà.
import json, pathlib
pathlib.Path("index").mkdir(exist_ok=True)
pathlib.Path("index/clip_gopt.json").write_text(json.dumps({
    "model": "ViT-gopt-16-SigLIP2-384",
    "pretrained": "webli",
    "chieu": 1536,
}), encoding="utf-8")

# Nạp model TRƯỚC, lúc chưa có đề — đây là phần lâu nhất (~90 giây tải trọng số).
import open_clip, torch
m, _, _ = open_clip.create_model_and_transforms(
    "ViT-gopt-16-SigLIP2-384", pretrained="webli")
del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None
print("✅ SẴN SÀNG — trọng số đã nằm trong cache đĩa, cell 2 sẽ nạp lại tức thì")
```

> **Vì sao nạp model rồi vứt đi?** `open_clip` lưu trọng số vào `~/.cache`. Lần
> nạp thứ hai đọc từ đĩa, không tải mạng. Làm việc này lúc chưa có đề là đổi
> 90 giây chờ mạng lấy 0 giây lúc đang tính từng phút.

---

## Cell 2 — CHẠY KHI ĐÃ CÓ ĐỀ

Dán nội dung từng gói vào `DE` bên dưới. **Tên gói phải đúng dạng
`query-p3-<số>-<loại>`** với loại ∈ `kis` / `qa` / `trake` — `run.py` đọc LOẠI
CÂU TỪ TÊN FILE, đặt sai tên là xử lý sai kiểu câu mà không có gì báo.

```python
DE = {
"query-p3-1-kis": """
Dán nguyên văn câu hỏi vào đây, giữ đủ dấu tiếng Việt.
""",
"query-p3-2-qa": """
Câu hỏi Q&A dán vào đây.
""",
"query-p3-3-trake": """
Câu TRAKE dán vào đây, giữ nguyên các câu mô tả từng sự kiện.
""",
# … dán đủ 25-30 gói …
}

# ---------------------------------------------------------------- ghi ra đĩa
import pathlib
d = pathlib.Path("/kaggle/working/aic/dev/SOTUYEN3-bo-de-thi")
d.mkdir(parents=True, exist_ok=True)
for ten, nd in DE.items():
    nd = nd.strip()
    assert nd and "Dán nguyên văn" not in nd, f"{ten}: chưa dán nội dung thật"
    (d / f"{ten}.txt").write_text(nd, encoding="utf-8")
print(f"✅ ghi {len(DE)} gói")

# ------------------------------------------------------- mã hoá + TỰ KIỂM
!cd /kaggle/working/aic && python scripts/25_ma_hoa_truy_van.py \
    --de dev/SOTUYEN3-bo-de-thi \
    --matrix clip_gopt.npy \
    --ra index/truy_van_dot3.npz

!cd /kaggle/working/aic && python scripts/119_kiem_truy_van.py \
    --de dev/SOTUYEN3-bo-de-thi \
    --cache index/truy_van_dot3.npz

# Đóng gói cả .npz lẫn thư mục đề — thư mục đề phải mang về theo, vì máy nhà
# cần ĐÚNG những file .txt này (khác một dấu cách là khác chuỗi, là trượt cache).
!cd /kaggle/working/aic && zip -qr /kaggle/working/dot3.zip \
    index/truy_van_dot3.npz dev/SOTUYEN3-bo-de-thi
print("\n📦 /kaggle/working/dot3.zip — bấm Output ở cột phải để tải về")
```

**Chỉ tải về khi cell in `✅ ĐỦ — kênh 1 sẽ chạy cho MỌI truy vấn`.** Nếu nó in
`❌ THIẾU` thì có gói bị hụt chuỗi; đọc tên gói nó chỉ ra, sửa nội dung trong
`DE` rồi chạy lại cell 2.

---

## Về tới máy nhà — ba lệnh

```powershell
# 1. giải nén vào đúng chỗ (ghi đè thư mục đề nếu đã có)
Expand-Archive dot3.zip -DestinationPath C:\Code\aic2026 -Force

# 2. gộp cache đợt 3 vào cache chính — KHÔNG cần model
.venv\Scripts\python.exe scripts\120_gop_cache.py `
    index\truy_van_gopt.npz index\truy_van_dot3.npz

# 3. TIỀN KIỂM lần cuối, trên đúng cache sẽ dùng để nộp
.venv\Scripts\python.exe scripts\119_kiem_truy_van.py --de dev\SOTUYEN3-bo-de-thi
```

Bước 3 in `✅ ĐỦ` thì mới chạy bài nộp. Đây là chốt đã bắt được `p2-22` khi
chạy thử trên đề Sơ tuyển 2 — câu duy nhất của đợt đó mất điểm vì **vận hành**,
không phải vì mô hình.

---

## Nếu Kaggle hỏng giữa chừng

Đề đã nằm trong `DE` của cell 2, nên chép sang notebook mới rồi chạy lại cell 1
+ cell 2 là xong. **Đừng gõ lại đề bằng tay** — tiếng Việt có dấu gõ lại là
gần như chắc chắn lệch một ký tự, mà lệch một ký tự là một chuỗi khác, là trượt
cache, là đúng lỗi `p2-22`.

Không có Kaggle thì Colab chạy được cùng hai cell (đổi `/kaggle/working` thành
`/content`).
