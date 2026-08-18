# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Hệ thống truy hồi video cho **AI Challenge HCMC 2026** (Textual KIS, Q&A, TRAKE).
Kho 873 video / 177.321 keyframe, nhóm 6 người, repo **công khai**.

**Mã nguồn, docstring, tài liệu và commit đều viết bằng tiếng Việt.** Tên hàm và
biến cũng vậy (`tim`, `cham`, `hop_nhat`, `be_chung`). Giữ nguyên quy ước đó.
Docstring dùng tiếng Việt **có dấu**; commit message dùng tiếng Việt **không
dấu** (ASCII) để hiển thị đúng trên mọi terminal.

## Lệnh

```powershell
.venv\Scripts\python.exe -m pytest tests -q              # toàn bộ (~1 phút)
.venv\Scripts\python.exe -m pytest tests/test_bm25.py -q # một file
.venv\Scripts\python.exe -m pytest tests -q -k "bigram"  # lọc theo tên test
```

**Luôn gọi `.venv\Scripts\python.exe`.** `python` trần trỏ vào Python hệ thống,
ở đó **không có `pyarrow`** nên mọi thứ đọc `master.parquet` sẽ chết giữa chừng
— và có thể chết *sau khi* đã ghi đè file.

Không có linter/formatter được cấu hình. Không có `pytest.ini`; test tự thêm
`src/` vào `sys.path` và tự `skipif` khi thiếu `index/`.

Terminal Windows mặc định cp1252 → in tiếng Việt sẽ `UnicodeEncodeError`. Đặt
`$env:PYTHONIOENCODING = "utf-8"`.

### Script đo (thứ chạy nhiều nhất)

```powershell
.venv\Scripts\python.exe scripts\15_do_bm25.py --nut        # kênh 2 metadata
.venv\Scripts\python.exe scripts\16_do_rrf.py               # hợp nhất kênh
.venv\Scripts\python.exe scripts\13_do_dedup.py --doi-chung # dedup
.venv\Scripts\python.exe src\tap_dev.py --kiem              # soát tập dev
```

`scripts/` đánh số theo thứ tự ra đời; `00`–`07` là Giai đoạn 0 (đã đóng), từ
`08` trở đi là Giai đoạn 1. Đặt script mới với số tiếp theo.

## Kiến trúc

**Mọi kênh truy hồi trả về `list[Candidate]`** (`src/schema.py`). Nhờ vậy `rrf.py`
không cần biết kênh nào sinh ra danh sách nào, và thêm kênh thứ sáu không phải
sửa dòng nào trong hợp nhất.

| Kênh | Module | Nguồn tín hiệu |
| --- | --- | --- |
| 1 — ảnh | `src/dense.py` | `clip.npy` / `clip_siglip2.npy` |
| 2 — metadata | `src/bm25.py` (`tu_metadata`) | title + description + keywords, **cấp video** |
| 3 — OCR/ASR | `src/bm25.py` (`tu_bang_khung`) | chờ bảng `(row_id, text)` |
| 4 — objects | `src/objects.py` | `objects.parquet` + IDF + bảng nhãn Việt–Anh |
| 5 — caption | `src/bm25.py` (`tu_bang_khung`) | `scripts/14_sinh_caption.py` |

`src/bm25.py` là **một bộ máy văn bản dùng chung cho ba kênh** — viết ba lần là
ba lần mắc lại cùng những lỗi tiếng Việt (dấu, từ ghép).

`src/dense.py` **không dính vào model nào**: số chiều đọc từ ma trận, tên model
đọc từ file `.json` cạnh nó. Đổi model = đổi tham số `--matrix`.

Hạ tầng: **không Milvus / Postgres / Elasticsearch**. Brute-force 177k vector
mất 16,7 ms — đo rồi. `master.parquet` + `numpy` là đủ.

## Kỷ luật đo — điều quan trọng nhất trong repo này

Dự án này chạy bằng **số đo, không bằng trực giác**, và đó không phải khẩu hiệu:
nhiều ý tưởng nghe rất hợp lý đã bị chính phép đo bác bỏ (dedup, RRF thô, hợp
nhất hai tầng — xem A11, A14, A14.1 trong kế hoạch). **Không tính năng nào được
bật mặc định trước khi thắng trên tập dev.**

`src/cham_diem.py` là thước đo, cài đúng công thức BTC
(`Final Score = trung bình R@{1,5,20,50,100}`). Bốn thứ bắt buộc:

- **`bao_cao_do_nhay()` chứ không phải điểm trung bình.** Nó chấm ở hai mức dung
  sai và tự kết luận `✅ ON DINH` / `🟡 YEU` / `❌ DAO DAU` / `⚪ KHÔNG ĐỔI GÌ`.
  Đảo dấu giữa hai mức = **không kết luận được**, không phải "hơi hơn".
- **So theo cặp, luôn báo thắng–thua–hòa.** Với ~97 câu, chênh lệch dưới ngưỡng
  nhiễu (in kèm) là vô nghĩa.
- **Mốc nền phải là cấu hình MẠNH NHẤT hiện có**, không phải cái tiện tay đặt
  trước. So với cái yếu thì gần như luôn thắng, mà thắng vậy chẳng nói gì.
- **Chỉ đổi MỘT thứ mỗi lần.** Đã vấp nhiều lần: đổi hai thứ rồi quy công cho
  nhầm cái. Khi phải so nhiều nút, dò 2×2 (xem `scripts/17_*`).

Thêm một **nhóm đối chứng** khi kết quả trông quá đẹp. `13_do_dedup.py` là ví dụ:
trên truy vấn tiếng Việt dedup bỏ 58,4/100 ứng viên — trông như phát hiện lớn,
thật ra nó đo *cái hỏng của kênh 1*; trên truy vấn tiếng Anh chỉ 0,5/100.

### Tập dev và tập test

`dev/tap_dev.jsonl` (97 câu) + `dev/tap_test.jsonl` (20 câu, **giữ kín**). Câu
mới thì `--gop` bình thường; `gop()` tự loại câu đã ở tập test. **Không bao giờ
chạy lại `--tach-test`** (nó cũng tự từ chối). Rò tập test là loại hỏng không tự
lộ ra — không crash, chỉ khiến con số kiểm cuối thành vô nghĩa.

## Bẫy đã cắn thật

- **`frame_idx` là giá trị nộp cho BTC.** Luôn lấy từ cột `frame_idx` của bảng
  cái, **đừng tính lại từ `pts_time`** — làm tròn lệch 1 frame.
- **fps có 4 giá trị** (25 / 26,44 / 29,97 / 30). Cấm hardcode. Cửa sổ thời gian
  dùng `pts_time`, không dùng số frame.
- **CLIP phải là tag `ViT-B-32-quickgelu`.** Tag thường làm cosine tụt
  0,9913 → 0,9513 mà chỉ in một dòng cảnh báo lẫn trong log.
- **Khóa bể ứng viên khi so hai ma trận có độ phủ khác nhau** (`dense.be_chung`).
  Không khóa thì bể nhỏ hơn thắng vì lý do không liên quan tới chất lượng: đo
  được **+0,2833**, lớn gấp bốn lần thứ cần đo.
- **Đường dẫn trong `master.parquet` là tuyệt đối** của máy dựng index. Máy khác
  phải chạy `scripts/12_va_duong_dan.py`.
- **PowerShell không bung `*` cho chương trình ngoài.** `--gop thu_muc\*.jsonl`
  nhận nguyên chuỗi `*.jsonl` và gộp ra 0 câu **mà không báo lỗi**. Tự bung
  trong Python.
- **`Get-Content | Set-Content -Encoding utf8`** trên PS 5.1 đọc UTF-8 không BOM
  thành ANSI → hỏng hết tiếng Việt. Dùng công cụ Edit/Write, đừng sửa file bằng
  pipeline PowerShell.
- **Đừng đặt hằng số trùng tên.** `MOC = (1,5,20,50,100)` từng bị một `MOC` khác
  đè, khiến `diem_cau()` lặng lẽ tính trung bình R@2 và R@15. Điểm sai, không có
  gì báo, và suýt kéo theo một tài liệu sai.

## Dữ liệu và git

`index/` (~395 MB) và dữ liệu thô **không nằm trong git** — đồng bộ qua Google
Drive. Dữ liệu thô ở ngoài repo (`C:\Code\aic_data`).

`.gitignore` phải là **`index/*`** chứ không phải `index/`: git không cho `!` moi
file ra lại nếu cả thư mục cha đã bị loại. Ba file báo cáo text là ngoại lệ có
chủ ý.

**Repo công khai.** Không commit `index/`, `*.mp4`, `*.jpg`, `*.npy`,
`*.parquet`, khóa API. Đã từng lọt 15 ảnh keyframe vì một luật `.gitignore` ghi
tên thư mục cũ sau khi đổi tên — đổi tên thư mục thì soát lại `.gitignore`.

Nhánh: tách từ `dev`, mở PR, **không push thẳng `main`/`dev`**. Không dùng
`git push --force`.

## Tài liệu

`docs/Ke_hoach_AIC2026_v4.md` là **nguồn sự thật duy nhất** — mọi phép đo, quyết
định và lý do đều ở đó, đánh số `A1`…`A15`. Đọc **PHẦN A** (sự thật về dữ liệu),
**PHẦN C** (cơ chế điểm chi phối mọi thiết kế) và **PHẦN H** (việc làm ngay)
trước khi sửa gì.

Đo được điều gì mới thì **thêm một mục `A<n>` kèm số liệu**, và ghi cả thứ đã
thử mà **không** hiệu quả — phần lớn giá trị của tài liệu này nằm ở chỗ đó.
