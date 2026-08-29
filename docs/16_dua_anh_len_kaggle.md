# 16 — Đưa ảnh keyframe lên Kaggle

Ảnh phải nằm sẵn trên Kaggle thì mới chạy được model nặng bằng GPU: mã hoá ảnh
bằng model thứ hai, sinh caption cho kênh 5, chạy VLM. Máy cá nhân không làm nổi
những việc đó.

Cả kho ≈ **34,5 GB / 177.321 ảnh**. Tài liệu này là cách đưa chúng lên, và cách
chia việc cho nhiều người.

---

## 0. Ai làm gì

Mỗi người đẩy **những nhóm L mình đang giữ ảnh**. Không ai phải tải về gói mà
mình chưa có — tải 16 GB rồi upload lại 16 GB là 32 GB đường truyền cho một
nhóm, trong khi người đang giữ nó chỉ tốn 16 GB.

Đẩy xong thì **add collaborator quyền `Can edit`** cho những người còn lại. Xem
mục 5 — đây là bước hay bị quên nhất, và quên là cả nhóm không dùng được.

---

## 1. Tài khoản

1. Đăng ký ở **kaggle.com**. Đăng nhập bằng Google cho nhanh.
2. **Settings → Phone Verification → xác thực số điện thoại.** Không xác thực
   thì Kaggle chặn phần lớn tính năng, kể cả bật Internet trong notebook.

---

## 2. Đăng nhập CLI

Cài vào chính venv của repo, đừng cài ra Python hệ thống:

```powershell
.venv\Scripts\python.exe -m pip install kaggle
```

Rồi đăng nhập. **Cách nhẹ nhất là OAuth — không phải tải file token nào:**

```powershell
.venv\Scripts\kaggle.exe auth login
```

Nó mở trình duyệt, bạn bấm đồng ý, credential được lưu lại. Xong.

> Nếu không muốn OAuth: vào **https://www.kaggle.com/settings/api** → mục
> **API** → **Generate New Token**, rồi đặt chuỗi token vào
> `C:\Users\<tên>\.kaggle\access_token`, hoặc `$env:KAGGLE_API_TOKEN = "…"`.
>
> Trang này **không nằm trong Settings chung** — Kaggle đã tách ra URL riêng, đó
> là lý do hay không tìm thấy.

Kiểm đăng nhập được chưa — lệnh này phải in ra bảng, không báo lỗi:

```powershell
.venv\Scripts\kaggle.exe datasets list --mine
```

---

## 3. Nén ảnh theo nhóm

```powershell
$env:PYTHONIOENCODING = "utf-8"

# xem trước: nhóm nào có ảnh, mỗi nhóm bao nhiêu MB
.venv\Scripts\python.exe scripts\45_dong_goi_anh_kaggle.py --tat-ca

# nén một nhóm
.venv\Scripts\python.exe scripts\45_dong_goi_anh_kaggle.py --nhom L23 --ghi --user <username-cua-ban>
```

Ra `kaggle_upload\aic2026-keyframes-l23\` gồm `Keyframes_L23.zip` và
`dataset-metadata.json` đã đặt sẵn `isPrivate: true`.

**Bắt đầu từ gói nhỏ nhất.** Gói đầu là để dò xem đường truyền chịu được không,
đứt thì chỉ mất vài trăm MB chứ không phải 6 GB.

| nhóm | ảnh | MB |
| --- | ---: | ---: |
| L23 | 2.326 | 483 |
| L27 | 4.914 | 1.041 |
| L30 | 7.915 | 1.345 |
| L21 | 7.800 | 1.379 |
| L22 | 9.096 | 1.639 |
| L24 | 6.781 | 1.646 |
| L28 | 10.683 | 2.063 |
| L29 | 10.771 | 2.392 |
| L25 | 37.445 | 5.804 |
| **L26** | **79.590** | **≈16.500** |

**Cần dư chỗ trống bằng đúng dung lượng ảnh** — file zip nằm cạnh ảnh gốc chứ
không thay thế nó. Nén hết 9 nhóm cần thêm ~17,8 GB.

Script dùng `ZIP_STORED`, cố ý không nén lại: JPEG đã nén sẵn, nén lại tốn hàng
chục phút CPU để giảm được vài phần trăm.

---

## 4. Tạo dataset

```powershell
.venv\Scripts\kaggle.exe datasets create -p kaggle_upload\aic2026-keyframes-l23 --dir-mode zip
```

Chạy lâu tuỳ đường truyền. **Đừng tắt cửa sổ PowerShell, và tắt chế độ ngủ của
máy trước** (Settings → Power) — máy sleep giữa chừng là đứt, phải chạy lại từ
đầu.

Đẩy xong Kaggle còn mất vài phút tới vài chục phút để giải nén phía nó. Dataset
ở trạng thái *processing* thì chưa mount được.

Cần đẩy lại nhóm đó về sau:

```powershell
.venv\Scripts\kaggle.exe datasets version -p kaggle_upload\aic2026-keyframes-l23 -m "cap nhat" --dir-mode zip
```

---

## 5. Add collaborator — bước hay bị quên nhất

Mở trang dataset → tab **Settings** (trong trang dataset, không phải Settings
tài khoản) → **Collaborators** → **Add collaborator**.

Thêm username của những người còn lại trong nhóm, quyền **`Can edit`**.

**Phải là `Can edit`, không phải `Can view`.** `Can view` thì người kia mount đọc
được nhưng **không đẩy được bản cập nhật lên cùng dataset** — tới lúc cần sửa
phải tạo dataset mới, và mọi notebook đang trỏ vào cái cũ đều sai chỗ.

---

## 6. Soát trước khi báo xong

| Mục | Phải là |
| --- | --- |
| Nhãn cạnh tên dataset | **Private** |
| Số file sau khi giải nén | đúng số ảnh ở bảng mục 3 |
| Collaborators | đủ người trong nhóm, quyền **Can edit** |

Rồi nhắn cả nhóm đường dẫn dạng
`kaggle.com/datasets/<username>/aic2026-keyframes-l23`.

---

## 7. Riêng L26 — nhóm nặng nhất

79.590 ảnh, ≈16,5 GB, 498 video. Đây là nhóm **to nhất kho và đang mù hoàn
toàn**: máy chính chưa có ảnh nhóm này, nên không soi tay được câu nào rơi vào
đó — mà đề sơ tuyển đợt 2 có 5 câu Q&A nằm trong L26.

Thứ tự nên thử:

1. **Người đang giữ L26 tự đẩy lên rồi add collaborator.** Rẻ nhất, không ai
   phải tải gì.
2. Tải thẳng từ Drive vào notebook Kaggle rồi tạo dataset từ output —
   `/kaggle/working` khoảng 20 GB nên 16,5 GB vừa, nhưng sát trần.
3. Đường cuối: tải về máy rồi upload như các nhóm khác. Tốn 33 GB đường truyền.

---

## ⚠️ Riêng tư — không được quên

Đây là **dữ liệu thi của BTC**. Mọi dataset phải **Private**, kể cả sau khi thi
xong.

`dataset-metadata.json` script sinh ra đã đặt `isPrivate: true`, nhưng vẫn phải
**mở trang dataset kiểm lại bằng mắt**. Lỡ tạo nhầm public thì đổi sang Private
ngay rồi báo cả nhóm biết nó đã public trong bao lâu.

Thư mục `kaggle_upload/` và `vqa/` đều đã bị `.gitignore` chặn — đừng gỡ luật đó.

---

## Vướng gì

| Hiện tượng | Nguyên nhân |
| --- | --- |
| `401` / `403` khi chạy lệnh | chưa `kaggle auth login`, hoặc chưa xác thực số điện thoại ở mục 1 |
| Upload đứt giữa chừng | chạy lại đúng lệnh ở mục 4, nó bắt đầu lại từ đầu |
| Kẹt ở *processing* rất lâu | file lớn thì Kaggle giải nén lâu; để yên 30–60 phút rồi tải lại trang |
| `Not enough space` khi nén | cần dư chỗ bằng đúng dung lượng ảnh, xem bảng mục 3 |
| Notebook không thấy dataset | dataset còn *processing*, hoặc quên **Add Input** trong notebook |
