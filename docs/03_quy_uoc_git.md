# 03 — Quy ước git

Sáu người trên một repo. Vài quy ước để không giẫm chân nhau.

## Cấu trúc nhánh

```text
main          bản ổn định, chỉ merge từ dev
 └── dev      nhánh làm việc chung — TÁCH NHÁNH TỪ ĐÂY
      ├── giai-doan-0
      ├── trich-day
      └── ocr-ticker
```

**Không ai push thẳng vào `main` hay `dev`.** Luôn tách nhánh riêng rồi mở PR.

## Vòng làm việc

```powershell
# 1. Lấy code mới nhất
git checkout dev
git pull

# 2. Tách nhánh cho việc của mình
git checkout -b ten-viec-cua-ban

# 3. Làm việc, commit
git add <file cụ thể>          # đừng git add -A cho quen tay
git commit -m "Mô tả việc đã làm"

# 4. Đẩy lên
git push -u origin ten-viec-cua-ban
```

Rồi vào GitHub mở Pull Request vào `dev`. Git sẽ in sẵn link sau khi push.

### Tên nhánh

Đặt theo việc, chữ thường, nối bằng gạch ngang, không dấu:
`trich-day`, `ocr-ticker`, `kenh-bm25`, `sua-loi-fps`.

### Commit

Viết bằng tiếng Việt cũng được, nhưng phải nói **đã làm gì**, không phải
"update", "fix", "sửa tí". Dòng đầu ngắn gọn, cần giải thích thêm thì cách
một dòng trống rồi viết tiếp.

## Cập nhật nhánh khi `dev` đã đi trước

```powershell
git checkout dev
git pull
git checkout ten-nhanh-cua-ban
git rebase dev
```

Rebase cho lịch sử thẳng, dễ đọc hơn merge. Nếu xung đột: sửa file, rồi

```powershell
git add <file đã sửa>
git rebase --continue
```

Rối quá thì `git rebase --abort` để quay lại như cũ, không mất gì.

## TUYỆT ĐỐI KHÔNG COMMIT

Repo này **public**. Đẩy nhầm là cả thế giới thấy, và xóa khỏi lịch sử git
rất phiền.

| Không commit | Vì sao |
| --- | --- |
| `index/` | 395 MB. Chia sẻ qua Drive |
| `*.mp4`, `*.jpg` | dữ liệu thi đấu, hàng trăm GB |
| `*.npy`, `*.parquet` | file nhị phân nặng, git không diff được |
| `.venv/` | môi trường máy bạn, `requirements.txt` là đủ |
| token, mật khẩu, khóa API | khỏi giải thích |

`.gitignore` đã chặn sẵn tất cả. **Đừng gỡ ra.** Ngoại lệ duy nhất đang mở
là ba file báo cáo text trong `index/` — chúng nhẹ và là bằng chứng bảng cái
đã được kiểm chứng.

Kiểm tra trước khi commit cho chắc:

```powershell
git status --short          # xem sẽ commit gì
git diff --cached --stat    # xem dung lượng
```

Thấy con số MB nào là dừng lại xem kỹ.

## Xử lý khi lỡ tay

**Lỡ commit file nặng, CHƯA push:**

```powershell
git reset HEAD~1            # bỏ commit, giữ nguyên file trên đĩa
```

**Đã push rồi:** đừng tự sửa lịch sử nhánh chung. Báo nhóm, xử lý cùng nhau.

**Muốn bỏ hết thay đổi chưa commit:**

```powershell
git checkout -- <file>      # một file
git stash                   # cất tạm, lấy lại bằng git stash pop
```

## PR

- Mô tả PR nói rõ **làm gì** và **kiểm tra thế nào**
- Nhờ ít nhất một người xem trước khi merge
- PR nhỏ, một việc — đừng gom ba tính năng vào một PR

## Cần dữ liệu, không phải code

`index/` và dữ liệu thô đi qua **Google Drive**, không qua git. Người giữ
`index/` chính thức đẩy lên Drive; ai cần thì tải về đặt vào
`C:\Code\aic2026\index\`. Xem [01_cai_dat.md](01_cai_dat.md).
