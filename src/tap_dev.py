"""
tap_dev.py — Đọc, ghi và kiểm tra tập dev.

Đọc kèm: docs/07_lam_tap_dev.md

Một câu hỏi dev lưu ở dạng JSONL, mỗi dòng một câu:

    {"id": "kis-001", "loai": "KIS",
     "cau_hoi": "người phụ nữ thái dứa trên thớt gỗ",
     "row_id_dung": [45231, 45232],
     "nguon": "duyệt sheet L25_V012"}

Vì sao lưu `row_id` chứ không lưu `frame_idx`: từ `row_id` suy ra được
video_id, frame_idx, pts_time, fps — ngược lại thì không.

Vì sao `row_id_dung` là DANH SÁCH: A5.6 đo được 11,83% keyframe có bản sao
cùng video ở cosine >= 0,99 (L25: 49,82%). Đánh dấu một cái rồi `no_cum()` nở
ra cả cụm — hệ thống trả về bất kỳ cái nào trong cụm đều phải tính là đúng.
Không làm vậy thì ta tự tạo ra kết quả sai giả, và mọi cấu hình đều bị chấm
thấp một cách ngẫu nhiên.

    python src/tap_dev.py --kiem            # soát tập dev
    python src/tap_dev.py --no-cum          # nở cụm trùng lặp, ghi đè file
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
MAC_DINH = GOC / "dev" / "tap_dev.jsonl"
LOAI_HOP_LE = {"KIS", "QA", "TRAKE"}


@dataclass
class CauHoi:
    id: str
    loai: str                          # KIS | QA | TRAKE
    cau_hoi: str
    row_id_dung: list                   # KIS/QA: list[int]. TRAKE: list[list[int]]
    dap_an: str = ""                    # chỉ QA. Sai định dạng -> 0 điểm
    nguon: str = ""                     # đã tìm ra bằng cách nào
    ghi_chu: str = ""

    def video_id(self, master: pd.DataFrame) -> str:
        r = self.row_id_dung[0]
        r = r[0] if isinstance(r, list) else r
        return master.video_id.iloc[r]

    def nhom(self, master: pd.DataFrame) -> str:
        return self.video_id(master)[:3]


def doc(f=MAC_DINH) -> list[CauHoi]:
    f = Path(f)
    if not f.exists():
        return []
    return [CauHoi(**json.loads(d)) for d in f.read_text("utf-8").splitlines() if d.strip()]


def ghi(cau: list[CauHoi], f=MAC_DINH) -> None:
    f = Path(f)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in cau) + "\n",
                 encoding="utf-8")


def no_cum(cau: list[CauHoi], index_dir=GOC / "index", nguong=0.99) -> list[CauHoi]:
    """Nở mỗi `row_id` đã đánh dấu ra thành cả cụm bản sao cùng video.

    Chỉ so trong CÙNG video: hai keyframe giống nhau ở hai video khác nhau là
    hai đáp án khác nhau, không được gộp.
    """
    master = pd.read_parquet(Path(index_dir) / "master.parquet")
    mat = np.load(Path(index_dir) / "clip.npy", mmap_mode="r")

    def no(rid: int) -> list[int]:
        vid = master.video_id.iloc[rid]
        anh_em = np.where((master.video_id == vid).values)[0]
        cos = np.asarray(mat[anh_em], np.float32) @ np.asarray(mat[rid], np.float32)
        return sorted(int(x) for x in anh_em[cos >= nguong])

    ra = []
    for c in cau:
        if c.loai == "TRAKE":
            moi = [sorted({r for x in b for r in no(x)}) for b in c.row_id_dung]
        else:
            moi = sorted({r for x in c.row_id_dung for r in no(x)})
        ra.append(CauHoi(**{**asdict(c), "row_id_dung": moi}))
    return ra


def kiem(cau: list[CauHoi], index_dir=GOC / "index") -> list[str]:
    """Trả về danh sách lỗi. Rỗng nghĩa là tập dev dùng được."""
    master = pd.read_parquet(Path(index_dir) / "master.parquet")
    n, loi = len(master), []
    thay = set()

    for c in cau:
        if c.id in thay:
            loi.append(f"{c.id}: id trùng")
        thay.add(c.id)
        if c.loai not in LOAI_HOP_LE:
            loi.append(f"{c.id}: loại '{c.loai}' không hợp lệ {sorted(LOAI_HOP_LE)}")
        if not c.cau_hoi.strip():
            loi.append(f"{c.id}: câu hỏi rỗng")
        if c.loai == "QA" and not c.dap_an.strip():
            loi.append(f"{c.id}: câu QA thiếu `dap_an` -> luôn 0 điểm")
        if not c.row_id_dung:
            loi.append(f"{c.id}: chưa có đáp án")
            continue

        phang = ([r for b in c.row_id_dung for r in b] if c.loai == "TRAKE"
                 else c.row_id_dung)
        if c.loai == "TRAKE" and not all(isinstance(b, list) for b in c.row_id_dung):
            loi.append(f"{c.id}: TRAKE cần list[list[int]], mỗi sự kiện một danh sách")
        for r in phang:
            if not isinstance(r, int) or not 0 <= r < n:
                loi.append(f"{c.id}: row_id {r} ngoài khoảng [0, {n})")

        vids = {master.video_id.iloc[r] for r in phang if isinstance(r, int) and 0 <= r < n}
        if len(vids) > 1:
            loi.append(f"{c.id}: đáp án nằm ở nhiều video {sorted(vids)} — "
                       f"KIS/QA chỉ có một video đúng")

        # Mã nhóm trong `id` phải khớp nhóm của đáp án.
        #
        # Không có chốt này thì câu bị đặt nhầm file KHÔNG AI KIỂM ĐƯỢC: người
        # giữ nhóm ghi trên `id` không có ảnh để mở, còn người giữ nhóm thật
        # thì không biết câu đó tồn tại. Đã bắt được hai ca thật — một câu
        # L22 nằm trong file L21, và `kis-L24-001` có đáp án ở L30_V075 trong
        # khi L24_V075 KHÔNG TỒN TẠI (0 khung).
        ma = re.search(r"L\d\d", c.id)
        if ma and vids and ma.group() != sorted(vids)[0][:3]:
            loi.append(f"{c.id}: id ghi nhóm {ma.group()} nhưng đáp án ở "
                       f"{sorted(vids)[0]} — đặt nhầm file, hoặc chép nhầm row_id")
    return loi


MAC_DINH_TEST = GOC / "dev" / "tap_test.jsonl"


def tach_test(cau: list[CauHoi], master, ty_le: float = 0.15,
              seed: int = 2026) -> tuple[list[CauHoi], list[CauHoi]]:
    """Chia thành (dev, test), **phân tầng theo (nhóm L × loại câu)**.

    Vì sao phân tầng: L26 chiếm 23/100 câu và tỷ lệ QA khác nhau giữa các nhóm.
    Chia ngẫu nhiên thuần có thể ra tập test không có câu QA nào, hoặc dồn hết
    vào một nhóm L — rồi con số kiểm cuối cùng chẳng đại diện cho cái gì.

    Chọn bằng **băm tất định trên `id`**, không dùng `random.shuffle`: kết quả
    giống nhau trên mọi máy, mọi lần chạy, không phụ thuộc thứ tự file.
    """
    import hashlib
    from collections import defaultdict

    # Chỉ câu HOÀN CHỈNH mới được vào tập giữ kín. Câu còn lỗi (vd QA thiếu
    # `dap_an`) mà lọt vào đó là kẹt vĩnh viễn: nó luôn chấm 0, mà `gop()` đã
    # loại câu test nên bản sửa ở file thành viên không bao giờ chảy tới được.
    # Giữ chúng lại ở tập dev cho tới khi tác giả sửa xong.
    xau = {x.split(":")[0] for x in kiem(cau)}
    if xau:
        print(f"Giữ {len(xau)} câu còn lỗi ở lại tập dev: {sorted(xau)}")

    tang = defaultdict(list)
    for c in cau:
        if c.id not in xau:
            tang[(c.nhom(master), c.loai)].append(c)

    test_id = set()
    for _, ds in sorted(tang.items()):
        k = round(len(ds) * ty_le)
        if k == 0:
            continue
        xep = sorted(ds, key=lambda c: hashlib.md5(
            f"{seed}:{c.id}".encode()).hexdigest())
        test_id |= {c.id for c in xep[:k]}

    return ([c for c in cau if c.id not in test_id],
            [c for c in cau if c.id in test_id])


def _bung(duong_dan: list) -> list[Path]:
    """Bung `*.jsonl` và thư mục thành danh sách file.

    ⚠️ PowerShell KHÔNG bung dấu sao cho chương trình ngoài (khác bash) — gõ
    `--gop thu_muc\\*.jsonl` thì Python nhận đúng chuỗi `*.jsonl`, đọc không ra
    file nào, và **gộp ra 0 câu mà không báo lỗi gì**. Phải tự bung ở đây.
    """
    ra = []
    for d in duong_dan:
        p = Path(d)
        if p.is_dir():
            ra += sorted(p.glob("*.jsonl"))
        elif any(k in str(d) for k in "*?["):
            ra += sorted(Path(p.parent or ".").glob(p.name))
        else:
            ra.append(p)
    if not ra:
        raise SystemExit(f"Không khớp file nào: {duong_dan}")
    return ra


def gop(cac_file: list, index_dir=GOC / "index") -> tuple[list[CauHoi], list[str]]:
    """Gộp tập dev của nhiều người thành một.

    Mỗi người soạn câu cho nhóm L MÌNH GIỮ — vì chỉ họ mới mở được ảnh gốc để
    kiểm lại (bước bắt buộc ở §4 của docs/07_lam_tap_dev.md). File `.jsonl`
    chỉ vài KB nên gửi qua chat cũng được, không cần Drive.

    Trùng `id` giữa hai người là lỗi phải sửa tay, không tự đổi tên: đổi ngầm
    thì sau này không ai truy được câu đó của ai. Đặt `id` theo nhóm L
    (`kis-L21-001`) là hết trùng mà còn nhìn ra phân bố ngay trên id.
    """
    # ⚠️ CHỐNG RÒ TẬP TEST. Không có đoạn này thì `--gop` chạy lại từ file thành
    # viên sẽ dựng lại ĐỦ cả trăm câu và nuốt mất tập giữ kín — im lặng, không
    # báo gì. Từ đó mọi con số "kiểm trên tập chưa từng nhìn" thành vô nghĩa.
    giu_kin = {c.id for c in doc(MAC_DINH_TEST)}

    # ⚠️ SOÁT TRÙNG `id` TRƯỚC, LỌC TẬP TEST SAU — thứ tự này quan trọng.
    #
    # Làm ngược lại thì một `id` bị nhân đôi mà tình cờ nằm trong tập test sẽ
    # bị `continue` cả hai lần và **không bao giờ được báo**. Đã vấp thật: một
    # câu được chép sang file khác nhưng bản cũ chưa xoá, `--gop` in "Bỏ 21
    # câu" trong khi tập test chỉ có 20 — con số lệch là dấu hiệu DUY NHẤT, mà
    # nó dễ bị đọc lướt qua.
    ra, thay, loi, bo = [], {}, [], 0
    for f in _bung(cac_file):
        for c in doc(f):
            if c.id in thay:
                loi.append(f"id '{c.id}' có ở cả {thay[c.id]} và {Path(f).name}")
                continue
            thay[c.id] = Path(f).name
            if c.id in giu_kin:
                bo += 1
                continue
            ra.append(c)
    if bo:
        print(f"Bỏ {bo} câu đã nằm trong tập test giữ kín ({MAC_DINH_TEST.name})")
    return ra, loi


def phan_bo(cau: list[CauHoi], index_dir=GOC / "index") -> pd.DataFrame:
    """Phân bố theo nhóm L và theo loại câu.

    A2: L26 chiếm 57% số video. Lấy mẫu ngẫu nhiên sẽ ra tập lệch hẳn về một
    nhóm, và mọi tinh chỉnh ở Giai đoạn 3 sẽ tối ưu cho một nhóm duy nhất.
    """
    master = pd.read_parquet(Path(index_dir) / "master.parquet")
    d = pd.DataFrame([{"nhom": c.nhom(master), "loai": c.loai} for c in cau])
    if d.empty:
        return d
    return pd.crosstab(d.nhom, d.loai, margins=True, margins_name="TỔNG")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=MAC_DINH, type=Path)
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--kiem", action="store_true")
    ap.add_argument("--no-cum", action="store_true", help="nở cụm trùng lặp rồi ghi đè")
    ap.add_argument("--gop", nargs="+", default=None,
                    help="gộp nhiều file .jsonl của các thành viên vào --file")
    ap.add_argument("--tach-test", action="store_true",
                    help="tách tập test giữ kín ra dev/tap_test.jsonl. "
                         "CHỈ CHẠY MỘT LẦN")
    ap.add_argument("--ty-le", type=float, default=0.15)
    a = ap.parse_args()

    if a.tach_test:
        if MAC_DINH_TEST.exists():
            raise SystemExit(
                f"❌ {MAC_DINH_TEST} ĐÃ TỒN TẠI — không tách lại.\n\n"
                "Tách lại là đổi thành phần tập test, mà những câu bị đẩy RA sẽ\n"
                "quay về tập dev sau khi ta đã dò trên chúng. Tập test hết còn\n"
                "'chưa từng nhìn', và con số kiểm cuối thành vô nghĩa.\n\n"
                "Thêm câu mới thì cứ `--gop` bình thường: câu mới vào tập dev,\n"
                "tập test giữ nguyên.")
        cau = doc(a.file)
        if not cau:
            raise SystemExit(f"Chưa có câu nào trong {a.file}")
        master = pd.read_parquet(Path(a.index) / "master.parquet")
        dev, test = tach_test(cau, master, a.ty_le)
        ghi(dev, a.file)
        ghi(test, MAC_DINH_TEST)
        print(f"Tách {len(cau)} câu -> dev {len(dev)} | test {len(test)}\n")
        print("PHÂN BỐ TẬP DEV");  print(phan_bo(dev, a.index).to_string())
        print("\nPHÂN BỐ TẬP TEST"); print(phan_bo(test, a.index).to_string())
        print(f"\n🔒 {MAC_DINH_TEST}")
        print("   ĐỪNG MỞ, ĐỪNG ĐO cho tới lượt kiểm cuối cùng ở Giai đoạn 3.")
        print("   `--gop` từ nay tự loại các câu này khỏi tập dev.")
        return

    if a.gop:
        cau, loi = gop(a.gop, a.index)
        if loi:
            print(f"❌ {len(loi)} id trùng — sửa tay rồi gộp lại:")
            for x in loi:
                print("   ", x)
            raise SystemExit(1)
        ghi(cau, a.file)
        print(f"Gộp {len(a.gop)} file -> {len(cau)} câu -> {a.file}\n")

    cau = doc(a.file)
    if not cau:
        raise SystemExit(f"Chưa có câu nào trong {a.file}.\n"
                         f"Xem docs/07_lam_tap_dev.md để bắt đầu.")
    print(f"{len(cau)} câu trong {a.file}\n")

    if a.no_cum:
        truoc = sum(len(x) if not isinstance(x[0], list) else sum(map(len, x))
                    for x in (c.row_id_dung for c in cau))
        cau = no_cum(cau, a.index)
        sau = sum(len(x) if not isinstance(x[0], list) else sum(map(len, x))
                  for x in (c.row_id_dung for c in cau))
        ghi(cau, a.file)
        print(f"Nở cụm trùng lặp: {truoc} -> {sau} row_id được tính là đúng\n")

    loi = kiem(cau, a.index)
    print(phan_bo(cau, a.index).to_string(), "\n")
    if loi:
        print(f"❌ {len(loi)} LỖI:")
        for x in loi:
            print("   ", x)
        raise SystemExit(1)
    print("✅ Tập dev hợp lệ.")
    if len(cau) < 30:
        print(f"\n⚠️  Mới {len(cau)} câu. Dưới 30 câu thì sai số lớn hơn mọi khác\n"
              "    biệt đáng quan tâm — xem docs/07_lam_tap_dev.md §6.")


if __name__ == "__main__":
    main()
