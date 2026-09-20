"""
124_dong_goi_dap_an.py — Ghép đáp án SOI TAY với đầu ra hệ thống thành bài nộp đúng định dạng BTC.

    python scripts/124_dong_goi_dap_an.py \\
        --dap-an dev/SOTUYEN3-bo-de-thi/_dap_an_nhom.txt \\
        --he-thong submission_he_thong \\
        --ra submission

VÌ SAO GHÉP CHỨ KHÔNG THAY

Nhóm soi tay ra đáp án cho phần lớn gói. Cách làm hiển nhiên là nộp đúng những
khung đó. Nhưng PHẦN C mục 1 nói rõ: **được nộp 100 dòng, không có điểm phạt,
dòng thứ 100 vẫn đáng 0,2**. Nộp 3 dòng soi tay rồi bỏ trống 97 dòng là **vứt
đi 97 vé số miễn phí** — nếu khung soi tay sai (đã xảy ra: A89 cho thấy BTC bác
6/20 nhãn tự soi ở Sơ tuyển 1) thì cả gói về 0, trong khi một dòng của hệ thống
có thể đã trúng.

Nên: **khung soi tay lên đầu, đầu ra hệ thống lấp phần còn lại.** Người soi
được ưu tiên tuyệt đối; hệ thống chỉ nhận những chỗ trống.

    hạng 1..k     khung nhóm soi tay, đúng thứ tự đã ghi
    hạng k+1..100 ứng viên hệ thống, bỏ những khung đã có ở trên

⚠️ `frame_idx` TRA TỪ CỘT CỦA BẢNG CÁI, không tính từ `kf_n × fps`. File đáp án
ghi `kf_n` (số keyframe, 1-based) vì đó là thứ người soi đọc trên giao diện;
`frame_idx` là thứ nộp cho BTC, và hai cái đó KHÔNG suy ra được từ nhau bằng
phép nhân — A5.7 đo được 614 keyframe dùng chung `frame_idx`.

⚠️ Q&A: `answer` gắn vào **MỌI** dòng, kể cả dòng lấp của hệ thống. Điểm Q&A
đòi đúng CẢ khung LẪN đáp án (PHẦN C mục 4); dòng lấp mang đáp án tốt nhất
đang có thì mới còn cơ hội, để trống là chắc chắn 0.

⚠️ TRAKE: dòng soi tay là MỘT dòng gồm N `frame_idx` theo đúng thứ tự sự kiện.
Các dòng còn lại lấy từ đầu ra hệ thống, giữ nguyên thứ tự của nó.

⚠️ Ghi qua `nop_bai.ghi_goi`, tức `soat()` chạy trước và **có lỗi thì không ghi
file nào** — sai định dạng vẫn tính là một lần nộp, mà chỉ có 3 lần (C7).
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import nop_bai                                        # noqa: E402
from schema import AnswerKIS, AnswerQA, AnswerTRAKE   # noqa: E402

TOI_DA = 100


def chuan_video(s: str) -> str:
    """`L26V424`, `L30_v023`, `L26_V424` -> `L26_V424`."""
    m = re.fullmatch(r"(L\d{2})_?[Vv](\d+)", s.strip())
    if not m:
        raise SystemExit(f"❌ video_id không đọc được: {s!r}")
    return f"{m.group(1)}_V{int(m.group(2)):03d}"


def doc_kf(s: str) -> list[int]:
    """`"108-120"` / `"123,124,127"` / `"161"` -> danh sách kf_n."""
    ra = []
    for phan in s.split(","):
        phan = phan.strip()
        if not phan:
            continue
        if "-" in phan:
            a, b = (int(x) for x in phan.split("-", 1))
            if b < a:
                raise SystemExit(f"❌ dải ngược: {phan!r}")
            ra += list(range(a, b + 1))
        else:
            ra.append(int(phan))
    return ra


def doc_dap_an(f: Path) -> dict:
    ra = {}
    for so, dong in enumerate(f.read_text("utf-8").splitlines(), 1):
        d = dong.split("#", 1)[0].strip()
        if not d:
            continue
        phan = [x.strip() for x in d.split(":")]
        if len(phan) < 3:
            raise SystemExit(f"❌ dòng {so} thiếu trường: {dong!r}")
        ten, vid, kf = phan[0], chuan_video(phan[1]), phan[2]
        tra = phan[3] if len(phan) > 3 else None
        if ten in ra:
            raise SystemExit(f"❌ dòng {so}: gói {ten} khai hai lần")
        if "+" in kf:                                  # TRAKE
            ra[ten] = {"video_id": vid, "trake": [int(x) for x in kf.split("+")]}
        else:
            ra[ten] = {"video_id": vid, "kf": doc_kf(kf), "answer": tra}
    return ra


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--dap-an", type=Path, required=True)
    ap.add_argument("--he-thong", type=Path, required=True,
                    help="thư mục CSV do run.py sinh")
    ap.add_argument("--ra", type=Path, default=GOC / "submission")
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet",
                             columns=["video_id", "kf_n", "frame_idx"])
    # (video_id, kf_n) -> frame_idx. Tra bảng, KHÔNG nhân fps.
    tra = {(v, int(k)): int(f) for v, k, f in
           zip(master.video_id.values, master.kf_n.values, master.frame_idx.values)}

    tay = doc_dap_an(a.dap_an)
    csv = sorted(a.he_thong.glob("query-*.csv"))
    if not csv:
        raise SystemExit(f"❌ không có query-*.csv trong {a.he_thong}")
    thua = set(tay) - {f.stem for f in csv}
    if thua:
        raise SystemExit(f"❌ đáp án khai gói KHÔNG có trong bộ đề: {sorted(thua)}")

    goi, so_su_kien, bao = {}, {}, []
    for f in csv:
        ten = f.stem
        loai = ten.rsplit("-", 1)[-1]
        ht = nop_bai.doc_csv(f)                        # đầu ra hệ thống
        t = tay.get(ten)

        if loai == "trake":
            n = len(ht[0].frame_idxs) if ht else 0
            dong = []
            if t:
                kf = t["trake"]
                fx = []
                for k in kf:
                    khoa = (t["video_id"], k)
                    if khoa not in tra:
                        raise SystemExit(
                            f"❌ {ten}: {t['video_id']} không có keyframe {k}")
                    fx.append(tra[khoa])
                # BTC đòi thứ tự thời gian; nếu hai sự kiện ra cùng frame_idx
                # thì đẩy lên 1 — hai khoảnh khắc khác nhau, nộp trùng là phí.
                for i in range(1, len(fx)):
                    if fx[i] <= fx[i - 1]:
                        fx[i] = fx[i - 1] + 1
                n = n or len(fx)
                if len(fx) != n:
                    raise SystemExit(f"❌ {ten}: soi tay {len(fx)} sự kiện, "
                                     f"hệ thống {n}")
                dong.append(AnswerTRAKE(t["video_id"], fx))
            da = {(d.video_id, tuple(d.frame_idxs)) for d in dong}
            for d in ht:
                if (d.video_id, tuple(d.frame_idxs)) not in da:
                    dong.append(d)
                    da.add((d.video_id, tuple(d.frame_idxs)))
                if len(dong) >= TOI_DA:
                    break
            goi[ten] = dong[:TOI_DA]
            so_su_kien[ten] = n
        else:
            dong, da = [], set()
            tra_loi = None
            if t:
                tra_loi = t.get("answer")
                if loai == "qa" and not tra_loi:
                    raise SystemExit(f"❌ {ten} là Q&A nhưng đáp án không có "
                                     f"`answer`")
                for k in t["kf"]:
                    khoa = (t["video_id"], k)
                    if khoa not in tra:
                        raise SystemExit(
                            f"❌ {ten}: {t['video_id']} không có keyframe {k}")
                    fi = tra[khoa]
                    if (t["video_id"], fi) in da:
                        continue          # hai kf_n cùng frame_idx (A5.7)
                    da.add((t["video_id"], fi))
                    dong.append(AnswerQA(t["video_id"], fi, tra_loi)
                                if loai == "qa" else
                                AnswerKIS(t["video_id"], fi))
            # Q&A chưa soi: giữ `answer` mà hệ thống đã đào.
            for d in ht:
                if (d.video_id, d.frame_idx) in da:
                    continue
                da.add((d.video_id, d.frame_idx))
                if loai == "qa":
                    at = tra_loi if tra_loi is not None else getattr(d, "answer", "")
                    dong.append(AnswerQA(d.video_id, d.frame_idx, at))
                else:
                    dong.append(AnswerKIS(d.video_id, d.frame_idx))
                if len(dong) >= TOI_DA:
                    break
            goi[ten] = dong[:TOI_DA]

        bao.append((ten, len(t["kf"]) if t and "kf" in t else
                    (1 if t else 0), len(goi[ten])))

    print(f"\n{'gói':<22}{'soi tay':>9}{'tổng dòng':>11}")
    print("-" * 42)
    n_tay = 0
    for ten, k, tong in bao:
        n_tay += k > 0
        print(f"{ten:<22}{(k or '—'):>9}{tong:>11}")
    print("-" * 42)
    print(f"{len(bao)} gói · {n_tay} gói có đáp án soi tay · "
          f"{len(bao) - n_tay} gói lấy nguyên hệ thống\n")

    d = nop_bai.ghi_goi(goi, thu_muc=str(a.ra), so_su_kien=so_su_kien)
    print(f"✅ {d}")
    for c in nop_bai.canh_bao("", []) or []:
        print("  ", c)


if __name__ == "__main__":
    main()
