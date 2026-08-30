r"""
tac_tu.py — Vòng lặp có CÔNG CỤ: tự soi ảnh, tự tra chữ, tự xếp lại 100 dòng.

    python src/tac_tu.py --de dev/SOTUYEN2-bo-de-thi --ung-vien vqa/p2_kenh1 \
        --ra vqa/tac_tu --so-goi 5

VÌ SAO CẦN THÊM MỘT TẦNG NỮA
============================

Đường ống hiện tại đi **một chiều**: truy vấn → kênh → 100 dòng → hết. Nó không
bao giờ tự hỏi *"khung này có đúng không"*. Người soát tay phải làm việc đó, và
chính bước ấy đã kéo bài nộp đợt 2 từ máy-thuần lên 11,6 điểm.

Tác tử làm đúng vòng lặp mà người soát làm: **tìm → mở ảnh xem → sai thì tra
chữ → tìm lại → soi khung lân cận → chốt**.

RÀNG BUỘC CỨNG: CHỈ ĐƯỢC XẾP LẠI, KHÔNG ĐƯỢC THÊM
=================================================

A27/A28 đo được, và A42 tìm ra cơ chế:

    kênh yếu XẾP LẠI thứ kênh mạnh đã chọn   ->  +1,6
    kênh yếu THAY THẾ / CHÈN ứng viên mới    ->  -0,4  (lớn nhất repo: -0,23)

Vì bể chỉ có **100 chỗ**, mà 25/38 câu có **ZERO** giao nhau giữa tập neo và
top-100 của kênh 1 — nên thêm một ứng viên là **đá một ứng viên khác ra**, chứ
không phải "thêm".

Nên mặc định tác tử **chỉ được hoán vị** danh sách vào. Cờ `--cho-them` mở khoá
việc thêm ứng viên mới, nhưng để TẮT cho tới khi đo được điều ngược lại trên
`dev/tap_de_that.jsonl`.

NGÂN SÁCH LÀ MỘT PHẦN CỦA THIẾT KẾ
==================================

Mỗi lần `xem_khung` là một lượt gọi VLM có ảnh. 30 gói × 20 lượt = 600 lượt, đủ
để hết giờ thi hoặc hết quota. `--tran-goi` chặn cứng số lượt mỗi gói, và script
in ra số lượt đã dùng để người sau ước được thời gian thật.

NÓ TRẢ VỀ GÌ
============

Với mỗi gói: danh sách 100 dòng **đã xếp lại**, kèm

    nhat_ky   từng bước: gọi công cụ gì, thấy gì, kết luận gì
    do_chac   "chac" | "ngo" | "khong-biet"

`do_chac` không phải trang trí — nó là thứ nói cho người soát biết **mở gói nào
trước** khi chỉ còn 20 phút.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                              # noqa: E402
from tra_loi_ocr import (MODEL_GEMINI, _goi_gemini,          # noqa: E402
                         _goi_gemini_anh, nap_khoa, thu_nho)

TRAN_GOI_MAC_DINH = 12


class Kho:
    """Bảng cái + OCR/ASR, tra theo `row_id`. Nạp một lần, dùng cho mọi gói."""

    def __init__(self, index: Path):
        self.master = pd.read_parquet(index / "master.parquet")
        m = self.master
        self.vid = m.video_id.values
        self.kfn = m.kf_n.values
        self.fidx = m.frame_idx.values
        self.pts = m.pts_time.values
        self.kfp = m.kf_path.values
        self.van_ban = {}
        p = index / "ocr_asr.parquet"
        if p.exists():
            b = pd.read_parquet(p)
            self.van_ban = dict(zip(b.row_id.values, b.text.values))
        # (video_id, frame_idx) -> row_id. A5.7: 614 keyframe trùng frame_idx
        # nên đây KHÔNG phải song ánh — giữ cái đầu tiên, đủ để soi ảnh.
        self.tra_nguoc = {}
        for r, (v, f) in enumerate(zip(self.vid, self.fidx)):
            self.tra_nguoc.setdefault((str(v), int(f)), r)

    def mo_ta(self, r: int) -> str:
        return (f"row_id={r} {self.vid[r]} kf{self.kfn[r]} "
                f"frame={self.fidx[r]} {self.pts[r]:.1f}s")

    def lan_can(self, r: int, ban_kinh: int = 4) -> list[int]:
        """Khung lân cận CÙNG video — hàng xóm ở biên bảng cái là video khác."""
        v = self.vid[r]
        return [i for i in range(max(0, r - ban_kinh),
                                 min(len(self.vid), r + ban_kinh + 1))
                if self.vid[i] == v]

    def khung_video(self, v: str, thua: int = 10) -> list[int]:
        idx = [i for i, x in enumerate(self.vid) if x == v]
        return idx[::thua]


# --------------------------------------------------------------- CÔNG CỤ

def cong_cu_xem_khung(kho: Kho, rows: list[int], cau_hoi: str, key: str,
                      model: str) -> str:
    """Gửi ảnh cho VLM, hỏi khung nào khớp câu hỏi. Đây là lượt gọi ĐẮT nhất."""
    anh, nhan = [], []
    for r in rows:
        p = kho.kfp[r]
        if not isinstance(p, str):
            continue
        b = thu_nho(p)
        if b:
            anh.append(b)
            nhan.append(kho.mo_ta(r))
    if not anh:
        return "Không khung nào có ảnh ở máy này."
    nhac = (
        "Bạn đang soát kết quả truy hồi video cho một cuộc thi.\n\n"
        f"CÂU HỎI:\n{cau_hoi}\n\n"
        f"Dưới đây là {len(anh)} keyframe, theo thứ tự:\n"
        + "\n".join(f"  [{i}] {t}" for i, t in enumerate(nhan))
        + "\n\nVới TỪNG ảnh, cho biết nó khớp câu hỏi tới đâu. Trả lời JSON:\n"
          '{"danh_gia":[{"i":0,"khop":0.0,"vi_sao":"..."}]}\n'
          "`khop` từ 0 (không liên quan) đến 1 (chắc chắn đúng). Chỉ chấm cao "
          "khi thấy ĐỦ các chi tiết câu hỏi nêu, không chấm cao vì cùng chủ đề."
    )
    # Một cú rớt mạng KHÔNG được giết cả gói. Thử lại vài lần, hết thì trả
    # chuỗi rỗng — gói đó mất một lượt soi chứ không mất cả kết quả.
    import time
    for lan in range(3):
        try:
            return _goi_gemini_anh(nhac, anh, model=model, key=key)
        except Exception as e:
            if lan == 2:
                return f"(lỗi gọi VLM sau 3 lần: {type(e).__name__})"
            time.sleep(2 * (lan + 1))
    return ""


def cong_cu_doc_van_ban(kho: Kho, rows: list[int]) -> str:
    ra = []
    for r in rows:
        t = str(kho.van_ban.get(r, "") or "").strip()
        ra.append(f"{kho.mo_ta(r)}: {t[:400] if t else '(không có chữ)'}")
    return "\n".join(ra) or "(không có gì)"


# --------------------------------------------------------------- VÒNG LẶP

def xu_ly_goi(kho: Kho, ten: str, cau_hoi: str, ung_vien: list[int],
              key: str, model: str, tran: int, cho_them: bool,
              nguong: float = 0.8) -> dict:
    """Một gói -> danh sách đã xếp lại + nhật ký + độ chắc."""
    nhat_ky, so_goi_vlm = [], 0
    diem = {r: 0.0 for r in ung_vien}

    # --- bước 1: cho VLM soi, theo lô nhỏ để mỗi lượt còn đọc được ---------
    #
    # KHÔNG chỉ soi top đầu. Đo trên 30 gói đợt 2: video đúng nằm trong top-100
    # ở 19 gói, nhưng chỉ 6 gói ở hạng 1 và 12 gói trong top-20 — số còn lại
    # rải ở hạng 38, 65, 70, 84, 87, 96. Bản đầu chỉ soi 18 dòng đầu nên không
    # bao giờ NHÌN THẤY chúng, tức vứt đi đúng những gói mà xếp lại có ích nhất.
    #
    # Nên: dày ở đầu (nơi mật độ đúng cao nhất), rồi RẢI ĐỀU hết phần còn lại.
    day = ung_vien[:12]
    con = ung_vien[12:]
    buoc = max(1, len(con) // max(1, (tran - 2) * 6 - len(day))) if con else 1
    quet = day + con[::buoc]
    lo = [quet[i:i + 6] for i in range(0, len(quet), 6)]
    for cum in lo:
        if so_goi_vlm >= tran:
            break
        tl = cong_cu_xem_khung(kho, cum, cau_hoi, key, model)
        so_goi_vlm += 1
        nhat_ky.append({"cong_cu": "xem_khung",
                        "khung": [kho.mo_ta(r) for r in cum],
                        "tra_loi": tl[:900]})
        for d in _doc_danh_gia(tl):
            i = d.get("i")
            if isinstance(i, int) and 0 <= i < len(cum):
                diem[cum[i]] = max(diem[cum[i]], float(d.get("khop", 0) or 0))

    # --- bước 2: khung nào điểm cao thì soi thêm LÂN CẬN của nó -------------
    tot = sorted((r for r in diem if diem[r] >= 0.6), key=lambda r: -diem[r])[:2]
    for r in tot:
        if so_goi_vlm >= tran:
            break
        lc = [x for x in kho.lan_can(r) if x in diem]     # CHỈ trong bể có sẵn
        if cho_them:
            lc = kho.lan_can(r)                            # được thêm mới
        if not lc:
            continue
        tl = cong_cu_xem_khung(kho, lc[:6], cau_hoi, key, model)
        so_goi_vlm += 1
        nhat_ky.append({"cong_cu": "lan_can", "quanh": kho.mo_ta(r),
                        "tra_loi": tl[:900]})
        for d in _doc_danh_gia(tl):
            i = d.get("i")
            if isinstance(i, int) and 0 <= i < len(lc[:6]):
                r2 = lc[i]
                diem[r2] = max(diem.get(r2, 0.0), float(d.get("khop", 0) or 0))

    # --- xếp lại, THẬN TRỌNG ------------------------------------------------
    #
    # Bản đầu sắp thuần theo điểm VLM. Đo trên 4 gói: TỆ ĐI cả hai gói có đáp án
    # trong bể (hạng 6->8 và 10->23). Hai lỗi cộng lại:
    #
    #   1. Ứng viên CHƯA ĐƯỢC SOI cũng nhận 0,0 rồi bị đẩy xuống dưới mọi dương
    #      tính giả. Chưa soi KHÔNG phải bằng chứng là sai.
    #   2. VLM chấm "trông hợp lý" chứ không chấm "đúng đúng cảnh này", nên nó
    #      cho 0,8-1,0 cho video cùng chủ đề. Một dương tính giả đủ để hất ứng
    #      viên đúng ra khỏi hạng 1.
    #
    # Nên: CHỈ điểm >= `nguong` mới được dịch chuyển. Mọi thứ khác giữ NGUYÊN
    # thứ tự gốc của kênh — tác tử chỉ được nhấc lên, không được dìm xuống.
    goc = {r: i for i, r in enumerate(ung_vien)}
    mo_rong = [r for r in diem if r not in goc] if cho_them else []
    day_len = lambda r: max(0.0, diem.get(r, 0.0) - nguong)
    xep = sorted(list(goc) + mo_rong,
                 key=lambda r: (-day_len(r), goc.get(r, 10**9)))

    cao = max(diem.values()) if diem else 0.0
    chac = "chac" if cao >= 0.8 else "ngo" if cao >= 0.5 else "khong-biet"
    return {"ten": ten, "xep": xep, "do_chac": chac, "diem_cao_nhat": round(cao, 2),
            "so_goi_vlm": so_goi_vlm, "nhat_ky": nhat_ky}


def _doc_danh_gia(tl: str) -> list[dict]:
    """VLM hay bọc JSON trong ```json … ``` hoặc kèm lời dẫn. Bóc cho chắc."""
    t = tl.strip()
    if "```" in t:
        t = t.split("```")[1]
        t = t[4:] if t.lstrip().startswith("json") else t
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < 0:
        return []
    try:
        return json.loads(t[i:j + 1]).get("danh_gia", [])
    except Exception:
        return []


def doc_ung_vien(f: Path, kho: Kho) -> list[int]:
    """CSV bài nộp -> row_id. Bỏ dòng không tra ngược được (A5.7)."""
    ra = []
    for d in f.read_text("utf-8").splitlines():
        p = d.split(",")
        if len(p) < 2:
            continue
        r = kho.tra_nguoc.get((p[0].strip(), int(p[1])))
        if r is not None and r not in ra:
            ra.append(r)
    return ra


def main():
    ap = argparse.ArgumentParser(description="tac tu tu soat ket qua truy hoi")
    ap.add_argument("--de", required=True, type=Path)
    ap.add_argument("--ung-vien", required=True, type=Path,
                    help="thư mục CSV kết quả truy hồi (đầu ra của src/run.py)")
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--ra", required=True, type=Path)
    ap.add_argument("--model", default=MODEL_GEMINI)
    ap.add_argument("--tran-goi", type=int, default=TRAN_GOI_MAC_DINH,
                    help="trần số lượt gọi VLM mỗi gói")
    ap.add_argument("--so-goi", type=int, default=0,
                    help="chỉ chạy N gói đầu — để đo thời gian trước khi chạy hết")
    ap.add_argument("--nguong", type=float, default=0.8,
                    help="điểm VLM tối thiểu để được nhấc lên. Dưới ngưỡng thì "
                         "GIỮ NGUYÊN thứ tự kênh — chưa soi không phải bằng "
                         "chứng là sai")
    ap.add_argument("--cho-them", action="store_true",
                    help="cho phép THÊM ứng viên mới. ⚠️ A27/A28 đo được thêm "
                         "ứng viên làm TỆ ĐI (-0,4). Chỉ bật để đo lại")
    a = ap.parse_args()

    key = nap_khoa()
    if not key:
        raise SystemExit("Chưa có GEMINI_API_KEY (xem .env)")

    kho = Kho(a.index)
    de = R.doc_de(a.de)
    ten_goi = sorted(de)
    if a.so_goi:
        ten_goi = ten_goi[:a.so_goi]

    a.ra.mkdir(parents=True, exist_ok=True)
    tong_goi_vlm = 0
    for ten in ten_goi:
        f = a.ung_vien / f"{ten}.csv"
        if not f.exists():
            print(f"  {ten}: không có {f.name} — bỏ qua")
            continue
        if R.loai_cua(ten) == "trake":
            print(f"  {ten}: TRAKE — bản này chưa xử lý, bỏ qua")
            continue
        uv = doc_ung_vien(f, kho)
        if not uv:
            print(f"  {ten}: không tra ngược được ứng viên nào")
            continue
        kq = xu_ly_goi(kho, ten, de[ten], uv, key, a.model, a.tran_goi,
                       a.cho_them, a.nguong)
        tong_goi_vlm += kq["so_goi_vlm"]
        (a.ra / f"{ten}.json").write_text(
            json.dumps(kq, ensure_ascii=False, indent=1), "utf-8")
        dau = {"chac": "✅", "ngo": "🟡", "khong-biet": "⚪"}[kq["do_chac"]]
        moi = kho.mo_ta(kq["xep"][0]) if kq["xep"] else "—"
        print(f"  {dau} {ten:<20} {kq['so_goi_vlm']:>2} lượt VLM  "
              f"cao nhất {kq['diem_cao_nhat']:.2f}  → {moi}")

    print(f"\n{len(ten_goi)} gói | {tong_goi_vlm} lượt gọi VLM | ra {a.ra}")
    print("So hạng trước/sau: python scripts/47_do_tac_tu.py "
          f"--truoc {a.ung_vien} --sau {a.ra}")


if __name__ == "__main__":
    main()
