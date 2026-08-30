"""
server.py — Máy chủ cục bộ cho giao diện truy vấn. Không thêm phụ thuộc nào.

    .venv\\Scripts\\python.exe web\\server.py
    .venv\\Scripts\\python.exe web\\server.py --cache index\\truy_van.npz

Rồi mở http://127.0.0.1:8000

VÌ SAO PHẢI CÓ MÁY CHỦ, KHÔNG THỂ LÀ HTML THUẦN
================================================

Bản thiết kế gốc là một **mẫu tĩnh**: dữ liệu bịa, trạng thái bịa. Muốn nó gõ
truy vấn ra kết quả thật thì phải có thứ đọc được `clip_siglip2.npy`
(177.321 × 1152 float16, 390 MB), chạy BM25, và mở file ảnh theo đường dẫn
tuyệt đối. Trình duyệt không làm được cả ba.

Nên: HTML lo phần nhìn, file này lo phần tính, và nó **gọi thẳng các module đã
có** (`run`, `bm25`, `objects`, `dense`, `nop_bai`) chứ không chép lại logic —
chép là mở đường cho hai bản lệch nhau, đúng cái bẫy `25_ma_hoa_truy_van.py`
đã cảnh báo.

Dùng `http.server` của thư viện chuẩn, không Flask/FastAPI: repo này cố tình
không có hạ tầng (PHẦN B), thêm một phụ thuộc chỉ để phục vụ 5 endpoint là đi
ngược lại điều đó.

BỐN CHỖ MÁY CHỦ NÀY PHẢI KHẮT KHE
=================================

**1. `frame_idx` luôn lấy từ cột của bảng cái.** Không nhân `pts_time × fps` ở
bất kỳ đâu — lệch 1 frame, và đó là con số nộp cho BTC.

**2. Ảnh có thể KHÔNG có, và phải nói thẳng.** Máy này chỉ 36.506/177.321 dòng
có `kf_path` (20,6% — A5.5). Ứng viên thiếu ảnh vẫn trả về, kèm cờ `co_anh:
false`, để giao diện vẽ khác đi thay vì hiện ô xám trông như đang tải.

**3. Kênh 1 có thể vắng mặt.** Máy 7,7 GB không nạp nổi SigLIP2. Không có
`--cache` thì `/api/trang_thai` báo rõ kênh nào đang chạy và điểm dev của
chúng, để người dùng biết mình đang nhìn bể ứng viên yếu hơn ~3 lần.

**4. Không tự ghi đè bài nộp.** `/api/nop` gọi `nop_bai.ghi_goi`, vốn `soat()`
trước khi ghi và từ chối hẳn nếu sai định dạng. Mỗi gói chỉ được nộp 3 lần.
"""

import argparse
import json
import mimetypes
import sys
import threading
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                        # noqa: E402
from bm25 import KenhVanBan                            # noqa: E402
from objects import KenhObjects                        # noqa: E402
from rrf import hop_nhat                               # noqa: E402
from schema import AnswerKIS, AnswerQA, AnswerTRAKE    # noqa: E402

WEB = Path(__file__).resolve().parent


class Kho:
    """Nạp một lần, dùng cho mọi truy vấn. Nạp lại mỗi lần là 20 giây/câu."""

    def __init__(self, index_dir: Path, cache: Path | None, matrix: str):
        self.index_dir = index_dir
        self.master = pd.read_parquet(index_dir / "master.parquet")
        self.kenh = {}
        self.ghi_chu = []

        # Kênh 1 — chỉ khi có cache vector truy vấn. Không cache thì KHÔNG nạp
        # model: `dense.kiem_ram` sẽ chặn, và trên máy 7,7 GB nó đã treo máy
        # hai lần.
        if cache and Path(cache).exists():
            from dense import KenhAnhCache
            self.kenh["anh"] = KenhAnhCache(str(index_dir), str(cache), matrix=matrix)
            self.ghi_chu.append("kênh 1 (SigLIP2) qua cache — 0,3258 trên dev")
        else:
            self.ghi_chu.append(
                "KHÔNG có kênh 1. Bể ứng viên yếu hơn ~3 lần bài nộp thật. "
                "Sinh cache: scripts/25_ma_hoa_truy_van.py --tap-dev --gop")

        p = index_dir / "ocr_asr.parquet"
        if p.exists():
            self.kenh["ocr"] = KenhVanBan.tu_bang_khung(
                self.master, pd.read_parquet(p), cot="text", ten="ocr_asr")
            self.ghi_chu.append("kênh 3 (OCR+ASR) — 0,1183 trên dev")

        try:
            self.kenh["objects"] = KenhObjects(str(index_dir), self.master)
            self.ghi_chu.append("kênh 4 (objects) — 0,0417 trên dev")
        except Exception as e:                       # thiếu objects.parquet
            self.ghi_chu.append(f"kênh 4 tắt: {e}")

        # Tra ngược row_id -> dòng, để đọc kf_path/pts_time trong O(1).
        # `row_id` trùng vị trí dòng (kiểm ở A39), nên dùng .values trực tiếp.
        self.kf_path = self.master.kf_path.values
        self.pts = self.master.pts_time.values
        self.fps = self.master.fps.values
        self.vid = self.master.video_id.values
        self.fidx = self.master.frame_idx.values
        self.kf_n = self.master.kf_n.values

        # Tra ngược (video_id, frame_idx) -> row_id, để dựng lại ảnh cho một
        # chuỗi TRAKE (`AnswerTRAKE` chỉ mang video_id + frame_idx).
        #
        # ⚠️ A5.7: KHÔNG phải song ánh — 614 keyframe dùng chung `frame_idx`
        # với dòng liền trước. Giữ dòng đầu tiên; chênh một dòng không đổi ảnh.
        self.tra_nguoc = {}
        for r in range(len(self.vid)):
            self.tra_nguoc.setdefault((self.vid[r], int(self.fidx[r])), r)

        self.van_ban = {}
        if "ocr" in self.kenh:
            b = pd.read_parquet(index_dir / "ocr_asr.parquet")
            for r in b.itertuples():
                t = str(getattr(r, "text", "") or "").strip()
                if t:
                    self.van_ban[int(r.row_id)] = t[:400]

    # ------------------------------------------------------------------ tìm

    def tim(self, cau: str, loai: str, k: int, dung_kenh: list[str]) -> dict:
        """Truy hồi một câu -> danh sách ứng viên đã hợp nhất.

        Câu dài bị `tach_truy_van` cắt thành mệnh đề trước khi vào kênh 1 —
        gọi lại đúng hàm của `run.py` để giao diện và bài nộp thấy CÙNG một
        bể ứng viên. Đây là chỗ dễ lệch nhất giữa hai đường chạy.
        """
        dung_kenh = [x for x in dung_kenh if x in self.kenh] or list(self.kenh)
        cac, da_dung, canh_bao = [], [], None
        for ten in dung_kenh:
            kn = self.kenh[ten]
            try:
                cac.append(kn.tim(R.tach_truy_van(cau), k=k))
                da_dung.append(ten)
            except KeyError:
                # Kênh 1 chạy từ cache vector, nên câu GÕ TAY gần như chắc chắn
                # không có sẵn. Trước đây chỗ này trả lỗi và giết cả lượt tìm —
                # tức giao diện chỉ dùng được cho câu đã nằm trong bộ đề. Nay bỏ
                # riêng kênh đó và chạy tiếp bằng kênh còn lại, có báo rõ.
                canh_bao = (
                    "Kênh 1 (ảnh) BỊ BỎ: truy vấn này chưa có trong "
                    "index/truy_van.npz. Kết quả dưới đây chỉ từ kênh văn bản "
                    "và objects — yếu hơn hẳn. Mã hoá thêm: "
                    'python scripts/25_ma_hoa_truy_van.py --them "…" --gop')
            except Exception:
                traceback.print_exc()

        cac = [c for c in cac if c]
        if not cac:
            return {"ung_vien": [], "kenh": da_dung, "canh_bao": canh_bao}

        # Một kênh thì khỏi RRF — hợp nhất một danh sách chỉ làm mất điểm gốc.
        ket = cac[0] if len(cac) == 1 else hop_nhat(cac, k=k)
        return {"ung_vien": [self._the(c, i) for i, c in enumerate(ket[:k])],
                "tho": ket[:k], "kenh": da_dung, "canh_bao": canh_bao}

    def _the(self, c, i: int) -> dict:
        r = int(c.row_id)
        return {
            "hang": i + 1,
            "row_id": r,
            "video_id": str(self.vid[r]),
            # ⚠️ frame_idx LẤY TỪ CỘT, không tính lại từ pts_time × fps.
            "frame_idx": int(self.fidx[r]),
            "pts_time": round(float(self.pts[r]), 2),
            "fps": float(self.fps[r]),
            "kf_n": int(self.kf_n[r]),
            "diem": round(float(c.score), 4),
            "nguon": c.source,
            "co_anh": bool(pd.notna(self.kf_path[r])),
            "van_ban": self.van_ban.get(r, ""),
        }

    def lap_chuoi(self, cac_su_kien: list, so_dong: int, **cau_hinh) -> list[dict]:
        """N danh sách ứng viên -> các dòng TRAKE, qua chính `run.dung_trake`.

        Không tự lắp chuỗi ở đây. `dung_trake` giữ đúng chỉ số sự kiện bằng
        quy hoạch động (`dong_hang_dp`) và ép tăng dần theo thời gian —
        dựng từ danh sách phẳng sẽ hoán đổi khung giữa các sự kiện, và
        `nop_bai.soat` chặn thẳng.

        Mỗi vị trí trong chuỗi được đánh dấu `that`: khung có mặt trong bảng
        cái, hay là khung **DP bịa ra** bằng nội suy / rải đều. Người soát
        phải thấy khác biệt đó — khung nội suy không có ảnh để soi, và nó là
        phỏng đoán chứ không phải kết quả truy hồi.
        """
        dong = R.dung_trake(cac_su_kien, self.master, so_dong=so_dong, **cau_hinh)
        ra = []
        for i, d in enumerate(dong):
            khung = []
            for f in d.frame_idxs:
                r = self.tra_nguoc.get((d.video_id, int(f)))
                khung.append({
                    "frame_idx": int(f),
                    "that": r is not None,
                    "row_id": r,
                    "co_anh": r is not None and bool(pd.notna(self.kf_path[r])),
                    # Chỉ đọc pts_time của khung THẬT. Khung nội suy thì để
                    # trống, không suy `frame_idx / fps` — con số đó trông như
                    # số đo mà thật ra là phỏng đoán chồng phỏng đoán.
                    "pts_time": round(float(self.pts[r]), 2) if r is not None else None,
                })
            ra.append({"hang": i + 1, "video_id": d.video_id,
                       "frame_idxs": [int(f) for f in d.frame_idxs],
                       "khung": khung,
                       "so_that": sum(1 for k in khung if k["that"])})
        return ra

    def lan_can(self, row_id: int, ban_kinh: int = 4) -> list[dict]:
        """Khung lân cận CÙNG VIDEO — bảng cái sắp liền nhau nên hàng xóm ở
        biên video là một video khác hẳn (xem `cua_so.bien_video`)."""
        v = self.vid[row_id]
        lo = max(0, row_id - ban_kinh)
        hi = min(len(self.vid), row_id + ban_kinh + 1)
        return [self._the(type("C", (), {"row_id": r, "score": 0.0,
                                         "source": "lan_can"})(), i)
                for i, r in enumerate(range(lo, hi)) if self.vid[r] == v]


# --------------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    kho: Kho = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):                        # bớt ồn
        pass

    def _json(self, obj, ma=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(ma)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _file(self, p: Path):
        if not p.exists():
            return self._json({"loi": "không có file"}, 404)
        b = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type",
                         mimetypes.guess_type(str(p))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        k = self.kho

        if u.path in ("/", "/index.html"):
            return self._file(WEB / "index.html")

        if u.path == "/api/trang_thai":
            de = sorted(x.name for x in (GOC / "dev").glob("*-bo-de-thi"))
            return self._json({
                "so_video": int(k.master.video_id.nunique()),
                "so_khung": int(len(k.master)),
                "so_co_anh": int(k.master.kf_path.notna().sum()),
                "kenh": list(k.kenh),
                "ghi_chu": k.ghi_chu,
                "bo_de": de,
            })

        if u.path == "/api/anh":
            r = int(q.get("row_id", [0])[0])
            p = k.kf_path[r]
            if pd.isna(p):
                return self._json({"loi": "máy này không có ảnh cho khung đó"}, 404)
            return self._file(Path(p))

        if u.path == "/api/lan_can":
            r = int(q.get("row_id", [0])[0])
            return self._json({"lan_can": k.lan_can(r)})

        if u.path == "/api/de":
            ten = q.get("bo", [""])[0]
            d = GOC / "dev" / ten
            if not d.is_dir():
                return self._json({"loi": "không có bộ đề"}, 404)
            cau = []
            for f in sorted(d.glob("query-*.txt")):
                cau.append({"ten": f.stem,
                            "loai": f.stem.rsplit("-", 1)[-1],
                            "noi_dung": f.read_text("utf-8").strip()})
            return self._json({"cau": cau})

        return self._json({"loi": "không có đường dẫn đó"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            d = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json({"loi": "JSON hỏng"}, 400)
        u = urllib.parse.urlparse(self.path)
        k = self.kho

        if u.path == "/api/tim":
            cau = (d.get("q") or "").strip()
            if not cau:
                return self._json({"loi": "truy vấn rỗng"}, 400)
            loai = d.get("loai", "kis")
            so = int(d.get("k", 100))
            kenh = d.get("kenh") or []
            if loai == "trake":
                # ⚠️ ĐẾM THEO DÒNG, KHÔNG THEO SỐ. Đề mẫu `query-p1-18-trake`
                # đánh nhầm `E1,E2,E2,E4` — tin vào con số thì ra 3 sự kiện,
                # mà sai số Frame ID là sai định dạng, MẤT TRẮNG cả gói.
                sk = R.tach_su_kien(cau)
                ds = [k.tim(s, loai, so, kenh) for s in sk]
                if any("loi" in x for x in ds):
                    return self._json(next(x for x in ds if "loi" in x), 400)
                # Bốn tham số A39 — mặc định giữ nguyên hành vi cũ, đúng kỷ
                # luật đo. Giao diện bật được để NHÌN chuỗi đổi thế nào, chứ
                # không phải để bật cho bài nộp trước khi thắng trên dev.
                a39 = {"dong_hang": d.get("dong_hang", "cu"),
                       "he_so_phat": float(d.get("he_so_phat", 0.0)),
                       "rai_hep": bool(d.get("rai_hep", False))}
                try:
                    chuoi = k.lap_chuoi([x["tho"] for x in ds], so, **a39)
                except Exception:
                    traceback.print_exc()
                    return self._json({"loi": "lắp chuỗi thất bại, xem log"}, 500)
                return self._json({
                    "so_su_kien": len(sk),
                    "su_kien": [{"text": s, "ung_vien": x["ung_vien"]}
                                for s, x in zip(sk, ds)],
                    "chuoi": chuoi, "a39": a39})
            j = k.tim(cau, loai, so, kenh)
            j.pop("tho", None)                        # Candidate không JSON được
            return self._json(j)

        if u.path == "/api/nop":
            return self._nop(d)

        return self._json({"loi": "không có đường dẫn đó"}, 404)

    def _nop(self, d):
        """Ghi một gói. `nop_bai.ghi_goi` tự `soat()` trước khi ghi và ném
        SystemExit nếu sai định dạng — bắt lại để trả lỗi ra giao diện thay vì
        làm chết máy chủ."""
        from nop_bai import ghi_goi
        ten = d.get("ten_goi", "")
        loai = ten.rsplit("-", 1)[-1] if "-" in ten else "kis"
        dong = d.get("dong", [])
        ra = GOC / d.get("ra", "submission")
        try:
            if loai == "qa":
                dap = [AnswerQA(x["video_id"], int(x["frame_idx"]),
                                x.get("answer", "")) for x in dong]
            elif loai == "trake":
                dap = [AnswerTRAKE(x["video_id"],
                                   [int(f) for f in x["frame_idxs"]]) for x in dong]
            else:
                dap = [AnswerKIS(x["video_id"], int(x["frame_idx"])) for x in dong]
            ghi_goi({ten: dap}, ra)
        except (Exception, SystemExit) as e:
            return self._json({"loi": str(e) or e.__class__.__name__}, 400)
        return self._json({"ok": True, "so_dong": len(dap),
                           "duong_dan": str(ra / f"{ten}.csv")})


def main():
    ap = argparse.ArgumentParser(description="may chu cuc bo cho giao dien")
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--cache", default=None, type=Path,
                    help="index/truy_van.npz — BẬT KÊNH 1 mà không nạp model. "
                         "Thiếu nó thì bể ứng viên yếu hơn ~3 lần")
    ap.add_argument("--matrix", default="clip_siglip2.npy")
    ap.add_argument("--cong", type=int, default=8000)
    a = ap.parse_args()

    print("Đang nạp bảng cái và các kênh...", flush=True)
    Handler.kho = Kho(a.index, a.cache, a.matrix)
    for g in Handler.kho.ghi_chu:
        print("  •", g)
    print(f"\n  http://127.0.0.1:{a.cong}\n")
    ThreadingHTTPServer(("127.0.0.1", a.cong), Handler).serve_forever()


if __name__ == "__main__":
    main()
