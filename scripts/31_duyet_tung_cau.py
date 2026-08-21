"""
31_duyet_tung_cau.py — Sinh trang HTML để NGƯỜI duyệt tay từng câu của bộ đề.

Mục đích khác hẳn `run.py`: không nhằm nộp bài, mà nhằm **đưa mắt người vào**.
Mỗi câu một khối, mỗi ứng viên một thẻ có ảnh thu nhỏ, `video_id`, `frame_idx`,
mốc thời gian và chữ OCR đọc được — đủ để một người ngồi xem và gật/lắc.

    python scripts/31_duyet_tung_cau.py --de dev/SOTUYEN1-bo-de-thi --ra duyet.html

⚠️ **KÊNH 1 (SigLIP2) KHÔNG CÓ Ở ĐÂY.** Máy 7,7 GB không nạp nổi model, và
`index/truy_van.npz` (cache vector truy vấn) chưa được đồng bộ về. Nên ứng viên
trong trang này đến từ **kênh 3 (OCR+ASR) và kênh 4 (objects)** — đo trên tập dev
lần lượt 0,1183 và 0,0417, so với **0,3258** của kênh 1.

Nói thẳng: đây là **bể ứng viên yếu hơn ~3 lần** so với thứ đã ăn 6,2 điểm. Trang
này dùng để **soi và bắt lỗi**, không phải để chốt bài nộp. Có `truy_van.npz` thì
chạy lại với `--cache` là ra đúng bể của bài nộp thật.

⚠️ **Chỉ 21% kho có ảnh trên máy này** (L21/L22/L24/L27/L30 — A5.5). Ứng viên
không có ảnh vẫn hiện, kèm chữ OCR làm bằng chứng thay thế, và được đánh dấu rõ
— để người duyệt không nhầm "không có ảnh" thành "không có kết quả".
"""

import argparse
import base64
import html
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                       # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from objects import KenhObjects                       # noqa: E402
from rrf import hop_nhat                              # noqa: E402
from tra_loi_ocr import thu_nho                       # noqa: E402


def mm_ss(giay: float) -> str:
    return f"{int(giay) // 60:d}:{int(giay) % 60:02d}"


def the(master, bang, c, hang: int, rong: int) -> str:
    """Một thẻ ứng viên: ảnh (nếu có) + số liệu + chữ OCR."""
    g = master.iloc[c.row_id]
    p = g.kf_path if isinstance(g.kf_path, str) else None
    anh = thu_nho(p, rong=rong, chat_luong=60) if p and Path(p).exists() else b""
    if anh:
        img = (f'<img src="data:image/jpeg;base64,{base64.b64encode(anh).decode()}" '
               f'alt="{html.escape(g.video_id)} {int(g.frame_idx)}">')
    else:
        img = ('<div class="khong-anh">chưa tải ảnh<br><span>nhóm '
               f'{html.escape(g.video_id[:3])} không có trên máy này</span></div>')

    x = bang.iloc[int(c.row_id)]
    ocr = " ".join(str(x.get("ocr_text", "") or "").split())[:150]
    nguon = ", ".join(c.meta.get("nguon", [])) if c.meta.get("nguon") else (c.source or "")

    return f"""<figure class="the">
  <div class="hang">#{hang}</div>
  {img}
  <figcaption>
    <span class="ma">{html.escape(g.video_id)} · {int(g.frame_idx)}
      <span class="gio">({mm_ss(float(g.pts_time))})</span></span>
    <div class="nguon">{html.escape(nguon)}</div>
    {'<div class="ocr">' + html.escape(ocr) + '</div>' if ocr else ''}
  </figcaption>
</figure>"""


def main():
    ap = argparse.ArgumentParser(description="sinh trang HTML duyet tay tung cau")
    ap.add_argument("--de", required=True, type=Path)
    ap.add_argument("--ra", default=Path("duyet_tung_cau.html"), type=Path)
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--so-ung-vien", type=int, default=8)
    ap.add_argument("--rong-anh", type=int, default=280)
    a = ap.parse_args()

    master = pd.read_parquet(a.index / "master.parquet")
    bang = pd.read_parquet(a.index / "ocr_asr.parquet")
    k3 = KenhVanBan.tu_bang_khung(master, bang, cot="text", ten="ocr_asr")
    k4 = KenhObjects(a.index, master)
    de = R.doc_de(a.de)
    print(f"{len(de)} gói | kênh 3 ({len(k3):,} khung có chữ) + kênh 4 objects\n")

    khoi = []
    for ten in sorted(de, key=lambda t: (R.loai_cua(t), t)):
        nd = de[ten]
        loai = R.loai_cua(ten)
        if loai == "trake":
            sk = R.tach_su_kien(nd)
            ds = [k3.tim(R.tach_truy_van(x), k=a.so_ung_vien) for x in sk]
            uv = [c for d in ds for c in d][:a.so_ung_vien]
        else:
            uv = hop_nhat([k3.tim(R.tach_truy_van(nd), k=50),
                           k4.tim(nd, k=50)])[:a.so_ung_vien]
        the_html = "\n".join(the(master, bang, c, i, a.rong_anh)
                             for i, c in enumerate(uv, 1))
        khoi.append(f"""<section id="{html.escape(ten)}">
  <div class="dau">
    <h2>{html.escape(ten)}</h2>
    <span class="loai">{loai.upper()}</span>
  </div>
  <blockquote>{html.escape(nd)}</blockquote>
  <div class="luoi">{the_html or '<p>Không kênh nào trả về ứng viên.</p>'}</div>
</section>""")
        print(f"  {ten:<22} {len(uv)} ứng viên")

    css = """
/* Bảng màu lấy từ phòng dựng phim: nền than có ánh lam, ảnh nổi lên trên đó;
   nhấn màu bút mỡ cam — thứ người ta khoanh lên tấm contact sheet.          */
:root{
  --nen:#faf9f7; --tam:#fff; --chu:#17181c; --mo:#6b6f76; --vien:#e2e0dc;
  --nhan:#f1efeb; --but:#c2410c; --but-mo:#fde6d3; --khung:#8a8f98;
}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
  --nen:#0e1013; --tam:#171a1f; --chu:#e9eaec; --mo:#8b9199; --vien:#262a31;
  --nhan:#1c2027; --but:#fb923c; --but-mo:#3a2213; --khung:#666c75;
}}
:root[data-theme=dark]{
  --nen:#0e1013; --tam:#171a1f; --chu:#e9eaec; --mo:#8b9199; --vien:#262a31;
  --nhan:#1c2027; --but:#fb923c; --but-mo:#3a2213; --khung:#666c75;
}
*{box-sizing:border-box}
body{background:var(--nen);color:var(--chu);margin:0;padding:28px 22px 80px;
  font:400 15px/1.6 "Be Vietnam Pro",system-ui,sans-serif;
  max-width:1460px;margin-inline:auto}
h1{font-size:clamp(24px,3vw,34px);font-weight:800;letter-spacing:-.02em;
  margin:0;text-wrap:balance}
.phu{color:var(--mo);font-size:14px;margin:6px 0 0}
.canh{background:var(--nhan);border-left:3px solid var(--but);
  padding:14px 18px;border-radius:4px;margin:22px 0 30px;font-size:13.5px;
  line-height:1.65;max-width:78ch}
.canh b{color:var(--but)}
/* mục lục: 25 gói thì phải nhảy được, không cuộn tay */
.muc-luc{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 34px}
.muc-luc a{font:600 11.5px/1 "JetBrains Mono",ui-monospace,monospace;
  text-decoration:none;color:var(--mo);background:var(--nhan);
  border:1px solid var(--vien);padding:6px 9px;border-radius:3px}
.muc-luc a:hover,.muc-luc a:focus-visible{color:var(--but);border-color:var(--but)}
section{margin-top:44px;scroll-margin-top:16px}
.dau{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;
  border-bottom:1px solid var(--vien);padding-bottom:10px;margin-bottom:14px}
h2{font:700 15px/1 "JetBrains Mono",ui-monospace,monospace;margin:0;
  letter-spacing:-.01em}
.loai{font:600 10.5px/1 "JetBrains Mono",monospace;letter-spacing:.09em;
  padding:4px 8px;border-radius:2px;background:var(--but-mo);color:var(--but)}
blockquote{margin:0 0 18px;padding:0 0 0 16px;border-left:2px solid var(--vien);
  white-space:pre-wrap;font-size:14.5px;line-height:1.7;max-width:82ch;
  color:var(--chu)}
/* lưới contact sheet */
.luoi{display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(250px,1fr))}
.the{margin:0;background:var(--tam);border:1px solid var(--vien);border-radius:3px;
  overflow:hidden;position:relative}
.the img{width:100%;display:block;aspect-ratio:16/9;object-fit:cover}
.hang{position:absolute;top:0;left:0;
  font:700 11px/1 "JetBrains Mono",monospace;letter-spacing:.05em;
  background:var(--but);color:#fff;padding:5px 9px;border-radius:0 0 3px 0}
.khong-anh{aspect-ratio:16/9;display:grid;place-content:center;text-align:center;
  gap:4px;background:repeating-linear-gradient(45deg,var(--nhan),
    var(--nhan) 9px,transparent 9px,transparent 18px);
  color:var(--mo);font-size:12.5px}
.khong-anh span{font-size:11px;color:var(--khung)}
figcaption{padding:10px 12px;font-size:12.5px}
.ma{font:600 12px/1.45 "JetBrains Mono",ui-monospace,monospace;
  font-variant-numeric:tabular-nums;display:block}
.gio{color:var(--mo);font-weight:400}
.nguon{color:var(--khung);font-size:10.5px;margin-top:5px;
  font-family:"JetBrains Mono",monospace;letter-spacing:.02em}
.ocr{margin-top:8px;padding-top:8px;border-top:1px dashed var(--vien);
  color:var(--mo);font-size:11.5px;line-height:1.5;word-break:break-word}
@media(prefers-reduced-motion:no-preference){
  .muc-luc a,.the{transition:border-color .15s,color .15s}}
"""
    muc_luc = "".join(
        f'<a href="#{html.escape(t)}">{html.escape(t.replace("query-p1-", ""))}</a>'
        for t in sorted(de, key=lambda t: (R.loai_cua(t), t)))

    trang = f"""<title>Contact Sheet Sơ Tuyển 1</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap">
<style>{css}</style>
<h1>Contact sheet — {html.escape(a.de.name)}</h1>
<p class="phu">{len(de)} gói · {a.so_ung_vien} ứng viên mỗi gói ·
xếp theo loại truy vấn</p>
<div class="canh">
<b>⚠️ Đây KHÔNG phải bể ứng viên của bài nộp thật.</b> Kênh 1 (SigLIP2, 0,3258)
không chạy được trên máy dựng trang này — thiếu RAM và chưa có
<code>index/truy_van.npz</code>. Ứng viên dưới đây đến từ kênh 3 (OCR+ASR,
<b>0,1183</b>) và kênh 4 (objects, <b>0,0417</b>), tức <b>yếu hơn khoảng ba
lần</b>. Dùng trang này để soi và bắt lỗi, đừng dùng để chốt bài nộp.
<br><br>
Chỉ <b>21% kho</b> có ảnh trên máy này (L21/L22/L24/L27/L30). Ứng viên không có
ảnh vẫn được liệt kê kèm chữ OCR làm bằng chứng thay thế.
</div>
<nav class="muc-luc">{muc_luc}</nav>
{"".join(khoi)}"""

    Path(a.ra).write_text(trang, encoding="utf-8")
    mb = Path(a.ra).stat().st_size / 1024 ** 2
    print(f"\n✅ {a.ra}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
