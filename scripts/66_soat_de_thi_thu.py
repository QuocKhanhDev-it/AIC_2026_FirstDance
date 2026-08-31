"""
66_soat_de_thi_thu.py — Trang soi ảnh để tìm đáp án cho 24 gói `de_thi_thu/`.

    python scripts/66_soat_de_thi_thu.py
    python scripts/66_soat_de_thi_thu.py --dau 60 --chi kis

VÌ SAO ĐÁNG LÀM TRƯỚC MỌI THỨ KHÁC

Tập đề thật có **52 câu**, và ngưỡng nhiễu ở cỡ đó là ±0,02–0,03. Ba kết quả
🟡 gần đây (ủng hộ video +0,0077, giữ 2 mệnh đề +0,0154, trọng số kênh 3
+0,0077) có thể đều đúng — tập dev quá nhỏ để biết. 24 gói này đưa 52 -> **76
câu (+46%)** và kéo ngưỡng nhiễu xuống, tự phân định những câu 🟡 đó mà không
cần thêm ý tưởng nào.

CÁCH LÀM: MÁY LỌC, NGƯỜI QUYẾT

Dò tay 177.321 keyframe là không tưởng. Nhưng A54 đo được: với bể 300, đáp án
nằm trong đó ở **41/49** câu đề thật. Nên máy đưa ra vài chục ứng viên đầu, còn
người chỉ việc **nhận ra cảnh** — việc mắt người làm trong một giây.

Trang HTML sinh ra dùng **ảnh thu nhỏ 256px** (`src/anh.py`), nên soi được cả
những video máy này không có ảnh gốc.

⚠️ ĐÁP ÁN TÌM THẤY LÀ NIỀM TIN, KHÔNG PHẢI SỰ THẬT — y như A46 đã ghi cho đợt
2. Trang này ghi mọi câu ở nhãn `do_chac: kha`; ai soát kỹ thì tự sửa thành
`xong`. Một đáp án sai nằm trong thước đo làm lệch mọi phép đo sau đó mà không
có gì báo.

⚠️ KHÔNG có ảnh thì KHÔNG đoán. Ô nào không có ảnh sẽ hiện xám — bấm chọn một
ô như thế là ghi vào tập dev một đáp án chưa ai nhìn thấy.
"""

import argparse
import html
import json
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import anh as ANH                                     # noqa: E402
import run as R                                       # noqa: E402
from bm25 import KenhVanBan                           # noqa: E402
from dense import KenhAnhCache                        # noqa: E402
from rrf import hop_nhat                              # noqa: E402

W3 = 0.5


def _phut(giay: float) -> str:
    return f"{int(giay) // 60}:{int(giay) % 60:02d}"


def _the(c, master, thu_tu: int) -> str:
    """Một ô ứng viên: ảnh + số liệu đủ để người soi quyết."""
    d = master.iloc[c.row_id]
    p, nho = ANH.tim(d.kf_path, d.video_id, d.kf_n)
    if p is None:
        img = '<div class="thieu">không có ảnh</div>'
    else:
        img = (f'<img loading="lazy" src="{Path(p).as_uri()}" '
               f'alt="{d.video_id}">')
    return (f'<div class="o{"" if p else " xam"}" data-rid="{c.row_id}">'
            f'{img}<div class="n">#{thu_tu} · {html.escape(str(d.video_id))} '
            f'· kf {int(d.kf_n)} · {_phut(float(d.pts_time))}'
            f'{" · <b>nhỏ</b>" if nho else ""}</div></div>')


CSS = """
body{font:14px system-ui;margin:0;background:#f6f6f7;color:#111}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;
 padding:10px 16px;z-index:9}
button{font:inherit;padding:6px 12px;cursor:pointer}
section{margin:20px 16px;background:#fff;border:1px solid #ddd;border-radius:8px}
h2{margin:0;padding:10px 14px;border-bottom:1px solid #eee;font-size:15px}
pre{margin:0;padding:10px 14px;white-space:pre-wrap;background:#fafafa;
 border-bottom:1px solid #eee;font:13px/1.5 system-ui}
.luoi{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));
 gap:8px;padding:12px}
.o{border:3px solid transparent;border-radius:6px;cursor:pointer;overflow:hidden;
 background:#f0f0f0}
.o img{width:100%;height:120px;object-fit:cover;display:block}
.o .n{font-size:11px;padding:3px 5px;color:#444}
.o.chon{border-color:#1a7f37;background:#e6f4ea}
.o.xam{opacity:.45}
.thieu{height:120px;display:grid;place-items:center;color:#999;font-size:12px}
#ra{width:calc(100% - 32px);height:180px;margin:16px;font:12px monospace}
"""

JS = """
const K='soat_de_thi_thu';
const luu=JSON.parse(localStorage.getItem(K)||'{}');
function ve(){
 document.querySelectorAll('.o').forEach(o=>{
  const s=o.closest('.luoi').dataset.khoa;
  o.classList.toggle('chon',(luu[s]||[]).includes(+o.dataset.rid));});
 document.querySelectorAll('.dem').forEach(e=>{
  e.textContent=(luu[e.dataset.khoa]||[]).length+' đã chọn';});}
document.addEventListener('click',e=>{
 const o=e.target.closest('.o'); if(!o)return;
 const s=o.closest('.luoi').dataset.khoa, r=+o.dataset.rid;
 const a=luu[s]||[]; const i=a.indexOf(r);
 i<0?a.push(r):a.splice(i,1); luu[s]=a;
 localStorage.setItem(K,JSON.stringify(luu)); ve();});
function xuat(){
 const ra=[];
 for(const c of CAU){
  const kh=c.khoa.map(k=>luu[k]||[]);
  if(kh.every(x=>x.length===0))continue;          // chưa chọn -> bỏ qua
  if(c.loai==='TRAKE'&&kh.some(x=>x.length===0)){
   ra.push('// '+c.id+': THIEU '+kh.filter(x=>!x.length).length+
           ' su kien — chua xuat');continue;}
  ra.push(JSON.stringify({id:c.id,loai:c.loai,cau_hoi:c.cau_hoi,
   row_id_dung:c.loai==='TRAKE'?kh:kh[0],dap_an:'',nguon:c.nguon,
   ghi_chu:'do_chac: kha — soi bang 66_soat_de_thi_thu.py, CHUA doi chieu anh goc'}));}
 document.getElementById('ra').value=ra.join('\\n');
 document.getElementById('ra').scrollIntoView();}
function xoa(){if(confirm('Xoá mọi lựa chọn?')){localStorage.removeItem(K);
 location.reload();}}
addEventListener('DOMContentLoaded',ve);
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--de", default=GOC / "de_thi_thu", type=Path)
    ap.add_argument("--dau", type=int, default=40, help="ứng viên mỗi câu")
    ap.add_argument("--be", type=int, default=300)
    ap.add_argument("--chi", default=None, help="lọc theo loại: kis | qa | trake")
    ap.add_argument("--ra", default=GOC / "dev" / "soat_de_thi_thu.html", type=Path)
    a = ap.parse_args()

    tep = sorted(a.de.glob("*.txt"))
    if a.chi:
        tep = [f for f in tep if R.loai_cua(f.stem) == a.chi]
    if not tep:
        raise SystemExit(f"không thấy .txt nào trong {a.de}")

    master = pd.read_parquet(a.index / "master.parquet")
    k1 = KenhAnhCache(str(a.index), str(a.index / "truy_van_gopt.npz"),
                      matrix="clip_gopt.npy")
    k3 = KenhVanBan.tu_bang_khung(
        master, pd.read_parquet(a.index / "ocr_asr.parquet"),
        cot="text", ten="ocr_asr")

    def tim(van: str):
        md = R.tach_truy_van(van)
        thieu = k1.co_du(md)
        if thieu:
            return None
        anh = hop_nhat([k1.tim(m, k=a.be) for m in md])
        return hop_nhat([anh, k3.tim(van, k=a.be)], trong_so=[1.0, W3])[:a.dau]

    phan, meta, bo_qua = [], [], []
    for f in tep:
        loai = {"kis": "KIS", "qa": "QA", "trake": "TRAKE"}[R.loai_cua(f.stem)]
        noi = f.read_text("utf-8").strip()
        cid = f"{R.loai_cua(f.stem)}-TT-{f.stem.replace('query-', '')}"
        muc = R.tach_su_kien(noi) if loai == "TRAKE" else [noi]

        khoa, than = [], []
        for i, van in enumerate(muc):
            ds = tim(van)
            if ds is None:
                bo_qua.append((cid, i + 1))
                continue
            k = f"{cid}#{i}"
            khoa.append(k)
            nhan = f"Sự kiện {i + 1}" if loai == "TRAKE" else "Ứng viên"
            than.append(
                f'<pre>{html.escape(van)}</pre>'
                f'<h2>{nhan} — <span class="dem" data-khoa="{k}"></span></h2>'
                f'<div class="luoi" data-khoa="{k}">'
                + "".join(_the(c, master, j) for j, c in enumerate(ds, 1))
                + "</div>")
        if not khoa:
            continue
        phan.append(f'<section><h2>{html.escape(cid)} · {loai} · '
                    f'{html.escape(f.name)}</h2>' + "".join(than) + "</section>")
        meta.append({"id": cid, "loai": loai, "cau_hoi": noi,
                     "nguon": f"de_thi_thu/{f.name}", "khoa": khoa})

    trang = (f"<!doctype html><meta charset=utf-8><title>Soi đề thi thử</title>"
             f"<style>{CSS}</style>"
             f"<header><b>{len(meta)} câu</b> · bấm vào ô để chọn/bỏ chọn "
             f"(chọn được nhiều ô cho một sự kiện) · "
             f'<button onclick="xuat()">Xuất JSONL</button> '
             f'<button onclick="xoa()">Xoá lựa chọn</button></header>'
             + "".join(phan)
             + f'<textarea id="ra" placeholder="Bấm Xuất JSONL rồi chép vào '
               f'dev/tap_de_thi_thu.jsonl"></textarea>'
               f"<script>const CAU={json.dumps(meta, ensure_ascii=False)};"
               f"{JS}</script>")
    a.ra.write_text(trang, encoding="utf-8", newline="\n")

    print(f"✅ {a.ra}  ({len(meta)} câu, {a.dau} ứng viên/sự kiện)")
    if bo_qua:
        print(f"⚠️ {len(bo_qua)} mục thiếu chuỗi trong cache truy vấn — "
              f"mã hoá thêm bằng scripts/25_ma_hoa_truy_van.py:")
        for cid, i in bo_qua[:5]:
            print(f"     {cid} sự kiện {i}")
    print("   Mở bằng trình duyệt. Lựa chọn lưu trong localStorage nên đóng "
          "tab không mất.")


if __name__ == "__main__":
    main()
