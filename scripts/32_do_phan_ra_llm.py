"""
32_do_phan_ra_llm.py — Việc 4 của 12_viec_cho_may_manh.md: phân rã truy vấn dài
bằng LLM thành 3 mệnh đề thị giác ngắn, đo trên tập dev trước khi nộp.

VÌ SAO — đề thi thật dài 63 từ/2,4 mệnh đề, SigLIP2 chỉ nhận 64 token
(`run.tach_truy_van`, A19/A20: 100% truy vấn đề mẫu bị cắt cụt). Hiện đang cắt
THEO DẤU CÂU — thô, không đảm bảo mỗi mảnh là một Ý THỊ GIÁC trọn vẹn. Thử để
LLM viết lại theo 3 góc nhìn cố định:

    1. cảnh tổng thể / không gian
    2. hành động của nhân vật
    3. vật thể đặc trưng cận cảnh

rồi gộp điểm:  Score(i) = max_j cos(v_i,q_j) + λ · Σ_j cos(v_i,q_j)

⚠️ CÂU DEV NGẮN HƠN ĐỀ THẬT tới 3 LẦN (dev tự soạn ~15-20 từ, đề thật 63 từ) —
`tach_truy_van` trên dev phần lớn KHÔNG cắt gì (câu đã dưới trần token), nên
mốc nền ở đây gần như là "câu gốc nguyên vẹn". Đây CÓ THỂ là một dạng mù khác
của tập dev (cùng họ A19/A20/A31): kỹ thuật nhắm đúng vào vấn đề CÂU DÀI mà
dev lại không có câu đủ dài để bộc lộ vấn đề đó. Đọc kết quả với lưu ý này.

Vector mới được lưu vào `index/truy_van_viec4.npz` — KHÔNG ghi đè
`index/truy_van.npz` đang dùng chung, để không ảnh hưởng máy khác khi kỹ
thuật này còn đang thử nghiệm.

    python scripts/32_do_phan_ra_llm.py --cache index/truy_van.npz --lam 0.15
"""

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "src"))

import run as R                                        # noqa: E402
import tap_dev                                         # noqa: E402
from cham_diem import bao_cao_do_nhay                  # noqa: E402
from dense import KenhAnhCache                          # noqa: E402
from schema import Candidate                            # noqa: E402
from tra_loi_ocr import MODEL_GEMINI, _goi_gemini       # noqa: E402


NHAC = """Câu truy vấn thị giác tiếng Việt sau đây mô tả một khoảnh khắc \
trong video. Viết lại thành ĐÚNG 3 câu tiếng Việt NGẮN (mỗi câu tối đa 20 \
từ), mỗi câu tả một góc nhìn:

  1. CẢNH TỔNG THỂ — không gian, bối cảnh chung
  2. HÀNH ĐỘNG — nhân vật đang làm gì
  3. VẬT THỂ CẬN CẢNH — vật/chi tiết đặc trưng nhất, nhìn rõ nhất

Giữ nguyên MÀU SẮC, SỐ LƯỢNG, CHỮ VIẾT nếu câu gốc có nhắc tới — đừng bỏ sót.
Chỉ trả về một mảng JSON đúng 3 chuỗi, không thêm chữ nào khác.

Câu gốc: {cau}

JSON:"""


def _nap_script_25():
    p = GOC / "scripts" / "25_ma_hoa_truy_van.py"
    spec = importlib.util.spec_from_file_location("ma_hoa_truy_van", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def phan_ra(cau_hoi: str, model: str) -> list[str]:
    """Gọi Gemini, trả về đúng 3 mệnh đề — rỗng nếu parse thất bại."""
    tho = _goi_gemini(NHAC.format(cau=cau_hoi), model=model)
    m = re.search(r"\[.*\]", tho or "", re.DOTALL)
    if not m:
        return []
    try:
        ra = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    ra = [str(x).strip() for x in ra if str(x).strip()]
    return ra[:3]


def tim_gop(kenh, mennh_de: list[str], k: int, lam: float) -> list[Candidate]:
    """Giống `KenhAnh.tim` nhưng gộp điểm max + λ·tổng thay vì chỉ lấy max."""
    sims = np.array([kenh._nhan(kenh.encode_text(c)) for c in mennh_de])
    sim = sims.max(axis=0) + lam * sims.sum(axis=0)

    lay = min(len(sim), k + 200)
    top = np.argpartition(-sim, lay - 1)[:lay]
    top = top[np.argsort(-sim[top])][:k]

    return [Candidate(row_id=int(i), video_id=r.video_id,
                      frame_idx=int(r.frame_idx), score=float(sim[i]),
                      source="phan_ra_llm",
                      meta={"pts_time": float(r.pts_time)})
            for i, r in zip(top, kenh.master.iloc[top].itertuples())]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--cache", type=Path, required=True,
                    help="index/truy_van.npz — cho mốc nền tach_truy_van")
    ap.add_argument("--matrix", default="clip_siglip2.npy")
    ap.add_argument("--lam", type=float, default=0.15, help="λ ở công thức")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--model", default=MODEL_GEMINI)
    ap.add_argument("--cho", type=float, default=4.5)
    ap.add_argument("--ra-cache", default=None, type=Path,
                    help="mặc định index/truy_van_viec4.npz — KHÔNG đè "
                         "truy_van.npz đang dùng chung")
    ap.add_argument("--fp16", action="store_true",
                    help="hạ ngưỡng RAM cần trống xuống ~3 GB")
    ap.add_argument("--phan-ra-cache", default=None, type=Path,
                    help="file JSON lưu/đọc kết quả phân rã — tránh gọi lại "
                         "Gemini khi chạy lại sau lỗi (mặc định "
                         "index/phan_ra_viec4.json)")
    a = ap.parse_args()

    index = GOC / "index"
    ra_cache = a.ra_cache or (index / "truy_van_viec4.npz")
    pr_cache = a.phan_ra_cache or (index / "phan_ra_viec4.json")

    k1 = KenhAnhCache(str(index), a.cache, matrix=a.matrix)
    cau = [c for c in tap_dev.doc() if c.loai == "KIS"]
    print(f"{len(cau)} câu KIS | phân rã bằng {a.model}, λ={a.lam}, "
          f"nghỉ {a.cho}s/lượt\n")

    mennh_de_theo_cau = json.loads(pr_cache.read_text("utf-8")) \
        if pr_cache.exists() else {}
    if mennh_de_theo_cau:
        print(f"Đã có {len(mennh_de_theo_cau)} câu phân rã sẵn trong "
              f"{pr_cache} — chỉ phân rã câu còn thiếu")

    goc, moi_cau = {}, []
    for i, c in enumerate(cau, 1):
        goc[c.id] = k1.tim(R.tach_truy_van(c.cau_hoi), k=a.k)
        if c.id not in mennh_de_theo_cau:
            mennh_de_theo_cau[c.id] = phan_ra(c.cau_hoi, a.model)
            pr_cache.write_text(
                json.dumps(mennh_de_theo_cau, ensure_ascii=False, indent=1),
                "utf-8")
            if i < len(cau):
                time.sleep(a.cho)
        md = mennh_de_theo_cau[c.id]
        moi_cau.extend(md)
        print(f"  [{i}/{len(cau)}] {c.id}  {len(md)} mệnh đề"
              + (f"  vd: {md[0][:40]!r}" if md else "  ⚠️ parse thất bại"))

    thieu = len([c for c in cau if not mennh_de_theo_cau[c.id]])
    if thieu:
        print(f"\n⚠️  {thieu}/{len(cau)} câu phân rã thất bại — giữ "
              f"nguyên tach_truy_van cho câu đó ở cấu hình 'phân rã'")

    print(f"\n{len(set(moi_cau))} mệnh đề mới cần mã hoá...")
    mht = _nap_script_25()
    cu = {}
    if a.cache.exists():
        z = np.load(a.cache, allow_pickle=False)
        cu = {str(c): np.asarray(z["vec"][i], dtype=np.float32)
              for i, c in enumerate(z["cau"])}
    can = [c for c in dict.fromkeys(moi_cau) if c not in cu]
    if can:
        vec, ghi_chu = mht.ma_hoa(can, a.matrix, index, a.fp16)
        for c, v in zip(can, vec):
            cu[c] = v
    else:
        ghi_chu = json.loads(str(np.load(a.cache, allow_pickle=False)["ghi_chu"]))

    cau_list = list(cu)
    np.savez_compressed(
        ra_cache, cau=np.array(cau_list, dtype=object).astype(str),
        vec=np.vstack([cu[c] for c in cau_list]).astype(np.float32),
        ghi_chu=json.dumps(ghi_chu, ensure_ascii=False))
    print(f"✅ {len(cau_list)} câu -> {ra_cache}")

    k1_ext = KenhAnhCache(str(index), str(ra_cache), matrix=a.matrix)
    phan_ra_kq = {}
    for c in cau:
        md = mennh_de_theo_cau[c.id]
        if not md:
            phan_ra_kq[c.id] = goc[c.id]
            continue
        phan_ra_kq[c.id] = tim_gop(k1_ext, md, a.k, a.lam)

    print("\n" + "=" * 70)
    print(bao_cao_do_nhay(
        cau,
        {
            "tach_truy_van (mốc nền)": lambda c: goc[c.id],
            f"LLM phân rã 3 mệnh đề, λ={a.lam}": lambda c: phan_ra_kq[c.id],
        },
        pd.read_parquet(index / "master.parquet"),
    ))


if __name__ == "__main__":
    main()
