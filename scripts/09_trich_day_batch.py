"""
09_trich_day_batch.py — TV2 "chạy phân tán": lấp các khoảng trống keyframe cho
CẢ nhóm video một máy đang giữ, xuất Parquet nhẹ để gộp qua Drive.

Vì sao cần thêm script này ngoài `src/trich_day.py`: hàm lõi ở đó trích quanh
MỘT khoảnh khắc (dùng lúc truy vấn, Bước 4 TRAKE). Script này khác — chạy
MỘT LẦN, quét mọi video máy này có sẵn, tự tìm những cặp keyframe cách nhau xa
(> `--nguong` frame — mặc định 10, đúng ngưỡng "cửa sổ 10 frame" ở A1) rồi lấp
thêm khung ở giữa, cách nhau `--buoc` frame. Kết quả là một bảng "keyframe dày
hơn" dùng chung cho MỌI truy vấn sau này, không phải trích lại từ đầu mỗi câu.

⚠️ ĐẮT dù đã tối ưu — đo trước khi chạy hết. 539 video / 86.830 keyframe local
(nhóm L23+L26+L27, đo 14/08) cần lấp ~430.000 khung ở nguong=10/buoc=10. Mốc
cũ ~45 ms/khung (16 luồng, một tiến trình `ffmpeg` MỖI KHUNG) ước ~5-6 GIỜ.

Tối ưu H2 #4 (A1, `trich_day.trich_nhieu`): mỗi KHOẢNG TRỐNG giữa hai keyframe
giờ trích bằng MỘT lệnh ffmpeg cho cả khoảng, không phải một lệnh mỗi khung —
đo được nhanh gấp 8,7 lần ở mức khung đơn (169 -> 19 ms/khung). Lợi ích thật
phụ thuộc độ rộng khoảng trống trung bình: khoảng càng nhiều khung thì càng
lời. LUÔN `--limit` một nhóm nhỏ để thử trước khi chạy cả đêm cho nhóm lớn.

Ảnh đã decode nằm trong `cache/` — NẶNG, chỉ ở lại máy này, không đẩy git,
không đồng bộ Drive (xem trich_day.py, cùng lý do "chỉ chuyển cái nhẹ" ở B4
kế hoạch: Parquet nhẹ, ảnh nặng). Parquet xuất ra CHỈ chứa toạ độ frame_idx +
đường dẫn cache cục bộ — người khác gộp Parquet để BIẾT khung nào đã có ở máy
nào, không phải để lấy ảnh.

Chạy:
    python scripts\\09_trich_day_batch.py --group L23 --workers 16
    python scripts\\09_trich_day_batch.py --group L23 --limit 3   # thử nhanh trước
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from trich_day import trich_nhieu, _duong_cache          # tái dùng lõi đã test


def cac_khoang_trong(g: pd.DataFrame, nguong: int, buoc: int):
    """`g`: keyframe của MỘT video, đã sort theo frame_idx. Trả về danh sách
    KHOẢNG TRỐNG cần lấp — mỗi khoảng nằm NGHIÊM NGẶT giữa hai keyframe cách
    nhau > `nguong` frame, neo thời gian theo keyframe BÊN TRÁI (sai số không
    cộng dồn qua cả video, xem docstring trich_day.py). Gom theo khoảng (thay
    vì làm phẳng thành danh sách khung rời) để trích cả khoảng bằng MỘT lệnh
    ffmpeg (H2 #4, A1).

    Trả về list các dict: {anchor_frame, anchor_pts_time, fidxs: [...]}.
    """
    ra = []
    truoc = None
    for r in g.itertuples():
        if truoc is not None and r.frame_idx - truoc.frame_idx > nguong:
            fidxs = list(range(truoc.frame_idx + buoc, r.frame_idx, buoc))
            if fidxs:
                ra.append({"anchor_frame": truoc.frame_idx,
                          "anchor_pts_time": truoc.pts_time, "fidxs": fidxs})
        truoc = r
    return ra


def cac_khung_can_lap(g: pd.DataFrame, nguong: int, buoc: int):
    """Danh sách phẳng `(frame_idx, pts_time)` cần trích thêm — dùng cho vòng
    đối chiếu cuối (khung đã cache từ lần chạy trước). Xem `cac_khoang_trong`
    cho bản gom theo khoảng, dùng khi thật sự gọi ffmpeg."""
    fps = float(g.fps.iloc[0])
    return [(fidx, round(khoang["anchor_pts_time"] + (fidx - khoang["anchor_frame"]) / fps, 3))
            for khoang in cac_khoang_trong(g, nguong, buoc) for fidx in khoang["fidxs"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="./index")
    ap.add_argument("--group", action="append", default=None,
                    help="lọc tiền tố video_id, vd L23. Lặp lại để nhiều nhóm. "
                         "Bỏ trống = MỌI video có sẵn .mp4 trên máy này.")
    ap.add_argument("--nguong", type=int, default=10,
                    help="chỉ lấp khoảng cách keyframe > nguong frame")
    ap.add_argument("--buoc", type=int, default=10, help="khoảng cách giữa các khung lấp thêm")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default=None,
                    help="đường dẫn Parquet xuất ra, mặc định index/dense_<group|toanbo>.parquet")
    ap.add_argument("--limit", type=int, default=None, help="giới hạn số video, để THỬ nhanh")
    a = ap.parse_args()

    m = pd.read_parquet(Path(a.index) / "master.parquet")
    m = m[m.video_path.notna() & m.kf_path.notna()]
    if a.group:
        m = m[m.video_id.str.startswith(tuple(a.group))]

    vids = sorted(v for v in m.video_id.unique() if Path(m[m.video_id == v].video_path.iloc[0]).exists())
    if not vids:
        raise SystemExit("Không có video nào có file .mp4 tồn tại thật trên máy này khớp bộ lọc.")
    if a.limit:
        vids = vids[:a.limit]

    # gom danh sách KHOẢNG TRỐNG cần trích của MỌI video trước (không phải
    # từng khung rời) — mỗi khoảng trích bằng MỘT lệnh ffmpeg (H2 #4, A1),
    # để in tổng số & ước lượng thời gian trước khi bắt đầu đốt CPU thật.
    viec = []
    tong_khung = 0
    for vid in vids:
        g = m[m.video_id == vid].sort_values("frame_idx")
        video_path = g.video_path.iloc[0]
        fps = float(g.fps.iloc[0])
        for khoang in cac_khoang_trong(g, a.nguong, a.buoc):
            fidxs_thieu = [f for f in khoang["fidxs"]
                          if not _duong_cache(a.cache, vid, f).exists()]
            if fidxs_thieu:            # đã cache từ lần chạy trước -> khỏi trích lại
                viec.append((vid, video_path, fps, khoang["anchor_frame"],
                            khoang["anchor_pts_time"], fidxs_thieu))
                tong_khung += len(fidxs_thieu)

    print(f"{len(vids)} video, {len(viec)} khoảng trống / {tong_khung} khung cần trích mới "
          f"(nguong={a.nguong}, buoc={a.buoc}, workers={a.workers})")
    if not viec:
        print("Không có khung nào cần trích (đã cache đủ từ trước).")

    def lam(item):
        vid, video_path, fps, anchor_frame, anchor_pts_time, fidxs = item
        anh_theo_fidx = trich_nhieu(video_path, fidxs, anchor_frame, anchor_pts_time, fps)
        ra = []
        for fidx in fidxs:
            anh = anh_theo_fidx.get(fidx)
            if anh is not None:
                anh.save(_duong_cache(a.cache, vid, fidx), quality=90)
                t = anchor_pts_time + (fidx - anchor_frame) / fps
                ra.append((vid, fidx, round(t, 3), fps, True))
            else:
                ra.append((vid, fidx, None, fps, False))
        return ra

    ket_qua, loi, t0, i = [], 0, time.perf_counter(), 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(lam, item) for item in viec]
        for fut in as_completed(futs):
            ket_qua_khoang = fut.result()
            i_truoc, i = i, i + len(ket_qua_khoang)
            for vid, fidx, t, fps, ok in ket_qua_khoang:
                if ok:
                    ket_qua.append({"video_id": vid, "frame_idx": fidx, "pts_time": t,
                                   "fps": fps, "nhom": vid[:3]})
                else:
                    loi += 1
            if i // 500 > i_truoc // 500 or i == tong_khung:  # gộp theo cửa sổ nên bớt in
                dt = time.perf_counter() - t0
                print(f"  [{i}/{tong_khung}]  {dt:.0f}s  {dt / max(i, 1) * 1000:.0f} ms/khung  loi={loi}")

    # Cộng thêm những khung đã cache sẵn từ lần chạy trước (không trích lại)
    # nhưng vẫn phải có mặt trong Parquet — nếu không, gộp qua Drive sẽ tưởng
    # máy này chưa có chúng.
    da_co = []
    for vid in vids:
        g = m[m.video_id == vid].sort_values("frame_idx")
        for fidx, t in cac_khung_can_lap(g, a.nguong, a.buoc):
            duong = _duong_cache(a.cache, vid, fidx)
            if duong.exists() and not any(k["video_id"] == vid and k["frame_idx"] == fidx for k in ket_qua):
                da_co.append({"video_id": vid, "frame_idx": fidx, "pts_time": t,
                             "fps": float(g.fps.iloc[0]), "nhom": vid[:3]})

    bang = pd.DataFrame(ket_qua + da_co).sort_values(["video_id", "frame_idx"]).reset_index(drop=True)

    ten_nhom = "_".join(a.group) if a.group else "toanbo"
    out = Path(a.out) if a.out else Path(a.index) / f"dense_{ten_nhom}.parquet"
    bang.to_parquet(out, index=False)

    dt = time.perf_counter() - t0
    print(f"\n{len(bang)} khung dày (mới: {len(ket_qua)}, đã cache từ trước: {len(da_co)}) "
          f"trên {bang.video_id.nunique()} video, lỗi {loi} khung, {dt:.0f}s.")
    print(f"Đã ghi -> {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print("Đưa file này lên Drive để gộp với các máy khác — KHÔNG cần đưa cache/, "
          "ảnh chỉ có ý nghĩa cục bộ trên máy đã trích.")


if __name__ == "__main__":
    main()
