"""
trich_day.py — TV2: Module trích xuất frame dày (đường găng).

Vì sao cần (A1, Ke_hoach_AIC2026_v4.md): trung vị khoảng cách 2 keyframe liên
tiếp là 55 frame (~1,8s ở 30fps) — xác suất một keyframe có sẵn rơi trúng cửa
sổ 10 frame quanh đáp án chỉ ~14,6%. Không trích dày thì R-Score trần ~0,15 dù
thuật toán dóng hàng hoàn hảo. Dùng cho Bước 4 của TRAKE (PHẦN D Mũi nhọn 2):
trích dày trong vùng ứng viên ĐÃ được Bước 2/2b/3 thu hẹp, KHÔNG chạy trên
toàn kho — 129,8 giờ video là khối lượng CPU lớn (B4).

KHÔNG cần GPU: chỉ gọi `ffmpeg` để seek + decode, không chạy model nào ở đây.

Trả về `list[KhungDay]`, KHÔNG PHẢI `list[Candidate]`. Cố ý: `Candidate.row_id`
là chỉ số dòng trong `master.parquet`/`clip.npy` — `rrf.hop_nhat` và
`dedup.gom_ban_sao` dùng nó để tra thẳng `ma_tran[row_id]`. Khung trích dày ở
đây chưa từng có trong bảng cái, KHÔNG có row_id thật. Nhét một row_id giả vào
Candidate là gài bẫy âm thầm y hệt kiểu A6: dedup sẽ lấy nhầm vector CLIP của
MỘT keyframe khác mà không ném lỗi nào. Muốn đưa khung dày vào RRF thì phải
encode CLIP/VLM cho nó trước rồi mới bọc thành Candidate thật.

`frame_idx` của khung dày là SỐ TA CHỌN (`center_frame ± k*stride`), KHÔNG
phải suy ngược từ `pts_time` mà ffmpeg trả về — đây là chiều AN TOÀN của phép
tính, ngược với bẫy ở `schema.py` (1061,36 × 25fps = 26534 nhưng giá trị đúng
26533, vì đó là suy NGƯỢC từ time có sẵn sang frame_idx). Ở đây ta có
frame_idx trước, chỉ suy `time = center_pts_time + (frame_idx - center_frame)
/ fps` để ra lệnh seek — sai số nếu có chỉ ảnh hưởng ẢNH lấy được (lệch khỏi
khung mong muốn một phần nhỏ giây), không làm sai giá trị frame_idx SẼ NỘP.

Cache theo TỪNG FRAME, không theo cả cửa sổ: `cache/<video_id>/<frame_idx>.jpg`.
Các truy vấn con của TRAKE (Bước 1) thường soi cùng một đoạn video từ nhiều
hướng — cache theo frame tận dụng được phần chồng lấn đó, cache theo cửa sổ
nguyên khối thì không. `cache/` không đẩy git (xem .gitignore), không đồng bộ
qua Drive — chỉ tồn tại trên máy đã trích, chạy phân tán theo nhóm L ai giữ.

**Tối ưu H2 #4 (đã đo, A1): khung nào cache trượt được trích bằng MỘT lệnh
`ffmpeg` cho CẢ cửa sổ còn thiếu**, không phải một tiến trình mỗi khung —
`trich_nhieu()` giải mã liên tục `-ss`+`-t` rồi tách luồng `image2pipe` theo
chữ ký PNG. Đo trên hai máy: 169 ms/khung (một tiến trình/khung) -> 19
ms/khung (một tiến trình/cửa sổ), nhanh gấp 8,7 lần. Chi phí nằm ở khởi động
tiến trình ffmpeg trên Windows, không phải decode — gộp lệnh xóa gần hết chi
phí đó. Cấu trúc cache theo-từng-frame KHÔNG đổi, chỉ đổi cách LẤY khi trượt.

Dùng:
    from trich_day import trich_day
    khung = trich_day(video_path, video_id="L23_V001", center_frame=1200,
                       center_pts_time=48.0, fps=25.0, radius_sec=2.0, stride=2)
    for k in khung:
        k.frame_idx, k.pts_time, k.anh   # anh: PIL.Image.Image, RGB

Tự đo chi phí trên máy mình (BẮT BUỘC theo A1 — đừng dùng mốc "3,1s/300 frame"
của v3, đo trên một máy khác):
    python src/trich_day.py --do-luong --index ./index
"""

import argparse
import io
import subprocess
import time
from pathlib import Path
from typing import NamedTuple

from PIL import Image


class KhungDay(NamedTuple):
    """Một frame trích thêm giữa hai keyframe. KHÔNG phải Candidate — xem
    docstring đầu file lý do không có row_id."""
    frame_idx: int
    pts_time: float
    anh: Image.Image


_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _tach_anh_png(data: bytes) -> list[bytes]:
    """Tách một luồng nhiều ảnh PNG nối liền nhau (ffmpeg `image2pipe`) thành
    từng ảnh riêng, dựa vào chữ ký 8 byte đầu mỗi PNG — không có khung nào
    khác chứa lại đúng chuỗi byte này nên tìm-và-cắt là đủ, không cần parse
    chunk IHDR/IEND."""
    moc = []
    start = 0
    while True:
        i = data.find(_PNG_SIG, start)
        if i == -1:
            break
        moc.append(i)
        start = i + len(_PNG_SIG)
    return [data[a:b] for a, b in zip(moc, moc[1:] + [len(data)])]


def trich_nhieu(video_path, frame_idxs, anchor_frame: int, anchor_pts_time: float,
                 fps: float, timeout: float = 60.0) -> dict:
    """Trích NHIỀU frame bằng MỘT tiến trình `ffmpeg` duy nhất (A1: đo được
    nhanh gấp 8,7 lần so với một tiến trình mỗi khung — 169 -> 19 ms/khung,
    xác nhận trên hai máy khác nhau).

    `-ss` (trước `-i`) + `-t` khiến ffmpeg giải mã LIÊN TỤC mọi frame trong
    khoảng `[min(frame_idxs), max(frame_idxs)]`, không nhảy cóc theo stride —
    nhảy cóc cần filter `select` riêng và phá tính liên tục của decode, mà
    chính tính liên tục đó là thứ giúp seek chỉ tốn chi phí MỘT lần cho cả cửa
    sổ. `image2pipe` trả về một LUỒNG ảnh PNG nối tiếp nhau, phải tách lại
    bằng `_tach_anh_png`. `-fps_mode passthrough` giữ nguyên số khung giải mã
    được, không cho ffmpeg tự rớt/nhân đôi khung để khớp một fps đầu ra khác
    (cờ cũ `-vsync 0` đã bị GỠ ở ffmpeg 9.0 — `Unrecognized option`, không chỉ
    deprecated; `-fps_mode` là cờ thay thế từ ffmpeg 5.1).

    Trả `dict[frame_idx, Image.Image]` — chỉ chứa khung tách/mở ảnh THÀNH
    CÔNG; khung nào ffmpeg không trả về (đầu/cuối video, file hỏng) vắng mặt
    trong dict, gọi hàng loạt không bị một khung hỏng làm dừng cả cửa sổ.
    """
    fidx_min, fidx_max = min(frame_idxs), max(frame_idxs)
    t_start = max(anchor_pts_time + (fidx_min - anchor_frame) / fps, 0.0)
    so_frame = fidx_max - fidx_min + 1
    thoi_luong = (so_frame + 0.5) / fps  # +nửa khung cho chắc khung cuối

    try:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{t_start:.3f}", "-i", str(video_path),
             "-t", f"{thoi_luong:.3f}", "-fps_mode", "passthrough",
             "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return {}
    if r.returncode != 0 or not r.stdout:
        return {}

    can_lay = set(frame_idxs)
    ra = {}
    for i, raw in enumerate(_tach_anh_png(r.stdout)):
        fidx = fidx_min + i
        if fidx not in can_lay:
            continue
        try:
            ra[fidx] = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            pass
    return ra


def _duong_cache(cache_dir, video_id: str, frame_idx: int) -> Path:
    d = Path(cache_dir) / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{frame_idx}.jpg"


def trich_day(video_path, video_id: str, center_frame: int, center_pts_time: float,
              fps: float, radius_sec: float = 2.0, stride: int = 2,
              cache_dir="cache") -> list[KhungDay]:
    """Trích các frame quanh `center_frame`, bán kính `radius_sec` giây, cách
    nhau `stride` frame (Bước 4 TRAKE khuyến nghị `stride=1..2`, vùng ~30s).

    `center_pts_time` PHẢI là thời điểm THẬT của `center_frame` — lấy thẳng từ
    cột `pts_time` của `master.parquet` cho keyframe đã chọn ở Bước 2/2b/3,
    KHÔNG suy từ `center_frame / fps`. Dùng nó làm neo để mọi khung lân cận
    chỉ lệch một khoảng NHỎ (tối đa `radius_sec` giây), không cộng dồn sai số
    qua hàng nghìn frame kể từ đầu video như tính thẳng `frame_idx / fps` sẽ
    gặp trên video có fps làm tròn (25.0 / 26.44 / 29.97 / 30.0 — README).

    `cache_dir=None` để tắt cache (dùng khi đo `--do-luong`, đo đúng chi phí
    ffmpeg thật, không lẫn phần đọc lại từ đĩa).

    Trả về sắp theo thời gian tăng dần, CÓ kèm chính `center_frame`. Khung nào
    ffmpeg không trích được (đầu/cuối video, file hỏng) bị bỏ qua lặng lẽ.
    """
    if fps <= 0:
        raise ValueError(f"fps không hợp lệ: {fps}")
    if stride < 1:
        raise ValueError(f"stride phải >= 1, nhận {stride}")

    ban_kinh_frame = max(1, round(radius_sec * fps))
    cac_frame = [f for f in
                 range(center_frame - ban_kinh_frame, center_frame + ban_kinh_frame + 1, stride)
                 if f >= 0]
    if not cac_frame:
        return []

    da_co = {}
    con_thieu = []
    for fidx in cac_frame:
        duong = _duong_cache(cache_dir, video_id, fidx) if cache_dir else None
        if duong is not None and duong.exists():
            try:
                da_co[fidx] = Image.open(duong).convert("RGB")
                continue
            except Exception:
                pass
        con_thieu.append(fidx)

    if con_thieu:
        moi = trich_nhieu(video_path, con_thieu, center_frame, center_pts_time, fps)
        for fidx, anh in moi.items():
            da_co[fidx] = anh
            if cache_dir:
                anh.save(_duong_cache(cache_dir, video_id, fidx), quality=90)

    ra = []
    for fidx in cac_frame:
        if fidx not in da_co:
            continue
        t = center_pts_time + (fidx - center_frame) / fps
        ra.append(KhungDay(frame_idx=fidx, pts_time=round(t, 3), anh=da_co[fidx]))
    return ra


def do_luong(index_dir="./index", video_id: str | None = None, n: int = 300,
             stride: int = 1) -> tuple[float, int]:
    """Đo chi phí THẬT trên máy này, tắt cache để đo đúng phần ffmpeg.

    Thay cho mốc "3,1s/300 frame" của v3 — mốc đó đo trên một máy khác, A1 nói
    rõ đừng dùng lại. Neo ở giữa video (tránh biên đầu/cuối) rồi chọn
    `radius_sec` sao cho vừa đúng khoảng `n` khung.
    """
    import pandas as pd

    m = pd.read_parquet(Path(index_dir) / "master.parquet")
    m = m[m.video_path.notna()]
    if video_id:
        m = m[m.video_id == video_id]
    if m.empty:
        raise SystemExit(f"Không có dòng nào có video_path (video_id={video_id!r}).")

    chon = None
    for vid in m.video_id.unique():
        r0 = m[m.video_id == vid].iloc[0]
        if Path(r0.video_path).exists():
            chon = vid
            break
    if chon is None:
        raise SystemExit(
            "Không tìm thấy video .mp4 nào TỒN TẠI TRÊN MÁY NÀY để đo. Cần ít "
            "nhất một video đã tải (xem README, mục Bố cục).")

    sub = m[m.video_id == chon].sort_values("kf_n")
    r = sub.iloc[len(sub) // 2]        # neo giữa video, tránh biên
    fps = float(r.fps)
    ban_kinh = (n - 1) * stride / (2 * fps)

    print(f"Đo trên {chon}, neo frame {int(r.frame_idx)} (fps={fps}), "
          f"mục tiêu ~{n} khung, stride={stride}, cache TẮT...")
    t0 = time.perf_counter()
    ra = trich_day(r.video_path, chon, int(r.frame_idx), float(r.pts_time),
                    fps, radius_sec=ban_kinh, stride=stride, cache_dir=None)
    dt = time.perf_counter() - t0

    moi_khung = dt / max(len(ra), 1)
    print(f"\n{len(ra)} khung trích được trong {dt:.2f}s "
          f"({moi_khung * 1000:.0f} ms/khung, ước tính {moi_khung * n:.1f}s cho {n} khung)")
    print("So sánh: mốc \"3,1s/300 frame\" của v3 (~10 ms/khung) đo trên MÁY "
          "KHÁC — đừng dùng lại, đây là số thật của máy này (A1).")
    return dt, len(ra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="./index")
    ap.add_argument("--do-luong", action="store_true",
                    help="đo chi phí ffmpeg thật trên máy này (bắt buộc theo A1)")
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--frame", type=int, default=None,
                    help="center_frame để trích thử, cần đi kèm --video-id")
    ap.add_argument("--radius-sec", type=float, default=2.0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--n", type=int, default=300, help="số khung mục tiêu khi --do-luong")
    ap.add_argument("--cache", default="cache", help="thư mục cache, rỗng để tắt")
    a = ap.parse_args()

    if a.do_luong:
        do_luong(a.index, a.video_id, a.n, a.stride)
        return

    if not a.video_id or a.frame is None:
        ap.print_help()
        raise SystemExit("\nCần --video-id VÀ --frame để trích thử, hoặc --do-luong để đo chi phí.")

    import pandas as pd
    m = pd.read_parquet(Path(a.index) / "master.parquet")
    sub = m[m.video_id == a.video_id].sort_values("kf_n")
    if sub.empty:
        raise SystemExit(f"Không có video_id={a.video_id!r} trong master.parquet.")
    neo = sub.iloc[(sub.frame_idx - a.frame).abs().to_numpy().argmin()]
    fps = float(neo.fps)
    center_pts_time = float(neo.pts_time) + (a.frame - int(neo.frame_idx)) / fps

    khung = trich_day(neo.video_path, a.video_id, a.frame, center_pts_time, fps,
                      radius_sec=a.radius_sec, stride=a.stride,
                      cache_dir=a.cache or None)
    print(f"{len(khung)} khung quanh frame {a.frame} của {a.video_id} "
          f"(neo thật gần nhất: frame {int(neo.frame_idx)}, fps={fps}):\n")
    for k in khung:
        danh_dau = "  <-- center" if k.frame_idx == a.frame else ""
        print(f"  frame_idx={k.frame_idx:>7}  pts_time={k.pts_time:>8.3f}s  "
              f"{k.anh.size[0]}x{k.anh.size[1]}{danh_dau}")


if __name__ == "__main__":
    main()
