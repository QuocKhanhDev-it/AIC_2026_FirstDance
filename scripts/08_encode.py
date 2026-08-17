"""
08_encode.py — Encode lại keyframe bằng model mạnh hơn ViT-B/32 của BTC.

Đọc kèm: docs/06_ke_hoach_encode_GPU.md

Sinh ra một ma trận THỨ HAI để ghép RRF với `clip.npy`. KHÔNG ghi đè `clip.npy`,
KHÔNG đụng `01_build_index.py` — nâng cấp là việc cộng thêm, ma trận mới tệ thì
xóa file là xong.

Chạy thử 100 video (đủ để đo tốc độ + kiểm lệch hàng):
    python scripts/08_encode.py --model ViT-SO400M-14-SigLIP2-378 \
        --pretrained webli --videos 100 --out index/clip_siglip2_thu.npy

Kiểm lệch hàng (BẮT BUỘC trước khi tin kết quả):
    python scripts/08_encode.py --kiem-lech-hang index/clip_siglip2_thu.npy

Toàn kho:
    python scripts/08_encode.py --model ViT-SO400M-14-SigLIP2-378 \
        --pretrained webli --out index/clip_siglip2.npy

BA CHỐT AN TOÀN cài sẵn trong file này:
  1. Lặp theo ĐÚNG THỨ TỰ row_id. Ảnh thiếu/hỏng -> GHI VECTOR 0, không bao giờ
     bỏ dòng. Bỏ dòng là mọi dòng sau dịch một bậc, không lỗi nào báo.
  2. Ghi model/pretrained ra .json cạnh file .npy (bẫy A6).
  3. Chuẩn hóa L2 trước khi lưu, nếu không `M @ q` không còn là cosine.

Ma trận luôn có đủ 177.321 dòng kể cả khi chỉ encode 100 video — dòng chưa
encode để 0 (cosine 0 với mọi thứ = không bao giờ được truy hồi). Nhờ vậy file
thử dùng thẳng được với `src/dense.py` mà không sửa gì.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
LUU_MOI = 10_000        # checkpoint: chạy 2 giờ mà mất điện thì không làm lại từ đầu


# ----------------------------------------------------------------- chọn video

def chon_theo_tap_dev(master: pd.DataFrame, f_dev) -> np.ndarray:
    """Encode TRỌN VẸN các video có chứa đáp án của tập dev.

    Đây là cách rẻ nhất để đo một model mới mà không thổi phồng điểm: bể ứng
    viên là **toàn bộ keyframe của những video đó**, chứ không phải vài ảnh
    rải rác quanh đáp án. Kết hợp với `dense.be_chung()` khi so sánh.
    """
    import json
    rid = []
    for d in Path(f_dev).read_text("utf-8").splitlines():
        if not d.strip():
            continue
        r = json.loads(d)["row_id_dung"]
        rid += [x for b in r for x in b] if isinstance(r[0], list) else r
    vids = sorted(set(master.video_id.iloc[rid]))
    print(f"Tập dev đụng {len(vids)} video -> encode trọn vẹn")
    return master[master.video_id.isin(vids)].row_id.values


def chon_video(master: pd.DataFrame, so_video: int | None) -> np.ndarray:
    """Trả về row_id cần encode. `so_video=None` -> toàn kho.

    Lấy mẫu PHÂN TẦNG theo nhóm L, không ngẫu nhiên: L26 chiếm 57% số video
    (A2) nên ngẫu nhiên sẽ ra một tập lệch hẳn về một nhóm, và mọi kết luận đo
    được trên đó không suy ra được cho kho thật.

    Lấy TRỌN VẸN mọi keyframe của video đã chọn — không lấy ảnh rải rác. Truy
    hồi là xếp hạng trên toàn kho; tập ứng viên vụn thì bài toán dễ giả tạo.
    """
    if so_video is None:
        return master.row_id.values

    vid = master[["video_id"]].drop_duplicates()
    vid["nhom"] = vid.video_id.str[:3]
    nhom = sorted(vid.nhom.unique())
    moi_nhom = max(1, so_video // len(nhom))

    chon = []
    for n in nhom:
        co = vid[vid.nhom == n].video_id.tolist()
        chon += co[:moi_nhom]                      # ổn định, không phụ thuộc seed
    chon = chon[:so_video]
    print(f"Chọn {len(chon)} video, phân tầng trên {len(nhom)} nhóm L "
          f"({moi_nhom}/nhóm)")
    return master[master.video_id.isin(chon)].row_id.values


# ----------------------------------------------------------------- bộ nạp ảnh

class BoAnh:
    """Đọc + tiền xử lý ảnh keyframe theo `row_id`.

    ⚠️ Lớp này PHẢI ở cấp module, không được lồng trong hàm. Windows tạo tiến
    trình con bằng `spawn` nên `DataLoader(num_workers>0)` phải pickle được
    dataset — lớp lồng trong hàm thì không pickle được và sẽ nổ ngay khi chạy
    đa luồng. Máy đích chạy Windows.
    """

    def __init__(self, duong_dan: list, row_ids: np.ndarray, pre, kich_thuoc: int):
        self.duong_dan, self.row_ids = duong_dan, row_ids
        self.pre, self.kich_thuoc = pre, kich_thuoc

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        rid = int(self.row_ids[i])
        try:
            return self.pre(Image.open(self.duong_dan[i]).convert("RGB")), rid, 1
        except Exception:
            # CHỐT 1: hỏng thì trả tensor rỗng + cờ 0, TUYỆT ĐỐI không bỏ dòng.
            # Dòng này thành vector 0 -> không bao giờ được truy hồi, nhưng vẫn
            # nằm đúng chỗ nên mọi dòng sau KHÔNG bị dịch.
            return torch.zeros(3, self.kich_thuoc, self.kich_thuoc), rid, 0


# ------------------------------------------------------------------- encode

def encode(a):
    import torch
    import open_clip

    master = pd.read_parquet(a.index / "master.parquet")
    row_ids = (chon_theo_tap_dev(master, a.theo_tap_dev) if a.theo_tap_dev
               else chon_video(master, a.videos))

    thiet_bi = "cuda" if torch.cuda.is_available() else "cpu"
    if thiet_bi == "cpu":
        print("\n  ⚠️  KHÔNG THẤY GPU — đang chạy bằng CPU.\n"
              "      Bản torch '+cpu' vẫn chạy bình thường, chỉ chậm ~100 lần và\n"
              "      không báo gì. Xem docs/06_ke_hoach_encode_GPU.md §5 việc 2.\n")
    else:
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    # `pretrained` là đường dẫn file cục bộ (không phải tag) -> open_clip
    # không tự merge preprocess_cfg của tag, ảnh sẽ bị resize sai kích cỡ
    # (mặc định 224) trong khi kiến trúc model đòi kích cỡ khác. --image-size
    # ép đúng kích cỡ theo bảng model ở docs/06_ke_hoach_encode_GPU.md §2.
    model, _, pre = open_clip.create_model_and_transforms(
        a.model, pretrained=a.pretrained,
        force_image_size=a.image_size if a.image_size else None)
    model = model.to(thiet_bi).eval()

    # cỡ ảnh của model, để tensor thăm dò/thay thế đều khớp shape (chốt bẫy:
    # hardcode 224 nổ ngay với model đòi kích cỡ khác, vd 378 của SigLIP2)
    cx = getattr(model.visual, "image_size", 224)
    cx = int(cx[0] if isinstance(cx, (tuple, list)) else cx)

    chieu = model.visual.output_dim if hasattr(model.visual, "output_dim") else \
        model.encode_image(torch.zeros(1, 3, cx, cx).to(thiet_bi)).shape[1]
    print(f"{a.model} / {a.pretrained}  ->  {chieu} chiều")

    kieu = np.float16 if a.fp16 else np.float32
    mat = np.zeros((len(master), chieu), dtype=kieu)
    xong = np.zeros(len(master), dtype=bool)

    # tiếp tục sau khi ngắt
    if a.out.exists() and (a.out.with_suffix(".tien_do.npy")).exists():
        mat = np.load(a.out)
        xong = np.load(a.out.with_suffix(".tien_do.npy"))
        print(f"Tiếp tục: đã có {int(xong.sum()):,} dòng")
        row_ids = np.array([r for r in row_ids if not xong[r]])
        if len(row_ids) == 0:
            print("Không còn dòng nào để làm.")
            return luu(a, mat, xong, chieu, 0.0, 0)

    # Dòng chưa tải ảnh thì để vector 0 luôn, đừng đẩy qua model cho tốn.
    # Vẫn đánh dấu là XONG để lần chạy tiếp không thử lại — và quan trọng hơn,
    # dòng vẫn nằm nguyên chỗ cũ trong ma trận (CHỐT 1).
    co_anh = master.kf_path.notna().values
    thieu = row_ids[~co_anh[row_ids]]
    row_ids = row_ids[co_anh[row_ids]]
    if len(thieu):
        xong[thieu] = True
        print(f"{len(thieu):,} keyframe chưa tải ảnh -> để vector 0, giữ nguyên dòng")

    ds = BoAnh(master.kf_path.values[row_ids].tolist(), row_ids, pre, cx)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=a.batch, num_workers=a.workers,
        pin_memory=(thiet_bi == "cuda"))

    print(f"Encode {len(row_ids):,} keyframe, batch {a.batch}, "
          f"{a.workers} luồng đọc ảnh\n")
    hong, da, t0 = 0, 0, time.perf_counter()

    with torch.no_grad():
        for lo, rid, ok in dl:
            lo = lo.to(thiet_bi, non_blocking=True)
            # Turing KHÔNG có bf16 phần cứng -> phải là float16
            with torch.autocast(thiet_bi, dtype=torch.float16,
                                enabled=(thiet_bi == "cuda")):
                v = model.encode_image(lo)
            v = v.float().cpu().numpy()
            v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)   # CHỐT 3

            r = rid.numpy()
            v[ok.numpy() == 0] = 0.0        # ảnh hỏng -> vector 0, dòng vẫn còn
            hong += int((ok.numpy() == 0).sum())
            mat[r] = v.astype(kieu)
            xong[r] = True

            da += len(r)
            if da % LUU_MOI < a.batch:
                np.save(a.out, mat)
                np.save(a.out.with_suffix(".tien_do.npy"), xong)
                toc = da / (time.perf_counter() - t0)
                con = (len(row_ids) - da) / toc / 60
                print(f"  {da:>7,}/{len(row_ids):,}  {toc:6.1f} ảnh/giây  "
                      f"còn ~{con:.0f} phút")

    giay = time.perf_counter() - t0
    toc = da / giay if giay else 0.0
    return luu(a, mat, xong, chieu, toc, hong)


def luu(a, mat, xong, chieu, toc, hong):
    np.save(a.out, mat)

    # `da_encode` KHÔNG phải kích thước bể ứng viên. Dòng chưa tải ảnh cũng
    # được đánh dấu `xong` (xem CHỐT 1 ở trên) nhưng vector vẫn là 0, nên nó
    # không tìm ra được gì. Đã vấp: sidecar ghi da_encode 18.635 trong khi ma
    # trận chỉ có 3.135 dòng thật — chênh 6 lần. Ai đọc file này để ước bể sẽ
    # ước sai, mà sai kích thước bể thì điểm lệch tới +0,2833 (xem be_chung()).
    co_vector = int((np.abs(mat[:, :8]).sum(1) > 0).sum())

    canh = a.out.with_suffix(".json")        # CHỐT 2
    canh.write_text(json.dumps({
        "model": a.model, "pretrained": a.pretrained, "chieu": int(chieu),
        "dtype": str(mat.dtype), "so_dong": int(mat.shape[0]),
        "da_encode": int(xong.sum()), "co_vector": co_vector,
        "anh_hong": int(hong),
        "toc_do_anh_moi_giay": round(toc, 1),
        "ngay": time.strftime("%Y-%m-%d %H:%M"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    tien = a.out.with_suffix(".tien_do.npy")
    if tien.exists() and int(xong.sum()) == mat.shape[0]:
        tien.unlink()                        # xong hết thì bỏ file tiến độ

    print(f"\n{a.out}  {mat.shape} {mat.dtype}  "
          f"({a.out.stat().st_size / 1024**2:.0f} MB)")
    print(f"Đã encode {int(xong.sum()):,}/{mat.shape[0]:,} dòng"
          + (f"  |  {hong:,} ảnh hỏng -> vector 0" if hong else ""))
    print(f"BỂ ỨNG VIÊN THẬT: {co_vector:,} dòng có vector khác 0"
          + (f"  ({int(xong.sum()) - co_vector:,} dòng đánh dấu xong nhưng "
             f"chưa tải ảnh -> vector 0, KHÔNG tìm ra được)"
             if int(xong.sum()) > co_vector else ""))
    if toc:
        print(f"Tốc độ thật: {toc:.1f} ảnh/giây"
              f"  (toàn kho 177.321 ảnh = {177321 / toc / 3600:.1f} giờ)")
    print(f"Ghi chú -> {canh}")
    print("\nBƯỚC TIẾP THEO — bắt buộc:")
    print(f"    python scripts/08_encode.py --kiem-lech-hang {a.out}")
    return 0


# ------------------------------------------------------- kiểm lệch hàng row_id

def kiem_lech_hang(a):
    """Chứng minh ma trận mới KHỚP HÀNG với master.parquet.

    Nguyên lý: hai keyframe thật sự giống nhau thì MODEL NÀO CŨNG thấy giống.
    `index/trung_lap.parquet` đã có sẵn danh sách keyframe có bản sao cùng video
    ở cosine >= 0,99 dưới ViT-B/32. Với mỗi cái, tìm bạn song sinh của nó trong
    ma trận CŨ, rồi đo lại chính cặp đó trong ma trận MỚI.

      * cặp vẫn giống nhau  -> hàng khớp
      * cặp hết giống nhau  -> ĐÃ LỆCH HÀNG, dừng ngay

    Nếu lệch một bậc thì `clip[row_id]` trỏ sang keyframe KHÁC, cosine rơi về
    mức ngẫu nhiên (0,3–0,7). Đây là phép kiểm rẻ mà bắt lỗi rất chắc.
    """
    master = pd.read_parquet(a.index / "master.parquet")
    cu = np.load(a.index / "clip.npy", mmap_mode="r")
    moi = np.load(a.kiem_lech_hang, mmap_mode="r")

    print(f"cũ  {a.index / 'clip.npy'}  {cu.shape}")
    print(f"mới {a.kiem_lech_hang}  {moi.shape}\n")
    if moi.shape[0] != len(master):
        raise SystemExit(f"❌ Số dòng LỆCH: master {len(master):,}, "
                         f"ma trận mới {moi.shape[0]:,}")

    da = np.abs(np.asarray(moi[:, :4], dtype=np.float32)).sum(1) > 0
    tl = pd.read_parquet(a.index / "trung_lap.parquet")
    ung = tl[(tl.max_cos >= 0.99) & da[tl.row_id.values]]
    if ung.empty:
        raise SystemExit("Không có keyframe trùng lặp nào trong phần đã encode. "
                         "Chạy với --videos nhiều hơn.")

    mau = ung.sample(min(a.so_mau, len(ung)), random_state=0)
    ket = []
    for r in mau.itertuples():
        anh_em = master.index[(master.video_id == r.video_id).values].values
        anh_em = anh_em[da[anh_em]]
        if len(anh_em) < 2:
            continue
        vc = np.asarray(cu[anh_em], dtype=np.float32) @ np.asarray(cu[r.row_id], np.float32)
        vc[anh_em == r.row_id] = -9
        ban = int(anh_em[int(np.argmax(vc))])          # bạn song sinh theo ma trận CŨ

        a1 = np.asarray(moi[r.row_id], dtype=np.float32)
        a2 = np.asarray(moi[ban], dtype=np.float32)
        ket.append((r.video_id, r.row_id, ban, float(vc.max()),
                    float(a1 @ a2 / ((np.linalg.norm(a1) * np.linalg.norm(a2)) + 1e-9))))

    d = pd.DataFrame(ket, columns=["video_id", "row_id", "ban_sao", "cos_cu", "cos_moi"])
    dat = d.cos_moi >= a.nguong_kiem
    print(f"Kiểm {len(d)} cặp keyframe trùng lặp (ngưỡng {a.nguong_kiem}):\n")
    print(d.sort_values("cos_moi").head(10).to_string(index=False))
    print(f"\n  trung vị cos_mới = {d.cos_moi.median():.4f}")
    print(f"  đạt: {int(dat.sum())}/{len(d)} ({dat.mean() * 100:.1f}%)")

    if dat.mean() >= 0.9:
        print("\n✅ HÀNG KHỚP. Ma trận mới dùng được.")
        return 0
    print("\n❌ NGHI LỆCH HÀNG. Cặp giống nhau ở ma trận cũ lại KHÔNG giống ở ma\n"
          "   trận mới. Khả năng cao script đã bỏ qua dòng nào đó thay vì ghi\n"
          "   vector 0. ĐỪNG dùng file này. Xem docs/06_ke_hoach_encode_GPU.md §3A.")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--index", default=GOC / "index", type=Path)
    ap.add_argument("--model", default="ViT-SO400M-14-SigLIP2-378")
    ap.add_argument("--pretrained", default="webli")
    ap.add_argument("--videos", type=int, default=None,
                    help="chỉ encode N video, phân tầng theo nhóm L")
    ap.add_argument("--theo-tap-dev", type=Path, default=None,
                    help="encode TRỌN VẸN các video có đáp án trong tập dev "
                         "— cách rẻ nhất để đo một model mới")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--fp32", dest="fp16", action="store_false")
    ap.add_argument("--out", type=Path, default=GOC / "index" / "clip_moi.npy")
    ap.add_argument("--image-size", type=int, default=None,
                    help="ép kích cỡ ảnh đầu vào — cần khi --pretrained là "
                         "file cục bộ (mất preprocess_cfg đi kèm tag)")
    ap.add_argument("--kiem-lech-hang", type=Path, default=None,
                    help="chạy riêng phép kiểm hàng, không encode")
    ap.add_argument("--so-mau", type=int, default=200)
    ap.add_argument("--nguong-kiem", type=float, default=0.90)
    a = ap.parse_args()

    raise SystemExit(kiem_lech_hang(a) if a.kiem_lech_hang else encode(a))


if __name__ == "__main__":
    main()
