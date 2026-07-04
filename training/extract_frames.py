"""Trích ảnh từ video để chuẩn bị dữ liệu gán nhãn (Giai đoạn 3).

Chạy:
  venv\\Scripts\\python.exe training\\extract_frames.py data\\videos\\camera.mp4 --every 15
Ảnh xuất ra data/training_raw/frames/.
"""
import os
import argparse
import cv2

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--every", type=int, default=15, help="lấy 1 ảnh mỗi N frame")
    ap.add_argument("--out", default=os.path.join(_BASE, "data", "training_raw", "frames"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Khong mo duoc video {args.video}")

    idx, saved = 0, 0
    base = os.path.splitext(os.path.basename(args.video))[0]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.every == 0:
            fn = os.path.join(args.out, f"{base}_{idx:06d}.jpg")
            cv2.imwrite(fn, frame)
            saved += 1
        idx += 1
    cap.release()
    print(f"Da trich {saved} anh tu {idx} frame -> {args.out}")


if __name__ == "__main__":
    main()
