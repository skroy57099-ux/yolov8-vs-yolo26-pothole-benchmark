import argparse
import csv
import time
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Compare YOLOv8n and YOLO26n on one road video."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (e.g. 0.25 or 0.50).",
    )
    parser.add_argument(
        "--video", default="Video/processed/chennai_short.mp4", help="Input video path."
    )
    parser.add_argument(
        "--yolov8-weights",
        default="runs/detect/benchmark_results/yolov8_training/weights/best.pt",
    )
    parser.add_argument(
        "--yolo26-weights",
        default="runs/detect/benchmark_results/yolo26_training/weights/best.pt",
    )
    parser.add_argument(
        "--output-dir", default="runs/detect/benchmark_results/chennai_comparison"
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0 < args.conf < 1:
        parser.error("--conf must be between 0 and 1")

    video = Path(args.video)
    output_dir = Path(args.output_dir) / f"conf_{args.conf:.2f}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_paths = {
        "YOLOv8n": Path(args.yolov8_weights),
        "YOLO26n": Path(args.yolo26_weights),
    }
    rows = []

    for name, weights in model_paths.items():
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        if not video.exists():
            raise FileNotFoundError(f"Video not found: {video}")
        model = YOLO(str(weights))
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_path = output_dir / f"{name.lower()}_conf{args.conf:.2f}.mp4"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create output video: {out_path}")

        frames = detections = 0
        inference_seconds = 0.0
        start = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            result = model.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            if torch.cuda.is_available() and str(args.device) != "cpu":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - t0
            detections += len(result.boxes)
            writer.write(result.plot())
            frames += 1
            if frames % 500 == 0:
                print(f"{name}: processed {frames}/{frame_total} frames")
        elapsed = time.perf_counter() - start
        cap.release()
        writer.release()
        row = {
            "model": name,
            "confidence": args.conf,
            "video": video.name,
            "frames_processed": frames,
            "total_frame_level_detections": detections,
            "inference_time_seconds": round(inference_seconds, 2),
            "end_to_end_time_seconds": round(elapsed, 2),
            "inference_only_fps": (
                round(frames / inference_seconds, 2) if inference_seconds else 0
            ),
            "end_to_end_fps": round(frames / elapsed, 2) if elapsed else 0,
            "output_video": str(out_path),
        }
        rows.append(row)
        print(row)

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {csv_path}")


if __name__ == "__main__":
    main()
