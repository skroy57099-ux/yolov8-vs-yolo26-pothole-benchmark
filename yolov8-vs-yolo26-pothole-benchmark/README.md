# YOLOv8n vs YOLO26n: Pothole Detection Benchmark

A focused benchmark comparing **YOLOv8n** and **YOLO26n** for pothole detection on a prepared subset of the RDD2022 road-damage dataset, followed by a qualitative video test on a 3,000-frame Chennai road clip.

> **Main takeaway:** YOLO26n scored higher on the prepared validation set, while YOLOv8n was faster in the single Chennai video run. Both models showed questionable detections on visually similar road features at confidence 0.25. At confidence 0.50, detections nearly disappeared. The video results are exploratory, not a measured accuracy benchmark, because the video has not been annotated with ground-truth pothole labels.

## Repository contents

- `results/validation_metrics.csv` — controlled validation metrics for both models.
- `results/video_comparison.csv` — video measurements at confidence thresholds 0.25 and 0.50.
- `results/limitations_and_conclusion.md` — interpretation and limitations.
- `figures/validation_metrics.png` — validation metric comparison.
- `figures/video_detections_by_confidence.png` — observed frame-level detection counts by confidence.
- `scripts/compare_chennai.py` — reproducible video comparison script; expects the trained weights and local video described below.
- `requirements.txt` — main Python dependencies.

The repository intentionally excludes the dataset, source videos, model weights, virtual environment, caches, and generated annotated MP4s. Those files are large, may have separate licensing restrictions, and are not required to understand the benchmark. Do not commit them accidentally.

## Experiment setup

- Models: YOLOv8n and YOLO26n, initialized from pretrained checkpoints.
- Training: 30 epochs for each model.
- Image size: 640.
- Batch size: 4.
- Device: NVIDIA GeForce GTX 1650 4 GB VRAM.
- Validation set: 520 images and 971 annotated pothole instances.
- Video test: one Chennai clip, 1920×1080, 60 FPS, 3,000 frames.
- Video inference: image size 640; confidence thresholds 0.25 and 0.50.

Both models used the same prepared train/validation split and training settings. Equal epochs do not guarantee equal convergence, and this is a single run per architecture rather than a multi-seed study.

## Validation results

| Metric | YOLOv8n | YOLO26n | Higher value |
|---|---:|---:|---|
| Precision | 60.5% | 59.6% | YOLOv8n |
| Recall | 49.7% | 52.4% | YOLO26n |
| mAP@0.50 | 54.1% | 56.8% | YOLO26n |
| mAP@0.50:0.95 | 22.4% | 24.4% | YOLO26n |
| Inference time per image (reported validation timing) | 7.6 ms | 7.5 ms | Effectively tied |
| Training time | 0.898 h | 1.308 h | YOLOv8n was faster to train |
| Fused parameter count | 3,132,707 | 2,389,735 | YOLO26n has fewer |
| Best checkpoint size | ~6.2 MB | ~5.2 MB | YOLO26n is smaller |

These validation metrics apply to the prepared validation set and do not guarantee performance on unseen road footage.

## Chennai video results

| Model | Confidence | Frame-level detections | Inference-only FPS | End-to-end FPS | End-to-end time |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 0.25 | 139 | 65.02 | 30.69 | 97.74 s |
| YOLO26n | 0.25 | 50 | 53.32 | 27.03 | 110.99 s |
| YOLOv8n | 0.50 | 3 | 59.33 | 27.22 | 110.21 s |
| YOLO26n | 0.50 | 0 | 48.58 | 24.26 | 123.65 s |

**Detection counts are individual boxes summed over frames, not unique potholes.** A count alone does not identify false positives or prove accuracy. Visual review at 0.25 found both models detecting some visually similar features such as repaired patches, speed breakers, and road markings. At 0.50, the detection count fell sharply, but the video must be annotated before recall or precision can be calculated.

## Reproducing the video test

1. Set up Python and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the source video at `Video/processed/chennai_short.mp4` in your local project folder.
3. Place trained checkpoints at:
   - `runs/detect/benchmark_results/yolov8_training/weights/best.pt`
   - `runs/detect/benchmark_results/yolo26_training/weights/best.pt`
4. Run both thresholds:
   ```bash
   python scripts/compare_chennai.py --conf 0.25
   python scripts/compare_chennai.py --conf 0.50
   ```

The script saves annotated MP4 files and a CSV under `runs/detect/benchmark_results/`. The paths are configurable through command-line arguments; run `python scripts/compare_chennai.py --help` for details.

## Limitations and next steps

- Only one video was tested, with no frame-level ground-truth annotations.
- Video detection totals cannot be treated as true positives or false positives without manual labels.
- The preparation pipeline may have excluded images without pothole labels, reducing background-only examples; verify the dataset before concluding this is the cause of false positives.
- More representative negative examples and difficult negatives (e.g., repaired road patches, speed breakers, markings) may improve discrimination.
- A stronger evaluation should annotate a held-out road video and report precision, recall, false positives per frame, and missed potholes at fixed confidence thresholds.
- Reported timing is from this machine and one run; it is not a universal speed ranking.

## Dataset and model licensing

This project uses a prepared subset of RDD2022 and pretrained YOLO checkpoints. Review the relevant dataset and model licenses/terms before redistributing data or weights. This repository contains only benchmark documentation, summary results, and a helper script.
