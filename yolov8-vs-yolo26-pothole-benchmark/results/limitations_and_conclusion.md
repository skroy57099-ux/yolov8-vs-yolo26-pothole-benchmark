# Findings, limitations, and next steps

## Findings

- YOLO26n achieved higher prepared-validation mAP@0.50 (56.8% vs 54.1%), mAP@0.50:0.95 (24.4% vs 22.4%), and recall (52.4% vs 49.7%). YOLOv8n had slightly higher precision (60.5% vs 59.6%).
- On the single Chennai video run, YOLOv8n was faster: 65.02 vs 53.32 inference-only FPS at confidence 0.25, and 59.33 vs 48.58 at confidence 0.50.
- Increasing confidence from 0.25 to 0.50 reduced detections from 139 to 3 for YOLOv8n and from 50 to 0 for YOLO26n.
- Visual review at 0.25 showed detections on some road features that were not potholes, including repaired patches, speed breakers, and road markings.

## What cannot be concluded

- The video counts do not prove which model has more false positives. No ground-truth labels exist for this video comparison.
- Zero detections at confidence 0.50 does not prove the model is inaccurate by itself; it could also mean the model was too conservative. Actual potholes need to be annotated and checked.
- Results from one video and one run per model do not establish universal speed or accuracy rankings.

## Potential dataset issue

The preparation pipeline reportedly skipped images with no D40 pothole annotations. If confirmed in the actual training split, this can reduce background-only examples and may contribute to false positives. Inspect the split and add difficult negative examples before retraining.

## Recommended next step

Annotate a representative held-out road video, then compare precision, recall, false positives per frame, and missed potholes at the same confidence thresholds. Include difficult negatives such as repaired patches, speed breakers, lane markings, shadows, and road debris.
