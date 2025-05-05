# General Workflow

- Drift check is performed using `helmet_drift.py` and `plate_drift.py`.
- If **no drift is detected**, training is **skipped** to conserve compute.
- If **drift is detected**, the system:
  - Loads the last saved model (`Final_Helmet.pt` or `Final_Plates.pt`)
  - Fine-tunes the model using the new test data
- Instead of downloading Kaggle datasets, the system now uses local structured test data at:
  - `/data/test_data/helmet/`
  - `/data/test_data/plates/`
- The training pipeline automatically splits this test data into train/val sets during execution.

---
# Docker Compatibility

- File structure, script names, and execution flow remain unchanged from the original architecture.
- The main entry point (`run_training.py`) is located in `/app/scripts/` and works without modification.
- If the drift detection modules (`helmet_drift.py`, `plate_drift.py`) are stored in a **separate folder** (e.g., `scripts-drift/`), include the following line in your Dockerfile:

  ```dockerfile
  # Copy drift detection modules (only if stored separately)
  COPY scripts-drift/*.py /app/scripts-drift/

- If you save the drift modules inside the scripts/ folder (alongside your training files), no additional Dockerfile changes are needed.

----
# Fine-Tuning Strategy

When drift is detected, the system uses the last trained model (if available) as a starting point for fine-tuning.

- If `Final_Helmet.pt` or `Final_Plates.pt` exists, it is loaded and trained for a short cycle (`20 epochs`) on the new data.
- If no previous model exists (e.g., first-time setup), the system falls back to the base YOLOv8 checkpoint (`yolov8m.pt` or `yolov8s.pt`).

---

# Helmet Detection Drift (helmet_drift.py)

Although our training pipeline includes various augmentations (e.g., brightness, contrast, scale, and geometric distortions), drift detection remains essential. Augmentations help the model generalize within expected ranges — but they do not guarantee robustness to shifts in the data distribution outside these bounds.
For example:

We use hsv_v=0.2 for brightness variation, but a sudden drop in average brightness in new data (e.g., night-time captures) could exceed this range.

We allow scale=0.1, yet a drastically different vehicle size distribution (e.g., more distant or zoomed-in views) can still cause performance degradation.

Our drift detection captures shifts in:

- Brightness: sensitivity to lighting conditions.

- Contrast: visual clarity of objects.

- Sharpness: focus and detail.

- Vehicle count per image: indicative of scene context and crowding.

By comparing these metrics against the training data baseline, we ensure retraining is triggered only when there's a meaningful distributional change, preventing unnecessary compute while maintaining accuracy.

# Plate Detection (plate_drift.py)

Our plate detection model is trained with basic augmentations such as minor brightness (hsv_v=0.2), hue/saturation adjustments, and small geometric distortions (e.g., scale=0.1, degrees=5.0). However, this setup limits the model's robustness to significant distribution shifts, such as:

- Changes in lighting (e.g., night vs. day)

- Lower-quality or blurred surveillance frames

- Changes in camera distance affecting object size

- Rare occlusion or background clutter patterns

To detect such shifts proactively, we compute key image-level metrics on incoming data:

Brightness: sensitivity to lighting conditions

Contrast: object-background separability

Sharpness: focus/clarity of plates
These are compared against the reference (training) distribution using relative thresholds. If a meaningful drift is detected, retraining is triggered to maintain accuracy and reliability.

