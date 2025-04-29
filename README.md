# SafeRide_Dtection

Helmet and Number Plate Detection with MLflow Experiment Tracking and MLOps Pipeline

---

## 🚦 Project Overview

SafeRide_Dtection is an end-to-end computer vision pipeline for road safety compliance, focusing on:
- **Helmet Detection**: Identifying whether a rider is wearing a helmet.
- **Number Plate Detection & OCR**: Detecting number plates and extracting license numbers if a helmet is not detected.
- **Experiment Tracking & MLOps**: All experiments are tracked with MLflow, and the pipeline is designed for reproducibility and deployment.

### Business Problem
Many road accidents involve riders not wearing helmets. Automating helmet and number plate detection helps enforce safety regulations and enables data-driven policy making.

### Scientific Methodology
- **YOLOv8** is used for object detection (helmets, number plates).
- **EasyOCR** is used for license plate text extraction.
- **MLflow** tracks experiments, metrics, and model versions.
- **MLOps** practices ensure reproducibility, modularity, and scalability.

---

## 🗂️ Project Structure

```
.
├── data/                   # Data storage (user-provided, not versioned)
├── models/                 # Trained model weights (.pt files)
├── notebooks/              # Experimentation and training notebooks
│   ├── helmet_detection/
│   └── plate_detection/
├── outputs/                # Inference results, logs, etc.
├── src/                    # Source code
│   ├── helmet_detection/
│   ├── number_plate_detection/
│   ├── ocr/
│   ├── pipeline/
│   └── utils/
├── tests/                  # Unit tests
├── Model Explainablity/    # Notebooks for model explainability
├── requirements.txt        # Python dependencies
├── pyproject.toml, poetry.lock # Alternative dependency management
├── config.yaml             # Main configuration file for pipeline
├── README.md               # Project documentation
└── ...
```

---

## 📂 External Artifacts Storage (OneDrive)

- [MLflow Experiments Folder](https://mcgill-my.sharepoint.com/:f:/g/personal/yash_sethi_mail_mcgill_ca/Eo-J_BRpWY9Cn8kF7lVSh78Be2IdwUYGAAuZNNyO0ht-Lg?e=j6TV5J)
- [YOLOv8 Detection Runs Folder](https://mcgill-my.sharepoint.com/:f:/g/personal/yash_sethi_mail_mcgill_ca/ElFE4BJzrOxHuGHyLKdpGyMB9Uhol2TbsiETPfeoPsLpWA?e=nxPPEw)
- [Best Trained Models (.pt)](https://mcgill-my.sharepoint.com/:f:/g/personal/yash_sethi_mail_mcgill_ca/Eui_X0SqhjVDnkSxGA_-sP4BSR6IuuP-VqYc6fPcl0jFIQ?e=69uJKB)

**Note:** Model weights (`helmet_detection_best.pt`, `number_plate_detection_best.pt`), MLflow experiment history (`mlruns/`), and YOLO runs (`runs/`) are hosted externally. Download these to run full inference or continue training.

---

## 🧪 Experimentation & Modeling

- **Notebooks**: All experimentation, training, and evaluation are in `notebooks/helmet_detection/` and `notebooks/plate_detection/`.
- **Modeling**: YOLOv8 models for helmet and number plate detection. Custom scripts in `src/helmet_detection/` and `src/number_plate_detection/`.
- **Data Preprocessing**: Data is expected in standard image formats. Preprocessing steps are detailed in the notebooks and utility scripts.
- **Experiment Tracking**: MLflow is used for tracking experiments, metrics, and model versions (see external link above).

---

## ⚙️ How to Use

### 1. **Setup**
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  # or, if using poetry
  poetry install
  ```
- Download model weights and experiment logs from the links above and place them in the appropriate folders (`models/`, etc).

### 2. **Configuration**
- Edit `config.yaml` to set:
  - Model paths (`helmet_model_path`, `plate_model_path`)
  - Inference thresholds (`helmet_confidence_threshold`, `plate_confidence_threshold`)
  - OCR settings (`ocr_languages`, `min_ocr_confidence`)
  - Output and log paths (`output_folder`, `log_file`)
  - Flags (e.g., `save_annotated_images`)
- **Example:**
  ```yaml
  helmet_model_path: "models/helmet_detection_best.pt"
  plate_model_path: "models/number_plate_detection_best.pt"
  helmet_confidence_threshold: 0.3
  plate_confidence_threshold: 0.3
  ocr_languages: ["en"]
  output_folder: "outputs/"
  log_file: "logs/inference.log"
  save_annotated_images: false
  ```
- **Note:** The pipeline script will read all settings from `config.yaml`. Make sure this file is present and correctly configured before running inference.

### 3. **Training**

- **Jupyter Notebooks:**
  - Use the notebooks in `notebooks/helmet_detection/` and `notebooks/plate_detection/` for step-by-step training, evaluation, and experiment tracking with MLflow.
  - These notebooks cover data loading, preprocessing, model training, evaluation, and saving best weights.

- **Custom Training via Scripts:**
  - Advanced users can adapt or extend the scripts in `src/helmet_detection/` and `src/number_plate_detection/` for custom training workflows or automation.
  - You may need to modify these scripts to use your own data or integrate with the configuration system.

- **Tips:**
  - Ensure your data is in the expected format (see notebook examples).
  - Track your experiments using MLflow for reproducibility.
  - Save your best model weights in the `models/` directory for later inference.

### 4. **Inference**
- Run the full pipeline script:
  ```bash
  python src/pipeline/full_inference_pipeline.py
  ```
  This will run helmet detection, number plate detection, and OCR on a sample image (edit the script to change the image path or add CLI support). All model paths, thresholds, and output locations are controlled via `config.yaml`.

### 5. **Testing**
- Run unit tests to validate the codebase:
  ```bash
  pytest tests/
  ```

---

## 🧑‍🔬 Model Explainability

- See the `Model Explainablity/` directory for Jupyter notebooks analyzing model bias and explainability.

---

## 📝 Notes for Developers
- The repository is modular and follows best practices for MLOps and reproducibility.
- All heavy artifacts are stored externally to keep the repo lightweight.
- For any issues, check the notebooks and scripts for detailed docstrings and comments.
- **Configuration-driven:** All key parameters (model paths, thresholds, outputs, logs) are set in `config.yaml` for easy reproducibility and deployment.

---
