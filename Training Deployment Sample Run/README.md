# SafeRide Training Deployment

This repository contains the Docker-based training pipeline for the SafeRide helmet and license plate detection models. The training setup is designed to work on both CPU and GPU environments, with configurable parameters for different compute resources.

## Overview

The SafeRide training deployment consists of:

- Helmet detection model training using YOLOv8
- License plate detection model training using YOLOv8
- MLflow integration for experiment tracking
- Docker-based deployment for reproducible training environments

## Requirements

- Docker (19.03 or higher)
- Docker Compose (optional, but recommended)
- Kaggle API credentials (for dataset download)

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/McGill-MMA-EnterpriseAnalytics/SafeRide_Dtection.git
cd Training Deployment Sample Run
```

### 2. Set Up Kaggle Credentials

Create a `.env` file with your Kaggle credentials:

```bash
# .env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

To obtain your Kaggle API credentials:
1. Log in to [Kaggle](https://www.kaggle.com/)
2. Click on your profile picture and select "Account"
3. Scroll to the "API" section and click "Create New API Token"
4. A `kaggle.json` file will be downloaded, containing your credentials

### 3. Build and Run the Training

#### Using Docker Compose (Recommended)

```bash
# Build the Docker image
docker-compose build

# Run the training (both helmet and plate detection)
docker-compose up
```


## Project Structure

```
Training Deployment Sample Run/
├── data/                  # Downloaded datasets and processed data
├── models/                # Trained model outputs
├── mlruns/                # MLflow experiment tracking data
├── runs/                  # Training run outputs
├── scripts/
│   ├── helmet_training.py # Helmet detection training script
│   ├── plate_training.py  # License plate detection training script
│   └── run_training.py    # Script to run both training pipelines
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
└── .env                   # Environment variables (Kaggle credentials)
```

helmet_training.py and plate_training.py are created based on ```Helmet_detection_v2_extended_training.ipynb``` and ```Number_plate_detection_v2_extended_training.ipynb``` from the notebooks folder. I used ```jupyter nbconvert``` to convert the ```.ipynb``` files to ```.py``` files and made additional changes to make it work for the deployment with docker.

## Sample Run Training Parameters

To ensure compatibility with a personal computer's CPU and avoid out-of-memory errors, I used the following parameters. For production-quality models, use the Training Deployment code.


- **Image Size**: 640px
- **Batch Size**: 4
- **Epochs**: 2 (for quick testing, increase for better results)
- **Sample Size**: 10 images (reduced for CPU training, increase for better results)
- **Devide**: 'cpu'
- **workers**: 1

To run on CPU, ensure ```CUDA_VISIBLE_DEVICES``` is set to 'cpu' instead of -1.

The data folder and .pt files in runs and mlruns are deleted due to the huge size of data, running the deployment will download the correct data to the /data folder. I kept the structure of the files to demonstrate that it ran successfully.

### Trained Models

After training completes, the final models will be available in the `models` directory:
- `models/Final_Helmet.pt` - Helmet detection model
- `models/Final_Plates.pt` - License plate detection model

These models can be used with the SafeRide inference application.