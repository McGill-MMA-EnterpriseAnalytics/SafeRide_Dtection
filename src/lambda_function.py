import os
import json
import boto3
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import logging
import easyocr

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize boto3 client
s3_client = boto3.client('s3')

# Configuration
MODEL_BUCKET = 'is2-project'
HELMET_MODEL_KEY = 'model/best_helmet_detection_model.pt'
PLATE_MODEL_KEY = 'model/best_plate_detection_model.pt'
BUCKET = 'is2-project'

# Initialize EasyOCR once (only when Lambda container warms up)
ocr_reader = easyocr.Reader(
    ['en'],
    model_storage_directory='/tmp/.EasyOCR',
    user_network_directory='/tmp/.EasyOCR/user_network'
)

def download_model_from_s3(model_key):
    logger.info(f"Downloading model {model_key}")
    response = s3_client.get_object(Bucket=MODEL_BUCKET, Key=model_key)
    model_data = response['Body'].read()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    temp_file.write(model_data)
    temp_file.close()
    return temp_file.name

def process_helmet_detection(image_data, helmet_model_path):
    model = YOLO(helmet_model_path)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model(img)

    detections = []
    for result in results:
        for box in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, class_id = box
            detections.append({
                'class': int(class_id),
                'label': 'Without Helmet' if int(class_id) == 1 else 'With Helmet',
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })

    return img, detections

def process_plate_detection(img, plate_model_path):
    model = YOLO(plate_model_path)
    results = model(img)

    plates = []
    for result in results:
        for box in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, _ = box
            plate_roi = img[int(y1):int(y2), int(x1):int(x2)]
            ocr_result = ocr_reader.readtext(plate_roi)
            text = ' '.join([r[1] for r in ocr_result]) if ocr_result else 'N/A'

            plates.append({
                'plate_number': text.strip(),
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
    return plates

def lambda_handler(event, context):
    logger.info("Lambda function started")
    try:
        body = json.loads(event['body']) if 'body' in event and event['body'] else {}
        input_prefix = body.get('input_prefix', 'images/')
        output_prefix = body.get('output_prefix', 'results/')

        if not input_prefix.endswith('/'):
            input_prefix += '/'
        if not output_prefix.endswith('/'):
            output_prefix += '/'

        helmet_model_path = download_model_from_s3(HELMET_MODEL_KEY)
        plate_model_path = download_model_from_s3(PLATE_MODEL_KEY)

        paginator = s3_client.get_paginator('list_objects_v2')
        processed_files = []

        for page in paginator.paginate(Bucket=BUCKET, Prefix=input_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.lower().endswith(('.jpg', '.jpeg', '.png')):
                        logger.info(f"Processing image: {key}")
                        response = s3_client.get_object(Bucket=BUCKET, Key=key)
                        image_data = response['Body'].read()

                        frame, helmet_detections = process_helmet_detection(image_data, helmet_model_path)

                        if any(d['class'] == 1 for d in helmet_detections):
                            plate_detections = process_plate_detection(frame, plate_model_path)
                        else:
                            plate_detections = []

                        output = {
                            'detections': helmet_detections,
                            'plates': plate_detections
                        }

                        relative_path = os.path.relpath(key, input_prefix)
                        output_key = os.path.join(output_prefix, os.path.splitext(relative_path)[0] + '.json')

                        s3_client.put_object(
                            Bucket=BUCKET,
                            Key=output_key,
                            Body=json.dumps(output),
                            ContentType='application/json'
                        )

                        processed_files.append({
                            'input_file': key,
                            'output_file': output_key,
                            'helmet_detections': len(helmet_detections),
                            'plate_detections': len(plate_detections)
                        })

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'message': 'Batch processing completed',
                'processed_files': processed_files
            })
        }

    except Exception as e:
        logger.error(f"Lambda function failed: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e)
            })
        }
