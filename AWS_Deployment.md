## What It Does

1. Reads images from the `images/` folder in your S3 bucket.
2. Detects whether people are wearing helmets.
3. If anyone is detected **without** a helmet, runs license plate detection and OCR.
4. Saves the result JSON files into the `results/` folder in the same bucket.

---

## Demo Video

Watch the step-by-step demo:  
🎥 [Click to view](https://mcgill-my.sharepoint.com/:v:/g/personal/maralmaa_batnasan_mail_mcgill_ca/EaasJtGIxp9IiyyYB68MOY8B9mxc6jcJZNRD_S4PSdyd1w?e=IGaSDs)

---

## How to Use (via Postman)

### 1. Upload Image to S3 (Manually or via AWS Console)

Place your image inside the `images/` folder in your S3 bucket.

### 2. Open Postman

- **Method**: `POST`
- **URL**:  https://6dismesldb.execute-api.us-east-2.amazonaws.com/default/yolo-image-processor


- **Headers**:
- `Content-Type`: `application/json`

- **Body**:  
Go to "Body" → select **raw** → set **JSON** type → paste:

{
  "input_prefix": "images/",
  "output_prefix": "results/"
}

### 3. Click "Send" - You will receive a JSON response like:

{
    "message": "Batch processing completed",
    "processed_files": [
        {
            "input_file": "images/test.png",
            "output_file": "results/test.json",
            "helmet_detections": 1,
            "plate_detections": 1
        }
    ]
}


### 4. Go to the results/ folder in S3 bucket to see the detailed JSON output.


