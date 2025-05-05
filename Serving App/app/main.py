from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles  # ✅ NEW
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
from app.model_handler import detect_plate_numbers
from datetime import datetime
import shutil

app = FastAPI()

# ✅ Mount /static
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="app/templates")

UPLOAD_FOLDER = "app/uploads"
CSV_FILE = "app/offenders.csv"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        records = df.to_dict(orient="records")
    else:
        records = []
    return templates.TemplateResponse("history.html", {"request": request, "records": records})

@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    plates = detect_plate_numbers(file_location)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_record = {
        "Image Name": file.filename,
        "Helmet Status": "No Helmet",
        "Plate Detected": "Yes" if plates else "No",
        "Plate Text": ", ".join(plates) if plates else "Unrecognized",
        "Timestamp": timestamp
    }

    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    else:
        df = pd.DataFrame([new_record])

    df.to_csv(CSV_FILE, index=False)

    shutil.copy(CSV_FILE, "app/static/offenders.csv")

    return templates.TemplateResponse("results.html", {
        "request": request,
        "plates": plates,
        "timestamp": timestamp
    })
