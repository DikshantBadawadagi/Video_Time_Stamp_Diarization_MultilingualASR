# app/main.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pathlib import Path

import os

# Point Whisper/ffmpeg-python to the correct ffmpeg.exe and add to PATH
FFMPEG_DIR = r"C:\ffmpeg-2025-12-07-git-c4d22f2d2c-full_build\bin"
os.environ["FFMPEG_BINARY"] = os.path.join(FFMPEG_DIR, "ffmpeg.exe")
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")


import whisper
import uuid
import asyncio
from starlette.background import BackgroundTasks

from app.controllers import process_video_file
from app.models import MongoDB

load_dotenv()  # load .env

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./tmp_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/whisper_app")

# create DB wrapper
db = MongoDB(MONGODB_URI)

# load whisper model at startup to avoid reloading per request
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")  # default base
print("Loading whisper model:", WHISPER_MODEL_NAME)
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
print("Whisper model loaded.")

app = FastAPI(title="Video Homework Coach - Transcribe & Suggest")

@app.on_event("startup")
async def startup_event():
    await db.connect()

@app.on_event("shutdown")
async def shutdown_event():
    await db.close()

def save_upload_file_tmp(upload_file: UploadFile, destination: str):
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    upload_file.file.close()

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    """
    Accepts multipart/form-data with field 'file' containing the video.
    Returns JSON with id and sentences list (text/start/end/improvement).
    """
    # basic checks
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    content_type = file.content_type or ""
    # allow many video types; if audio-only is uploaded, it's fine
    allowed_prefixes = ("video/", "audio/")
    if not any(content_type.startswith(p) for p in allowed_prefixes):
        # still allow if it looks like a binary with extension
        pass

    tmp_filename = str(UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}")
    try:
        # save file to disk (blocking) in threadpool
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, save_upload_file_tmp, file, tmp_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to save upload: {str(e)}")

    try:
        result = await process_video_file(tmp_filename, whisper_model, db)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"processing error: {str(e)}")
    finally:
        # cleanup the uploaded video file
        try:
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)
        except Exception:
            pass
