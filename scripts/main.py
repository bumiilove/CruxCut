from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.requests import Request
from fastapi.exceptions import RequestValidationError
import os
import uuid
import shutil
from pathlib import Path
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import json
from datetime import datetime
from typing import Dict, Any
import asyncio
from starlette.exceptions import HTTPException as StarletteHTTPException
from video_cropper_oop import cropper, VideoCropper, CropperConfig  # 비디오 크롭퍼와 설정 클래스 임포트


app = FastAPI(title="CruxCut Video Processing API", version="1.0.0")

@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": exc.errors(),
            "body": exc.body
        }
    )
    
# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Next.js 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 디렉토리 설정
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
TEMP_DIR = Path("temp")

# 디렉토리 생성
for directory in [UPLOAD_DIR, PROCESSED_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# 작업 상태 저장소 (실제 환경에서는 Redis나 데이터베이스 사용)
job_status: Dict[str, Dict[str, Any]] = {}

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """비디오 파일 업로드 및 처리 시작"""
    
    # 파일 형식 검증
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="Only video files are allowed")
    
    # 고유 작업 ID 생성
    job_id = str(uuid.uuid4())[:8]
    
    # 파일 저장
    file_extension = Path(file.filename).suffix
    input_filename = f"{job_id}_input{file_extension}"
    input_path = UPLOAD_DIR / input_filename
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 출력 파일 경로
    output_filename = f"{job_id}_processed.mp4"
    output_path = PROCESSED_DIR / output_filename
    
    # 작업 상태 초기화
    job_status[job_id] = {
        "status": "processing",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "input_file": str(input_path),
        "output_file": str(output_path)
    }
    

    # 백그라운드에서 비디오 처리 시작
    background_tasks.add_task(
        # 원래의 더미 코드
        # VideoProcessor.crop_and_track_video,
        # str(input_path),
        # str(output_path),
        # job_id
        
        cropper.process_video,
        str(input_path),
        str(output_path),
        job_status,
        job_id
    )
    return {"job_id": job_id, "message": "Video processing started"}

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in job_status:
        return {"status_code": 404, "content": {"error": f"Job ID '{job_id}' not found"}}
    return JSONResponse(content=job_status[job_id])

@app.get("/download/{job_id}")
async def download_processed_video(job_id: str):
    """처리된 비디오 다운로드"""
    
    file_path = PROCESSED_DIR / f"{job_id}_processed.mp4"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Processed file not found")

    return FileResponse(path=file_path, media_type='video/mp4', filename=f"cruxcut_processed_{job_id}.mp4")
    
    job = job_status[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    output_path = job["output_file"]
    
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Processed file not found")
    
    return FileResponse(
        path=output_path,
        media_type='video/mp4',
        filename=f"cruxcut_processed_{job_id}.mp4"
    )

@app.get("/")
async def root():
    return {"message": "CruxCut Video Processing API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
