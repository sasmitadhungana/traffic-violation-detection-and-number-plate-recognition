from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uuid
import os
import shutil
from datetime import datetime

from .database import get_db
from .inference_engine import engine
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="TrafficGuard API")

# Mount the parent output directory to serve evidence citation and plate images
parent_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
os.makedirs(parent_output_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=parent_output_dir), name="static")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def process_video_background(video_id: str, video_path: str):
    """Background task to run inference on the uploaded video."""
    db = get_db()
    try:
        # Run inference
        results = await engine.process_video(video_id, video_path, db)
        
        # Update video analysis record
        db["video_analyses"].update_one(
            {"_id": video_id},
            {"$set": {
                "status": results["status"],
                "violation_count": results["violation_count"]
            }}
        )
            
        # Add mock violations to DB
        if results["violations"]:
            # Insert all violations at once
            db["violations"].insert_many(results["violations"])
            
    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        db["video_analyses"].update_one(
            {"_id": video_id},
            {"$set": {"status": "Failed"}}
        )

@app.post("/api/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    db = get_db()
    video_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{video_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Create DB record
    new_analysis = {
        "_id": video_id,
        "filename": file.filename,
        "status": "Processing",
        "upload_time": datetime.utcnow(),
        "violation_count": 0
    }
    db["video_analyses"].insert_one(new_analysis)
    
    # Trigger inference in background
    background_tasks.add_task(process_video_background, video_id, file_path)
    
    return {"message": "Upload successful, processing started.", "video_id": video_id}

from pydantic import BaseModel
from passlib.context import CryptContext
from fastapi import HTTPException

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/api/login")
def login(request: LoginRequest):
    db = get_db()
    user = db["users"].find_one({"email": request.email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
        
    if not pwd_context.verify(request.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
        
    return {"message": "Login successful", "token": "dummy_jwt_token_for_now"}


from typing import Optional

@app.get("/api/analysis")
def get_recent_analysis(limit: Optional[int] = 100):
    db = get_db()
    analyses = list(db["video_analyses"].find().sort("upload_time", -1).limit(limit))
    for doc in analyses:
        doc["id"] = doc.pop("_id")
    return analyses

@app.get("/api/stats")
def get_stats():
    db = get_db()
    total_videos = db["video_analyses"].count_documents({})
    total_violations = db["violations"].count_documents({})
    
    # Get breakdown by type for pie chart
    pipeline = [
        {"$group": {"_id": "$type", "count": {"$sum": 1}}}
    ]
    type_counts = list(db["violations"].aggregate(pipeline))
    pie_data = [{"name": item["_id"], "value": item["count"]} for item in type_counts]
    
    return {
        "total_videos": total_videos,
        "total_violations": total_violations,
        "high_priority": total_violations, # simplify logic for now
        "pie_data": pie_data
    }

@app.get("/api/reports")
def get_reports(limit: Optional[int] = 100):
    db = get_db()
    violations = list(db["violations"].find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
    return violations


@app.get("/api/analysis/{video_id}")
def get_analysis_status(video_id: str):
    db = get_db()
    record = db["video_analyses"].find_one({"_id": video_id})
    if not record:
        return {"error": "Video not found"}
        
    violations = list(db["violations"].find({"video_id": video_id}, {"_id": 0}))
    return {
        "status": record.get("status"),
        "filename": record.get("filename"),
        "violation_count": record.get("violation_count", 0),
        "violations": violations
    }

@app.get("/")
def read_root():
    return {"status": "TrafficGuard API is running with MongoDB."}
