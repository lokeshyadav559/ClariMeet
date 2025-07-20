#!/usr/bin/env python
"""
FastAPI Web Server for Audio Processing Pipeline
Integrates the AudioPipelineMVP with a web interface using WebSockets for real-time updates
"""

import os
import json
import asyncio
import uuid
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import your existing pipeline (assuming it's in the same directory)
from audio_pipeline_mvp import AudioPipelineMVP, logger

# FastAPI app initialization
@asynccontextmanager
async def lifespan(app):
    # Startup logic
    logger.info("FastAPI Audio Processing Server starting up...")
    os.makedirs("temp_processing", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    logger.info("Server startup completed")
    yield
    # Shutdown logic
    logger.info("FastAPI Audio Processing Server shutting down...")
    for client_id in list(manager.active_connections.keys()):
        manager.disconnect(client_id)
    logger.info("Server shutdown completed")

app = FastAPI(title="ClariMeet Audio Processing API", version="1.0.0", lifespan=lifespan)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state management
active_connections: Dict[str, WebSocket] = {}
processing_tasks: Dict[str, Dict[str, Any]] = {}
pipeline_config = {
    "whisper_model": "large-v3",
    "language": None,
    "lm_studio_url": "http://localhost:1234",
    "lm_studio_model": None,
    "chunk_length_ms": 30000
}

# Initialize the pipeline
audio_pipeline = AudioPipelineMVP(pipeline_config)

# Pydantic models for request/response
class ProcessingStatus(BaseModel):
    task_id: str
    status: str  # 'queued', 'processing', 'completed', 'error'
    progress: float  # 0-100
    current_step: str
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class StartSessionRequest(BaseModel):
    link: str

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket connection established for client: {client_id}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket connection closed for client: {client_id}")

    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: Dict[str, Any]):
        disconnected_clients = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)

manager = ConnectionManager()

# Helper function to update task status
async def update_task_status(task_id: str, status: str, progress: float, 
                           current_step: str, message: str, result: Optional[Dict] = None, 
                           error: Optional[str] = None):
    """Update task status and notify connected clients"""
    processing_tasks[task_id] = {
        "task_id": task_id,
        "status": status,
        "progress": progress,
        "current_step": current_step,
        "message": message,
        "result": result,
        "error": error,
        "updated_at": datetime.now().isoformat()
    }

    # Broadcast update to all connected clients
    await manager.broadcast({
        "type": "status_update",
        "task_id": task_id,
        "data": processing_tasks[task_id]
    })

# Background task to process audio files
async def process_audio_background(task_id: str, audio_file_path: str, output_dir: str):
    """Background task to process audio files with progress updates"""
    try:
        await update_task_status(task_id, "processing", 0, "starting", "Initializing audio processing...")

        # Step 1: Audio preprocessing (20% of progress)
        await update_task_status(task_id, "processing", 10, "preprocessing", "Preprocessing audio file...")
        await asyncio.sleep(0.1)  # Allow other tasks to run

        # We need to modify the existing pipeline to support progress callbacks
        # For now, we'll simulate the process with status updates

        processed_files = audio_pipeline.audio_processor.preprocess_audio(audio_file_path, 
                                                                         os.path.join(output_dir, "temp"))
        await update_task_status(task_id, "processing", 20, "preprocessing", "Audio preprocessing completed")

        # Step 2: Transcription (60% of progress - this is the longest step)
        await update_task_status(task_id, "processing", 30, "transcription", "Starting audio transcription...")
        transcription_results = audio_pipeline.transcriber.batch_transcribe(processed_files)
        await update_task_status(task_id, "processing", 80, "transcription", "Audio transcription completed")

        # Step 3: Text combination (5% of progress)
        await update_task_status(task_id, "processing", 85, "combining", "Combining transcriptions...")
        combined_text = audio_pipeline._combine_transcriptions(transcription_results)

        # Step 4: Summary generation (15% of progress)
        await update_task_status(task_id, "processing", 90, "summarization", "Generating summaries...")

        summaries = {}
        summary_types = ["comprehensive", "actionable", "bullet_points"]
        for i, summary_type in enumerate(summary_types):
            summaries[summary_type] = audio_pipeline.lm_client.generate_summary(combined_text, summary_type)
            progress = 90 + (5 * (i + 1) / len(summary_types))
            await update_task_status(task_id, "processing", progress, "summarization", 
                                   f"Generated {summary_type} summary")

        # Compile final results
        result = {
            "task_id": task_id,
            "audio_file": audio_file_path,
            "processing_time": datetime.now().isoformat(),
            "transcription": {
                "full_text": combined_text,
                "word_count": len(combined_text.split())
            },
            "summaries": summaries,
            "metadata": {
                "whisper_model": pipeline_config.get("whisper_model"),
                "chunks_processed": len(processed_files)
            }
        }

        # Save results
        audio_pipeline._save_results({
            "audio_file": audio_file_path,
            "processing_time": datetime.now().isoformat(),
            "transcription": {"full_text": combined_text, "segments": transcription_results},
            "summaries": summaries,
            "metadata": result["metadata"],
            "performance": {}
        }, output_dir)

        await update_task_status(task_id, "completed", 100, "completed", "Processing completed successfully", result)
        logger.info(f"Audio processing completed successfully for task {task_id}")

    except Exception as e:
        logger.error(f"Error processing audio for task {task_id}: {e}")
        await update_task_status(task_id, "error", 0, "error", f"Processing failed: {str(e)}", error=str(e))

# API Routes
@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    """Serve the main HTML interface"""
    try:
        with open("index2.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>ClariMeet</h1><p>Frontend not found. Please ensure index2.html exists.</p>", 
                          status_code=404)

@app.post("/start-session")
async def start_session(request: StartSessionRequest):
    """Start a new meeting session"""
    logger.info(f"Starting new session with link: {request.link}")
    return {"status": "success", "message": "Session started", "link": request.link}

@app.post("/upload-audio")
async def upload_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process audio file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate file type
    allowed_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.mp4', '.webm'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

    # Generate unique task ID
    task_id = str(uuid.uuid4())

    # Create temp directory for this task
    output_dir = os.path.join("temp_processing", task_id)
    os.makedirs(output_dir, exist_ok=True)

    # Save uploaded file
    temp_file_path = os.path.join(output_dir, file.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File uploaded successfully: {temp_file_path}")

        # Initialize task status
        await update_task_status(task_id, "queued", 0, "queued", "File uploaded, queued for processing")

        # Start background processing
        background_tasks.add_task(process_audio_background, task_id, temp_file_path, output_dir)

        return {
            "task_id": task_id,
            "status": "queued",
            "message": "File uploaded successfully and queued for processing",
            "filename": file.filename
        }

    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get current status of a processing task"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return processing_tasks[task_id]

@app.get("/results/{task_id}")
async def get_task_results(task_id: str):
    """Get detailed results of a completed task"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = processing_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")

    return task["result"]

@app.get("/download/{task_id}/{file_type}")
async def download_result_file(task_id: str, file_type: str):
    """Download result files (transcript, summaries, etc.)"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = processing_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")

    output_dir = os.path.join("temp_processing", task_id)

    file_mapping = {
        "transcript": "transcript.txt",
        "comprehensive": "summary_comprehensive.txt",
        "actionable": "summary_actionable.txt",
        "bullet_points": "summary_bullet_points.txt",
        "results": "results.json"
    }

    if file_type not in file_mapping:
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_path = os.path.join(output_dir, file_mapping[file_type])
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=file_mapping[file_type],
        media_type='application/octet-stream'
    )

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket, client_id)

    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types
            if message.get("type") == "ping":
                await manager.send_personal_message({"type": "pong"}, client_id)
            elif message.get("type") == "subscribe":
                # Client wants to subscribe to specific task updates
                task_id = message.get("task_id")
                if task_id and task_id in processing_tasks:
                    await manager.send_personal_message({
                        "type": "status_update",
                        "task_id": task_id,
                        "data": processing_tasks[task_id]
                    }, client_id)

    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_tasks": len(processing_tasks),
        "active_connections": len(manager.active_connections)
    }

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
