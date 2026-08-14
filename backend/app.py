import os
import time
import logging
import io
import requests
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from PIL import Image
import numpy as np
from ultralytics import YOLO

# setup basic logging so it doesnt fail silently
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# classes we care about. 
# TODO: move this to a config file eventually
PPE_CLASSES = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

# hex colors for frontend mapping
CLASS_COLORS = {
    'Hardhat': '#00FF88',       # green-ish
    'Mask': '#00CCFF',          
    'NO-Hardhat': '#FF4444',    # red
    'NO-Mask': '#FF6B35',       
    'NO-Safety Vest': '#FF1744',
    'Person': '#FFEB3B',        # yellow
    'Safety Cone': '#FF9800',   
    'Safety Vest': '#4CAF50',   
    'machinery': '#9C27B0',     
    'vehicle': '#2196F3',       
}

app = FastAPI(title="PPE Detection backend")

# allow frontend to call us during local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # FIXME: don't use * in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "weights/best.pt")
model = None # global model to avoid cold starts

@app.on_event("startup")
def load_yolo():
    global model
    try:
        logger.info(f"trying to load weights from {MODEL_PATH}")
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            logger.info("loaded model!")
        else:
            logger.warning("couldn't find best.pt. API calls will crash!")
    except Exception as e:
        logger.error(f"model load failed: {e}")

# --- AUTH ---
security = HTTPBearer()

def check_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # super basic auth for now until we add a real db
    if credentials.credentials != "fake-jwt-token-123":
        raise HTTPException(status_code=401, detail="bad token")
    return credentials.credentials

class LoginReq(BaseModel):
    username: str
    password: str

@app.post("/api/login")
def do_login(req: LoginReq):
    # hardcoded admin for testing
    if req.username == "admin" and req.password == "admin":
        return {"token": "fake-jwt-token-123"}
    raise HTTPException(status_code=401, detail="wrong user/pass")

# --- API ---
class UrlReq(BaseModel):
    url: str # just using str instead of HttpUrl cause it's easier
    confidence: float = 0.25

class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

        detections=dets,
        image_width=w,
        image_height=h,
        inference_time_ms=inf_time,
        total_detections=len(dets)
