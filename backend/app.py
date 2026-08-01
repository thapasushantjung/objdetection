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

