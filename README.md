# PPE Shield AI

Building a quick object detection app to track PPE (hardhats, vests, etc) on construction sites using YOLOv8. 

I'm mostly doing this to learn FastAPI and test out the new YOLO models. 

## Setup

1. Make sure you have python 3 installed (I'm using 3.10 but whatever).
2. Install the backend stuff: `pip install -r backend/requirements.txt`
3. Download the weights `best.pt` from the colab notebook (or train your own) and stick it in `backend/weights/`. 
4. Run the backend: `python backend/run.py`

## Structure
- `backend/` - FastAPI junk. The ML inference happens here.
- `frontend/` - Basic vanilla JS and Tailwind frontend. Nothing fancy like React, didn't need it.
- `notebooks/` - The training script. If you want to retrain the model, run this in Colab.

## TODO
- [ ] Make the webcam feed less laggy 
- [ ] Hook up a real database for users instead of hardcoding `admin` lol
- [ ] Better error handling when the model crashes on weird images
