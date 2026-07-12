import uvicorn
import os

if __name__ == "__main__":
    # check if they set a custom port in env, otherwise 8000
    p = int(os.environ.get("PORT", 8000))
    
    # print("starting dev server on port", p) # uncomment if debugging startup
    
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=p, 
        reload=True # handy for local dev, maybe turn off for prod?
    )
