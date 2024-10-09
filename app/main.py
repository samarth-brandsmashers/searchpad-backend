from fastapi import FastAPI
from app.routes import video
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="AI Video Generation API",
    description="API for generating videos using AI based on user prompts.",
    version="1.0.0"
)

# Include video routes
app.include_router(video.router, prefix="/api/v1", tags=["Video Generation"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Video Generation API!"}
