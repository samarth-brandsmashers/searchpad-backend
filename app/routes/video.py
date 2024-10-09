from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services import video_service
from app.models.video_model import VideoRequest, VideoResponse

router = APIRouter()

@router.post("/generate-video", response_model=VideoResponse)
async def generate_video(request: VideoRequest):
    """
    Endpoint to generate a video from an image using RunwayML's API.

    Args:
        request (VideoRequest): The video generation request payload.

    Returns:
        VideoResponse: Contains the task ID and status of the video generation.
    """
    try:
        response = video_service.create_video(
            prompt_image=request.promptImage,
            model=request.model,
            seed=request.seed,
            prompt_text=request.promptText,
            watermark=request.watermark
        )
        return VideoResponse(task_id=response["task_id"], status=response["status"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
