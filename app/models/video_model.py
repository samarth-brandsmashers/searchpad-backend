from pydantic import BaseModel, HttpUrl, Field
from typing import Optional

class VideoRequest(BaseModel):
  promptImage: str = Field(..., description="HTTPS URL to a JPEG, PNG, or WebP image (max 16MB).")
  model: str = Field(..., description='Model variant to use. Accepted value: "gen3a_turbo".')
  seed: Optional[int] = Field(None, ge=0, le=999999999, description="Seed integer [0..999999999].")
  promptText: Optional[str] = Field(None, max_length=512, description="Detailed description for the video.")
  watermark: Optional[bool] = Field(False, description="Whether to include a Runway watermark.")

class VideoResponse(BaseModel):
  task_id: str = Field(..., description="ID of the initiated video generation task.")
  status: str = Field(..., description="Status of the video generation task.")
