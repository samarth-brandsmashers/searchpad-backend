import os
import requests
from typing import Optional

def create_video(
    prompt_image: str,
    model: str,
    seed: Optional[int] = None,
    prompt_text: Optional[str] = None,
    watermark: bool = False
) -> dict:
    """
    Sends a request to RunwayML's image_to_video API to generate a video from an image.

    Args:
        prompt_image (str): HTTPS URL to the input image.
        model (str): Model variant to use ("gen3a_turbo").
        seed (Optional[int]): Seed integer [0..999999999].
        prompt_text (Optional[str]): Detailed description for the video.
        watermark (bool): Whether to include a Runway watermark.

    Returns:
        dict: Response from RunwayML API containing task_id and status.

    Raises:
        Exception: If API key or URL is missing, or if the request fails.
    """
    api_key = os.getenv("RUNWAYML_API_KEY")
    api_url = os.getenv("RUNWAYML_API_URL")
    runway_version = os.getenv("RUNWAYML_VERSION")

    if not api_key:
        raise Exception("RunwayML API key not found in environment variables.")
    if not api_url:
        raise Exception("RunwayML API URL not found in environment variables.")
    if not runway_version:
        raise Exception("RunwayML Version not found in environment variables.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Runway-Version": runway_version
    }

    payload = {
        "promptImage": prompt_image,
        "model": model,
        "watermark": watermark
    }

    if seed is not None:
        payload["seed"] = seed
    if prompt_text:
        payload["promptText"] = prompt_text

    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code in [200, 201]:
        data = response.json()
        # Assuming the API returns a task_id and status
        return {
            "task_id": data.get("task_id"),
            "status": data.get("status", "initiated")
        }
    else:
        raise Exception(f"Video generation failed: {response.status_code} - {response.text}")
