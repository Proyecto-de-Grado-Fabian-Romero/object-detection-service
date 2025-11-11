from typing import Dict, List

from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    request_id: str = Field(alias="RequestId")
    image_urls: List[str] = Field(alias="ImageUrls")
    reply_to: str = Field(alias="ReplyTo")

    class Config:
        populate_by_name = True


class DetectionResponse(BaseModel):
    request_id: str = Field(alias="RequestId")
    detected_objects: Dict[str, int] = Field(alias="DetectedObjects")
    success: bool = Field(alias="Success", default=True)
    error: str = Field(alias="Error", default="")

    class Config:
        populate_by_name = True
