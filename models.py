from pydantic import BaseModel, Field, HttpUrl


class ImageInput(BaseModel):
    id: str = Field(..., min_length=1, description="Stable identifier for the image.")
    url: HttpUrl


class GroupImagesRequest(BaseModel):
    images: list[ImageInput] = Field(..., min_length=1)
    hash_distance_threshold: int = Field(
        10,
        ge=0,
        le=64,
        description="Max Hamming distance between p-hashes to treat as the same group.",
    )


class ImageGroup(BaseModel):
    image_ids: list[str]
    recommended_id: str


class GroupImagesResponse(BaseModel):
    groups: list[ImageGroup]
