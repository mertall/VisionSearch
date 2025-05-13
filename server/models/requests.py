from pydantic import BaseModel

class IndexBuildRequest(BaseModel):
    dataset_repo: str
    split: str = "train"  # default to 'train'
    image_column: str = "image"