from pydantic import BaseModel

class IndexBuildRequest(BaseModel):
    dataset_repo: str ="AI-Lab-Makerere/beans"
    split: str = "train"  # default to 'train'
    image_column: str = "image"