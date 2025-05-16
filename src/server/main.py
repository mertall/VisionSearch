import logging
from fastapi import FastAPI, HTTPException, Query
from server.index_store import HNSWIndexSingleton
from server.sage_maker import CLIPSageMakerClient
from server.models.search import SearchResponse, SearchResult
from server.models.status import StatusResponse
from server.models.requests import IndexBuildRequest
from scripts.build_index import build_index

# Configure root logger to output to console
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("server")

app = FastAPI(
    title="VisionSearch API",
    description="API for building and querying an HNSW index of CLIP embeddings over HuggingFace datasets.",
    version="1.0.0",
)

@app.get(
    "/index/status", 
    response_model=StatusResponse,
    summary="Get index build status",
    response_description="Indicates whether the index is ready or still building. Invokes building of index if not ready"
)
def index_status():
    """
    Check the current status of the HNSW index.

    Returns:
        StatusResponse: Contains a single field 'status', which is 'ready' if the index
                        has been built, or 'building' otherwise.
    """
    HNSWIndexSingleton.ensure_ready()
    status = "ready" if HNSWIndexSingleton.is_ready() else "building"
    return StatusResponse(status=status)
@app.post(
    "/index/build", 
    summary="Trigger index build",
    response_description="Confirmation that index build has completed and how many passed/failed."
)
def build_index(
    req: IndexBuildRequest
):
    """
    Build the HNSW index from a HuggingFace dataset.

    Args:
        req (IndexBuildRequest): Contains:
            - dataset_repo (str): HF repo identifier (e.g., 'huggingface/xyz').
            - split (str): Dataset split to use (e.g., 'train').
            - image_column (str): Column name containing image paths or URLs.

    Returns:
        dict: {"status": "completed", "dataset": <repo_id>} on success.

    Raises:
        HTTPException(500): If the index build fails.
    """
    try:
        build_index(req.dataset_repo, req.split, req.image_column)
        return {"status": "completed", "dataset": req.dataset_repo}
    except Exception as e:
        logger.error(f"❌ Build index error: {e}")
        raise HTTPException(status_code=500, detail="Failed to build index.")

@app.get(
    "/search", 
    response_model=SearchResponse,
    summary="Search the index",
    response_description="Return top-k closest images for a given text query."
)
def search(
    query: str = Query(..., description="Text query to encode and search over the index."),
    k: int = Query(5, ge=1, le=100, description="Number of nearest neighbors to retrieve.")
):
    """
    Encode a text query using CLIP and search the HNSW index.

    Args:
        query (str): The input text to encode.
        k (int): Number of nearest neighbors to return.

    Returns:
        SearchResponse: Contains a list of image paths and their similarity scores.
    """
    if not HNSWIndexSingleton.is_ready():
        raise HTTPException(status_code=503, detail="Index is still building.")
    try:
        client = CLIPSageMakerClient()
        vec = client.encode_text(text=query)
    except Exception as e:
        logger.error(f"❌ Text encoding error: {e}")
        raise HTTPException(status_code=500, detail="Failed to encode query text.")

    results, scores = HNSWIndexSingleton.query(vec, k)
    return SearchResponse(
        results=[SearchResult(image_path=p, score=s) for p, s in zip(results, scores)]
    )
