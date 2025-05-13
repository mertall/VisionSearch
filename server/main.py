import threading
from fastapi import FastAPI, HTTPException, Query
from server.index_store import HNSWIndexSingleton
from server.triton_client import query_triton_text_encoder
from server.models.search import SearchResponse, SearchResult
from server.models.status import StatusResponse
from server.models.requests import IndexBuildRequest
from scripts.build_index import enqueue_hf_dataset

app = FastAPI()

@app.get("/index/status", response_model=StatusResponse)
def check_status():
    HNSWIndexSingleton.ensure_ready()
    return StatusResponse(status="ready" if HNSWIndexSingleton.is_ready() else "building")

@app.get("/search", response_model=SearchResponse)
def search(query: str = Query(...), k: int = Query(default=5, ge=1, le=100)):
    if not HNSWIndexSingleton.is_ready():
        raise HTTPException(status_code=503, detail="Index is still building.")
    
    vec = query_triton_text_encoder(query)
    results, scores = HNSWIndexSingleton.query(vec, k)
    return SearchResponse(
        results=[SearchResult(image_path=p, score=s) for p, s in zip(results, scores)]
    )

@app.post("/index/build")
def build_index(req: IndexBuildRequest):
    try:
        # Run the enqueueing in a background thread to stream asynchronously
        thread = threading.Thread(
            target=enqueue_hf_dataset,
            args=(req.dataset_repo, req.split, req.image_column),
            daemon=True
        )
        thread.start()
        return {"status": "started", "dataset": req.dataset_repo}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
