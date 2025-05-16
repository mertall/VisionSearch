# VisionSearch

Docker container with text and image CLIP-based embeddings served on Sagemaker Inference Endpoint, HNSW indexing, and RAG-powered search, exposed via FastAPI router.

## Components

1. **Model Packaging & SageMaker Endpoint**

   * Download `openai/clip-vit-base-patch32` artifacts.
   * Copy custom code (`inference.py`, `requirements.txt`) under `model/code/`.
   * Archive `model/` → `model.tar.gz`, upload to S3.
   * Deploy a HuggingFaceModel endpoint in SageMaker serving both vision/text embeddings.

2. **Data Ingestion & HNSW Indexing**

   * API endpoint (FastAPI/FastAPI) to pull a HF dataset (images or text), embed via the sagemaker inference endpoint, and build an HNSW index (hnswlib).
   * Simple pipeline: read dataset → embed → insert into index.

3. **Retrieval-Augmented Generation (RAG)**

   * Query-time: given a text prompt, query the index for top-k nearest image/text embeddings.

## Local Setup

1. **AWS Setup**

   ```bash
   brew install awscli       # macOS
   aws configure             
   ```

   You must set these yourself.  
   Standard Sagemaker Execution role with S3 bucket priveleges, find ARN on AWS or pull it down with boto3 - iam

   ```bash
    AWS_ACCESS_KEY_ID=""
    AWS_SECRET_ACCESS_KEY=""
    SAGEMAKER_ROLE_ARN = "" 
    HF_TOKEN = ""
   ```

2. **AWS Cloud Infrastructure**  
   `cloud/sagemaker_deploy.py`  

   Sets up our AWS infra, as long as we provision a Sagemaker execution role given all s3 bucket privleges.  

    ```bash
    cd cloud
    python3 sagemaker_deploy.py
    ```

3. **Docker**

   ```bash
   docker-compose build
   docker-compose up
   ```
4. **Fast API docs**

   `localhost:8000/docs`  

      ---

      ### GET /index/status

      **Summary:** Check Status

      **Response (200):**

      * **Content:** `application/json`
      * **Schema:** [StatusResponse](#statusresponse)

      ---

      ### POST /index/build

      **Summary:** Build Index Endpoint
      **Operation ID:** `build_index_endpoint_index_build_post`

      #### Request Body

      * **Content:** `application/json`
      * **Schema:** [IndexBuildRequest](#indexbuildrequest)

      #### Responses

      * **200 (Successful Response)**

      * **Content:** `application/json`
      * **Schema:** *empty* (acknowledgement)
      * **422 (Validation Error)**

      * **Content:** `application/json`
      * **Schema:** [HTTPValidationError](#httpvalidationerror)

      ---

      ### GET /search

      **Summary:** Search

      **Responses:**

      * **200 (Successful Response)**

      * **Content:** `application/json`
      * **Schema:** [SearchResponse](#searchresponse)
      * **422 (Validation Error)**

      * **Content:** `application/json`
      * **Schema:** [HTTPValidationError](#httpvalidationerror)

      ---

## Future Work

* **Improve Data Pipeline**  
   Utilize celery workers and tasks, so we can load index quickly with ~100 embeddings and then continue to add more embeddings in background.

* **Cloudformation Template**  
   Make deployment of cloud services even easier.  
   Former2 is a great way to quickly spin up a Cloudformation script for use by anyone, simplifies creating a role to use with Sagemaker.

* **Frontend**  
  A lightweight React interface for:

  * Searching by text or uploading an image.
  * Viewing nearest neighbors images.
