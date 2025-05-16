# VisionSearch

Docker container with text and image CLIP-based embeddings served on Sagemaker Inference Endpoint, HNSW indexing, and RAG-powered search, exposed via FastAPI router.

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
      **Operation ID:** `check_status_index_status_get`

      **Response (200):**

      * **Content:** `application/json`
      * **Schema:** [StatusResponse](#statusresponse)

      ---

      ### GET /search

      **Summary:** Search
      **Operation ID:** `search_search_get`

      #### Query Parameters

      | Name  | Type    | Required | Default | Constraints | Description                                     |
      | ----- | ------- | -------- | ------- | ----------- | ----------------------------------------------- |
      | query | string  | Yes      | —       | —           | Text query to encode and search over the index. |
      | k     | integer | No       | 5       | 1 ≤ k ≤ 100 | Number of nearest neighbors to retrieve.        |

      **Responses:**

      * **200 (Successful Response)**

      * **Content:** `application/json`
      * **Schema:** [SearchResponse](#searchresponse)
      * **422 (Validation Error)**

      * **Content:** `application/json`
      * **Schema:** [HTTPValidationError](#httpvalidationerror)

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

      ## Component Schemas

      ### IndexBuildRequest

      | Field         | Type   | Required | Default | Description                                |
      | ------------- | ------ | -------- | ------- | ------------------------------------------ |
      | dataset\_repo | string | Yes      | —       | HF repo identifier (e.g., `user/dataset`). |
      | split         | string | No       | `train` | Dataset split to use.                      |
      | image\_column | string | No       | `image` | Column name for image paths.               |

      ---

      #### SearchResponse

      | Field   | Type                                   | Required | Description             |
      | ------- | -------------------------------------- | -------- | ----------------------- |
      | results | array of [SearchResult](#searchresult) | Yes      | List of search results. |

      ---

      #### SearchResult

      | Field       | Type   | Required | Description                               |
      | ----------- | ------ | -------- | ----------------------------------------- |
      | image\_path | string | Yes      | Path or URL to the image.                 |
      | score       | number | Yes      | Similarity score (higher = more similar). |

      ---

      #### StatusResponse

      | Field  | Type   | Required | Description            |
      | ------ | ------ | -------- | ---------------------- |
      | status | string | Yes      | `ready` or `building`. |

      ---

      #### HTTPValidationError

      | Field  | Type                                         | Description                           |
      | ------ | -------------------------------------------- | ------------------------------------- |
      | detail | array of [ValidationError](#validationerror) | List of individual validation errors. |

      ---

      #### ValidationError

      | Field | Type                | Description                    |
      | ----- | ------------------- | ------------------------------ |
      | loc   | array of string/int | Location of the invalid field. |
      | msg   | string              | Error message.                 |
      | type  | string              | Error type identifier.         |

      ---

*Generated from OpenAPI 3.1.0 specification.*



## Components

1. **Model Packaging & SageMaker Endpoint**

   * Download `openai/clip-vit-base-patch32` artifacts.
   * Copy custom code (`inference.py`, `requirements.txt`) under `model/code/`.
   * Archive `model/` → `model.tar.gz`, upload to S3.
   * Deploy a HuggingFaceModel endpoint in SageMaker serving both vision/text embeddings.

2. **Data Ingestion & HNSW Indexing**

   * API endpoints (FastAPI/FastAPI + Celery initially) to pull a HF dataset (images or text), embed via the endpoint, and build an HNSW index (hnswlib).
   * Simple pipeline: read dataset → batch embed → insert into index → persist.

3. **Retrieval-Augmented Generation (RAG)**

   * Query-time: given a text prompt, query the index for top-k nearest image/text embeddings.

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
