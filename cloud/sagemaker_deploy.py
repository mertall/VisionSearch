import os
import tarfile
import boto3
import botocore
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from transformers import CLIPModel, CLIPProcessor
import shutil

# ────────────────────────────────────────────────────────────────
# 1. CONFIG
# ────────────────────────────────────────────────────────────────

sess      = sagemaker.Session()
region    = sess.boto_region_name
role      = "arn:aws:iam::018194649765:role/service-role/AmazonSageMaker-ExecutionRole-20250408T155211"
bucket    = "my-clip-32-bucket"
prefix    = "clip32-sagemaker"

file_name = "model" # < Do not change - necessary for sagemaker inference endpoint to pick up our model and use it
workdir   = "./model" # < Do not change - -
model_dir = workdir
code_dir  = os.path.join(workdir, "code")

# Local inference script and requirements
inference_src     = "/inference_deployment/inference.py"
requirements_src  = "/inference_deployment/requirements.txt"

os.makedirs(code_dir, exist_ok=True)

# ────────────────────────────────────────────────────────────────
# 2. DOWNLOAD PRETRAINED CLIP + PROCESSOR
# ────────────────────────────────────────────────────────────────

clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clip_model.save_pretrained(model_dir)
clip_processor.save_pretrained(model_dir)

# Copy inference.py and requirements.txt into code/
shutil.copy(inference_src, os.path.join(code_dir, "inference.py"))
shutil.copy(requirements_src, os.path.join(code_dir, "requirements.txt"))

# ────────────────────────────────────────────────────────────────
# 3. TAR + UPLOAD TO S3 (FLATTENED STRUCTURE)
# ────────────────────────────────────────────────────────────────

# Optional: Print directory structure
print("\n📁 Directory structure inside TAR archive:")
for root, dirs, files in os.walk(model_dir):
    for f in files:
        relpath = os.path.relpath(os.path.join(root, f), start=model_dir)
        print("├──", relpath)

# Create TAR
model_tar = f"{file_name}.tar.gz"
with tarfile.open(model_tar, "w:gz") as tar:
    for root, _, files in os.walk(model_dir):
        for fname in files:
            fullpath = os.path.join(root, fname)
            arcname = os.path.relpath(fullpath, model_dir)
            tar.add(fullpath, arcname=arcname)

print(f"\n✅ Created archive: {model_tar}")

# Upload to S3
s3 = boto3.client("s3", region_name=region)
try:
    s3.head_bucket(Bucket=bucket)
except botocore.exceptions.ClientError as e:
    if int(e.response["Error"]["Code"]) == 404:
        s3.create_bucket(Bucket=bucket)
    else:
        raise

s3_key = f"{prefix}/model.tar.gz"
s3.upload_file(model_tar, bucket, s3_key)
model_s3_uri = f"s3://{bucket}/{s3_key}"

print("📦 Uploaded to S3:", model_s3_uri)

# ────────────────────────────────────────────────────────────────
# 4. DEPLOY TO SAGEMAKER
# ────────────────────────────────────────────────────────────────

hf_model = HuggingFaceModel(
    model_data=model_s3_uri,
    role=role,
    transformers_version="4.49",
    pytorch_version="2.6",
    py_version="py312",
    source_dir=code_dir,      # Required: inference script lives here
    entry_point="inference.py",
    sagemaker_session=sess
)

predictor = hf_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    endpoint_name=os.getenv("CLIP_ENDPOINT_NAME")
)

print("🚀 Deployed endpoint:", predictor.endpoint_name)
