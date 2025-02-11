# step1_upload_tiles_async.py
import os
import asyncio
import aiohttp
from hypha_rpc import connect_to_server
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()  # Loads environment variables, e.g. AGENT_LENS_WORKSPACE_TOKEN

SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_TOKEN = os.getenv("AGENT_LENS_WORKSPACE_TOKEN")
ZARR_PATH = os.getenv("ZARR_PATH")
ARTIFACT_ALIAS = "microscopy-tiles"
SCALE_RANGE = range(7, 11)  # Upload only scales 3 through 10
CONCURRENCY_LIMIT = 5       # Max number of concurrent uploads
UPLOAD_RECORD_FILE = "upload_record.json"  # File to track uploaded files

def load_upload_record():
    """Load the record of previously uploaded files"""
    if os.path.exists(UPLOAD_RECORD_FILE):
        with open(UPLOAD_RECORD_FILE, 'r') as f:
            return json.load(f)
    return {
        "uploaded_files": set(),
        "last_update": None,
        "total_files": 0,
        "completed_files": 0
    }

def save_upload_record(record):
    """Save the record of uploaded files"""
    # Convert set to list for JSON serialization
    record_copy = record.copy()
    record_copy["uploaded_files"] = list(record["uploaded_files"])
    record_copy["last_update"] = datetime.now().isoformat()

    with open(UPLOAD_RECORD_FILE, 'w') as f:
        json.dump(record_copy, f, indent=2)

async def upload_single_file(artifact_manager, artifact_alias, local_file, relative_path, semaphore, session, upload_record):
    """
    Requests a presigned URL from artifact_manager, then does an async PUT to upload the file.
    """
    # Skip if file was already uploaded
    if relative_path in upload_record["uploaded_files"]:
        print(f"Skipping already uploaded file: {relative_path}")
        return

    async with semaphore:
        try:
            # 1) Get the presigned URL
            put_url = await artifact_manager.put_file(artifact_alias, file_path=relative_path)

            # 2) Use aiohttp session to PUT the data
            async with session.put(put_url, data=open(local_file, "rb")) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"File upload failed for {local_file}, status={resp.status}")

            # 3) Record successful upload
            upload_record["uploaded_files"].add(relative_path)
            upload_record["completed_files"] += 1

            # 4) Save progress periodically (every 10 files)
            if upload_record["completed_files"] % 10 == 0:
                save_upload_record(upload_record)

            print(f"Uploaded file: {relative_path} ({upload_record['completed_files']}/{upload_record['total_files']})")

        except Exception as e:
            print(f"Error uploading {relative_path}: {str(e)}")
            raise

async def main():
    # Load upload record
    upload_record = load_upload_record()
    if isinstance(upload_record["uploaded_files"], list):
        upload_record["uploaded_files"] = set(upload_record["uploaded_files"])

    # 0) Connect to Artifact Manager
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL, "token": WORKSPACE_TOKEN})
    artifact_manager = await api.get_service("public/artifact-manager")

    # 1) Prepare a list of (local_file, relative_path) to upload
    to_upload = []
    for root_dir, dirs, files in os.walk(ZARR_PATH):
        current_path = os.path.relpath(root_dir, ZARR_PATH)
        path_parts = current_path.split(os.sep)

        # Decide if we skip based on scale number
        if len(path_parts) >= 2 and path_parts[1].startswith("scale"):
            try:
                scale_num = int(path_parts[1].replace("scale", ""))
                if scale_num not in SCALE_RANGE:
                    continue
            except ValueError:
                pass

        # Collect files
        for filename in files:
            local_file = os.path.join(root_dir, filename)
            relative_path = os.path.relpath(local_file, ZARR_PATH)
            to_upload.append((local_file, relative_path))

    # Update total files count
    upload_record["total_files"] = len(to_upload)
    save_upload_record(upload_record)

    # 2) Create tasks to upload each file in parallel, with concurrency limit
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []

    async with aiohttp.ClientSession() as session:
        for local_file, relative_path in to_upload:
            task = asyncio.create_task(
                upload_single_file(
                    artifact_manager,
                    ARTIFACT_ALIAS,
                    local_file,
                    relative_path,
                    semaphore,
                    session,
                    upload_record
                )
            )
            tasks.append(task)

        # 3) Run tasks concurrently
        await asyncio.gather(*tasks)

    # 4) Save final record and commit the dataset
    save_upload_record(upload_record)
    await artifact_manager.commit(ARTIFACT_ALIAS)
    print("Dataset committed successfully.")
    print(f"Total files uploaded: {upload_record['completed_files']}/{upload_record['total_files']}")

if __name__ == "__main__":
    asyncio.run(main())

# Created/Modified files during execution:
print(["step1_upload_tiles_async.py", "upload_record.json"])