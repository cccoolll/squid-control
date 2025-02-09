# step1_upload_tiles.py
import os
import asyncio
import requests
from hypha_rpc import connect_to_server
from dotenv import load_dotenv

load_dotenv()  # Loads your environment variables, e.g. AGENT_LENS_WORKSPACE_TOKEN

SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_TOKEN = os.getenv("AGENT_LENS_WORKSPACE_TOKEN")
ZARR_PATH = os.getenv("ZARR_PATH")
ARTIFACT_ALIAS = "microscopy-tiles"                # Name of your new dataset
SCALE_RANGE = range(3, 11)  # This will include scales 3 to 10

async def main():
    # 1) Connect to Artifact Manager
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL, "token": WORKSPACE_TOKEN})
    artifact_manager = await api.get_service("public/artifact-manager")


    # 2) Recursively walk your local Zarr directory and upload files
    for root_dir, dirs, files in os.walk(ZARR_PATH):
        # Check if this directory contains a scale we want to upload
        current_path = os.path.relpath(root_dir, ZARR_PATH)
        path_parts = current_path.split(os.sep)
        
        # Skip if this is not a scale directory or not in our desired range
        if len(path_parts) >= 2 and path_parts[1].startswith('scale'):
            try:
                scale_num = int(path_parts[1].replace('scale', ''))
                if scale_num not in SCALE_RANGE:
                    continue
            except ValueError:
                pass  # Not a scale directory, continue normal processing
        
        for filename in files:
            local_file = os.path.join(root_dir, filename)
            relative_path = os.path.relpath(local_file, ZARR_PATH)
            
            # Request a pre-signed URL for uploading this file
            put_url = await artifact_manager.put_file(ARTIFACT_ALIAS, file_path=relative_path)
            with open(local_file, "rb") as f:
                response = requests.put(put_url, data=f)
                assert response.ok, f"File upload failed for {local_file}"
                print(f"Uploaded file: {relative_path}")

    # 4) Commit the dataset so it becomes accessible
    await artifact_manager.commit(ARTIFACT_ALIAS)
    print("Dataset committed successfully.")

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
