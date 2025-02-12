import asyncio
from hypha_rpc import connect_to_server
import os
from dotenv import load_dotenv

load_dotenv()

SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_TOKEN = os.getenv("AGENT_LENS_WORKSPACE_TOKEN")
ARTIFACT_ALIAS = "microscopy-tiles-complete"

async def list_files_with_limit(artifact_manager, channel, scale, limit):
    dir_path = f"{channel}/scale{scale}"
    files = await artifact_manager.list_files(ARTIFACT_ALIAS, dir_path=dir_path, limit=limit)
    return files

async def list_files_without_limit(artifact_manager, channel, scale):
    dir_path = f"{channel}/scale{scale}"
    files = await artifact_manager.list_files(ARTIFACT_ALIAS, dir_path=dir_path)
    return files

async def main():
    api = await connect_to_server({
        "name": "test-client",
        "server_url": SERVER_URL,
        "token": WORKSPACE_TOKEN
    })
    artifact_manager = await api.get_service("public/artifact-manager")

    channel = "BF_LED_matrix_full"
    scale = 0

    print("Listing files with limit=3000...")
    files_with_limit = await list_files_with_limit(artifact_manager, channel, scale, 3000)
    print(f"Number of tiles with limit=3000: {len(files_with_limit)}")

    print("Listing files without limit...")
    files_without_limit = await list_files_without_limit(artifact_manager, channel, scale)
    print(f"Number of tiles without limit: {len(files_without_limit)}")

    file_path = f"{channel}/scale{scale}/55.81"

    print("Checking for specific tile with limit...")
    if any(f['name'] == "55.81" for f in files_with_limit):
        print(f"Tile found with limit: {file_path}")
    else:
        print(f"Tile not found with limit: {file_path}")

    print("Checking for specific tile without limit...")
    if any(f['name'] == "55.81" for f in files_without_limit):
        print(f"Tile found without limit: {file_path}")
    else:
        print(f"Tile not found without limit: {file_path}")

if __name__ == "__main__":
    asyncio.run(main())