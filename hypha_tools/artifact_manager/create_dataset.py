import asyncio
import requests
from hypha_rpc import connect_to_server
from dotenv import load_dotenv
import os
load_dotenv()
token = os.getenv('AGENT_LENS_WORKSPACE_TOKEN')
SERVER_URL = "https://hypha.aicell.io"

async def main():
    # Connect to the Artifact Manager API
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL, "token": token})
    artifact_manager = await api.get_service("public/artifact-manager")

    # Add a dataset to the gallery
    dataset_manifest = {
        "name": "Squid Tile Dataset",
        "description": "A dataset containing imaging map tiles of a microscopy sample",
    }
    dataset = await artifact_manager.create(
        parent_id= "agent-lens/microscopy-data",
        alias="microscopy-tiles-complete",
        manifest=dataset_manifest,
        version="stage",
        overwrite=True
    )
    print("Dataset added to the gallery.")
    #detele the json file
    if os.path.exists("upload_record.json"):
        os.remove("upload_record.json")
    

asyncio.run(main())