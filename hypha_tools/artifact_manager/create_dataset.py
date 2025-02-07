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
        "name": "Squid Dataset",
        "description": "A dataset containing imaging data",
    }
    dataset = await artifact_manager.create(
        parent_id= "agent-lens/microscopy-data",
        alias="squid-dataset",
        manifest=dataset_manifest,
        version="stage"
    )
    print("Dataset added to the gallery.")
    

asyncio.run(main())