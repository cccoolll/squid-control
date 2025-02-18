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

    # Upload a file to the dataset
    put_url = await artifact_manager.put_file("squid-dataset", file_path="data.bmp", download_weight=0.5)
    print("File upload URL:", put_url)
    file_path = os.path.abspath(os.path.join(os.getcwd(), "hypha_tools/artifact_manager/example-data/A3_12_0_BF_LED_matrix_full.bmp"))
    with open(file_path, "rb") as f:
        response = requests.put(put_url, data=f)
        assert response.ok, "File upload failed"
    print("File uploaded to the dataset.")

    # Commit the dataset
    await artifact_manager.commit("squid-dataset")
    print("Dataset committed.")

    # List all datasets in the gallery
    datasets = await artifact_manager.list("agent-lens/microscopy-data")
    print("Datasets in the gallery:", datasets)

asyncio.run(main())