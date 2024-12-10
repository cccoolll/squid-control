import asyncio  
import os  
import dotenv  
from hypha_rpc import connect_to_server  

# Import your existing streaming module  
import streaming_imaging_map  

# Load environment variables  
dotenv.load_dotenv()  
token = os.getenv("SQUID_WORKSPACE_TOKEN")  

async def start_server(server_url):  
    """  
    Start the Hypha server and register the tile streaming service.  
    """  
    # Connect to the Hypha server  
    server = await connect_to_server({  
        "server_url": server_url,  
        "token": token,  
        "workspace": "squid-control",  # Specify the workspace if needed  
    })  

    # Define the RPC function for serving tiles  
    async def get_tile(z: int, x: int, y: int) -> bytes:  
        """  
        Serve a tile for the fixed channel and z, x, y parameters.  
        """  
        try:  
            # Use the existing function from streaming_imaging_map  
            tile = streaming_imaging_map.get_tile(z, x, y)  
            return tile.read()  # Return the binary content of the tile  
        except Exception as e:  
            return {"error": str(e)}  

    # Define the RPC function for serving the main webpage  
    async def index() -> str:  
        """  
        Serve the main webpage with OpenLayers.  
        """  
        try:  
            # Use the existing function from streaming_imaging_map  
            return streaming_imaging_map.index()  
        except Exception as e:  
            return {"error": str(e)}  

    # Register the service with Hypha  
    service_info = await server.register_service({  
        "name": "Tile Streaming Service",  
        "id": "tile-streaming",  
        "config": {  
            "visibility": "public",  # Make the service publicly accessible  
            "require_context": False,  # No specific context required  
            "run_in_executor": True,  # Run in an executor for compatibility  
        },  
        "get_tile": get_tile,  # Expose the get_tile function  
        "index": index,        # Expose the index function  
    })  

    print(f"Service registered successfully!")  
    print(f"Service ID: {service_info.id}")  
    print(f"Workspace: {server.config.workspace}")  
    print(f"Test the service at: {server_url}/{server.config.workspace}/services/{service_info.id}/get_tile?z=0&x=0&y=0")  

    # Keep the server running  
    await server.serve()  

if __name__ == "__main__":  
    # Replace with your Hypha server URL  
    server_url = "https://hypha.aicell.io"  
    asyncio.run(start_server(server_url))  