import asyncio    
from hypha_rpc import connect_to_server    
import os    
import dotenv    
from PIL import Image    
import io    
"""
Description: Test the tile hypha service by fetching tiles from the hypha server.
"""
async def test_tile_service():    
    try:    
        # Load environment variables    
        dotenv.load_dotenv()    
        token = os.getenv("AGENT_LENS_WORKSPACE_TOKEN")    

        # Connect to server    
        server = await connect_to_server({    
            "server_url": "https://hypha.aicell.io",    
            "token": token,    
            "workspace": "agent-lens",    
        })    

        print("Connected to server successfully")    

        # Get the tile service    
        service = await server.get_service("microscope-tile-service")    
        print("Got tile service successfully")    

        # Test parameters    
        test_coords = [    
            (3, 0, 0),  # zoom level 0, top-left tile    
            (7, 5, 1),  # zoom level 7, top-left tile    
            (9, 1, 1),  # zoom level 9, bottom-right tile    
        ]
        channel_name = "BF_LED_matrix_full"

        for z, x, y in test_coords:    
            print(f"\nTesting tile at z={z}, x={x}, y={y}")    
            try:    
                # Get tile data    
                tile_bytes = await service.get_tile(channel_name, z, x, y)     

                if tile_bytes is None:    
                    print(f"No tile data received for z={z}, x={x}, y={y}")    
                    continue    
                filename = f"tile_z{z}_x{x}_y{y}.png"  
                with open(filename, "wb") as f:  
                    f.write(tile_bytes)  
                print(f"Tile saved to {filename}, size={len(tile_bytes)} bytes")  

                try:    
                    img = Image.open(io.BytesIO(tile_bytes))    
                    print(f"Valid image received: size={img.size}, mode={img.mode}")    
                except Exception as e:    
                    print(f"Failed to open image: {str(e)}")    

            except Exception as e:    
                print(f"Error getting tile: {str(e)}")    

    except Exception as e:  # Added this missing except block  
        print(f"Connection error: {str(e)}")    

if __name__ == "__main__":    
    asyncio.run(test_tile_service())    