import asyncio  
import fractions  
import logging  
import numpy as np  
from av import VideoFrame  
from aiortc import MediaStreamTrack  
from hypha_rpc import connect_to_server, login, register_rtc_service  
from streaming_imaging_map import get_tile_from_zarr, ZARR_FOLDER, CHANNEL_NAME  
import os  
from io import BytesIO  
import dotenv  
dotenv.load_dotenv()  
ENV_FILE = dotenv.find_dotenv()  
if ENV_FILE:  
    dotenv.load_dotenv(ENV_FILE)  
  
logger = logging.getLogger("tile_streaming")

class TileStreamTrack(MediaStreamTrack):  
    """  
    A video stream track that streams tiles based on received coordinates  
    """  
    kind = "video"  
    def __init__(self):  
        super().__init__()  
        self.count = 0  
        self.current_channel = CHANNEL_NAME  
        self.current_coords = {"z": 0, "x": 0, "y": 0}  
        self._running = True  

    def update_coords(self, z, x, y):  
        """Update the current coordinates for tile streaming"""  
        self.current_coords = {"z": z, "x": x, "y": y}  

    def update_channel(self, channel_name):  
        """Update the current channel"""  
        self.current_channel = channel_name  

    async def recv(self):  
        if not self._running:  
            return None  

        try:  
            # Get tile based on current coordinates  
            zarr_path = os.path.join(ZARR_FOLDER, f"{self.current_channel}.zarr")  
            tile_buffer = get_tile_from_zarr(  
                zarr_path,   
                self.current_coords["z"],  
                self.current_coords["x"],  
                self.current_coords["y"]  
            )  

            # Convert tile to numpy array for video frame  
            tile_array = np.array(Image.open(tile_buffer))  

            # Ensure the array is in the correct format (BGR)  
            if len(tile_array.shape) == 2:  # If grayscale  
                tile_array = np.stack([tile_array] * 3, axis=-1)  

            # Create video frame  
            new_frame = VideoFrame.from_ndarray(tile_array, format="bgr24")  
            new_frame.pts = self.count  
            self.count += 1  
            new_frame.time_base = fractions.Fraction(1, 1000)  

            return new_frame  

        except Exception as e:  
            logger.error(f"Error in tile streaming: {str(e)}")  
            # Return a blank frame in case of error  
            blank_frame = np.zeros((256, 256, 3), dtype=np.uint8)  
            new_frame = VideoFrame.from_ndarray(blank_frame, format="bgr24")  
            new_frame.pts = self.count  
            self.count += 1  
            new_frame.time_base = fractions.Fraction(1, 1000)  
            return new_frame  

async def start_tile_service(server_url):  
    try:  
        token = os.environ.get("WORKSPACE_TOKEN")  
    except:  
        token = await login({"server_url": server_url})  

    # Connect to the Hypha server  
    server = await connect_to_server({  
        "server_url": server_url,  
        "token": token,  
        "workspace": "agent-lens",  
    })  

    # Create a shared tile stream track instance  
    tile_track = TileStreamTrack()  

    async def on_init(peer_connection):  
        """  
        This function is called when a new WebRTC peer connection is initialized.  
        """  
        @peer_connection.on("datachannel")  
        def on_datachannel(channel):  
            @channel.on("message")  
            def on_message(message):  
                try:  
                    data = json.loads(message)  
                    if "coords" in data:  
                        tile_track.update_coords(  
                            data["coords"]["z"],  
                            data["coords"]["x"],  
                            data["coords"]["y"]  
                        )  
                    if "channel" in data:  
                        tile_track.update_channel(data["channel"])  
                except Exception as e:  
                    logger.error(f"Error processing message: {str(e)}")  

        @peer_connection.on("track")  
        def on_track(track):  
            logger.info(f"Track {track.kind} received")  
            peer_connection.addTrack(tile_track)  

            @track.on("ended")  
            def on_ended():  
                logger.info(f"Track {track.kind} ended")  
                tile_track._running = False  

    # Register the WebRTC service  
    await register_rtc_service(  
        server,  
        service_id="microscopy-tile-stream",  
        config={  
            "visibility": "public",  
            "on_init": on_init,  # Pass the corrected on_init function  
        }  
    )  

    print(f"WebRTC service registered at workspace: {server.config.workspace}")  

    # Keep the service running  
    await server.serve()  

if __name__ == "__main__":  
    server_url = "https://hypha.aicell.io"  
    asyncio.run(start_tile_service(server_url))  