from bioimageio_chatbot.utils import ChatbotExtension
# from bioimageio_chatbot.hypha_store import HyphaDataStore
from openai import AsyncOpenAI
from schema_agents import schema_tool
import base64
from pydantic import Field, BaseModel
from typing import Optional, List
import httpx
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import random
# make sure matplotlib is operating headless (no GUI)
plt.switch_backend("agg")

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

async def aask(images, messages, max_tokens=1024):
    aclient = AsyncOpenAI()
    user_message = []
    # download the images and save it into a list of PIL image objects
    img_objs = []
    for image in images:
        async with httpx.AsyncClient() as client:
            response = await client.get(image.url)
            response.raise_for_status()
        try:
            img = Image.open(BytesIO(response.content))
        except Exception as e:
            raise ValueError(f"Failed to read image {image.title or ''} from {image.url}. Error: {e}")
        img_objs.append(img)
    
    if len(img_objs) == 1:
        # plot the image with matplotlib
        plt.imshow(img_objs[0])
        if images[0].title:
            plt.title(images[0].title)
        fig = plt.gcf()
    else:
        # plot them in subplots with matplotlib in a row
        fig, ax = plt.subplots(1, len(img_objs), figsize=(15, 5))
        for i, img in enumerate(img_objs):
            ax[i].imshow(img)
            if images[0].title:
                ax[i].set_title(images[i].title)
    # save the plot to a buffer as png format and convert to base64
    buffer = BytesIO()
    fig.tight_layout()
    # if the image size (width or height) is smaller than 512, use the original size and aspect ratio
    # otherwise set the maximun width of the image to n*512 pixels, where n is the number of images; the maximum total width is 1024 pixels
    fig_width = min(1024, len(img_objs)*512, fig.get_figwidth()*fig.dpi)
    # make sure the pixel size (not inches)
    fig.set_size_inches(fig_width/fig.dpi, fig.get_figheight(), forward=True)
    
    # save fig
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.read()).decode("utf-8")
    # append the image to the user message
    user_message.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_image}"
        }
    })
    
    for message in messages:
        assert isinstance(message, str), "Message must be a string."
        user_message.append({"type": "text", "text": message})

    response = await aclient.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant that help user to inspect the provided images visually based on the context, make insightful comments and answer questions about the provided images."
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

class ImageInfo(BaseModel):
    """Image information."""
    url: str=Field(..., description="The URL of the image.")
    title: Optional[str]=Field(None, description="The title of the image.")

@schema_tool
async def inspect_tool(images: List[ImageInfo]=Field(..., description="A list of images to be inspected, each with a http url and title"), 
                       query: str=Field(..., description="user query about the image"),  
                       context_description: str=Field(..., description="describe the context for the visual inspection task")) -> str:
    """Inspect an image using GPT4-Vision, return 'NO SAMPLE' if no sample is found in the image."""
    # assert image_url.startswith("http"), "Image URL must start with http."
    for image in images:
        assert image.url.startswith("http"), "Image URL must start with http."
    
    response = await aask(images, [context_description, query])
    return response

    
@schema_tool
async def move_stage(
    x: float=Field(..., description="x offset"),
    y: float=Field(..., description="y offset")
    ) -> str:
    """Move the microscope stage."""
    # config = MoveStageInput(**kwargs)
    print(x, y)
    return "success"

@schema_tool
async def snap_image(
    exposure: float=Field(..., description="exposure time")
    ) -> ImageInfo:
    """Snap an image with the microscope camera."""
    # config = SnapImageInput(**kwargs)
    print(exposure)
    # TODO: implement the image snapping logic
    # randomly pick a number from {0,1}
    number = random.choice([0, 1,2])
    if number == 1:
        # with sample
        url = "https://chat.bioimage.io/google-oauth2%7C103047988474094226050/apps/data-store/get?id=4e3505c5-a91e-4515-83a8-9461ee70c071"
    else:
        url = "https://chat.bioimage.io/google-oauth2%7C103047988474094226050/apps/data-store/get?id=d38e0186-9d14-42a9-b208-46fee7b900b3"
    title = "snaped image"
    return ImageInfo(url=url, title=title)
    
def get_extension():
    return ChatbotExtension(
        id="microscope_control",
        name="Microscope Control",
        description="Control the microscope stage and snap images, perform visual inspection on images using GPT4-Vision model, used for describing images and answer image related questions. The images will be plotted using matplotlib and then sent to the GPT4-Vision model for inspection.",
        tools=dict(
            inspect=inspect_tool,
            move_stage=move_stage,
            snap_image=snap_image,
        )
    )

if __name__ == "__main__":
    import asyncio
    async def main():
        extension = get_extension()
        print(await extension.tools["inspect"](images=[ImageInfo(url="https://chat.bioimage.io/google-oauth2%7C103047988474094226050/apps/data-store/get?id=4e3505c5-a91e-4515-83a8-9461ee70c071", title="snaped image"), ImageInfo(url="https://bioimage.io/static/img/bioimage-io-logo.png", title="BioImage.io Logo")], query="What are these?", context_description="Inspect the image and tell me if you see a sample."))
        # test only one image
        # print(await extension.tools["inspect"](images=[ImageInfo(url="https://bioimage.io/static/img/bioimage-io-icon.png", title="BioImage.io Icon")], query="What is this?", context_description="Inspect the BioImage.io icon."))
    # Run the async function
    asyncio.run(main())