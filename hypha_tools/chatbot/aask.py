
# Initialize chatpt vision
from openai import AsyncOpenAI
import base64
import httpx
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt




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
