import asyncio
from imjoy_rpc import api
def get_schema():
    return {
        "move_by_distance": {
            "type": "bioimageio-chatbot-extension",
            "title": "move_by_distance",
            "description": "Move the stage by a specified distance in millimeters.",
            "properties": {
                "x": {"type": "number", "description": "Move the stage along X axis."},
                "y": {"type": "number", "description": "Move the stage along Y axis."},
                "z": {"type": "number", "description": "Move the stage along Z axis."},
            },
        },
        "snap_image": {
            "type": "bioimageio-chatbot-extension",
            "title": "snap_image",
            "description": "Snap an image from the microscope with specified exposure time.",
            "properties": {
                "exposure": {"type": "number", "description": "Set the microscope camera's exposure time in milliseconds."},
            },
        }
    }

def move_by_distance(config):
    
    squid_svc.move_by_distance(config["x"], config["y"], config["z"])
    return {"result": "Moved the stage!"}

def snap_image(config):
    
    squid_image = squid_svc.snap()
    viewer = api.createWindow(src="https://kitware.github.io/itk-vtk-viewer/app/")
    viewer.setImage(squid_image)
    return {"result": "Image snapped"}

async def setup():
    global squid_svc
    from imjoy_rpc.hypha import connect_to_server

    squid_server = await connect_to_server({"server_url": "https://ai.imjoy.io/"})
    squid_svc = await squid_server.get_service("microscope-control-squid")

    chatbot_extension = {
        "_rintf": True,
        "id": "squid-control",
        "type": "bioimageio-chatbot-extension",
        "name": "Squid Microscope Control",
        "description": "Control the microscope based on the user's request. Now you can move the microscope stage, and snap an image.",
        "get_schema": get_schema,
        "tools": {
            "move_by_distance": move_by_distance,
            "snap_image": snap_image,
        }
    }

    chatbot = await api.createWindow(
        src="https://bioimage.io/chat",
        name="Microscope-Control Chatbot",
    )
    #chatbot_extension._rintf = True  # make the chatbot extension as an interface
    await chatbot.registerExtension(chatbot_extension)
    print('Chatbot extension registered.')

api.export({"setup": setup})