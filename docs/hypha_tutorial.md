# Hypha Tutorial for Microscope Control

This tutorial provides detailed instructions on how to use the Hypha service for microscope control. The Hypha service allows users to interact with the microscope through a chatbot interface, enabling real-time control and automation of the microscope.

## Connect to the Microscope Service in Hypha

To connect to the microscope service in Hypha, follow this tutorial: [Connecting to a Service in Hypha](https://docs.amun.ai/#/?id=using-the-service).

Example code to connect to the microscope service:

```javascript
    const [microscopeControl, setMicroscopeControl] = useState(null);
    const server_url = "https://hypha.aicell.io";

    const server = await hyphaWebsocketClient.connectToServer({
        name: "js-client",
        server_url,
        method_timeout: 10,
        token,
    });

    const microscopeControlService = await getService(server, "Microscope Control", "agent-lens/microscope-control-squid-test");
    setMicroscopeControl(microscopeControlService);

```

## Control the Microscope via Hypha

### Basic Commands

In the microscope service, you can control the microscope using various commands. You can read the annotated code for the microscope service in the `start_hypha_service.py` file. And here are some examples of commands you can use to control the microscope:

```javascript
    // Move the microscope stage to a specific position
    await microscopeControlService.move_to_position(...);

    // Capture an image using the microscope camera
    await microscopeControlService.snap(...);

    // Do autofocus
    await microscopeControlService.auto_focus();

```
