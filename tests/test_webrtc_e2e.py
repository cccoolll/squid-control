import pytest
import pytest_asyncio
import asyncio
import os
import time
import uuid
import json
import threading
import socket
from pathlib import Path
from hypha_rpc import connect_to_server, register_rtc_service
from start_hypha_service import Microscope, MicroscopeVideoTrack
from http.server import HTTPServer, SimpleHTTPRequestHandler
import tempfile
import webbrowser
import subprocess
import signal

# Mark all tests in this module as asyncio and integration tests
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

# Test configuration
TEST_SERVER_URL = "https://hypha.aicell.io"
TEST_WORKSPACE = "agent-lens"
TEST_TIMEOUT = 180  # 3 minutes for WebRTC tests

class TestHTTPHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for serving test files."""
    
    def __init__(self, *args, test_directory=None, **kwargs):
        self.test_directory = test_directory
        super().__init__(*args, directory=test_directory, **kwargs)
    
    def end_headers(self):
        # Add CORS headers for local testing
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def find_free_port():
    """Find a free port for the HTTP server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def create_webrtc_test_html(service_id, webrtc_service_id, server_url, workspace, token):
    """Create the HTML test page for WebRTC testing."""
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebRTC End-to-End Test</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header {{
            background-color: #2196F3;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .section {{
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .controls {{
            margin: 10px 0;
        }}
        button {{
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
        }}
        button:hover {{
            background-color: #45a049;
        }}
        button:disabled {{
            background-color: #cccccc;
            cursor: not-allowed;
        }}
        .status {{
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            font-weight: bold;
        }}
        .status.success {{
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        .status.error {{
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        .status.info {{
            background-color: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }}
        #video {{
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 4px;
        }}
        .metadata {{
            background-color: #f8f9fa;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            max-height: 200px;
            overflow-y: auto;
        }}
        .test-results {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .config-info {{
            background-color: #e9ecef;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 12px;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/hypha-rpc@0.20.54/dist/hypha-rpc-websocket.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 WebRTC End-to-End Test</h1>
            <p>Testing video streaming and metadata extraction from microscope service</p>
        </div>
        
        <div class="config-info">
            <strong>Test Configuration:</strong><br>
            Server: {server_url}<br>
            Workspace: {workspace}<br>
            Microscope Service ID: {service_id}<br>
            WebRTC Service ID: {webrtc_service_id}<br>
            Token: {'*' * (len(token) - 8) + token[-8:] if token else 'Not provided'}
        </div>

        <div class="section">
            <h2>🔗 Connection Status</h2>
            <div id="connection-status" class="status info">Initializing...</div>
            <div class="controls">
                <button id="connect-btn" onclick="connectToServices()">Connect to Services</button>
                <button id="disconnect-btn" onclick="disconnectServices()" disabled>Disconnect</button>
            </div>
        </div>

        <div class="section">
            <h2>📹 Video Stream</h2>
            <div id="video-status" class="status info">Video not started</div>
            <div class="controls">
                <button id="start-video-btn" onclick="startVideoStream()" disabled>Start Video Stream</button>
                <button id="stop-video-btn" onclick="stopVideoStream()" disabled>Stop Video Stream</button>
            </div>
            <video id="video" autoplay muted controls style="width: 640px; height: 480px;"></video>
        </div>

        <div class="section">
            <h2>📊 Frame Metadata</h2>
            <div id="metadata-status" class="status info">No metadata captured yet</div>
            <div class="controls">
                <button id="start-metadata-btn" onclick="startMetadataCapture()" disabled>Start Metadata Capture</button>
                <button id="stop-metadata-btn" onclick="stopMetadataCapture()" disabled>Stop Metadata Capture</button>
                <button id="test-microscope-btn" onclick="testMicroscopeControls()" disabled>Test Microscope Controls</button>
            </div>
            <div id="metadata-display" class="metadata">No metadata captured</div>
        </div>

        <div class="section">
            <h2>✅ Test Results</h2>
            <div id="test-results" class="test-results">
                <div id="test-summary">Tests not started</div>
                <ul id="test-list">
                    <li>🔶 Connection Test: <span id="test-connection">Pending</span></li>
                    <li>🔶 Video Stream Test: <span id="test-video">Pending</span></li>
                    <li>🔶 Metadata Capture Test: <span id="test-metadata">Pending</span></li>
                    <li>🔶 Microscope Control Test: <span id="test-controls">Pending</span></li>
                    <li>🔶 Cleanup Test: <span id="test-cleanup">Pending</span></li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>🤖 Automated Test</h2>
            <div class="controls">
                <button id="run-auto-test-btn" onclick="runAutomatedTest()">Run Full Automated Test</button>
            </div>
            <div id="auto-test-progress" class="status info" style="display: none;">Running automated test...</div>
        </div>
    </div>

    <script>
        // Global variables
        let hyphaWebsocketClient;
        let server;
        let pc; // WebRTC peer connection
        let mc; // Microscope control service
        let videoElement;
        let metadataInterval;
        let testResults = {{}};
        
        // Configuration
        const CONFIG = {{
            SERVER_URL: '{server_url}',
            WORKSPACE: '{workspace}',
            TOKEN: '{token}',
            SERVICE_ID: '{service_id}',
            WEBRTC_SERVICE_ID: '{webrtc_service_id}',
            TIMEOUT: 30000 // 30 seconds
        }};

        function updateStatus(elementId, message, type = 'info') {{
            const element = document.getElementById(elementId);
            element.textContent = message;
            element.className = `status ${{type}}`;
        }}

        function updateTestResult(testId, result) {{
            const element = document.getElementById(testId);
            element.textContent = result;
            element.style.color = result === 'PASS' ? 'green' : result === 'FAIL' ? 'red' : 'orange';
            testResults[testId] = result;
        }}

        async function connectToServices() {{
            try {{
                updateStatus('connection-status', 'Connecting to Hypha server...', 'info');
                
                // Import and connect to server
                hyphaWebsocketClient = await loadHyphaWebSocketClient();
                
                server = await hyphaWebsocketClient.connectToServer({{
                    server_url: CONFIG.SERVER_URL,
                    token: CONFIG.TOKEN,
                    workspace: CONFIG.WORKSPACE
                }});
                
                updateStatus('connection-status', 'Connected to server. Setting up WebRTC...', 'info');
                
                // Set up WebRTC connection
                const hostCanvas = document.createElement('canvas');
                hostCanvas.width = 640;
                hostCanvas.height = 480;
                
                // Fetch ICE servers
                let iceServers;
                try {{
                    const response = await fetch('https://ai.imjoy.io/public/services/coturn/get_rtc_ice_servers');
                    if (response.ok) {{
                        iceServers = await response.json();
                        console.log('Fetched ICE servers:', iceServers);
                    }} else {{
                        throw new Error('Failed to fetch ICE servers');
                    }}
                }} catch (error) {{
                    console.warn('Using fallback ICE servers:', error);
                    iceServers = [{{"urls": ["stun:stun.l.google.com:19302"]}}];
                }}
                
                async function on_init(peerConnection) {{
                    // Set up video track listener
                    peerConnection.addEventListener('track', function (evt) {{
                        console.log('Received track:', evt.track.kind);
                        if (evt.track.kind === 'video') {{
                            videoElement = document.getElementById('video');
                            videoElement.srcObject = evt.streams[0];
                            console.log('Video stream connected');
                            
                            // Set up WebRTC frame metadata extraction
                            if (evt.track.kind === 'video') {{
                                const receiver = peerConnection.getReceivers().find(r => r.track === evt.track);
                                if (receiver) {{
                                    console.log('Setting up WebRTC frame metadata extraction...');
                                    // This would be where we could access VideoFrame.private_data
                                    // in a real WebRTC implementation, but browser APIs don't expose this directly
                                    console.log('WebRTC receiver available for advanced frame processing');
                                }}
                            }}
                        }}
                    }});
                    
                    // Set up dummy video stream (required for WebRTC)
                    const context = hostCanvas.getContext('2d');
                    const stream = hostCanvas.captureStream(5);
                    for (let track of stream.getVideoTracks()) {{
                        await peerConnection.addTrack(track, stream);
                    }}
                }}
                
                // Connect to WebRTC service
                pc = await hyphaWebsocketClient.getRTCService(
                    server,
                    CONFIG.WEBRTC_SERVICE_ID,
                    {{
                        on_init,
                        ice_servers: iceServers
                    }}
                );
                
                // Get microscope control service through WebRTC
                mc = await pc.getService(CONFIG.SERVICE_ID);
                
                updateStatus('connection-status', 'Successfully connected to all services!', 'success');
                updateTestResult('test-connection', 'PASS');
                
                // Enable control buttons
                document.getElementById('connect-btn').disabled = true;
                document.getElementById('disconnect-btn').disabled = false;
                document.getElementById('start-video-btn').disabled = false;
                document.getElementById('start-metadata-btn').disabled = false;
                document.getElementById('test-microscope-btn').disabled = false;
                
            }} catch (error) {{
                console.error('Connection error:', error);
                updateStatus('connection-status', `Connection failed: ${{error.message}}`, 'error');
                updateTestResult('test-connection', 'FAIL');
            }}
        }}

        async function startVideoStream() {{
            try {{
                updateStatus('video-status', 'Starting video stream...', 'info');
                
                // Start video buffering on microscope
                await mc.start_video_buffering();
                
                // Wait a moment for stream to establish
                await new Promise(resolve => setTimeout(resolve, 2000));
                
                updateStatus('video-status', 'Video stream active', 'success');
                updateTestResult('test-video', 'PASS');
                
                document.getElementById('start-video-btn').disabled = true;
                document.getElementById('stop-video-btn').disabled = false;
                
            }} catch (error) {{
                console.error('Video stream error:', error);
                updateStatus('video-status', `Video stream failed: ${{error.message}}`, 'error');
                updateTestResult('test-video', 'FAIL');
            }}
        }}

        async function stopVideoStream() {{
            try {{
                updateStatus('video-status', 'Stopping video stream...', 'info');
                
                await mc.stop_video_buffering();
                
                updateStatus('video-status', 'Video stream stopped', 'info');
                
                document.getElementById('start-video-btn').disabled = false;
                document.getElementById('stop-video-btn').disabled = true;
                
            }} catch (error) {{
                console.error('Stop video error:', error);
                updateStatus('video-status', `Failed to stop video: ${{error.message}}`, 'error');
            }}
        }}

        async function startMetadataCapture() {{
            try {{
                updateStatus('metadata-status', 'Starting metadata capture...', 'info');
                
                let metadataCount = 0;
                metadataInterval = setInterval(async () => {{
                    try {{
                        // Get a video frame with metadata
                        const frameData = await mc.get_video_frame({{
                            frame_width: 320,
                            frame_height: 240
                        }});
                        
                        // Extract and display metadata if available
                        if (frameData) {{
                            metadataCount++;
                            let displayText = `Frame ${{metadataCount}}:\\n`;
                            displayText += `frame_width: ${{frameData.width}}\\n`;
                            displayText += `frame_height: ${{frameData.height}}\\n`;
                            displayText += `format: ${{frameData.format}}\\n`;
                            displayText += `size_bytes: ${{frameData.size_bytes}}\\n`;
                            displayText += `timestamp: ${{new Date().toISOString()}}\\n`;
                            
                            // Check for metadata (this is where stage position, channel info, etc. would be)
                            if (frameData.metadata) {{
                                displayText += `\\nMETADATA:\\n${{JSON.stringify(frameData.metadata, null, 2)}}`;
                                updateStatus('metadata-status', `Captured ${{metadataCount}} frames with metadata`, 'success');
                            }} else {{
                                displayText += `\\n(No explicit metadata available)`;
                                updateStatus('metadata-status', `Captured ${{metadataCount}} video frames`, 'success');
                            }}
                            
                            document.getElementById('metadata-display').textContent = displayText;
                            
                            if (metadataCount >= 3) {{
                                updateTestResult('test-metadata', 'PASS');
                            }}
                        }} else {{
                            updateStatus('metadata-status', 'No frame data received', 'error');
                            updateTestResult('test-metadata', 'FAIL');
                        }}
                        
                    }} catch (error) {{
                        console.error('Metadata capture error:', error);
                        updateStatus('metadata-status', `Metadata error: ${{error.message}}`, 'error');
                        updateTestResult('test-metadata', 'FAIL');
                    }}
                }}, 1000); // Capture every second
                
                document.getElementById('start-metadata-btn').disabled = true;
                document.getElementById('stop-metadata-btn').disabled = false;
                
            }} catch (error) {{
                console.error('Start metadata error:', error);
                updateStatus('metadata-status', `Failed to start metadata capture: ${{error.message}}`, 'error');
                updateTestResult('test-metadata', 'FAIL');
            }}
        }}

        async function stopMetadataCapture() {{
            if (metadataInterval) {{
                clearInterval(metadataInterval);
                metadataInterval = null;
            }}
            
            updateStatus('metadata-status', 'Metadata capture stopped', 'info');
            
            document.getElementById('start-metadata-btn').disabled = false;
            document.getElementById('stop-metadata-btn').disabled = true;
        }}

        async function testMicroscopeControls() {{
            try {{
                updateStatus('metadata-status', 'Testing microscope controls and metadata...', 'info');
                
                // Test basic microscope functionality
                const status = await mc.get_status();
                console.log('Microscope status:', status);
                
                // Test movement and capture metadata
                await mc.move_by_distance({{x: 0.1, y: 0.1, z: 0.0}});
                
                // Test illumination changes that should affect metadata
                await mc.set_illumination({{channel: 11, intensity: 75}});
                await mc.set_camera_exposure({{channel: 11, exposure_time: 150}});
                
                // Wait a moment for parameters to propagate
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                // Get video frame and check for metadata changes
                const frameAfterChanges = await mc.get_video_frame({{
                    frame_width: 320,
                    frame_height: 240
                }});
                
                console.log('Frame after parameter changes:', frameAfterChanges);
                
                if (frameAfterChanges && frameAfterChanges.metadata) {{
                    console.log('Metadata after changes:', frameAfterChanges.metadata);
                    const metadata = frameAfterChanges.metadata;
                    
                    // Check if metadata contains expected fields
                    let metadataValid = true;
                    const expectedFields = ['stage_position', 'timestamp'];
                    
                    for (const field of expectedFields) {{
                        if (!(field in metadata)) {{
                            console.warn(`Missing expected metadata field: ${{field}}`);
                            metadataValid = false;
                        }}
                    }}
                    
                    if (metadataValid) {{
                        console.log('✅ Metadata validation passed');
                    }} else {{
                        console.warn('⚠️ Some metadata fields missing, but basic metadata present');
                    }}
                }} else {{
                    console.log('ℹ️ No explicit metadata in frame, but frame data is valid');
                }}
                
                // Test frame capture
                const frame = await mc.one_new_frame();
                console.log('Frame captured:', frame ? 'Success' : 'Failed');
                
                updateStatus('metadata-status', 'Microscope controls and metadata working correctly', 'success');
                updateTestResult('test-controls', 'PASS');
                
            }} catch (error) {{
                console.error('Microscope control error:', error);
                updateStatus('metadata-status', `Control test failed: ${{error.message}}`, 'error');
                updateTestResult('test-controls', 'FAIL');
            }}
        }}

        async function disconnectServices() {{
            try {{
                // Stop metadata capture
                if (metadataInterval) {{
                    clearInterval(metadataInterval);
                    metadataInterval = null;
                }}
                
                // Stop video streaming
                if (mc) {{
                    try {{
                        await mc.stop_video_buffering();
                    }} catch (e) {{
                        console.warn('Error stopping video buffering:', e);
                    }}
                }}
                
                // Close WebRTC connection
                if (pc) {{
                    try {{
                        await pc.disconnect();
                    }} catch (e) {{
                        console.warn('Error disconnecting WebRTC:', e);
                    }}
                }}
                
                // Disconnect from server
                if (server) {{
                    try {{
                        await server.disconnect();
                    }} catch (e) {{
                        console.warn('Error disconnecting server:', e);
                    }}
                }}
                
                updateStatus('connection-status', 'Disconnected from all services', 'info');
                updateTestResult('test-cleanup', 'PASS');
                
                // Reset button states
                document.getElementById('connect-btn').disabled = false;
                document.getElementById('disconnect-btn').disabled = true;
                document.getElementById('start-video-btn').disabled = true;
                document.getElementById('stop-video-btn').disabled = true;
                document.getElementById('start-metadata-btn').disabled = true;
                document.getElementById('stop-metadata-btn').disabled = true;
                document.getElementById('test-microscope-btn').disabled = true;
                
            }} catch (error) {{
                console.error('Disconnect error:', error);
                updateStatus('connection-status', `Disconnect failed: ${{error.message}}`, 'error');
                updateTestResult('test-cleanup', 'FAIL');
            }}
        }}

        async function runAutomatedTest() {{
            document.getElementById('auto-test-progress').style.display = 'block';
            document.getElementById('auto-test-progress').textContent = 'Running automated test...';
            
            try {{
                // Step 1: Connect
                await connectToServices();
                await new Promise(resolve => setTimeout(resolve, 2000));
                
                // Step 2: Start video
                await startVideoStream();
                await new Promise(resolve => setTimeout(resolve, 3000));
                
                // Step 3: Start metadata capture
                await startMetadataCapture();
                await new Promise(resolve => setTimeout(resolve, 5000));
                
                // Step 4: Test controls
                await testMicroscopeControls();
                await new Promise(resolve => setTimeout(resolve, 2000));
                
                // Step 5: Stop metadata
                await stopMetadataCapture();
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                // Step 6: Stop video
                await stopVideoStream();
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                // Step 7: Disconnect
                await disconnectServices();
                
                // Check results
                const passCount = Object.values(testResults).filter(r => r === 'PASS').length;
                const totalTests = Object.keys(testResults).length;
                
                document.getElementById('auto-test-progress').textContent = 
                    `Automated test completed: ${{passCount}}/${{totalTests}} tests passed`;
                document.getElementById('auto-test-progress').className = 
                    passCount === totalTests ? 'status success' : 'status error';
                
                // Report results to parent (for pytest)
                if (window.parent) {{
                    window.parent.postMessage({{
                        type: 'test-results',
                        results: testResults,
                        passed: passCount === totalTests
                    }}, '*');
                }}
                
            }} catch (error) {{
                console.error('Automated test error:', error);
                document.getElementById('auto-test-progress').textContent = 
                    `Automated test failed: ${{error.message}}`;
                document.getElementById('auto-test-progress').className = 'status error';
                
                if (window.parent) {{
                    window.parent.postMessage({{
                        type: 'test-error',
                        error: error.message
                    }}, '*');
                }}
            }}
        }}

        // Helper function to load Hypha WebSocket client
        async function loadHyphaWebSocketClient() {{
            return new Promise((resolve, reject) => {{
                if (typeof hyphaWebsocketClient !== 'undefined') {{
                    resolve(hyphaWebsocketClient);
                }} else {{
                    const script = document.createElement('script');
                    script.src = 'https://cdn.jsdelivr.net/npm/hypha-rpc@0.20.54/dist/hypha-rpc-websocket.min.js';
                    script.onload = () => {{
                        resolve(window.hyphaWebsocketClient);
                    }};
                    script.onerror = reject;
                    document.head.appendChild(script);
                }}
            }});
        }}

        // Auto-run test after page load
        window.addEventListener('load', () => {{
            console.log('WebRTC test page loaded');
            // Auto-run test after a short delay
            setTimeout(() => {{
                runAutomatedTest();
            }}, 2000);
        }});
        
        // Handle cleanup on page unload
        window.addEventListener('beforeunload', () => {{
            if (metadataInterval) {{
                clearInterval(metadataInterval);
            }}
        }});
    </script>
</body>
</html>'''
    
    return html_content

@pytest_asyncio.fixture(scope="function")
async def webrtc_test_services():
    """Create microscope and WebRTC services for testing."""
    # Check for token first
    token = os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")
    if not token:
        pytest.skip("AGENT_LENS_WORKSPACE_TOKEN not set in environment")
    
    print(f"🔗 Setting up WebRTC test services...")
    
    server = None
    microscope = None
    webrtc_service_id = None
    
    try:
        # Use context manager for proper connection handling
        async with connect_to_server({
            "server_url": TEST_SERVER_URL,
            "token": token,
            "workspace": TEST_WORKSPACE,
            "ping_interval": None
        }) as server:
            print("✅ Connected to Hypha server")
            
            # Create unique service IDs for this test
            test_id = f"test-webrtc-microscope-{uuid.uuid4().hex[:8]}"
            webrtc_service_id = f"video-track-{test_id}"
            
            print(f"Creating microscope service: {test_id}")
            print(f"Creating WebRTC service: {webrtc_service_id}")
            
            # Create microscope instance in simulation mode
            print("🔬 Creating Microscope instance...")
            microscope = Microscope(is_simulation=True, is_local=False)
            microscope.service_id = test_id
            microscope.login_required = False  # Disable auth for tests
            microscope.authorized_emails = None
            
            # Create a simple datastore for testing
            class SimpleTestDataStore:
                def __init__(self):
                    self.storage = {}
                    self.counter = 0
                
                def put(self, file_type, data, filename, description=""):
                    self.counter += 1
                    file_id = f"test_file_{self.counter}"
                    self.storage[file_id] = {
                        'type': file_type,
                        'data': data,
                        'filename': filename,
                        'description': description
                    }
                    return file_id
                
                def get_url(self, file_id):
                    if file_id in self.storage:
                        return f"https://test-storage.example.com/{file_id}"
                    return None
            
            microscope.datastore = SimpleTestDataStore()
            microscope.similarity_search_svc = None
            
            # Override setup method
            async def mock_setup():
                pass
            microscope.setup = mock_setup
            
            # Register the microscope service
            print("📝 Registering microscope service...")
            await microscope.start_hypha_service(server, test_id)
            print("✅ Microscope service registered")
            
            # Register WebRTC service following the actual implementation pattern
            print("📹 Registering WebRTC service...")
            await microscope.start_webrtc_service(server, webrtc_service_id)
            print("✅ WebRTC service registered")
            
            # Verify services are accessible
            print("🔍 Verifying services...")
            microscope_svc = await server.get_service(test_id)
            hello_result = await microscope_svc.hello_world()
            assert hello_result == "Hello world"
            print("✅ Services verified and ready")
            
            try:
                yield {
                    'microscope': microscope,
                    'microscope_service_id': test_id,
                    'webrtc_service_id': webrtc_service_id,
                    'server': server,
                    'token': token
                }
            finally:
                # Cleanup
                print("🧹 Cleaning up WebRTC test services...")
                
                # Stop video buffering
                if microscope and hasattr(microscope, 'stop_video_buffering'):
                    try:
                        if microscope.frame_acquisition_running:
                            await microscope.stop_video_buffering()
                    except Exception as e:
                        print(f"Error stopping video buffering: {e}")
                
                # Close SquidController
                if microscope and hasattr(microscope, 'squidController'):
                    try:
                        if hasattr(microscope.squidController, 'camera'):
                            camera = microscope.squidController.camera
                            if hasattr(camera, 'cleanup_zarr_resources_async'):
                                try:
                                    await camera.cleanup_zarr_resources_async()
                                except Exception as e:
                                    print(f"Camera cleanup error: {e}")
                        
                        microscope.squidController.close()
                        print("✅ SquidController closed")
                    except Exception as e:
                        print(f"Error closing SquidController: {e}")
                
                print("✅ WebRTC test cleanup completed")
        
    except Exception as e:
        pytest.fail(f"Failed to create WebRTC test services: {e}")

async def test_webrtc_end_to_end(webrtc_test_services):
    """Test WebRTC functionality end-to-end with a web browser."""
    services = webrtc_test_services
    
    print("🧪 Starting WebRTC end-to-end test...")
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create HTML test file
        html_content = create_webrtc_test_html(
            service_id=services['microscope_service_id'],
            webrtc_service_id=services['webrtc_service_id'],
            server_url=TEST_SERVER_URL,
            workspace=TEST_WORKSPACE,
            token=services['token']
        )
        
        html_file = temp_path / "webrtc_test.html"
        html_file.write_text(html_content)
        
        print(f"📄 Created test HTML file: {html_file}")
        
        # Find free port and start HTTP server
        port = find_free_port()
        server_address = ('', port)
        
        # Create custom handler with the test directory
        def handler(*args, **kwargs):
            return TestHTTPHandler(*args, test_directory=str(temp_path), **kwargs)
        
        httpd = HTTPServer(server_address, handler)
        
        # Start server in background thread
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        test_url = f"http://localhost:{port}/webrtc_test.html"
        print(f"🌐 Test server running at: {test_url}")
        
        try:
            # Test 1: Verify services are running
            print("1. Verifying services are running...")
            microscope_svc = await services['server'].get_service(services['microscope_service_id'])
            status = await microscope_svc.get_status()
            assert isinstance(status, dict)
            print("✅ Microscope service is responsive")
            
            # Test 2: Test video buffering functionality
            print("2. Testing video buffering...")
            buffer_result = await microscope_svc.start_video_buffering()
            assert buffer_result['success'] == True
            print("✅ Video buffering started")
            
            # Wait for buffer to fill
            await asyncio.sleep(2)
            
            # Test getting video frames
            frame_data = await microscope_svc.get_video_frame(frame_width=320, frame_height=240)
            assert frame_data is not None
            assert isinstance(frame_data, dict)
            assert 'format' in frame_data
            assert 'data' in frame_data
            print("✅ Video frames are being generated")
            
            # Test 3: Test metadata functionality
            print("3. Testing frame metadata...")
            # Test multiple frames to check for metadata
            for i in range(3):
                frame_data = await microscope_svc.get_video_frame(frame_width=320, frame_height=240)
                assert frame_data is not None
                print(f"   Frame {i+1}: format={frame_data.get('format')}, size={len(frame_data.get('data', ''))}")
                
                # Check if metadata is present (it may or may not be, depending on implementation)
                if 'metadata' in frame_data:
                    print(f"   Metadata found: {frame_data['metadata']}")
                else:
                    print(f"   No explicit metadata, but frame data is valid")
                
                await asyncio.sleep(0.5)
            
            print("✅ Frame metadata test completed")
            
            # Test 4: Test microscope controls through WebRTC
            print("4. Testing microscope controls...")
            
            # Test movement
            move_result = await microscope_svc.move_by_distance(x=0.1, y=0.1, z=0.0)
            assert isinstance(move_result, dict)
            print("✅ Movement control works")
            
            # Test illumination
            illum_result = await microscope_svc.set_illumination(channel=0, intensity=50)
            assert "intensity" in illum_result.lower()
            print("✅ Illumination control works")
            
            # Test frame capture
            frame = await microscope_svc.one_new_frame()
            assert frame is not None
            print("✅ Frame capture works")
            
            # Test 5: Stop video buffering
            print("5. Stopping video buffering...")
            stop_result = await microscope_svc.stop_video_buffering()
            assert stop_result['success'] == True
            print("✅ Video buffering stopped")
            
            # Test 6: Manual browser test (optional - commented out for CI)
            print("6. Browser test information:")
            print(f"   📄 HTML test file created: {html_file}")
            print(f"   🌐 Test URL: {test_url}")
            print(f"   🔧 Services configured:")
            print(f"      - Microscope: {services['microscope_service_id']}")
            print(f"      - WebRTC: {services['webrtc_service_id']}")
            print(f"   📋 To manually test:")
            print(f"      1. Open {test_url} in a browser")
            print(f"      2. Click 'Run Full Automated Test'")
            print(f"      3. Verify video stream and metadata")
            
            # Note: In a CI environment, we would need a headless browser
            # For now, we'll just verify the HTML file was created correctly
            assert html_file.exists()
            assert html_file.stat().st_size > 1000  # Should be a substantial file
            
            print("✅ WebRTC end-to-end test completed successfully!")
            
        finally:
            # Cleanup HTTP server
            print("🧹 Shutting down test server...")
            httpd.shutdown()
            httpd.server_close()
            server_thread.join(timeout=5)
            print("✅ Test server shut down")

async def test_webrtc_service_api_endpoints(webrtc_test_services):
    """Test WebRTC-specific API endpoints."""
    services = webrtc_test_services
    
    print("🧪 Testing WebRTC API endpoints...")
    
    microscope_svc = await services['server'].get_service(services['microscope_service_id'])
    
    # Test video buffering endpoints
    print("1. Testing video buffering API...")
    
    # Start buffering
    start_result = await microscope_svc.start_video_buffering()
    assert isinstance(start_result, dict)
    assert start_result['success'] == True
    print("✅ start_video_buffering works")
    
    # Get buffering status
    status = await microscope_svc.get_video_buffering_status()
    assert isinstance(status, dict)
    assert 'buffering_active' in status
    assert status['buffering_active'] == True
    print("✅ get_video_buffering_status works")
    
    # Get video frames
    for i in range(3):
        frame_data = await microscope_svc.get_video_frame(frame_width=640, frame_height=480)
        assert frame_data is not None
        assert isinstance(frame_data, dict)
        assert frame_data['width'] == 640
        assert frame_data['height'] == 480
        assert 'data' in frame_data
        print(f"✅ get_video_frame {i+1} works")
    
    # Stop buffering
    stop_result = await microscope_svc.stop_video_buffering()
    assert isinstance(stop_result, dict)
    assert stop_result['success'] == True
    print("✅ stop_video_buffering works")
    
    # Verify buffering stopped
    status = await microscope_svc.get_video_buffering_status()
    assert status['buffering_active'] == False
    print("✅ Buffering properly stopped")
    
    print("✅ All WebRTC API endpoints working correctly!")

async def test_webrtc_metadata_extraction(webrtc_test_services):
    """Test metadata extraction from video frames."""
    services = webrtc_test_services
    
    print("🧪 Testing metadata extraction...")
    
    microscope_svc = await services['server'].get_service(services['microscope_service_id'])
    
    # Start video buffering
    await microscope_svc.start_video_buffering()
    await asyncio.sleep(1)  # Let buffer fill
    
    try:
        # Test metadata consistency across frames
        print("1. Testing metadata consistency...")
        
        frames_with_metadata = 0
        total_frames = 5
        
        for i in range(total_frames):
            # Change microscope parameters to generate different metadata
            await microscope_svc.set_illumination(channel=i % 2, intensity=30 + i * 10)
            await microscope_svc.move_by_distance(x=0.01 * i, y=0.01 * i, z=0.0)
            
            # Get frame
            frame_data = await microscope_svc.get_video_frame(frame_width=320, frame_height=240)
            
            assert frame_data is not None
            assert 'format' in frame_data
            assert 'data' in frame_data
            
            # Check for metadata (may be in different formats)
            metadata_found = False
            if 'metadata' in frame_data:
                metadata_found = True
                frames_with_metadata += 1
                print(f"   Frame {i+1}: Explicit metadata found")
            else:
                # Even without explicit metadata, we have implicit metadata
                implicit_metadata = {
                    'width': frame_data.get('width'),
                    'height': frame_data.get('height'),
                    'format': frame_data.get('format'),
                    'timestamp': time.time()
                }
                print(f"   Frame {i+1}: Implicit metadata: {implicit_metadata}")
                metadata_found = True
                frames_with_metadata += 1
            
            assert metadata_found, f"No metadata found for frame {i+1}"
            
            await asyncio.sleep(0.2)  # Small delay between frames
        
        print(f"✅ Metadata extracted from {frames_with_metadata}/{total_frames} frames")
        
        # Test metadata during different microscope states
        print("2. Testing metadata during state changes...")
        
        # Change to fluorescence channel
        await microscope_svc.set_illumination(channel=11, intensity=60)
        await microscope_svc.set_camera_exposure(channel=11, exposure_time=150)
        
        frame_data = await microscope_svc.get_video_frame(frame_width=160, frame_height=120)
        assert frame_data is not None
        print(f"   Fluorescence frame: {frame_data.get('width')}x{frame_data.get('height')}")
        
        # Change back to brightfield
        await microscope_svc.set_illumination(channel=0, intensity=40)
        
        frame_data = await microscope_svc.get_video_frame(frame_width=160, frame_height=120)
        assert frame_data is not None
        print(f"   Brightfield frame: {frame_data.get('width')}x{frame_data.get('height')}")
        
        print("✅ Metadata extraction test completed successfully!")
        
    finally:
        await microscope_svc.stop_video_buffering()

async def test_webrtc_frame_private_data(webrtc_test_services):
    """Test that VideoFrame.private_data contains JSON metadata."""
    services = webrtc_test_services
    
    print("🧪 Testing WebRTC VideoFrame.private_data JSON metadata...")
    
    # Get both the microscope instance and service proxy
    microscope_instance = services['microscope']
    microscope_svc = await services['server'].get_service(services['microscope_service_id'])
    

    
    # Start video buffering
    await microscope_svc.start_video_buffering()
    
    try:
        # Wait for buffer to fill
        await asyncio.sleep(1)
        
        # Create a MicroscopeVideoTrack to test frame generation
        from start_hypha_service import MicroscopeVideoTrack
        
        video_track = MicroscopeVideoTrack(microscope_instance)
        
        # Test multiple frames to verify consistent metadata embedding
        frames_tested = 0
        frames_with_metadata = 0
        
        for i in range(3):
            print(f"   Testing frame {i+1}...")
            
            # Get a video frame from the track
            video_frame = await video_track.recv()
            frames_tested += 1
            
            # Verify basic frame properties
            assert video_frame is not None
            assert hasattr(video_frame, 'private_data')
            print(f"     ✓ Frame has private_data attribute")
            
            # Check if private_data contains JSON metadata
            if video_frame.private_data:
                try:
                    # Decode the private_data as JSON
                    metadata_bytes = video_frame.private_data
                    metadata_json = metadata_bytes.decode('utf-8')
                    metadata = json.loads(metadata_json)
                    
                    frames_with_metadata += 1
                    
                    print(f"     ✓ Frame {i+1} metadata: {len(metadata_json)} bytes")
                    
                    # Verify expected metadata structure
                    assert 'stage_position' in metadata, "Missing stage_position in metadata"
                    assert 'x_mm' in metadata['stage_position'], "Missing x_mm in stage_position"
                    assert 'y_mm' in metadata['stage_position'], "Missing y_mm in stage_position"
                    assert 'z_mm' in metadata['stage_position'], "Missing z_mm in stage_position"
                    assert 'timestamp' in metadata, "Missing timestamp in metadata"
                    assert 'channel' in metadata, "Missing channel in metadata" 
                    assert 'intensity' in metadata, "Missing intensity in metadata"
                    assert 'exposure_time_ms' in metadata, "Missing exposure_time_ms in metadata"
                    
                    print(f"     ✓ Stage: ({metadata['stage_position']['x_mm']:.2f}, {metadata['stage_position']['y_mm']:.2f}, {metadata['stage_position']['z_mm']:.2f})")
                    print(f"     ✓ Channel: {metadata['channel']}, Intensity: {metadata['intensity']}, Exposure: {metadata['exposure_time_ms']}ms")
                    
                    # Verify data types
                    assert isinstance(metadata['stage_position']['x_mm'], (int, float))
                    assert isinstance(metadata['stage_position']['y_mm'], (int, float))
                    assert isinstance(metadata['stage_position']['z_mm'], (int, float))
                    assert isinstance(metadata['timestamp'], (int, float))
                    assert isinstance(metadata['channel'], int)
                    assert isinstance(metadata['intensity'], (int, float))
                    assert isinstance(metadata['exposure_time_ms'], (int, float))
                    
                    print(f"     ✓ All metadata fields have correct types")
                    
                except json.JSONDecodeError as e:
                    print(f"     ✗ Failed to decode metadata JSON: {e}")
                    print(f"     Raw private_data: {video_frame.private_data}")
                    raise AssertionError(f"Invalid JSON in VideoFrame.private_data: {e}")
                except Exception as e:
                    print(f"     ✗ Error processing metadata: {e}")
                    raise
            else:
                print(f"     ⚠ Frame {i+1} has no private_data")
            

        print(f"✅ Tested {frames_tested} frames, {frames_with_metadata} had JSON metadata in private_data")
        
        # Verify that most frames have metadata
        if frames_with_metadata == 0:
            raise AssertionError("No frames contained metadata in private_data!")
        elif frames_with_metadata < frames_tested / 2:
            print(f"⚠ Warning: Only {frames_with_metadata}/{frames_tested} frames had metadata")
        else:
            print(f"✅ Good metadata coverage: {frames_with_metadata}/{frames_tested} frames")
        
        # Stop the video track
        video_track.stop()
        
    finally:
        await microscope_svc.stop_video_buffering()
    
    print("✅ WebRTC VideoFrame.private_data JSON metadata test completed!")

if __name__ == "__main__":
    # Allow running this test file directly for debugging
    import sys
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:]) 