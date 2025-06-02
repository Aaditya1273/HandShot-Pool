import cv2
import numpy as np
import time
import threading
from flask import Flask, Response, render_template, jsonify
from flask_cors import CORS
import json
import os
from web_game_controller import MasterHitController

# Initialize Flask app
app = Flask(__name__, template_folder=os.path.abspath('templates'))
CORS(app)  # Enable CORS for all routes

# Global variables
controller = None
camera = None
output_frame = None
lock = threading.Lock()
is_running = False

def initialize_camera():
    """Initialize the camera and return the video stream."""
    global camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(2.0)  # Allow camera to warm up
    return camera

def initialize_controller():
    """Initialize the gesture controller."""
    global controller
    controller = MasterHitController()
    controller.start_calibration()
    return controller

def process_frames():
    """Process frames from the camera and update the output frame."""
    global output_frame, lock, is_running, camera, controller
    
    # Initialize camera and controller if not already done
    if camera is None:
        camera = initialize_camera()
    
    if controller is None:
        controller = initialize_controller()
    
    is_running = True
    
    # Count frames for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    while is_running:
        # Read frame from camera
        success, frame = camera.read()
        if not success:
            break
        
        # Flip the frame horizontally for a more intuitive experience
        frame = cv2.flip(frame, 1)
        
        # Process the frame with the controller
        processed_frame = controller.process_frame(frame)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:  # Update FPS every second
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Add FPS counter to the frame
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", 
                   (processed_frame.shape[1] - 120, processed_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Acquire the lock, update the output frame, and release the lock
        with lock:
            output_frame = processed_frame.copy()

def generate():
    """Generate frames for the video feed."""
    global output_frame, lock
    
    while True:
        # Wait until there's a valid output frame
        with lock:
            if output_frame is None:
                continue
            
            # Encode the frame as JPEG
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            
            if not flag:
                continue
        
        # Yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encoded_image) + b'\r\n')

@app.route("/")
def index():
    """Return the rendered index page."""
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    """Return the video feed response."""
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start", methods=["GET"])
def start():
    """Start the video processing thread."""
    global is_running
    
    if not is_running:
        # Start the video processing thread
        t = threading.Thread(target=process_frames)
        t.daemon = True
        t.start()
        return jsonify({"status": "started"})
    
    return jsonify({"status": "already running"})

@app.route("/stop", methods=["GET"])
def stop():
    """Stop the video processing thread."""
    global is_running, camera, controller
    
    is_running = False
    
    # Release resources
    if controller is not None:
        controller.release()
        controller = None
    
    if camera is not None:
        camera.release()
        camera = None
    
    return jsonify({"status": "stopped"})

@app.route("/toggle_debug", methods=["GET"])
def toggle_debug():
    """Toggle debug information display."""
    global controller
    
    if controller is not None:
        controller.toggle_debug()
        return jsonify({"status": "debug toggled"})
    
    return jsonify({"status": "controller not initialized"})

def create_templates_directory():
    """Create the templates directory if it doesn't exist."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(current_dir, 'templates')
    if not os.path.exists(templates_dir):
        print(f"Creating templates directory at {templates_dir}")
        os.makedirs(templates_dir)
    return templates_dir

def create_index_html():
    """Create the index.html template."""
    templates_dir = create_templates_directory()
    index_path = os.path.join(templates_dir, 'index.html')
    
    with open(index_path, 'w') as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Master Hit: Boss Hunter - Hand Gesture Controller</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #1a1a2e;
            color: #ffffff;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            color: #ff9900;
            margin-bottom: 20px;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
            max-width: 1200px;
        }
        .video-container {
            position: relative;
            margin-bottom: 20px;
            border: 3px solid #ff9900;
            border-radius: 10px;
            overflow: hidden;
            width: 100%;
            max-width: 1280px;
        }
        .video-feed {
            width: 100%;
            height: auto;
            display: block;
        }
        .controls {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }
        .button {
            background-color: #ff9900;
            color: #1a1a2e;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .button:hover {
            background-color: #ffcc00;
        }
        .button:disabled {
            background-color: #666666;
            cursor: not-allowed;
        }
        .instructions {
            background-color: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            width: 100%;
            max-width: 1280px;
        }
        .game-container {
            width: 100%;
            max-width: 1280px;
            height: 720px;
            border: 3px solid #ff9900;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        .game-iframe {
            width: 100%;
            height: 100%;
            border: none;
        }
        .status {
            color: #ff9900;
            margin-bottom: 10px;
        }
        .gesture-display {
            display: flex;
            justify-content: space-around;
            width: 100%;
            margin-top: 20px;
        }
        .gesture {
            text-align: center;
            padding: 10px;
            background-color: #16213e;
            border-radius: 10px;
            width: 30%;
        }
        .gesture img {
            width: 80px;
            height: 80px;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Master Hit: Boss Hunter - Hand Gesture Controller</h1>
        
        <div class="instructions">
            <h2>Instructions</h2>
            <p>Control the game using hand gestures captured by your webcam:</p>
            <ul>
                <li><strong>Open Hand:</strong> Move the eye cursor</li>
                <li><strong>Index Finger Only:</strong> Shoot</li>
                <li><strong>Closed Fist:</strong> Dodge attacks or collect power-ups</li>
            </ul>
            <p>Click "Start Camera" to begin tracking your hand gestures. The game will be controlled by your movements.</p>
        </div>
        
        <div class="gesture-display">
            <div class="gesture">
                <h3>Move</h3>
                <p>Open hand</p>
            </div>
            <div class="gesture">
                <h3>Shoot</h3>
                <p>Index finger only</p>
            </div>
            <div class="gesture">
                <h3>Dodge/Collect</h3>
                <p>Closed fist</p>
            </div>
        </div>
        
        <div class="controls">
            <button class="button" id="startButton">Start Camera</button>
            <button class="button" id="stopButton" disabled>Stop Camera</button>
            <button class="button" id="toggleDebugButton">Toggle Debug Info</button>
        </div>
        
        <p class="status" id="statusText">Camera is stopped. Click "Start Camera" to begin.</p>
        
        <div class="video-container">
            <img class="video-feed" id="videoFeed" src="" alt="Webcam Feed" style="display: none;">
        </div>
        
        <div class="game-container">
            <iframe class="game-iframe" id="gameIframe" src="https://www.crazygames.com/game/master-hit-boss-hunter-btt" allowfullscreen></iframe>
        </div>
    </div>
    
    <script>
        const startButton = document.getElementById('startButton');
        const stopButton = document.getElementById('stopButton');
        const toggleDebugButton = document.getElementById('toggleDebugButton');
        const videoFeed = document.getElementById('videoFeed');
        const statusText = document.getElementById('statusText');
        
        startButton.addEventListener('click', async () => {
            try {
                const response = await fetch('/start');
                const data = await response.json();
                
                if (data.status === 'started' || data.status === 'already running') {
                    videoFeed.src = '/video_feed';
                    videoFeed.style.display = 'block';
                    statusText.textContent = 'Camera is running. Use hand gestures to control the game.';
                    startButton.disabled = true;
                    stopButton.disabled = false;
                }
            } catch (error) {
                console.error('Error starting camera:', error);
                statusText.textContent = 'Error starting camera. Please try again.';
            }
        });
        
        stopButton.addEventListener('click', async () => {
            try {
                const response = await fetch('/stop');
                const data = await response.json();
                
                if (data.status === 'stopped') {
                    videoFeed.src = '';
                    videoFeed.style.display = 'none';
                    statusText.textContent = 'Camera is stopped. Click "Start Camera" to begin.';
                    startButton.disabled = false;
                    stopButton.disabled = true;
                }
            } catch (error) {
                console.error('Error stopping camera:', error);
                statusText.textContent = 'Error stopping camera. Please try again.';
            }
        });
        
        toggleDebugButton.addEventListener('click', async () => {
            try {
                const response = await fetch('/toggle_debug');
                await response.json();
            } catch (error) {
                console.error('Error toggling debug info:', error);
            }
        });
        
        // Handle page unload to stop the camera
        window.addEventListener('beforeunload', async () => {
            try {
                await fetch('/stop');
            } catch (error) {
                console.error('Error stopping camera on page unload:', error);
            }
        });
    </script>
</body>
</html>
""")

if __name__ == "__main__":
    # Create the index.html template
    templates_dir = create_templates_directory()
    create_index_html()
    print(f"Templates directory: {templates_dir}")
    print(f"Index.html created: {os.path.exists(os.path.join(templates_dir, 'index.html'))}")
    
    # Start the Flask app
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
