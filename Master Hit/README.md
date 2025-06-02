# Master Hit: Boss Hunter - Hand Gesture Controller

A hand gesture controller for playing the web-based game "Master Hit: Boss Hunter" using your webcam. This controller allows you to throw knives at targets by using simple hand gestures.

## Features

- Control the game using hand gestures detected via webcam
- Automatic hand calibration for different users
- Smooth cursor movement with position smoothing
- Reliable detection of open and closed hand gestures
- Visual feedback with debug information
- Simple and intuitive gesture system:
  - **Open hand**: Move the cursor to aim
  - **Closed fist → Open hand**: Throw knife (click)

## Requirements

- Python 3.7+
- Webcam
- Dependencies (installed automatically):
  - OpenCV
  - MediaPipe
  - NumPy
  - PyAutoGUI

## Installation

1. Clone the repository or download the files
2. Navigate to the project directory
3. Run the main script:

```bash
python main_fixed.py
```

The script will check for dependencies and offer to install them if needed.

## Usage

1. Run `python main_fixed.py` to start the controller
2. Open the Master Hit game in your browser: https://www.crazygames.com/game/master-hit-boss-hunter-btt
3. Arrange your windows so you can see both the game and the webcam window
4. The controller will start with a calibration phase - move your hand around naturally
5. Once calibrated, you can control the game with hand gestures

## Controls

| Gesture | Action |
|---------|--------|
| Open hand (2+ fingers extended) | Move the cursor to aim |
| Closed fist → Open hand | Throw knife (triggers mouse click) |

## Troubleshooting

- **Camera not working**: Make sure your webcam is connected and not being used by another application
- **Gestures not detected**: Try adjusting lighting conditions and ensure your hand is clearly visible
- **Cursor not moving**: Make sure your hand is clearly visible and within the camera frame
- **Throw not detected**: Make sure you fully close your fist and then open it clearly
- **Game not responding**: Make sure the game window is in focus when using the controller

## License

MIT License

---

Built with ❤️ using OpenCV and MediaPipe
