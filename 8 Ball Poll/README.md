# HandShot-Pool: Gesture-Controlled 8 Ball Billiards:

A computer vision-based 8 Ball Billiards game that allows players to control the game using hand gestures.

## Features

- Hand gesture recognition for controlling the cue stick
- Physics-based ball movement and interactions
- Intuitive interface with visual feedback
- Real-time gesture tracking with minimal latency
- Gesture calibration for personalized experience

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Pygame

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd HandShot-Pool
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## How to Play

1. Run the game:
   ```
   python main.py
   ```

2. Calibrate your hand gestures by following on-screen instructions
3. Use the following gestures to control the game:
   - Open hand: Position the cue stick
   - Move hand left/right: Adjust the cue stick angle
   - Closed fist followed by open hand: Strike the cue ball (power depends on movement speed)

## Controls

- Press 'C' to recalibrate hand gestures
- Press 'R' to reset the game
- Press 'ESC' to exit the game
