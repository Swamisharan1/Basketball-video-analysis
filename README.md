---

# Basketball Video Analysis

This repository contains Python scripts for analyzing a basketball video using computer vision techniques. The analysis includes counting the number of dribbles performed by the player and measuring the speed of the player's hand movement.

## Files

1. `dribble-count.py`: This script performs an analysis on the basketball video to count the number of dribbles performed by the player. It uses the YOLO (You Only Look Once) object detection model to detect the basketball in each frame and updates the dribble count based on the change in the y-coordinate of the basketball's center.

2. `Players-hand-speed-analysis.py`: This script measures the speed of the player's hand movement. It calculates the distance moved by the hand between consecutive frames and divides it by the time taken. The speed is then stored for later analysis.

## Usage

To run the scripts, you need to have Python installed on your machine along with the necessary libraries such as OpenCV and Ultralytics.

You can run the scripts using the following commands:

```bash
python dribble-count.py
python Players-hand-speed-analysis.py
```

Please make sure to replace the video path in the scripts with the path to your own video file.

## Dependencies

- OpenCV
- Ultralytics

## Credits

The pretrained models used in this project for basketball and pose detection were obtained from this [GitHub repository](https://github.com/ayushpai/AI-Basketball-Referee).

## Contributing

Contributions are welcome. Please feel free to submit a pull request or open an issue.

---
