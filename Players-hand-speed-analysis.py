import cv2
from ultralytics import YOLO
import numpy as np
import time
from google.colab.patches import cv2_imshow

class DribbleCounter:
    def __init__(self, video_path):
        # Load the YOLO models for pose estimation and ball detection
        self.pose_model = YOLO("/content/drive/MyDrive/basketball-video-analysis/yolov8s-pose.pt")
        self.ball_model = YOLO("/content/drive/MyDrive/basketball-video-analysis/basketballModel.pt")

        # Open the video file
        self.cap = cv2.VideoCapture(video_path)

        # Define the body part indices. Switch left and right to account for the mirrored image.
        self.body_index = {
            "left_wrist": 10,  # switched
            "right_wrist": 9,  # switched
        }

        # Initialize variables to store the previous position of the basketball
        self.prev_x_center = None
        self.prev_y_center = None
        self.prev_time = None

        # Initialize the dribble counter
        self.dribble_count = 0

        # Threshold for the y-coordinate change to be considered as a dribble
        self.dribble_threshold = 3

        # Define the coordinates of the top left and bottom right corners of the cropping rectangle
        self.crop_x1, self.crop_y1, self.crop_x2, self.crop_y2 = 40, 110, 700, 700  # replace with your coordinates

        # Initialize list to store hand speeds
        self.hand_speeds = []

    def run(self):
        # Process frames from the video until the user quits
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                # Crop the frame before processing it
                frame = frame[self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2]

                # Get pose estimation results
                pose_results = self.pose_model(frame, verbose=False, conf=0.65)

                # Get ball detection results
                ball_results = self.ball_model(frame, verbose=False, conf=0.65)

                for results in ball_results:
                    for bbox in results.boxes.xyxy:
                        x1, y1, x2, y2 = bbox[:4]

                        # Calculate the center of the bounding box
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2

                        self.prev_x_center = x_center
                        self.prev_y_center = y_center

                # Analyze player's hand movement
                self.analyze_hand_movement(pose_results)

    def analyze_hand_movement(self, pose_results):
        # Analyze player's hand movement
        for results in pose_results:
            for bbox in results.boxes.xyxy:
                x1, y1, x2, y2 = bbox[:4]

                # Calculate the center of the bounding box
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                # If this is not the first frame, calculate the hand speed
                if self.prev_x_center is not None and self.prev_y_center is not None and self.prev_time is not None:
                    # Calculate the distance moved by the hand
                    distance = np.sqrt((x_center - self.prev_x_center)**2 + (y_center - self.prev_y_center)**2)
                    # Get the current time
                    current_time = time.time()
                    # Calculate the speed of the hand
                    speed = distance / (current_time - self.prev_time)
                    print(f"Hand speed: {speed:.2f} units per second")
                    # Store the hand speed for later analysis
                    self.hand_speeds.append(speed)

                # Store the current position and time for the next frame
                self.prev_x_center = x_center
                self.prev_y_center = y_center
                self.prev_time = time.time()

if __name__ == "__main__":
    # Specify the path to the video file
    video_path = '/content/drive/MyDrive/basketball-video-analysis/WHATSAAP ASSIGNMENT.mp4'  # replace with your video path

    # Create a DribbleCounter object and run the analysis
    dribble_counter = DribbleCounter(video_path)
    dribble_counter.run()
