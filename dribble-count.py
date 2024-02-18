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
        self.prev_delta_y = None

        # Initialize the dribble counter
        self.dribble_count = 0

        # Threshold for the y-coordinate change to be considered as a dribble
        self.dribble_threshold = 3

        # Define the coordinates of the top left and bottom right corners of the cropping rectangle
        self.crop_x1, self.crop_y1, self.crop_x2, self.crop_y2 = 40, 110, 700, 700  # replace with your coordinates

    def run(self):
        # Process frames from the video until the user quits
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                # Crop the frame before processing it
                frame = frame[self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2]

                # Use the ball detection model to detect the basketball in the cropped frame
                results_list = self.ball_model(frame, verbose=False, conf=0.65)

                for results in results_list:
                    for bbox in results.boxes.xyxy:
                        x1, y1, x2, y2 = bbox[:4]

                        # Calculate the center of the bounding box
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2

                        # Print the ball coordinates and the current dribble count
                        print(f"Ball coordinates: (x={x_center:.2f}, y={y_center:.2f})")
                        print(f"Dribble Count:{self.dribble_count}")

                        # Update the dribble count based on the y-coordinate of the basketball's center
                        self.update_dribble_count(x_center, y_center)

                        # Store the current position of the basketball for the next frame
                        self.prev_x_center = x_center
                        self.prev_y_center = y_center

                    # Annotate the frame with the bounding boxes and labels
                    annotated_frame = results.plot()

                    # Draw the dribble count on the frame
                    cv2.putText(
                        annotated_frame,
                        f"Dribble Count: {self.dribble_count}",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    # Display the annotated frame
                    cv2_imshow(annotated_frame)

                # Break the loop if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        # Release the video file and destroy the windows
        self.cap.release()
        cv2.destroyAllWindows()

    def update_dribble_count(self, x_center, y_center):
        # Check if this is not the first frame
        if self.prev_y_center is not None:
            # Calculate the change in the y-coordinate of the basketball's center
            delta_y = y_center - self.prev_y_center

            # Check if the basketball's y-coordinate has changed by more than the threshold
            if (
                self.prev_delta_y is not None
                and self.prev_delta_y > self.dribble_threshold
                and delta_y < -self.dribble_threshold
            ):
                # If so, increment the dribble count
                self.dribble_count += 1

            # Store the change in the y-coordinate for the next frame
            self.prev_delta_y = delta_y


if __name__ == "__main__":
    # Specify the path to the video file
    video_path = '/content/drive/MyDrive/basketball-video-analysis/WHATSAAP ASSIGNMENT.mp4'  # replace with your video path

    # Create a DribbleCounter object and run the analysis
    dribble_counter = DribbleCounter(video_path)
    dribble_counter.run()
