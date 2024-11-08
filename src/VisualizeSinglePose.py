import cv2
import json
import numpy as np
import os
import matplotlib.pyplot as plt

# OpenPose body keypoint connections for drawing skeletons
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Head, neck, left arm
    (1, 5), (5, 6), (6, 7),  # Right arm
    (1, 8), (8, 9), (8, 12),  # Torso
    (9, 10), (10, 11),  # Left leg
    (12, 13), (13, 14),  # Right leg
    (11, 24), (11, 22), (22, 23),  # Left foot
    (14, 21), (14, 19), (19, 20),  # Right foot
    (0, 15), (0, 16), (15, 17), (16, 18)  # Eyes and ears
]

def load_keypoints_from_json(json_file):
    """Load pose keypoints from a JSON file."""
    print("Attempting to load JSON from:", json_file)
    with open(json_file, 'r') as f:
        data = json.load(f)
        if len(data["people"]) == 0:
            return None
        return data["people"][0]["pose_keypoints_2d"]

def plot_keypoints_on_frame(frame, keypoints):
    """Draw pose keypoints on a video frame."""
    keypoints = np.array(keypoints).reshape(-1, 3)  # Reshape to (25, 3)

    # Draw keypoints
    for (start, end) in POSE_CONNECTIONS:
        # Only plot lines between points with confidence > 0
        if keypoints[start, 2] > 0 and keypoints[end, 2] > 0:
            start_point = (int(keypoints[start, 0]), int(keypoints[start, 1]))
            end_point = (int(keypoints[end, 0]), int(keypoints[end, 1]))
            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

    for point in keypoints:
        if point[2] > 0:  # Confidence check
            cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

    return frame

def display_frame_with_pose(video_number, frame_number):
    # Paths for video and pose estimation data
    src_dir = os.path.dirname(__file__)
    video_path = os.path.join(src_dir, f'../dataset/Session 1/kick {video_number}.mp4')
    json_file = os.path.join(src_dir, f'../output/pose_estimation_results_1/Kick_{video_number}_0000000000{str(frame_number).zfill(2)}_keypoints.json')

    # Load pose data
    keypoints = load_keypoints_from_json(json_file)
    if keypoints is None:
        print(f"No pose data available for frame {frame_number} in video {video_number}.")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Set to the desired frame
    ret, frame = cap.read()
    if not ret:
        print("Could not retrieve the specified frame.")
        cap.release()
        return

    # Draw pose keypoints on the frame
    frame_with_pose = plot_keypoints_on_frame(frame, keypoints)

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(frame_with_pose, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Video {video_number}, Frame {frame_number}")
    plt.show()

    cap.release()

# Example usage:
contact_frames = np.load("contact_frames.npy")
video_number = 1
frame_number = 45
display_frame_with_pose(video_number, frame_number)