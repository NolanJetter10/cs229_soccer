import numpy as np
import cv2
import os
import json


def load_contact_frames(filename="contact_frames.npy"):
    """Load the array representing frames of ball contact."""
    try:
        return np.load(filename)
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return None


def build_video_path(src_dir, video_number):
    """Build the video path based on the video number."""
    return os.path.join(src_dir, f'../dataset/Session 1/kick {video_number}.mp4')


def build_pose_estimation_path(src_dir, video_number, frame_number):
    """Build the path for the pose estimation JSON file based on video and frame number."""
    frame_str = str(frame_number).zfill(2)  # Ensure consistent zero-padding
    return os.path.join(src_dir, f'../output/pose_estimation_results_1/Kick_{video_number}_0000000000{frame_str}_keypoints.json')


def open_video(video_path, target_frame=0):
    """Open the video and set to the target frame."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        print("Could not retrieve the specified frame.")
        cap.release()
    return cap, frame


def load_pose_data(pose_estimation_path):
    """Load pose estimation data from a JSON file."""
    with open(pose_estimation_path, 'r') as f:
        pose_data = json.load(f)
    return pose_data


def get_foot_position(pose_data, joint_indices):
    """Calculate the average position of the specified foot joints."""
    all_points = pose_data['people'][0]['pose_keypoints_2d']
    x_avg = y_avg = count = 0

    for i in joint_indices:
        if i * 3 + 2 < len(all_points):
            x, y, confidence = all_points[i * 3], all_points[i * 3 + 1], all_points[i * 3 + 2]
            if confidence > 0 and x > 0 and y > 0:
                x_avg += x
                y_avg += y
                count += 1

    if count > 0:
        return x_avg / count, y_avg / count
    return -1, -1  # Return -1, -1 if no valid points were found


def detect_circles(frame):
    """Detect circles using the Hough Circle Transform."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=60,
        param2=30,
        minRadius=10,
        maxRadius=50
    )
    return np.round(circles[0, :]).astype("int") if circles is not None else None


def find_closest_ball(circles, left_foot, right_foot):
    """Find the circle closest to either the left or right foot."""
    closest_ball = None
    min_distance = float('inf')

    # foot needs to be this close to the ball for it to count.
    threshold_distance = 100

    for (x, y, r) in circles:
        left_distance = np.sqrt((x - left_foot[0]) ** 2 + (y - left_foot[1]) ** 2) if left_foot[0] != -1 else 1000
        right_distance = np.sqrt((x - right_foot[0]) ** 2 + (y - right_foot[1]) ** 2) if right_foot[0] != -1 else 1000
        distance = min(left_distance, right_distance)
        if distance < min_distance and distance < threshold_distance:
            min_distance = distance
            closest_ball = (x, y, r)

    return closest_ball


def draw_annotations(frame, left_foot, right_foot, closest_ball):
    """Draw circles around the detected feet and the closest ball on the frame."""
    if left_foot[0] != -1:
        cv2.circle(frame, (int(left_foot[0]), int(left_foot[1])), 10, (255, 0, 0), 4)
    if right_foot[0] != -1:
        cv2.circle(frame, (int(right_foot[0]), int(right_foot[1])), 10, (255, 0, 0), 4)
    if closest_ball:
        (x, y, r) = closest_ball
        cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
    return frame


def user_find_ball(video_path):
    curr_frame = 0
    cap, frame = open_video(video_path, curr_frame)
    if frame is None:
        return None

    coordinates = []
    ball_found = False  # Flag to exit after finding the ball location

    def click_event(event, x, y, flags, param):
        nonlocal coordinates, ball_found
        if event == cv2.EVENT_LBUTTONDOWN:
            if coordinates:
                last_x, last_y = coordinates[-1]
                distance = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)
                if distance < 50:
                    avg_x = (last_x + x) // 2
                    avg_y = (last_y + y) // 2
                    print(f"Ball location determined at: ({avg_x}, {avg_y})")
                    coordinates.append((avg_x, avg_y))
                    ball_found = True  # Set flag to exit loop
                    return
            coordinates.append((x, y))


    # Set up the mouse callback once for the window
    cv2.namedWindow("Select Ball Location")
    cv2.setMouseCallback("Select Ball Location", click_event)

    while not ball_found:
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        ret, frame = cap.read()
        if not ret:
            print("End of video reached or error loading frame.")
            break

        # Display the current frame
        cv2.imshow("Select Ball Location", frame)

        # Wait for user input
        key = cv2.waitKey(0)
        if key == ord('q'):  # Exit if 'q' is pressed
            break
        elif key == ord(' '):  # Move to next frame if space key is pressed
            curr_frame += 1

    # Clean up resources after loop
    cap.release()
    cv2.destroyWindow("Select Ball Location")  # Close specific window
    cv2.destroyAllWindows()

    # Return the last determined location if found, or None otherwise
    return coordinates[-1]


def FindBallLocation(video_number, annotate_image=False):
    """Main function to perform ball location analysis on a specified video."""
    print("finding ball location on video ", video_number)
    contact_frames = load_contact_frames()
    if contact_frames is None or video_number < 1 or video_number > len(contact_frames):
        print(f"Invalid video number {video_number}.")
        return

    frame_number = contact_frames[video_number - 1]  # Adjust for zero-indexing
    src_dir = os.path.dirname(__file__)
    video_path = build_video_path(src_dir, video_number)
    pose_estimation_path = build_pose_estimation_path(src_dir, video_number, frame_number)

    # Open video and load pose data
    cap, frame = open_video(video_path, target_frame=0)
    if frame is None:
        return
    pose_data = load_pose_data(pose_estimation_path)

    # Determine foot positions
    left_foot_joints = [11, 22, 23, 24]  # Left ankle, heel, big toe
    right_foot_joints = [14, 19, 20, 21]  # Right ankle, heel, big toe
    left_foot = get_foot_position(pose_data, left_foot_joints)
    right_foot = get_foot_position(pose_data, right_foot_joints)

    # Detect circles and find the closest ball
    circles = detect_circles(frame)
    if circles is None:
        print("No circles detected.")
        cap.release()
        return
    closest_ball = find_closest_ball(circles, left_foot, right_foot)

    if not annotate_image:
        if closest_ball is not None:
            return closest_ball
        else:
            x, y = user_find_ball(video_path)
            return x, y, 1

    else:
        annotated_frame = draw_annotations(frame, left_foot, right_foot, closest_ball)
        cv2.imshow('Detected Soccer Ball', annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cap.release()

