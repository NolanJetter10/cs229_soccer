import cv2
import numpy as np


def preprocess_image(image, gamma=1.2):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Apply CLAHE to the V (brightness) channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)

    # Gamma correction for further brightness control
    v = np.array(255 * (v / 255) ** gamma, dtype=np.uint8)

    # Merge channels back and convert to BGR
    hsv = cv2.merge([h, s, v])
    processed_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return processed_image


# Load in the video directory
kick_number = 15
session_number = 3
video_directory = "/Users/nolanjetter/Documents/GitHub/cs229_soccer/dataset/Session " + str(session_number) +"/Kick " + str(kick_number) + ".mp4"
ball_coordinates = "/Users/nolanjetter/Documents/GitHub/cs229_soccer/output/Session " + str(session_number) + "/Ball_coordinates.npy"
ball_coordinate = ball_coordinates[kick_number]
# Load the video
video = cv2.VideoCapture(video_directory)

# Grab the first frame from the video
ret, frame = video.read()
if not ret:
    print("Failed to read video")
    video.release()
    exit()

# Convert the frame to grayscale (required for edge detection)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Perform Canny edge detection
edges = cv2.Canny(gray, 100, 200)  # Adjust the thresholds as needed

# Perform the Hough Line Transform to get lines above a certain threshold
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
processed_image = preprocess_image(frame)


# Function to check if a line is white


def is_white_line(line, image, white_threshold=100):
    x1, y1, x2, y2 = line[0]
    num_white_pixels = 0
    num_checked_pixels = 0

    if x2 - x1 == 0:  # Vertical line
        for y in range(min(y1, y2), max(y1, y2) + 1):
            for dx in range(-2, 3):  # Check 2 pixels to the left and right
                nx = x1 + dx
                if 0 <= nx < image.shape[1] and 0 <= y < image.shape[0]:
                    pixel = image[y, nx]
                    if np.linalg.norm(pixel - np.array([255, 255, 255])) < white_threshold:
                        num_white_pixels += 1
                    num_checked_pixels += 1

    elif y2 - y1 == 0:  # Horizontal line
        for x in range(min(x1, x2), max(x1, x2) + 1):
            for dy in range(-2, 3):  # Check 2 pixels above and below
                ny = y1 + dy
                if 0 <= x < image.shape[1] and 0 <= ny < image.shape[0]:
                    pixel = image[ny, x]
                    if np.linalg.norm(pixel - np.array([255, 255, 255])) < white_threshold:
                        num_white_pixels += 1
                    num_checked_pixels += 1

    else:  # Diagonal line
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        for x in range(min(x1, x2), max(x1, x2) + 1):
            y = int(slope * x + intercept)
            for dy in range(-2, 3):  # Check 2 pixels above and below
                ny = y + dy
                if 0 <= x < image.shape[1] and 0 <= ny < image.shape[0]:
                    pixel = image[ny, x]
                    if np.linalg.norm(pixel - np.array([255, 255, 255])) < white_threshold:
                        num_white_pixels += 1
                    num_checked_pixels += 1

    return num_checked_pixels > 0 and (num_white_pixels / num_checked_pixels) >= 0.5


def is_vertical_line(line, vert_threshold=5):
    x1, y1, x2, y2 = line[0]
    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
    return abs(90 - angle) <= vert_threshold


def is_horizontal_line(line, horiz_threshold=5):
    x1, y1, x2, y2 = line[0]
    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
    return angle <= horiz_threshold


def capture_goal_posts(vertical_sets, horizontal_lines, alpha=1.0):
    """
    Finds the best match of vertical lines and a horizontal line to represent a goal structure,
    adding a penalty for horizontal lines deviating from the intended y-coordinate of the crossbar.

    :param vertical_sets: List of pairs of vertical lines, each potentially representing two goalposts.
    :param horizontal_lines: List of horizontal lines, each potentially representing a crossbar.
    :param alpha: Penalty multiplier for deviation in y-coordinate of the horizontal line.
    :return: Best matched set of [left_post, right_post, crossbar] with minimal cost.
    """
    best_match = None
    best_cost = float('inf')  # Initialize with a large value to find the minimum cost

    # Iterate through each horizontal line
    for h_line in horizontal_lines:
        x1_h, y1_h, x2_h, y2_h = h_line[0]
        horizontal_y_avg = (y1_h + y2_h) / 2  # Calculate the y-coordinate of the horizontal line's midpoint

        # Iterate through each vertical set
        for vertical_pair in vertical_sets:
            left_post, right_post = vertical_pair

            # Extract the "upper" endpoints of the vertical lines
            upper_left = (left_post[0][0], min(left_post[0][1], left_post[0][3]))
            upper_right = (right_post[0][0], min(right_post[0][1], right_post[0][3]))

            # Calculate the intended y-coordinate as the average y of the two upper points
            intended_y = (upper_left[1] + upper_right[1]) / 2

            # Compute the penalty based on deviation of the horizontal line's midpoint y from the intended y
            y_deviation_penalty = alpha * (horizontal_y_avg - intended_y) ** 2

            # Calculate the minimum distance from each horizontal endpoint to the closest vertical endpoint
            vertical_endpoints = [
                (left_post[0][0], left_post[0][1]), (left_post[0][2], left_post[0][3]),
                (right_post[0][0], right_post[0][1]), (right_post[0][2], right_post[0][3])
            ]
            min_distance_1 = min(np.linalg.norm(np.array([x1_h, y1_h]) - np.array(endpoint)) for endpoint in vertical_endpoints)
            min_distance_2 = min(np.linalg.norm(np.array([x2_h, y2_h]) - np.array(endpoint)) for endpoint in vertical_endpoints)

            # Calculate the total cost as the sum of distances plus the y-deviation penalty
            total_cost = min_distance_1 + min_distance_2 + y_deviation_penalty

            # Update best match if the total cost is lower than the current best cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_match = [left_post, right_post, h_line]

    return best_match


def process_goal_structure(left_post, right_post, crossbar):
    """
    Processes the goal structure by creating adjusted vertical lines (goalposts) and a connected crossbar.

    :param left_post: Line representing the left vertical post (in the format [[x1, y1, x2, y2]]).
    :param right_post: Line representing the right vertical post (in the format [[x1, y1, x2, y2]]).
    :param crossbar: Line representing the crossbar (in the format [[x1, y1, x2, y2]]).
    :return: Processed left_post, right_post, and crossbar lines in the same format.
    """

    # Determine the intersection points of the horizontal line (crossbar) with each vertical line
    x1_left, y1_left, x2_left, y2_left = left_post[0]
    x1_right, y1_right, x2_right, y2_right = right_post[0]
    x1_cross, y1_cross, x2_cross, y2_cross = crossbar[0]

    # The intersection points of the crossbar with each vertical line
    intersection_left = (int(x1_left), int(y1_cross)) if y1_left < y2_left else (int(x2_left), int(y1_cross))
    intersection_right = (int(x1_right), int(y1_cross)) if y1_right < y2_right else (int(x2_right), int(y1_cross))

    # Create a new crossbar line between the intersection points
    processed_crossbar = [[intersection_left[0], intersection_left[1], intersection_right[0], intersection_right[1]]]

    # Calculate the length of the post between the two intersections
    post_length = abs(intersection_right[0] - intersection_left[0])

    # Calculate the length of each vertical segment as one-third of the post length
    segment_length = int(post_length / 3)

    # Draw vertical lines extending downward from each intersection point by one-third of the post length
    processed_left_post = [[
        intersection_left[0], intersection_left[1],
        intersection_left[0], intersection_left[1] + segment_length
    ]]

    processed_right_post = [[
        intersection_right[0], intersection_right[1],
        intersection_right[0], intersection_right[1] + segment_length
    ]]

    return processed_left_post, processed_right_post, processed_crossbar


def capture_vertical_pairs(vertical_lines, y_threshold=20, spread_threshold=30):
    """
    Identifies pairs of vertical lines with similar y-coordinates of their midpoints,
    suggesting they may represent goalposts.

    :param vertical_lines: List of vertical white lines.
    :param y_threshold: Maximum allowed difference in y-coordinate of the midpoints to consider as a pair.
    :return: List of detected vertical line pairs [line1, line2].
    """
    vertical_pairs = []

    # Loop through each pair of vertical lines
    for i, line1 in enumerate(vertical_lines):
        x1_1, y1_1, x2_1, y2_1 = line1[0]
        midpoint_y1 = (y1_1 + y2_1) / 2  # Calculate midpoint y-coordinate of line1

        for j, line2 in enumerate(vertical_lines):
            if i >= j:  # Avoid duplicate pairs and self-pairing
                continue

            x1_2, y1_2, x2_2, y2_2 = line2[0]
            midpoint_y2 = (y1_2 + y2_2) / 2  # Calculate midpoint y-coordinate of line2

            # Check if the y-coordinates of the midpoints are within the specified threshold
            if abs(midpoint_y1 - midpoint_y2) <= y_threshold and abs(x1_1 - x1_2) > spread_threshold:
                # Add this pair of vertical lines as a potential goalpost pair
                vertical_pairs.append([line1, line2])

    return vertical_pairs


# Check each line to see if it is white, and if so, whether it's vertical or horizontal
white_lines = []
white_vertical_lines = []
white_horizontal_lines = []

for line in lines:
    if is_white_line(line, processed_image):
        white_lines.append(line)
        if is_vertical_line(line):
            white_vertical_lines.append(line)
        elif is_horizontal_line(line):
            white_horizontal_lines.append(line)

# create pairs of lines based on how close their y-coordinate midpoints are.
vert_sets = capture_vertical_pairs(white_vertical_lines)
left_post, right_post, crossbar = capture_goal_posts(vert_sets, white_horizontal_lines)
left_post, right_post, crossbar = process_goal_structure(left_post, right_post, crossbar)

# Draw left post in blue
x1, y1, x2, y2 = left_post[0]
cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Draw right post in blue
x1, y1, x2, y2 = right_post[0]
cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Draw crossbar in red
x1, y1, x2, y2 = crossbar[0]
cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the result
cv2.imshow("Edges", edges)
cv2.imshow("Goal Structure Detected", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
video.release()