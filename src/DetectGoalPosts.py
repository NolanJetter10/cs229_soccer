import cv2
import numpy as np

# Load in the video directory
directory = "/Users/nolanjetter/Documents/GitHub/cs229_soccer/dataset/Session 1/Kick 1.mp4"

# Load the video
video = cv2.VideoCapture(directory)

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


# Function to check if a line is white
def is_white_line(line, image, white_threshold=225):
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


def capture_goal_posts(vertical_sets, horizontal_lines, distance_threshold=400):
    """
    Finds the best match of vertical lines and a horizontal line to represent a goal structure.

    :param vertical_sets: List of pairs of vertical lines, each potentially representing two goalposts.
    :param horizontal_lines: List of horizontal lines, each potentially representing a crossbar.
    :param distance_threshold: Maximum allowed distance for endpoints to be considered a close match.
    :return: Best matched set of [left_post, right_post, crossbar] with minimal endpoint distances.
    """
    best_match = None
    best_fit_value = float('inf')  # Initialize with a large value to find the minimum fit

    # Iterate through each pair of vertical lines
    for vertical_pair in vertical_sets:
        left_post, right_post = vertical_pair

        # Extract endpoints of the vertical lines (goalposts)
        vertical_endpoints = [
            (left_post[0][0], left_post[0][1]), (left_post[0][2], left_post[0][3]),
            (right_post[0][0], right_post[0][1]), (right_post[0][2], right_post[0][3])
        ]

        # Iterate through each horizontal line
        for horizontal_line in horizontal_lines:
            x1_h, y1_h, x2_h, y2_h = horizontal_line[0]

            # Find the closest endpoint in the vertical pair to the first endpoint of the horizontal line
            min_distance_1 = min(np.linalg.norm(np.array([x1_h, y1_h]) - np.array(endpoint)) for endpoint in vertical_endpoints)

            # Find the closest endpoint in the vertical pair to the second endpoint of the horizontal line
            min_distance_2 = min(np.linalg.norm(np.array([x2_h, y2_h]) - np.array(endpoint)) for endpoint in vertical_endpoints)

            # Aggregate the minimum distances to get a "fit score" for this combination
            fit_value = min_distance_1 + min_distance_2

            # Check if this combination is the best match so far and meets the distance threshold
            if fit_value < best_fit_value and min_distance_1 <= distance_threshold and min_distance_2 <= distance_threshold:
                best_fit_value = fit_value
                best_match = [left_post, right_post, horizontal_line]

    return best_match[0], best_match[1], best_match[2]

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
    if is_white_line(line, frame):
        white_lines.append(line)
        if is_vertical_line(line):
            white_vertical_lines.append(line)
        elif is_horizontal_line(line):
            white_horizontal_lines.append(line)

# create pairs of lines based on how close their y-coordinate midpoints are.
vert_sets = capture_vertical_pairs(white_vertical_lines)
print(len(vert_sets))
left_post, right_post, crossbar = capture_goal_posts(vert_sets, white_horizontal_lines)

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