import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# Global Variables
ax = None
limits = None
master_array = None
paused = False

# Constants
FILENAME = 'contact_frames.npy'
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


# Data Loading
def load_contact_frames(filename=FILENAME):
    """Load the array representing frames of ball contact."""
    try:
        contact_frames = np.load(filename)
        print(f"Successfully loaded {filename}. Shape: {contact_frames.shape}")
        return contact_frames
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return None


def load_keypoints_from_json(json_file):
    """Load pose keypoints from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
        if len(data["people"]) == 0:
            return None
        return np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)


# Plotting
def plot_keypoints(ax, pose_keypoints):
    """Plot the 2D pose keypoints on the given axis."""
    xs, ys, confidences = pose_keypoints[:, 0], pose_keypoints[:, 1], pose_keypoints[:, 2]

    ax.clear()
    ax.set_xlim(limits[0] - 100, limits[1] + 100)
    ax.set_ylim(limits[2] - 100, limits[3] + 100)
    ax.scatter(xs, ys, c='red', s=30, label='Keypoints')

    for (start, end) in POSE_CONNECTIONS:
        if confidences[start] > 0 and confidences[end] > 0:
            ax.plot([xs[start], xs[end]], [ys[start], ys[end]], 'b-', linewidth=2)

    ax.invert_yaxis()
    ax.axis('off')


# Animation Functions
def on_click(event):
    global paused
    paused = not paused


def frame_generator(num_frames):
    i = 0
    while i < num_frames:
        if not paused:
            yield i
            i += 1
        else:
            yield None


def animate_combined(i, master_array, joint_groups, names, fig, ax_pose, ax_time_series_x, ax_time_series_y,
                     lines_x, lines_y, frame_counter, skeleton_scale):
    global paused
    if paused:
        return lines_x, lines_y, frame_counter

    plot_keypoints(ax_pose, master_array[i])

    for idx, joints in enumerate(joint_groups):
        # Calculate the average position of the specified joints for the time series plot
        avg_x = np.mean(master_array[:i + 1, joints, 0], axis=1)
        avg_y = np.mean(master_array[:i + 1, joints, 1], axis=1)

        # Update the time-series plot with averaged trajectories
        lines_x[idx].set_data(range(i + 1), avg_x)
        lines_y[idx].set_data(range(i + 1), avg_y)

    frame_counter.set_text(f"Frame: {i}")

    return lines_x + lines_y + [frame_counter]


def plot_combined_animation_with_time_series(master_array, joint_groups, names, skeleton_scale=2, interval=1500):
    num_frames = master_array.shape[0]

    fig, (ax_pose, ax_time_series_x, ax_time_series_y) = plt.subplots(1, 3, figsize=(15, 6))

    # Initialize time-series lines for each group
    lines_x = []
    lines_y = []

    for name in names:
        line_x, = ax_time_series_x.plot([], [], label=f"{name} X")
        line_y, = ax_time_series_y.plot([], [], label=f"{name} Y")
        lines_x.append(line_x)
        lines_y.append(line_y)

    # Time-series plot settings for X coordinates
    ax_time_series_x.set_xlim(0, num_frames)
    ax_time_series_x.set_ylim(np.min(master_array[:, :, 0]) - 50,
                              np.max(master_array[:, :, 0]) + 50)
    ax_time_series_x.set_title("Average Joint X Coordinates Over Time")
    ax_time_series_x.legend()

    # Time-series plot settings for Y coordinates
    ax_time_series_y.set_xlim(0, num_frames)
    ax_time_series_y.set_ylim(np.min(master_array[:, :, 1]) - 50,
                              np.max(master_array[:, :, 1]) + 50)
    ax_time_series_y.set_title("Average Joint Y Coordinates Over Time")
    ax_time_series_y.legend()

    # Frame counter for pose plot
    frame_counter = ax_pose.text(0.5, 0.9, '', transform=ax_pose.transAxes, ha='center', va='center', fontsize=12)

    fig.canvas.mpl_connect('button_press_event', on_click)
    anim = FuncAnimation(
        fig, animate_combined, frames=frame_generator(num_frames), interval=interval,
        fargs=(master_array, joint_groups, names, fig, ax_pose, ax_time_series_x, ax_time_series_y,
               lines_x, lines_y, frame_counter, skeleton_scale),
        blit=False
    )
    plt.tight_layout()
    plt.show()


def plot_joint_positions_over_time(master_array, joint_groups, names):
    """Plot x and y positions of joint groups over time with color gradients representing time in side-by-side plots,
    display the mean position as a black point, and show a 1-standard deviation ellipse based on the covariance matrix.
    Only points with confidence greater than 0 are included."""
    num_frames = master_array.shape[0]
    fig, axes = plt.subplots(1, len(joint_groups), figsize=(8 * len(joint_groups), 8))

    # Ensure axes is iterable even if there's only one joint group
    if len(joint_groups) == 1:
        axes = [axes]

    # Color schemes for different joint groups
    color_schemes = [(0, 0, 1), (1, 0, 0)]  # RGB basis for each group
    for idx, (joints, ax) in enumerate(zip(joint_groups, axes)):
        color_base = np.array(color_schemes[idx % len(color_schemes)])  # Cycle through color schemes if needed

        # Plot each frame's position with a color gradient, only including points with confidence > 0
        frame_positions_x = []
        frame_positions_y = []
        for frame in range(num_frames):
            valid_x = []
            valid_y = []
            for joint in joints:
                # Check confidence for each joint; add only if confidence > 0
                if master_array[frame, joint, 2] > 0:
                    valid_x.append(master_array[frame, joint, 0])
                    valid_y.append(master_array[frame, joint, 1])

            # Compute mean x and y for valid points in this frame
            if valid_x and valid_y:  # Only if there are valid points
                avg_x = np.mean(valid_x)
                avg_y = np.mean(valid_y)
                frame_positions_x.append(avg_x)
                frame_positions_y.append(avg_y)

                # Color varies with time using the base color
                color = color_base * (frame / num_frames)
                ax.scatter(avg_x, avg_y, color=color, label=names[idx] if frame == 0 else "")

        # Calculate and plot the mean position for the joint group, based only on valid points
        mean_x = np.mean(frame_positions_x)
        mean_y = np.mean(frame_positions_y)
        ax.scatter(mean_x, mean_y, color='black', s=60, label='Mean Position')  # Black mean marker, larger size

        # Calculate the covariance matrix for valid points and create a 1-std deviation ellipse
        if frame_positions_x and frame_positions_y:  # Ensure there are points to calculate covariance
            covariance_matrix = np.cov(frame_positions_x, frame_positions_y)
            print(np.linalg.det(covariance_matrix))
            eigvals, eigvecs = np.linalg.eigh(covariance_matrix)  # Get eigenvalues and eigenvectors

            # Scale eigenvalues to represent 1 standard deviation
            std_devs = np.sqrt(eigvals)

            # Get the angle of the ellipse from the eigenvectors
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

            # Create the ellipse patch
            ellipse = patches.Ellipse(
                (mean_x, mean_y),
                width=2 * std_devs[0],
                height=2 * std_devs[1],
                angle=angle,
                edgecolor='green',
                facecolor='none',
                linestyle='--',
                linewidth=2,
                label='1 Std Dev'
            )
            ax.add_patch(ellipse)

        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(f"{names[idx]} Positions Over Time with Color Gradient")
        ax.legend()

    plt.tight_layout()
    plt.show()

# Main Processing
def adjust_keypoints_to_ball_location(pose_keypoints, ball_location):
    """Adjust all keypoints so the ball stays at the origin (0,0)."""
    ball_x, ball_y = ball_location
    translation = np.array([0, 0]) - np.array([ball_x, ball_y])
    pose_keypoints[:, :2] += translation  # Apply translation to x and y only
    return pose_keypoints


def calculate_limits(keypoints_array):
    """Calculate global min/max x and y values for the entire sequence of keypoints."""
    all_x = []
    all_y = []

    for keypoints in keypoints_array:
        xs, ys = keypoints[:, 0], keypoints[:, 1]
        all_x.extend(xs)
        all_y.extend(ys)

    return min(all_x), max(all_x), min(all_y), max(all_y)


def main_func(kick_number, joint_groups, names, interval=1500):
    global master_array, limits

    contact_frames = load_contact_frames()
    if contact_frames is None:
        return

    master_array = []
    ball_location = [0, 0]  # Define ball location here as needed
    still_frames = True
    i = 0
    while still_frames:
        try:
            json_file = os.path.join(os.path.dirname(__file__),
                                     f'../output/pose_estimation_results_1/Kick_{kick_number}_0000000000{str(i).zfill(2)}_keypoints.json')
            pose_keypoints = load_keypoints_from_json(json_file)
            if pose_keypoints is not None:
                adjusted_keypoints = adjust_keypoints_to_ball_location(pose_keypoints, ball_location)
                master_array.append(adjusted_keypoints)
        except:
            i = 0
            still_frames = False
        i += 1

    master_array = np.array(master_array)
    limits = calculate_limits(master_array)

    # Plot combined animation with time series
    plot_combined_animation_with_time_series(master_array, joint_groups, names, skeleton_scale=3, interval=interval)

    # Plot joint positions over time
    plot_joint_positions_over_time(master_array, joint_groups, names)


if __name__ == "__main__":
    kick_number = 1
    joint_groups = [[11, 22, 23, 24], [14, 19, 20, 21]]  # Define groups of joints to average
    names = ["left foot", "right foot"]  # Define names for each group
    main_func(kick_number, joint_groups, names, interval=200)