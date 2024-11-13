import numpy as np
import os
import json
from FindBallLocation import FindBallLocation
from FindFirstPhase import find_foot_plant_information

save_path = 'output/Session 2'
base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
full_save_path = os.path.join(base_dir, save_path)

# ---------------------------------------------------------------
# Using the video sound analysis module, generate a list corresponding to the frame of impact.
# num_kicks = 20  # Adjust this to the number of kick videos you have
# batch_number = 1  # Set your batch number here
# output_dir = f"..\\output\\contact_frames_{batch_number}"
# os.makedirs(output_dir, exist_ok=True)
# contact_frames_array = process_kick_videos(num_kicks, batch_number)
# np.save(os.path.join(output_dir, "contact_frames.npy"), contact_frames_array)
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# # Using the Soccer Ball Location module, find the location of the soccer ball
# ball_location = []
# for i in range(1, 21):
#     ball_location.append(FindBallLocation(i))
#     print(ball_location[i - 1])
# ball_location = np.array(ball_location)
# # **save the ball_location array in Session 1 folder. call the file Ball_coordinates
# np.save(os.path.join(full_save_path, "Ball_coordinates.npy"), ball_location)
# # ---------------------------------------------------------------
#
# ---------------------------------------------------------------
# find which foot is the plant foot and the frame the foot is planted.
plant_foot_info = []
for i in range(1, 21):
    plant_foot_info.append(find_foot_plant_information(i))
    print(plant_foot_info[-1])
plant_foot_info = np.array(plant_foot_info)
# **save to session 1 folder. call Plant_foot_info
np.save(os.path.join(full_save_path, "Plant_foot_info.npy"), plant_foot_info)
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# helper functions
def preprocess_keypoints(keypoints, ball_location):
    ball_loc = ball_location[0:2]
    # Translate keypoints to set ball location at (0, 0)
    translated_keypoints = []
    print(keypoints)
    for i in range(25):
        translated_keypoints.append(keypoints[i] - ball_loc)
    translated_keypoints = np.array(translated_keypoints)
    # Normalize keypoints to range [-1, 1]
    max_val = np.max(np.abs(translated_keypoints), axis=0)
    normalized_keypoints = np.divide(translated_keypoints, max_val, where=max_val != 0)

    return normalized_keypoints


def load_keypoints_from_json(json_file):
    """Load pose keypoints from a JSON file."""
    print("Attempting to load JSON from:", json_file)
    with open(json_file, 'r') as f:
        data = json.load(f)
        if len(data["people"]) == 0:
            return None
        return np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)

# ---------------------------------------------------------------

"""""
1. gather the pose estimation keypoints corresponding to the frame of plant foot and frame of contact.
2. translate the points such that the ball is located at 0,0
3. normalize the data such that points lie from (-1) to 1.
"""""
processed_keypoints = []

for i in range(20):
    contact_keypoints = np.load(os.path.join(full_save_path, "contact_frames.npy"))
    plant_keypoints = np.load(os.path.join(full_save_path, "Plant_foot_info.npy"))
    ball_coordinates = np.load(os.path.join(full_save_path, "Ball_coordinates.npy"))
    # load in the keypoints corresponding to contact frame and plant keypoints
    plant_frame_json = os.path.join(
        base_dir,
        f'output/pose_estimation_results_1/Kick_{i+1}_0000000000{str(plant_keypoints[i, 1]).zfill(2)}_keypoints.json'
    )
    contact_frame_json = os.path.join(
        base_dir,
        f'output/pose_estimation_results_1/Kick_{i+1}_0000000000{str(contact_keypoints[i]).zfill(2)}_keypoints.json'
    )
    # load in the pose keypoints.
    plant_frame_keypoints = load_keypoints_from_json(plant_frame_json)
    contact_frame_keypoints = load_keypoints_from_json(contact_frame_json)
    # convert to np.float datatype for later calculations.
    plant_frame_keypoints = np.array(plant_frame_keypoints, dtype=np.float32)
    plant_frame_keypoints = plant_frame_keypoints[:, 0:2]
    print(plant_frame_keypoints.shape)
    contact_frame_keypoints = np.array(contact_frame_keypoints, dtype=np.float32)
    contact_frame_keypoints = contact_frame_keypoints[:, 0:2]
    print(contact_frame_keypoints.shape)
    # Translate and normalize each set of keypoints
    processed_contact = preprocess_keypoints(contact_frame_keypoints, ball_coordinates[i])
    processed_plant = preprocess_keypoints(plant_frame_keypoints, ball_coordinates[i])

    # Add to list for further processing or model training
    processed_keypoints.append((processed_contact, processed_plant))

processed_keypoints = np.array(processed_keypoints)
print(np.shape(processed_keypoints))
np.save(os.path.join(full_save_path, "Processed_keypoints.npy"), processed_keypoints)

print("Preprocessing complete. Data saved in the Session 1 folder.")

"""
4. gather labels from the annotations 
5. organize everything in a numpy array and write to local.
6. train some models :)
    - we can try using both the contact frame and the plant frame, or just the plant frame and see if
      there is a difference in performance. 
"""