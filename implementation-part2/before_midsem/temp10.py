import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("WIFS2018_In_Ictu_Oculi\dlib_model\shape_predictor_68_face_landmarks.dat")

def extract_eyebrow_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        right_eyebrow = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(17, 22)]
        left_eyebrow = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(22, 27)]
        return right_eyebrow, left_eyebrow
    return None, None

def calculate_velocity(current_landmarks, previous_landmarks, time_delta):
    if previous_landmarks is None:
        return [0] * len(current_landmarks)
    velocities = [np.linalg.norm(np.array(curr) - np.array(prev)) / time_delta 
                  for curr, prev in zip(current_landmarks, previous_landmarks)]
    return velocities

def calculate_displacement(current_landmarks, previous_landmarks):
    if previous_landmarks is None:
        return [0] * len(current_landmarks)
    displacements = [np.linalg.norm(np.array(curr) - np.array(prev)) 
                     for curr, prev in zip(current_landmarks, previous_landmarks)]
    return displacements

def process_video(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_delta = 1 / fps

    prev_right, prev_left = None, None
    right_velocities, left_velocities = [], []
    right_displacements, left_displacements = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        right_eyebrow, left_eyebrow = extract_eyebrow_landmarks(frame)
        if right_eyebrow is None or left_eyebrow is None:
            continue

        # Calculate velocity
        right_vel = calculate_velocity(right_eyebrow, prev_right, time_delta)
        left_vel = calculate_velocity(left_eyebrow, prev_left, time_delta)
        right_velocities.append(np.mean(right_vel))
        left_velocities.append(np.mean(left_vel))

        # Calculate displacement
        right_disp = calculate_displacement(right_eyebrow, prev_right)
        left_disp = calculate_displacement(left_eyebrow, prev_left)
        right_displacements.append(np.mean(right_disp))
        left_displacements.append(np.mean(left_disp))

        prev_right, prev_left = right_eyebrow, left_eyebrow

    cap.release()

    # Replace None or NaN values with mean
    right_velocities = np.array(right_velocities)
    left_velocities = np.array(left_velocities)
    right_displacements = np.array(right_displacements)
    left_displacements = np.array(left_displacements)

    for arr in [right_velocities, left_velocities, right_displacements, left_displacements]:
        mean_val = np.nanmean(arr)
        arr[np.isnan(arr)] = mean_val

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(right_velocities, label='Right Eyebrow')
    plt.plot(left_velocities, label='Left Eyebrow')
    plt.title('Eyebrow Velocity')
    plt.ylabel('Velocity (pixels/second)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(right_displacements, label='Right Eyebrow')
    plt.plot(left_displacements, label='Left Eyebrow')
    plt.title('Eyebrow Displacement')
    plt.xlabel('Frame')
    plt.ylabel('Displacement (pixels)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_velocity_displacement.png"))
    plt.close()

    # Save to CSV
    csv_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_data.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame', 'Right Velocity', 'Left Velocity', 'Right Displacement', 'Left Displacement'])
        for i in range(len(right_velocities)):
            writer.writerow([i, right_velocities[i], left_velocities[i], right_displacements[i], left_displacements[i]])

def process_multiple_videos(video_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for video_file in os.listdir(video_folder):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(video_folder, video_file)
            print(f"Processing {video_file}...")
            process_video(video_path, output_folder)
            print(f"Finished processing {video_file}")

# Usage
video_folder = "videos/deepfake"
output_folder = "output_7/deepfake"
process_multiple_videos(video_folder, output_folder)
