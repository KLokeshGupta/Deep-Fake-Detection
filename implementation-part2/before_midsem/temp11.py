import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("WIFS2018_In_Ictu_Oculi\dlib_model\shape_predictor_68_face_landmarks.dat")

def extract_eyebrow_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame=cv2.equalizeHist(frame) #for normalization if you want
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        right_eyebrow = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(17, 22)]
        left_eyebrow = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(22, 27)]
        return right_eyebrow, left_eyebrow
    return None, None

def calculate_asymmetry(left_eyebrow, right_eyebrow):
    asymmetry = np.mean([abs(left[1] - right[1]) for left, right in zip(left_eyebrow, right_eyebrow)])
    return asymmetry

def calculate_velocity(current_landmarks, previous_landmarks, time_delta):
    if previous_landmarks is None:
        return [0] * len(current_landmarks)
    velocities = [np.linalg.norm(np.array(curr) - np.array(prev)) / time_delta 
                  for curr, prev in zip(current_landmarks, previous_landmarks)]
    return velocities

def calculate_temporal_jitter(landmark_positions):
    if len(landmark_positions) < 3:
        return 0
    
    diffs = [np.linalg.norm(np.array(landmark_positions[i+1]) - np.array(landmark_positions[i])) 
             for i in range(len(landmark_positions)-1)]
    second_diffs = [abs(diffs[i+1] - diffs[i]) for i in range(len(diffs)-1)]
    return np.var(second_diffs)

def process_video(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_delta = 1 / fps

    asymmetry_values = []
    left_velocities = []
    right_velocities = []
    left_jitter_values = []
    right_jitter_values = []

    prev_left = None
    prev_right = None
    left_positions = []
    right_positions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        right_eyebrow, left_eyebrow = extract_eyebrow_landmarks(frame)
        if right_eyebrow is None or left_eyebrow is None:
            continue

        asymmetry = calculate_asymmetry(left_eyebrow, right_eyebrow)
        asymmetry_values.append(asymmetry)

        left_vel = np.mean(calculate_velocity(left_eyebrow, prev_left, time_delta))
        right_vel = np.mean(calculate_velocity(right_eyebrow, prev_right, time_delta))
        left_velocities.append(left_vel)
        right_velocities.append(right_vel)

        left_positions.append(left_eyebrow)
        right_positions.append(right_eyebrow)
        if len(left_positions) > 10:
            left_jitter = calculate_temporal_jitter(left_positions[-10:])
            right_jitter = calculate_temporal_jitter(right_positions[-10:])
            left_jitter_values.append(left_jitter)
            right_jitter_values.append(right_jitter)

        prev_left, prev_right = left_eyebrow, right_eyebrow

    cap.release()

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(asymmetry_values)
    plt.title("Eyebrow Asymmetry")
    plt.ylabel("Asymmetry")

    plt.subplot(3, 1, 2)
    plt.plot(left_velocities, label='Left Eyebrow')
    plt.plot(right_velocities, label='Right Eyebrow')
    plt.title("Eyebrow Velocity")
    plt.ylabel("Velocity (pixels/s)")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(left_jitter_values, label='Left Eyebrow')
    plt.plot(right_jitter_values, label='Right Eyebrow')
    plt.title("Temporal Jitter")
    plt.xlabel("Frame")
    plt.ylabel("Jitter")
    plt.legend()

    plt.tight_layout()
    output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_analysis.png")
    plt.savefig(output_path)
    plt.close()

    csv_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_data.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame', 'Asymmetry', 'Left Velocity', 'Right Velocity', 'Left Jitter', 'Right Jitter'])
        for i in range(len(asymmetry_values)):
            writer.writerow([i, asymmetry_values[i], left_velocities[i], right_velocities[i], 
                             left_jitter_values[i] if i < len(left_jitter_values) else '',
                             right_jitter_values[i] if i < len(right_jitter_values) else ''])

def process_multiple_videos(video_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for video_file in os.listdir(video_folder):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(video_folder, video_file)
            print(f"Processing {video_file}...")
            process_video(video_path, output_folder)
            print(f"Finished processing {video_file}")

# Usage
video_folder = "videos/original"
output_folder = "output_8/original"
process_multiple_videos(video_folder, output_folder)
