import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("WIFS2018_In_Ictu_Oculi\dlib_model\shape_predictor_68_face_landmarks.dat")

def extract_eyebrow_landmarks(frame):
    """Extract normalized eyebrow landmarks from a video frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Extract eyebrow landmarks
        right_eyebrow = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(17, 22)]
        left_eyebrow = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(22, 27)]
        
        # Normalize coordinates relative to face size for distance invariance
        face_width = face.right() - face.left()
        face_height = face.bottom() - face.top()
        face_size = max(face_width, face_height)
        
        # Normalize coordinates
        right_eyebrow = [((x - face.left()) / face_size, (y - face.top()) / face_size) for x, y in right_eyebrow]
        left_eyebrow = [((x - face.left()) / face_size, (y - face.top()) / face_size) for x, y in left_eyebrow]
        
        return right_eyebrow, left_eyebrow
    
    return None, None

def calculate_asymmetry(left_eyebrow, right_eyebrow):
    """Calculate asymmetry between left and right eyebrows."""
    if not left_eyebrow or not right_eyebrow:
        return 0
    
    # Calculate vertical differences between corresponding points
    vertical_diffs = [abs(left[1] - right[1]) for left, right in zip(left_eyebrow, right_eyebrow)]
    
    # Return mean vertical difference as asymmetry score
    return np.mean(vertical_diffs)

def calculate_velocity(current_landmarks, previous_landmarks, time_delta):
    """Calculate velocity of eyebrow landmarks between frames."""
    if not previous_landmarks:
        return None
    
    velocities = []
    for i in range(len(current_landmarks)):
        curr_x, curr_y = current_landmarks[i]
        prev_x, prev_y = previous_landmarks[i]
        
        # Calculate displacement
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        
        # Calculate velocity (normalized units per second)
        velocity = np.sqrt(dx**2 + dy**2) / time_delta
        velocities.append(velocity)
    
    return velocities

def analyze_eyebrow_features(video_path, output_folder):
    """Process video and extract combined eyebrow features."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_delta = 1 / fps
    
    # Initialize variables
    frame_count = 0
    prev_right = None
    prev_left = None
    asymmetry_values = []
    right_velocity_values = []
    left_velocity_values = []
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract eyebrow landmarks
        right_eyebrow, left_eyebrow = extract_eyebrow_landmarks(frame)
        if not right_eyebrow or not left_eyebrow:
            continue
        
        # Calculate asymmetry
        asymmetry = calculate_asymmetry(left_eyebrow, right_eyebrow)
        asymmetry_values.append(asymmetry)
        
        # Calculate velocity
        if prev_right and prev_left:
            right_velocities = calculate_velocity(right_eyebrow, prev_right, time_delta)
            left_velocities = calculate_velocity(left_eyebrow, prev_left, time_delta)
            
            if right_velocities and left_velocities:
                right_velocity_values.append(np.mean(right_velocities))
                left_velocity_values.append(np.mean(left_velocities))
        
        # Update previous landmarks
        prev_right = right_eyebrow
        prev_left = left_eyebrow
        frame_count += 1
    
    cap.release()
    
    # Create combined feature plot
    plt.figure(figsize=(12, 8))
    
    # Plot asymmetry
    plt.subplot(2, 1, 1)
    plt.plot(asymmetry_values, color='purple')
    plt.title("Eyebrow Asymmetry Over Time")
    plt.ylabel("Normalized Asymmetry")
    plt.grid(True)
    
    # Plot velocities
    plt.subplot(2, 1, 2)
    plt.plot(right_velocity_values, label='Right Eyebrow', color='green')
    plt.plot(left_velocity_values, label='Left Eyebrow', color='blue')
    plt.title("Eyebrow Velocity Over Time")
    plt.xlabel("Frame Index")
    plt.ylabel("Normalized Velocity")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_combined_features.png")
    plt.savefig(output_path)
    plt.close()
    
    # Save data to CSV
    csv_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_combined_data.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame', 'Asymmetry', 'Right_Velocity', 'Left_Velocity'])
        
        # Align data lengths
        max_length = max(len(asymmetry_values), len(right_velocity_values), len(left_velocity_values))
        
        for i in range(max_length):
            asym = asymmetry_values[i] if i < len(asymmetry_values) else ''
            right_vel = right_velocity_values[i] if i < len(right_velocity_values) else ''
            left_vel = left_velocity_values[i] if i < len(left_velocity_values) else ''
            writer.writerow([i, asym, right_vel, left_vel])
    
    print(f"Analysis complete for {video_path}")
    print(f"Results saved to {output_folder}")

def process_multiple_videos(video_folder, output_folder):
    """Process all videos in a folder."""
    for video_file in os.listdir(video_folder):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(video_folder, video_file)
            print(f"Processing {video_file}...")
            analyze_eyebrow_features(video_path, output_folder)
            print(f"Finished processing {video_file}")

# Example usage
if __name__ == "__main__":
    video_folder = "videos/original"
    output_folder = "output_6/original"
    process_multiple_videos(video_folder, output_folder)
