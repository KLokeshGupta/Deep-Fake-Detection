import dlib
import cv2
import numpy as np
import os
import csv
from pathlib import Path
from collections import deque

# Initialize dlib components
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("WIFS2018_In_Ictu_Oculi\dlib_model\shape_predictor_68_face_landmarks.dat")

# Configuration
WINDOW_SIZE = 90  # 3 seconds at 30 fps
EAR_THRESHOLD = 0.2
MIN_BLINK_FRAMES = 2
MAX_BLINK_FRAMES = 8

def eye_aspect_ratio(eye):
    """Calculate normalized eye aspect ratio"""
    vertical1 = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    vertical2 = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    horizontal = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (vertical1 + vertical2) / (2.0 * horizontal) if horizontal != 0 else 0.0

def analyze_temporal_patterns(ear_history):
    """Extract key temporal features from EAR history"""
    blinks = []
    in_blink = False
    start_frame = 0
    
    # Detect blink intervals
    for i, ear in enumerate(ear_history):
        if ear < EAR_THRESHOLD and not in_blink:
            in_blink = True
            start_frame = i
        elif ear >= EAR_THRESHOLD and in_blink:
            in_blink = False
            if MIN_BLINK_FRAMES <= (i - start_frame) <= MAX_BLINK_FRAMES:
                blinks.append((start_frame, i))
    
    # Calculate temporal features
    features = {}
    if blinks:
        durations = [end-start for start,end in blinks]
        intervals = [blinks[i][0]-blinks[i-1][1] for i in range(1,len(blinks))]
        
        features.update({
            'blink_freq': len(blinks) / (WINDOW_SIZE/30),  # Blinks per second
            'avg_duration': np.mean(durations),
            'avg_interval': np.mean(intervals) if intervals else 0,
            'duration_var': np.var(durations),
            'interval_var': np.var(intervals) if intervals else 0
        })
    else:
        features.update({
            'blink_freq': 0,
            'avg_duration': 0,
            'avg_interval': 0,
            'duration_var': 0,
            'interval_var': 0
        })
    
    return features

def extract_core_features(frame, history):
    """Extract essential eye features with temporal context"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    features = {
        'ear_left': 0.0,
        'ear_right': 0.0,
        'ear_diff': 0.0,
        'blink_freq': 0.0,
        'avg_duration': 0.0,
        'avg_interval': 0.0
    }
    
    if faces:
        landmarks = predictor(gray, faces[0])
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36,42)]
        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42,48)]
        
        ear_right = eye_aspect_ratio(right_eye)
        ear_left = eye_aspect_ratio(left_eye)
        ear_diff = abs(ear_left - ear_right)
        
        # Update history
        history['left'].append(ear_left)
        history['right'].append(ear_right)
        
        # Keep window size consistent
        for eye in ['left', 'right']:
            if len(history[eye]) > WINDOW_SIZE:
                history[eye].popleft()
        
        # Calculate temporal features when enough data
        if len(history['left']) == WINDOW_SIZE:
            temp_left = analyze_temporal_patterns(history['left'])
            temp_right = analyze_temporal_patterns(history['right'])
            
            # Use weighted average of both eyes
            features.update({
                'blink_freq': (temp_left['blink_freq'] + temp_right['blink_freq'])/2,
                'avg_duration': (temp_left['avg_duration'] + temp_right['avg_duration'])/2,
                'avg_interval': (temp_left['avg_interval'] + temp_right['avg_interval'])/2,
                'duration_var': (temp_left['duration_var'] + temp_right['duration_var'])/2,
                'interval_var': (temp_left['interval_var'] + temp_right['interval_var'])/2
            })
        
        features.update({
            'ear_left': ear_left,
            'ear_right': ear_right,
            'ear_diff': ear_diff
        })
    
    return features

def process_video(video_path, output_csv, is_fake=False):
    """Process video into temporal eye feature sequences"""
    cap = cv2.VideoCapture(video_path)
    history = {'left': deque(maxlen=WINDOW_SIZE), 'right': deque(maxlen=WINDOW_SIZE)}
    sequence = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        features = extract_core_features(frame, history)
        sequence.append([
            features['ear_left'],
            features['ear_right'],
            features['ear_diff'],
            features['blink_freq'],
            features['avg_duration'],
            features['avg_interval']
        ])
    
    cap.release()
    
    # Create fixed-length sequences
    seq_length = 75  #224*6=1344 per row+1 class label
    for i in range(0, len(sequence), seq_length):
        chunk = sequence[i:i+seq_length]
        if len(chunk) < seq_length:
            chunk += [[0.0]*6]*(seq_length - len(chunk))
        
        # Flatten and add label
        flat_chunk = [val for sublist in chunk for val in sublist]
        flat_chunk.append("fake" if is_fake else "real")
        
        # Write to CSV
        with open(output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(flat_chunk)

def process_dataset(real_dir, fake_dir, output_file):
    """Batch process video dataset"""


    # Process real videos
    if os.path.exists(real_dir):
        for vid in Path(real_dir).glob('*'):
            if vid.suffix.lower() in ('.mp4', '.avi', '.mov'):
                print(f"Processing {vid}...")
                process_video(str(vid), output_file, False)
                print(f"Finished processing {vid}")

    # Process fake videos
    if os.path.exists(fake_dir):
        for vid in Path(fake_dir).glob('*'):
            if vid.suffix.lower() in ('.mp4', '.avi', '.mov'):
                print(f"Processing {vid}...")
                process_video(str(vid), output_file, True)
                print(f"Finished processing {vid}")

# Example usage
if __name__ == "__main__":
    process_dataset("videos/empty", "videos/DeepFakeDetection-1", "eyeblink_dataset-DeepFakeDetection.csv")
