import cv2
import dlib
import numpy as np
import os
import csv
from scipy.spatial import distance as dist
from collections import deque

class EnhancedFeatureExtractor:
    def __init__(self, predictor_path="WIFS2018_In_Ictu_Oculi\dlib_model\shape_predictor_68_face_landmarks.dat"): #required to download from internet or take from other codes
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Landmark indices
        self.RIGHT_EYE = list(range(36, 42))
        self.LEFT_EYE = list(range(42, 48))
        self.RIGHT_EYEBROW = list(range(17, 22))
        self.LEFT_EYEBROW = list(range(22, 27))
        
        # Temporal tracking
        self.velocity_history = deque(maxlen=10)
        self.position_history = deque(maxlen=10)

    def _calculate_jitter(self, current_velocity):
        """Calculate temporal jitter from velocity history"""
        if len(self.velocity_history) < 2:
            return 0.0
            
        # Calculate second derivative of velocity
        diffs = [self.velocity_history[i+1] - self.velocity_history[i] 
                for i in range(len(self.velocity_history)-1)]
        return np.var(diffs)

    def extract_features(self, frame):
        """Extract all required landmarks from frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if not faces:
            return None

        landmarks = self.predictor(gray, faces[0])
        return {
            'right_eyebrow': [(landmarks.part(n).x, landmarks.part(n).y) 
                            for n in self.RIGHT_EYEBROW],
            'left_eyebrow': [(landmarks.part(n).x, landmarks.part(n).y) 
                           for n in self.LEFT_EYEBROW],
        }

    def calculate_frame_features(self, current, prev, time_delta):
        """Calculate all features for a single frame"""
        # 1. Asymmetry calculation
        vert_diffs = [abs(l[1]-r[1]) for l,r in zip(current['left_eyebrow'], current['right_eyebrow'])]
        asymmetry = np.mean(vert_diffs)
        
        # 2. Mean velocity
        if prev:
            right_vel = np.mean([np.linalg.norm(np.array(c)-np.array(p)) / time_delta
                               for c,p in zip(current['right_eyebrow'], prev['right_eyebrow'])])
            left_vel = np.mean([np.linalg.norm(np.array(c)-np.array(p)) / time_delta
                              for c,p in zip(current['left_eyebrow'], prev['left_eyebrow'])])
            mean_velocity = (right_vel + left_vel) / 2
        else:
            mean_velocity = 0.0
        
        # 3. Temporal jitter
        self.velocity_history.append(mean_velocity)
        jitter = self._calculate_jitter(mean_velocity)

        return [asymmetry, mean_velocity, jitter]

    def process_video(self, video_path, output_csv, is_fake=False):
        """Process video and generate dataset rows"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        time_delta = 1 / fps
        
        features = []
        prev_landmarks = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_landmarks = self.extract_features(frame)
            if not current_landmarks:
                continue
                
            frame_features = self.calculate_frame_features(
                current_landmarks, 
                prev_landmarks,
                time_delta
            )
            
            features.extend(frame_features)
            prev_landmarks = current_landmarks
        
        cap.release()
        self._save_features(features, output_csv, is_fake)

    def _save_features(self, features, output_csv, label, row_size=225):
        """Save features with padding and labels"""
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        with open(output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Split into chunks of 225 features (75 frames × 3 features)
            for i in range(0, len(features), row_size):
                chunk = features[i:i+row_size]
                if len(chunk) < row_size:
                    chunk += [0.0] * (row_size - len(chunk))
                chunk.append(label)
                writer.writerow(chunk)

    def process_dataset(self, real_dir, fake_dir, output_file):
        """Batch process videos"""
        if os.path.exists(output_file):
            os.remove(output_file)
            
        # Process real videos
        for vid in self._find_videos(real_dir):
            print(f"Processing REAL: {vid}")
            self.process_video(vid, output_file, False)
            
        # Process fake videos
        for vid in self._find_videos(fake_dir):
            print(f"Processing FAKE: {vid}")
            self.process_video(vid, output_file, True)

    def _find_videos(self, directory):
        """Find supported video files"""
        return [os.path.join(directory, f) for f in os.listdir(directory) 
              if f.lower().endswith(('.mp4', '.avi', '.mov'))]

# Usage
if __name__ == "__main__":
    processor = EnhancedFeatureExtractor()
    processor.process_dataset(
        real_dir="videos/original",
        fake_dir="videos/DeepFakeDetection-1", #modify the dataset as you require
        output_file="part2_4/eyebrow_features_DeepFakeDetection.csv"
    )
