import cv2
import dlib
import numpy as np
import os
import csv
from collections import deque
from pathlib import Path

class ComprehensiveEyeFeatureExtractor:
    def __init__(self, predictor_path="WIFS2018_In_Ictu_Oculi\dlib_model\shape_predictor_68_face_landmarks.dat"):
        """Initialize the feature extractor with all necessary components"""
        # Detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Landmark indices
        self.RIGHT_EYE = list(range(36, 42))
        self.LEFT_EYE = list(range(42, 48))
        self.RIGHT_EYEBROW = list(range(17, 22))
        self.LEFT_EYEBROW = list(range(22, 27))
        
        # Configuration for blink detection
        self.WINDOW_SIZE = 90  # 3 seconds at 30 fps
        self.EAR_THRESHOLD = 0.2
        self.MIN_BLINK_FRAMES = 2
        self.MAX_BLINK_FRAMES = 8
        
        # Temporal tracking
        self.velocity_history = deque(maxlen=10)
        self.ear_history = {'left': deque(maxlen=self.WINDOW_SIZE), 
                           'right': deque(maxlen=self.WINDOW_SIZE)}
        
    def _eye_aspect_ratio(self, eye):
        """Calculate EAR for a single eye"""
        # Convert to numpy arrays for consistency
        eye_points = [np.array(point) for point in eye]
        
        # Vertical distances (avg of two)
        vertical1 = np.linalg.norm(eye_points[1] - eye_points[5])
        vertical2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal distance
        horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Return the ratio
        return (vertical1 + vertical2) / (2.0 * horizontal) if horizontal > 0 else 0.0
    
    def _calculate_jitter(self, current_velocity):
        """Calculate temporal jitter from velocity history"""
        if len(self.velocity_history) < 2:
            return 0.0
            
        # Calculate second derivative of velocity (acceleration changes)
        diffs = [self.velocity_history[i+1] - self.velocity_history[i]
                for i in range(len(self.velocity_history)-1)]
                
        return np.var(diffs)
    
    def _analyze_temporal_patterns(self, ear_history):
        """Extract key temporal features from EAR history"""
        blinks = []
        in_blink = False
        start_frame = 0
        
        # Detect blink intervals
        for i, ear in enumerate(ear_history):
            if ear < self.EAR_THRESHOLD and not in_blink:
                in_blink = True
                start_frame = i
            elif ear >= self.EAR_THRESHOLD and in_blink:
                in_blink = False
                if self.MIN_BLINK_FRAMES <= (i - start_frame) <= self.MAX_BLINK_FRAMES:
                    blinks.append((start_frame, i))
        
        # Calculate temporal features
        features = {}
        if blinks:
            durations = [end-start for start,end in blinks]
            intervals = [blinks[i][0]-blinks[i-1][1] for i in range(1,len(blinks))]
            
            features.update({
                'blink_freq': len(blinks) / (self.WINDOW_SIZE/30),  # Blinks per second
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
    
    def extract_landmarks(self, frame):
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
            'right_eye': [(landmarks.part(n).x, landmarks.part(n).y) 
                         for n in self.RIGHT_EYE],
            'left_eye': [(landmarks.part(n).x, landmarks.part(n).y) 
                        for n in self.LEFT_EYE]
        }
    
    def calculate_frame_features(self, current, prev, time_delta):
        """Calculate all features for a single frame"""
        # Skip if no landmarks
        if not current:
            return None
            
        # 1. Basic EAR values
        right_ear = self._eye_aspect_ratio(current['right_eye'])
        left_ear = self._eye_aspect_ratio(current['left_eye'])
        mean_ear = (right_ear + left_ear) / 2
        ear_diff = abs(left_ear - right_ear)
        
        # Update EAR history
        self.ear_history['left'].append(left_ear)
        self.ear_history['right'].append(right_ear)
        
        # 2. Asymmetry calculation
        vert_diffs = [abs(l[1]-r[1]) for l,r in zip(current['left_eyebrow'], current['right_eyebrow'])]
        asymmetry = np.mean(vert_diffs)
        
        # 3. Mean velocity (only when we have previous frame)
        if prev:
            right_vel = np.mean([np.linalg.norm(np.array(c)-np.array(p)) / time_delta
                              for c,p in zip(current['right_eyebrow'], prev['right_eyebrow'])])
            left_vel = np.mean([np.linalg.norm(np.array(c)-np.array(p)) / time_delta
                             for c,p in zip(current['left_eyebrow'], prev['left_eyebrow'])])
            mean_velocity = (right_vel + left_vel) / 2
        else:
            mean_velocity = 0.0
            
        # 4. Temporal jitter
        self.velocity_history.append(mean_velocity)
        jitter = self._calculate_jitter(mean_velocity)
        
        # 5. Blink temporal features (if enough history data)
        if len(self.ear_history['left']) == self.WINDOW_SIZE and len(self.ear_history['right']) == self.WINDOW_SIZE:
            temp_left = self._analyze_temporal_patterns(self.ear_history['left'])
            temp_right = self._analyze_temporal_patterns(self.ear_history['right'])
            
            # Use weighted average of both eyes
            blink_freq = (temp_left['blink_freq'] + temp_right['blink_freq'])/2
            avg_duration = (temp_left['avg_duration'] + temp_right['avg_duration'])/2
            avg_interval = (temp_left['avg_interval'] + temp_right['avg_interval'])/2
        else:
            blink_freq = 0
            avg_duration = 0
            avg_interval = 0
            
        # Return all features
        return {
            'asymmetry': asymmetry,
            'mean_velocity': mean_velocity,
            'jitter': jitter,
            'mean_ear': mean_ear,
            'blink_freq': blink_freq,
            'avg_duration': avg_duration,
            'avg_interval': avg_interval
        }
    
    def process_video(self, video_path, output_csv, is_fake=False):
        """Process video and extract all features"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        time_delta = 1 / fps if fps > 0 else 1/30  # Default to 30 FPS if not detected
        
        # Reset temporal tracking
        self.velocity_history.clear()
        self.ear_history = {'left': deque(maxlen=self.WINDOW_SIZE), 
                           'right': deque(maxlen=self.WINDOW_SIZE)}
        
        feature_sequence = []
        prev_landmarks = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            current_landmarks = self.extract_landmarks(frame)
            if not current_landmarks:
                continue
                
            features = self.calculate_frame_features(
                current_landmarks,
                prev_landmarks,
                time_delta
            )
            
            if features:
                # Add features to sequence
                feature_sequence.append([
                    features['asymmetry'], 
                    features['mean_velocity'],
                    features['jitter'],
                    features['mean_ear'],
                    features['blink_freq'],
                    features['avg_duration'],
                    features['avg_interval']
                ])
                
            prev_landmarks = current_landmarks
            
        cap.release()
        
        # Save features to CSV
        self._save_features(feature_sequence, output_csv, is_fake)
        
    def _save_features(self, features, output_csv, is_fake, row_size=224):
        """Save features with padding and labels"""
        # Make directory if it doesn't exist
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        # No features extracted
        if not features:
            return
            
        # Process feature sequences
        for i in range(0, len(features), row_size):
            chunk = features[i:i+row_size]
            
            # If chunk is smaller than row_size, pad with zeros
            if len(chunk) < row_size:
                chunk += [[0.0] * len(features[0])] * (row_size - len(chunk))
                
            # Flatten the chunk
            flat_chunk = []
            for frame_features in chunk:
                flat_chunk.extend(frame_features)
                
            # Add label
            flat_chunk.append("fake" if is_fake else "real")
            
            # Write to CSV
            with open(output_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(flat_chunk)
    
    def process_dataset(self, real_dir, fake_dir, output_file):
        """Batch process video dataset"""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Ensure fresh file
        if os.path.exists(output_file):
            os.remove(output_file)
            
        # Process real videos
        if os.path.exists(real_dir):
            for vid in self._find_videos(real_dir):
                print(f"Processing REAL: {vid}")
                self.process_video(vid, output_file, False)
            
        # Process fake videos
        if os.path.exists(fake_dir):  
            for vid in self._find_videos(fake_dir):
                print(f"Processing FAKE: {vid}")
                self.process_video(vid, output_file, True)
            
    def _find_videos(self, directory):
        """Find supported video files"""
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return []
            
        return [os.path.join(directory, f) for f in os.listdir(directory)
               if f.lower().endswith(('.mp4', '.avi', '.mov'))]

# Example usage
if __name__ == "__main__":
    extractor = ComprehensiveEyeFeatureExtractor()
    extractor.process_dataset(
        real_dir="videos/empty",
        fake_dir="videos/DeepFakeDetection-1",
        output_file="part2_4/comprehensive_features-DeepFakeDetection.csv"
    )
