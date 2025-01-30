import cv2
import numpy as np
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import dlib
from deepface import DeepFace
import mediapipe as mp
import time

class PersonalityAnalyzer:
    def __init__(self):
        # Initialize main window
        self.root = Tk()
        self.root.title("Personality & Mood Analyzer")
        self.root.geometry("1200x800")
        
        # Initialize detectors
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open camera")
            self.root.destroy()
            return
            
        # Personality traits mapping
        self.personality_traits = {
            'openness': ['raised_eyebrows', 'wide_eyes', 'relaxed_mouth'],
            'conscientiousness': ['focused_gaze', 'neutral_expression', 'straight_posture'],
            'extraversion': ['smiling', 'animated_expression', 'open_posture'],
            'agreeableness': ['soft_smile', 'relaxed_eyes', 'head_tilt'],
            'neuroticism': ['tense_expression', 'rapid_blinking', 'furrowed_brow']
        }
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frames
        self.video_frame = Frame(self.root)
        self.video_frame.pack(side=LEFT, padx=10, pady=10)
        
        self.analysis_frame = Frame(self.root)
        self.analysis_frame.pack(side=RIGHT, fill=Y, padx=10, pady=10)
        
        # Video display
        self.video_label = Label(self.video_frame)
        self.video_label.pack()
        
        # Analysis sections
        self.create_analysis_sections()
        
    def create_analysis_sections(self):
        # Emotion section
        emotion_frame = ttk.LabelFrame(self.analysis_frame, text="Emotional State")
        emotion_frame.pack(fill=X, pady=5, padx=5)
        self.emotion_label = Label(emotion_frame, text="Current Emotion: --")
        self.emotion_label.pack(pady=5)
        
        # Personality section
        personality_frame = ttk.LabelFrame(self.analysis_frame, text="Personality Traits")
        personality_frame.pack(fill=X, pady=5, padx=5)
        self.personality_labels = {}
        for trait in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']:
            label = Label(personality_frame, text=f"{trait}: --")
            label.pack(pady=2)
            self.personality_labels[trait.lower()] = label
            
        # Mood section
        mood_frame = ttk.LabelFrame(self.analysis_frame, text="Mood Analysis")
        mood_frame.pack(fill=X, pady=5, padx=5)
        self.mood_label = Label(mood_frame, text="Current Mood: --")
        self.mood_label.pack(pady=5)
        
        # Body language section
        body_frame = ttk.LabelFrame(self.analysis_frame, text="Body Language")
        body_frame.pack(fill=X, pady=5, padx=5)
        self.body_text = Text(body_frame, height=4, width=30)
        self.body_text.pack(pady=5)
        
    def analyze_emotion(self, frame):
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if analysis:
                return analysis[0]['dominant_emotion'], analysis[0]['emotion']
            return "Unknown", {}
        except:
            return "Unknown", {}
            
    def analyze_personality(self, landmarks, emotion_data):
        personality_scores = {
            'openness': 0,
            'conscientiousness': 0,
            'extraversion': 0,
            'agreeableness': 0,
            'neuroticism': 0
        }
        
        # Example analysis based on facial features and emotions
        if emotion_data:
            # Extraversion correlates with positive emotions
            personality_scores['extraversion'] += emotion_data.get('happy', 0) * 0.5
            
            # Neuroticism correlates with negative emotions
            personality_scores['neuroticism'] += (
                emotion_data.get('sad', 0) + 
                emotion_data.get('angry', 0) + 
                emotion_data.get('fear', 0)
            ) * 0.3
            
            # Agreeableness correlates with positive expressions
            personality_scores['agreeableness'] += (
                emotion_data.get('happy', 0) + 
                emotion_data.get('neutral', 0)
            ) * 0.4
        
        return personality_scores
        
    def analyze_body_language(self, landmarks):
        observations = []
        
        # Example body language analysis
        if landmarks is not None:
            # Head position analysis
            head_pose = self.estimate_head_pose(landmarks)
            if head_pose:
                observations.append(f"Head Position: {head_pose}")
            
            # Eye contact analysis
            eye_contact = self.analyze_eye_contact(landmarks)
            if eye_contact:
                observations.append(f"Eye Contact: {eye_contact}")
            
            # Facial tension analysis
            tension = self.analyze_facial_tension(landmarks)
            if tension:
                observations.append(f"Facial Tension: {tension}")
        
        return observations
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            # Detect faces with dlib
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray)
            
            for face in faces:
                # Get landmarks
                landmarks = self.predictor(gray, face)
                landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
                
                # Analyze emotion
                emotion, emotion_data = self.analyze_emotion(frame)
                self.emotion_label.config(text=f"Current Emotion: {emotion.title()}")
                
                # Analyze personality
                personality_scores = self.analyze_personality(landmarks, emotion_data)
                for trait, score in personality_scores.items():
                    self.personality_labels[trait].config(
                        text=f"{trait.title()}: {score:.1f}%")
                
                # Analyze body language
                body_observations = self.analyze_body_language(landmarks)
                self.body_text.delete(1.0, END)
                self.body_text.insert(END, "\n".join(body_observations))
                
                # Draw landmarks and annotations
                self.draw_analysis_visualization(frame, face, landmarks)
            
            # Convert frame for tkinter
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = ImageTk.PhotoImage(image=img)
            self.video_label.img = img
            self.video_label.configure(image=img)
        
        self.root.after(100, self.update_frame)
        
    def draw_analysis_visualization(self, frame, face, landmarks):
        # Draw face rectangle
        cv2.rectangle(frame, (face.left(), face.top()),
                     (face.right(), face.bottom()), (0, 255, 0), 2)
        
        # Draw landmarks
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
    def estimate_head_pose(self, landmarks):
        # Simplified head pose estimation
        try:
            nose_bridge = landmarks[27]
            nose_tip = landmarks[30]
            
            if nose_tip[0] - nose_bridge[0] > 5:
                return "Turned Right"
            elif nose_tip[0] - nose_bridge[0] < -5:
                return "Turned Left"
            else:
                return "Straight"
        except:
            return None
            
    def analyze_eye_contact(self, landmarks):
        try:
            left_eye = np.mean(landmarks[36:42], axis=0)
            right_eye = np.mean(landmarks[42:48], axis=0)
            
            # Simplified eye contact analysis
            if abs(left_eye[1] - right_eye[1]) < 3:
                return "Direct"
            else:
                return "Averted"
        except:
            return None
            
    def analyze_facial_tension(self, landmarks):
        try:
            # Analyze distances between specific landmarks to detect tension
            brow_dist = np.linalg.norm(landmarks[21] - landmarks[22])
            jaw_dist = np.linalg.norm(landmarks[5] - landmarks[11])
            
            if brow_dist < 20:  # Threshold values need adjustment
                return "Tense"
            else:
                return "Relaxed"
        except:
            return None
            
    def run(self):
        self.update_frame()
        self.root.mainloop()
        
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

if __name__ == "__main__":
    analyzer = PersonalityAnalyzer()
    analyzer.run() 