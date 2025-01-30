import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import mediapipe as mp
from deepface import DeepFace

class PersonAnalyzer:
    def __init__(self):
        print("Starting initialization...")
        
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("Person Analyzer")
        self.root.geometry("1200x700")
        
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize camera
        print("Opening camera...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("ERROR: Could not open camera!")
            self.root.destroy()
            return
            
        print("Camera opened successfully!")
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Create frames
        self.video_frame = ttk.Frame(self.root)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.analysis_frame = ttk.Frame(self.root)
        self.analysis_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # Video display
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()
        
        # Analysis display
        self.emotion_label = ttk.Label(self.analysis_frame, text="Emotion: --")
        self.emotion_label.pack(pady=5)
        
        self.posture_label = ttk.Label(self.analysis_frame, text="Posture: --")
        self.posture_label.pack(pady=5)
        
        self.status_label = ttk.Label(self.analysis_frame, text="Status: Running")
        self.status_label.pack(pady=5)
        
    def analyze_emotion(self, frame):
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if analysis:
                return analysis[0]['dominant_emotion']
            return "Unknown"
        except:
            return "Unknown"
            
    def analyze_posture(self, results):
        if results.pose_landmarks:
            # Basic posture analysis
            landmarks = results.pose_landmarks.landmark
            
            # Check if shoulders are level
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            if abs(left_shoulder.y - right_shoulder.y) < 0.05:
                return "Good posture"
            else:
                return "Poor posture"
        return "No posture detected"
        
    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                # Process frame with MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(rgb_frame)
                
                # Draw pose landmarks
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_holistic.POSE_CONNECTIONS
                    )
                
                # Analyze emotion and posture
                emotion = self.analyze_emotion(frame)
                posture = self.analyze_posture(results)
                
                # Update labels
                self.emotion_label.config(text=f"Emotion: {emotion}")
                self.posture_label.config(text=f"Posture: {posture}")
                
                # Convert frame for display
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
            # Schedule next update
            self.root.after(100, self.update_frame)
            
        except Exception as e:
            print(f"Error in update_frame: {e}")
            self.status_label.config(text=f"Error: {str(e)}")
            
    def run(self):
        print("Starting main loop...")
        self.update_frame()
        self.root.mainloop()
        
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'holistic'):
            self.holistic.close()

if __name__ == "__main__":
    app = PersonAnalyzer()
    app.run() 