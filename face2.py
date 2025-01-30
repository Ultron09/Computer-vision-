import cv2
import numpy as np
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import os

class FaceDetectionSystem:
    def __init__(self):
        self.root = Tk()
        self.root.title("Age and Gender Detection System")
        self.root.geometry("800x600")
        
        # Check for model files
        if not self.check_model_files():
            messagebox.showerror("Error", "Model files are missing! Please run download_models.py first.")
            self.root.destroy()
            return
        
        # Load age and gender models
        self.load_age_gender_models()
        
        # Load face cascade
        cascade_path = 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            messagebox.showerror("Error", "Cascade file missing! Please run download_cascade.py first.")
            self.root.destroy()
            return
            
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            messagebox.showerror("Error", "Failed to load face cascade classifier!")
            self.root.destroy()
            return
        
        # Create UI elements
        self.setup_ui()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera!")
            self.root.destroy()
            return

    def check_model_files(self):
        required_files = [
            "age_deploy.prototxt",
            "age_net.caffemodel",
            "gender_deploy.prototxt",
            "gender_net.caffemodel"
        ]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print("Missing files:", missing_files)
            return False
        return True
        
    def load_age_gender_models(self):
        try:
            # Load models
            self.age_net = cv2.dnn.readNet(
                "age_net.caffemodel",
                "age_deploy.prototxt"
            )
            self.gender_net = cv2.dnn.readNet(
                "gender_net.caffemodel",
                "gender_deploy.prototxt"
            )
            
            # Define age ranges with exact years
            self.age_ranges = {
                0: (0, 2),    # Baby
                1: (4, 6),    # Child
                2: (8, 12),   # Youth
                3: (15, 20),  # Teen
                4: (25, 32),  # Young Adult
                5: (38, 43),  # Adult
                6: (48, 53),  # Middle Age
                7: (60, 100)  # Senior
            }
            
            self.gender_list = ['Male', 'Female']
            self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        except Exception as e:
            messagebox.showerror("Error", f"Error loading models: {str(e)}")
            self.root.destroy()
            return
        
    def setup_ui(self):
        # Video frame
        self.video_label = Label(self.root)
        self.video_label.pack(pady=10)
        
    def get_exact_age(self, age_index, confidence):
        age_range = self.age_ranges[age_index]
        min_age, max_age = age_range
        # Calculate a more precise age within the range based on confidence
        exact_age = min_age + (max_age - min_age) * confidence
        return round(exact_age)
        
    def predict_age_gender(self, face_img):
        try:
            # Prepare face image for age/gender prediction
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                        self.MODEL_MEAN_VALUES, swapRB=False)
            
            # Gender prediction
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]
            gender_confidence = float(gender_preds[0].max())
            
            # Age prediction
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age_index = age_preds[0].argmax()
            age_confidence = float(age_preds[0][age_index])
            
            # Get exact age
            exact_age = self.get_exact_age(age_index, age_confidence)
            
            return f"{gender}, {exact_age} years"
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "Unknown"
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Loop through each face
            for (x, y, w, h) in faces:
                # Get face ROI for age/gender prediction
                face_img = frame[y:y+h, x:x+w].copy()
                age_gender = self.predict_age_gender(face_img)
                
                # Draw rectangle and labels
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Display age/gender
                cv2.putText(frame, age_gender, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            # Convert frame for tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img = ImageTk.PhotoImage(image=img)
            self.video_label.img = img
            self.video_label.configure(image=img)
        
        self.root.after(10, self.update_frame)
    
    def run(self):
        self.update_frame()
        self.root.mainloop()
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

if __name__ == "__main__":
    face_system = FaceDetectionSystem()
    face_system.run() 