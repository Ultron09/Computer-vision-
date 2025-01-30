import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
import dlib
from scipy.spatial import distance

class FaceRatingSystem:
    def __init__(self):
        # Initialize main window
        self.root = Tk()
        self.root.title("Face Attractiveness Calculator")
        self.root.geometry("1000x800")
        
        # Initialize face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Define attractiveness categories
        self.categories = {
            (0, 40): ("Below Average", "red"),
            (40, 60): ("Average", "orange"),
            (60, 80): ("Attractive", "green"),
            (80, 100): ("Very Attractive", "purple")
        }
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open camera")
            self.root.destroy()
            return
            
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Create frames
        self.video_frame = Frame(self.root)
        self.video_frame.pack(side=LEFT, padx=10, pady=10)
        
        self.info_frame = Frame(self.root)
        self.info_frame.pack(side=RIGHT, padx=10, pady=10)
        
        # Video display
        self.video_label = Label(self.video_frame)
        self.video_label.pack()
        
        # Rating display
        self.rating_label = Label(self.info_frame, 
                                text="Attractiveness Score: --",
                                font=('Arial', 16, 'bold'))
        self.rating_label.pack(pady=10)
        
        # Category display
        self.category_label = Label(self.info_frame,
                                  text="Category: --",
                                  font=('Arial', 14))
        self.category_label.pack(pady=5)
        
        # Features display
        self.features_frame = Frame(self.info_frame)
        self.features_frame.pack(pady=10)
        
        self.feature_labels = {}
        features = ['Symmetry', 'Proportions', 'Eye Distance', 'Facial Ratio']
        for feature in features:
            label = Label(self.features_frame, 
                         text=f"{feature}: --",
                         font=('Arial', 12))
            label.pack(pady=2)
            self.feature_labels[feature] = label
        
    def get_landmarks(self, gray, face):
        landmarks = self.predictor(gray, face)
        return np.array([[p.x, p.y] for p in landmarks.parts()])
        
    def calculate_attractiveness(self, landmarks):
        try:
            # Calculate facial features
            
            # 1. Symmetry (compare left and right side distances)
            left_eye = np.mean(landmarks[36:42], axis=0)
            right_eye = np.mean(landmarks[42:48], axis=0)
            nose_tip = landmarks[33]
            
            symmetry_score = 100 - min(100, abs(
                distance.euclidean(left_eye, nose_tip) -
                distance.euclidean(right_eye, nose_tip)
            ) * 100)
            
            # 2. Golden ratio of face length to width
            face_width = distance.euclidean(landmarks[0], landmarks[16])
            face_height = distance.euclidean(landmarks[8], landmarks[27])
            golden_ratio = 1.618
            proportion_score = 100 - min(100, abs(
                (face_height / face_width) - golden_ratio
            ) * 50)
            
            # 3. Eye distance
            eye_distance = distance.euclidean(left_eye, right_eye)
            eye_width = distance.euclidean(landmarks[36], landmarks[39])
            eye_ratio_score = 100 - min(100, abs(
                (eye_distance / eye_width) - 2.5
            ) * 50)
            
            # 4. Facial third proportions
            upper_third = distance.euclidean(landmarks[27], landmarks[30])
            middle_third = distance.euclidean(landmarks[30], landmarks[33])
            lower_third = distance.euclidean(landmarks[33], landmarks[8])
            
            thirds_avg = np.mean([upper_third, middle_third, lower_third])
            facial_ratio_score = 100 - min(100, (
                abs(upper_third - thirds_avg) +
                abs(middle_third - thirds_avg) +
                abs(lower_third - thirds_avg)
            ) * 0.5)
            
            # Update feature labels
            scores = {
                'Symmetry': symmetry_score,
                'Proportions': proportion_score,
                'Eye Distance': eye_ratio_score,
                'Facial Ratio': facial_ratio_score
            }
            
            for feature, score in scores.items():
                self.feature_labels[feature].config(
                    text=f"{feature}: {score:.1f}/100")
            
            # Calculate overall score (weighted average)
            weights = {
                'Symmetry': 0.3,
                'Proportions': 0.3,
                'Eye Distance': 0.2,
                'Facial Ratio': 0.2
            }
            
            overall_score = sum(scores[f] * weights[f] for f in scores)
            
            return overall_score, scores
            
        except Exception as e:
            print(f"Error calculating attractiveness: {e}")
            return 0, {}
            
    def get_category(self, score):
        for (min_score, max_score), (category, color) in self.categories.items():
            if min_score <= score < max_score:
                return category, color
        return "Unknown", "black"
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray)
            
            for face in faces:
                # Get facial landmarks
                landmarks = self.get_landmarks(gray, face)
                
                # Calculate attractiveness
                score, feature_scores = self.calculate_attractiveness(landmarks)
                
                # Get category
                category, color = self.get_category(score)
                
                # Update labels
                self.rating_label.config(
                    text=f"Attractiveness Score: {score:.1f}/100")
                self.category_label.config(
                    text=f"Category: {category}",
                    fg=color)
                
                # Draw rectangle and score
                cv2.rectangle(frame, (face.left(), face.top()),
                            (face.right(), face.bottom()), (0, 255, 0), 2)
                
                # Draw landmarks
                for (x, y) in landmarks:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                # Draw score above face
                cv2.putText(frame, f"Score: {score:.1f}",
                          (face.left(), face.top() - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Convert frame for tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img = ImageTk.PhotoImage(image=img)
            
            # Update video label
            self.video_label.img = img
            self.video_label.configure(image=img)
        
        # Schedule next update
        self.root.after(10, self.update_frame)
        
    def run(self):
        self.update_frame()
        self.root.mainloop()
        
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

if __name__ == "__main__":
    app = FaceRatingSystem()
    app.run() 