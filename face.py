import cv2
import face_recognition
import numpy as np
import json
import os
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.data_file = "face_data.json"
        
        # Load existing face data
        self.load_face_data()
        
        # Initialize UI
        self.root = Tk()
        self.root.title("Face Recognition System")
        self.root.geometry("800x600")
        
        # Create UI elements
        self.setup_ui()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
    def setup_ui(self):
        # Video frame
        self.video_label = Label(self.root)
        self.video_label.pack(pady=10)
        
        # Register button
        self.register_btn = Button(self.root, text="Register New Face", 
                                 command=self.register_face)
        self.register_btn.pack(pady=10)
        
        # Name entry
        self.name_label = Label(self.root, text="Enter Name:")
        self.name_label.pack()
        self.name_entry = Entry(self.root)
        self.name_entry.pack(pady=5)
        
    def load_face_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.known_face_encodings = [np.array(enc) for enc in data['encodings']]
                self.known_face_names = data['names']
    
    def save_face_data(self):
        data = {
            'encodings': [enc.tolist() for enc in self.known_face_encodings],
            'names': self.known_face_names
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f)
    
    def register_face(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name")
            return
        
        # Capture frame
        ret, frame = self.cap.read()
        if ret:
            # Find faces in frame
            face_locations = face_recognition.face_locations(frame)
            if len(face_locations) == 0:
                messagebox.showerror("Error", "No face detected")
                return
            if len(face_locations) > 1:
                messagebox.showerror("Error", "Multiple faces detected")
                return
            
            # Get face encoding
            face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
            
            # Save face data
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            self.save_face_data()
            
            messagebox.showinfo("Success", f"Face registered for {name}")
            self.name_entry.delete(0, END)
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Find faces in frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            # Loop through each face
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, 
                                                       face_encoding)
                name = "Unknown"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                
                # Draw rectangle and name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), 
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
        self.cap.release()

if __name__ == "__main__":
    # First update environment.yml to include required packages
    face_system = FaceRecognitionSystem()
    face_system.run() 