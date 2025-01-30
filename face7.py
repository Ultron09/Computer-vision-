import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
import dlib
from deepface import DeepFace
import mediapipe as mp
import time
from datetime import datetime
import threading
from scipy.spatial import distance
import math
from collections import Counter
import os
import logging
import tensorflow as tf

class BehaviorAnalyzer:
    def __init__(self):
        try:
            # Initialize main window
            self.root = Tk()
            self.root.title("Advanced Behavioral Analysis System")
            self.root.geometry("1400x900")
            
            # Initialize tracking variables
            self.eye_contact_history = []
            self.confidence_indicators = []
            self.micro_expressions = []
            self.emotion_history = []
            self.emotion_intensities = []
            self.posture_history = []
            self.gesture_history = []
            self.attention_scores = []
            self.stress_scores = []
            self.previous_hand_positions = None
            self.previous_brow_positions = None
            self.previous_mouth_positions = None
            self.previous_eye_positions = None
            
            # Initialize detectors
            self.face_detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Setup camera
            self.setup_camera()
            print("Camera initialized successfully")
            
            # Setup UI
            self.setup_ui()
            print("UI setup completed successfully")
            
            # Analysis state
            self.is_analyzing = True
            self.start_time = time.time()
            
            print("Initialization completed successfully")
            print("Analysis started - Press the Stop button to end and generate report")
            
        except Exception as e:
            print(f"Initialization Error: {str(e)}")
            if hasattr(self, 'root'):
                self.root.destroy()
            raise

    def setup_camera(self):
        """Initialize and setup camera"""
        try:
            # Try different camera indices
            for i in range(2):  # Try first two camera indices
                self.cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow on Windows
                if self.cap.isOpened():
                    # Test reading a frame
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"Camera initialized successfully on index {i}")
                        
                        # Set camera properties
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        # Store camera index
                        self.camera_index = i
                        return
                    else:
                        self.cap.release()
            
            raise Exception("No working camera found")
            
        except Exception as e:
            print(f"Camera setup error: {str(e)}")
            raise

    def setup_ui(self):
        """Setup the user interface"""
        try:
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
            
            # Stop button
            self.stop_button = Button(self.root, text="Stop Analysis", 
                                    command=self.stop_analysis,
                                    bg='red', fg='white',
                                    font=('Arial', 12, 'bold'))
            self.stop_button.pack(pady=5)
            
            print("UI setup completed successfully")
            
        except Exception as e:
            print(f"Error setting up UI: {str(e)}")
            raise

    def create_analysis_sections(self):
        """Create sections for different types of analysis"""
        try:
            sections = [
                ("Emotional State", "emotion_labels"),
                ("Body Language", "posture_labels"),
                ("Mental State", "mental_health_labels"),
                ("Attention Patterns", "attention_labels"),
                ("Stress Indicators", "stress_labels")
            ]
            
            for title, attr_name in sections:
                frame = LabelFrame(self.analysis_frame, text=title, 
                                 font=('Arial', 12, 'bold'))
                frame.pack(fill="x", padx=5, pady=5)
                setattr(self, attr_name, {})
                
                if title == "Emotional State":
                    for emotion in ['Happy', 'Sad', 'Angry', 'Fearful', 
                                  'Surprised', 'Disgusted', 'Neutral']:
                        label = Label(frame, text=f"{emotion}: 0%", 
                                    font=('Arial', 10))
                        label.pack(anchor="w", padx=5, pady=2)
                        getattr(self, attr_name)[emotion.lower()] = label
                else:
                    label = Label(frame, text="Analyzing...", 
                                font=('Arial', 10))
                    label.pack(anchor="w", padx=5, pady=2)
                    getattr(self, attr_name)['main'] = label
            
            print("Analysis sections created successfully")
            
        except Exception as e:
            print(f"Error creating analysis sections: {str(e)}")
            raise

    def collect_baseline(self):
        """Collect baseline behavioral data for first 5 seconds"""
        self.baseline_duration = 5
        self.baseline_start_time = time.time()
        self.baseline_data = {
            'emotions': [],
            'posture': [],
            'gestures': [],
            'facial_movements': [],
            'eye_movements': []
        } 

    def collect_baseline_data(self, landmarks, holistic_results):
        """Collect baseline behavioral data"""
        try:
            # Analyze facial features
            facial_data = self.analyze_facial_landmarks(landmarks)
            if facial_data:
                self.baseline_data['facial_movements'] = self.baseline_data.get('facial_movements', [])
                self.baseline_data['facial_movements'].append(facial_data)
            
            # Analyze posture
            if holistic_results.pose_landmarks:
                posture_data = self.analyze_posture(holistic_results.pose_landmarks)
                self.baseline_data['posture'] = self.baseline_data.get('posture', [])
                self.baseline_data['posture'].append(posture_data)
            
            # Analyze eye movements
            eye_data = self.analyze_eye_patterns(landmarks)
            if eye_data:
                self.baseline_data['eye_movements'] = self.baseline_data.get('eye_movements', [])
                self.baseline_data['eye_movements'].append(eye_data)
            
            print(f"Collecting baseline data... {int(time.time() - self.baseline_start_time)}s")
            
        except Exception as e:
            print(f"Error collecting baseline data: {str(e)}")

    def analyze_posture(self, pose_landmarks):
        """Analyze posture from pose landmarks"""
        try:
            # Convert landmarks to numpy array
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark])
            
            # Calculate key posture metrics
            shoulder_alignment = self.calculate_shoulder_alignment(landmarks)
            head_position = self.calculate_head_position(landmarks)
            spine_angle = self.calculate_spine_angle(landmarks)
            
            return {
                'shoulder_alignment': shoulder_alignment,
                'head_position': head_position,
                'spine_angle': spine_angle,
                'is_straight': spine_angle > 0.8
            }
        except Exception as e:
            print(f"Error analyzing posture: {str(e)}")
            return None

    def calculate_shoulder_alignment(self, landmarks):
        """Calculate shoulder alignment score"""
        try:
            left_shoulder = landmarks[11]  # Left shoulder landmark
            right_shoulder = landmarks[12]  # Right shoulder landmark
            
            # Calculate alignment (0 = perfectly aligned, 1 = maximum misalignment)
            alignment = abs(left_shoulder[1] - right_shoulder[1])
            return 1 - min(alignment * 10, 1)  # Normalize to 0-1 range
        except Exception as e:
            print(f"Error calculating shoulder alignment: {str(e)}")
            return 0.5

    def calculate_head_position(self, landmarks):
        """Calculate head position relative to shoulders"""
        try:
            nose = landmarks[0]  # Nose landmark
            shoulder_center = (landmarks[11] + landmarks[12]) / 2  # Mid-point between shoulders
            
            # Calculate relative position (0 = forward, 1 = upright)
            forward_tilt = abs(nose[2] - shoulder_center[2])
            return 1 - min(forward_tilt * 2, 1)  # Normalize to 0-1 range
        except Exception as e:
            print(f"Error calculating head position: {str(e)}")
            return 0.5

    def calculate_spine_angle(self, landmarks):
        """Calculate spine angle from vertical"""
        try:
            shoulder_center = (landmarks[11] + landmarks[12]) / 2
            hip_center = (landmarks[23] + landmarks[24]) / 2
            
            # Calculate angle from vertical
            angle = abs(math.atan2(shoulder_center[0] - hip_center[0],
                                  shoulder_center[1] - hip_center[1]))
            return 1 - min(angle / (math.pi/4), 1)  # Normalize to 0-1 range
        except Exception as e:
            print(f"Error calculating spine angle: {str(e)}")
            return 0.5

    def draw_analysis_visualization(self, frame, face, landmarks, holistic_results):
        """Draw visual indicators for analysis"""
        try:
            # Draw face rectangle
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw facial landmarks
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            
            # Draw pose landmarks if available
            if holistic_results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    holistic_results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS
                )
            
        except Exception as e:
            print(f"Error drawing visualization: {str(e)}")

    def update_ui(self, frame, remaining_time):
        """Update UI elements"""
        try:
            # Update timer
            self.timer_label.config(text=f"Time remaining: {int(remaining_time)}s")
            
            # Convert frame for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = ImageTk.PhotoImage(image=img)
            self.video_label.img = img
            self.video_label.configure(image=img)
            
        except Exception as e:
            print(f"Error updating UI: {str(e)}")

    def analyze_facial_landmarks(self, landmarks):
        """Analyze facial features and their movements"""
        try:
            facial_data = {
                'asymmetry': self.calculate_facial_asymmetry(landmarks),
                'muscle_tension': self.analyze_muscle_tension(landmarks),
                'micro_movements': self.detect_micro_movements(landmarks),
                'eye_patterns': self.analyze_eye_patterns(landmarks),
                'mouth_patterns': self.analyze_mouth_patterns(landmarks)
            }
            return facial_data
        except Exception as e:
            print(f"Error in facial landmark analysis: {e}")
            return None

    def calculate_facial_asymmetry(self, landmarks):
        """Calculate facial asymmetry score"""
        try:
            # Get midline of face
            nose_bridge = landmarks[27]
            
            # Compare left and right side features
            asymmetry_scores = {
                'eyes': self.compare_eye_symmetry(landmarks),
                'eyebrows': self.compare_eyebrow_symmetry(landmarks),
                'mouth': self.compare_mouth_symmetry(landmarks),
                'cheeks': self.compare_cheek_symmetry(landmarks)
            }
            
            # Weight and combine scores
            weights = {'eyes': 0.3, 'eyebrows': 0.2, 'mouth': 0.3, 'cheeks': 0.2}
            total_asymmetry = sum(score * weights[feature] 
                                for feature, score in asymmetry_scores.items())
            
            return total_asymmetry
        except Exception as e:
            print(f"Error calculating facial asymmetry: {e}")
            return 0

    def analyze_muscle_tension(self, landmarks):
        """Analyze facial muscle tension patterns"""
        try:
            tension_indicators = {
                'forehead': self.measure_forehead_tension(landmarks),
                'jaw': self.measure_jaw_tension(landmarks),
                'eyes': self.measure_eye_tension(landmarks),
                'mouth': self.measure_mouth_tension(landmarks)
            }
            
            # Calculate overall tension score
            overall_tension = sum(tension_indicators.values()) / len(tension_indicators)
            
            return {
                'score': overall_tension,
                'areas': tension_indicators
            }
        except Exception as e:
            print(f"Error analyzing muscle tension: {e}")
            return None

    def detect_micro_movements(self, landmarks, threshold=0.1):
        """Detect and analyze micro movements in facial features"""
        try:
            if not hasattr(self, 'previous_landmarks'):
                self.previous_landmarks = landmarks
                return None
            
            micro_movements = []
            
            # Check each landmark for subtle movements
            for i, (curr, prev) in enumerate(zip(landmarks, self.previous_landmarks)):
                movement = distance.euclidean(curr, prev)
                if 0 < movement < threshold:
                    feature = self.get_feature_name(i)
                    micro_movements.append((feature, movement))
            
            self.previous_landmarks = landmarks
            return micro_movements
        except Exception as e:
            print(f"Error detecting micro movements: {e}")
            return None

    def analyze_eye_patterns(self, landmarks):
        """Analyze detailed eye movement patterns"""
        try:
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            
            eye_data = {
                'blink_rate': self.calculate_blink_rate(left_eye, right_eye),
                'gaze_direction': self.estimate_gaze_direction(left_eye, right_eye),
                'eye_openness': self.measure_eye_openness(left_eye, right_eye),
                'pupil_dilation': self.estimate_pupil_dilation(left_eye, right_eye)
            }
            
            return eye_data
        except Exception as e:
            print(f"Error analyzing eye patterns: {e}")
            return None

    def analyze_breathing_patterns(self, landmarks):
        """Analyze breathing patterns through chest movement"""
        try:
            if self.holistic.pose_landmarks:
                shoulders = self.get_shoulder_positions()
                chest_movement = self.calculate_chest_movement(shoulders)
                
                breathing_data = {
                    'rate': self.calculate_breathing_rate(chest_movement),
                    'depth': self.calculate_breathing_depth(chest_movement),
                    'regularity': self.analyze_breathing_regularity(chest_movement)
                }
                
                return breathing_data
        except Exception as e:
            print(f"Error analyzing breathing patterns: {e}")
            return None

    def analyze_emotional_leakage(self, facial_data, emotion_data):
        """Detect inconsistencies between expressed and leaked emotions"""
        try:
            expressed_emotion = emotion_data.get('dominant_emotion', None)
            micro_expressions = facial_data.get('micro_movements', [])
            
            leakage_indicators = []
            
            # Check for contradictory micro-expressions
            if expressed_emotion and micro_expressions:
                for feature, movement in micro_expressions:
                    if self.is_contradictory_movement(feature, movement, expressed_emotion):
                        leakage_indicators.append({
                            'type': 'micro_expression',
                            'feature': feature,
                            'expressed': expressed_emotion,
                            'leaked': self.interpret_micro_movement(feature, movement)
                        })
            
            return leakage_indicators
        except Exception as e:
            print(f"Error analyzing emotional leakage: {e}")
            return None

    def analyze_personality_indicators(self, facial_data, emotion_data, posture_data):
        """Analyze personality traits based on behavioral patterns"""
        try:
            personality_scores = {trait: [] for trait in self.personality_traits}
            
            # Analyze openness
            personality_scores['openness'].extend([
                self.analyze_expression_variety(emotion_data),
                self.analyze_gestural_expressiveness(posture_data),
                self.analyze_emotional_responsiveness(emotion_data)
            ])
            
            # Analyze conscientiousness
            personality_scores['conscientiousness'].extend([
                self.analyze_movement_control(facial_data),
                self.analyze_attention_patterns(facial_data),
                self.analyze_behavioral_consistency(posture_data)
            ])
            
            # Analyze extraversion
            personality_scores['extraversion'].extend([
                self.analyze_emotional_expressiveness(emotion_data),
                self.analyze_social_engagement(facial_data),
                self.analyze_energy_level(posture_data)
            ])
            
            # Calculate final scores
            final_scores = {
                trait: sum(scores) / len(scores) 
                for trait, scores in personality_scores.items() 
                if scores
            }
            
            return final_scores
        except Exception as e:
            print(f"Error analyzing personality indicators: {e}")
            return None

    def analyze_deception_indicators(self, facial_data, emotion_data, baseline_data):
        """Analyze potential deception indicators"""
        try:
            deception_scores = {
                'baseline_deviation': self.calculate_baseline_deviation(facial_data, baseline_data),
                'emotional_inconsistency': self.analyze_emotional_consistency(emotion_data),
                'behavioral_stress': self.analyze_stress_indicators(facial_data),
                'cognitive_load': self.analyze_cognitive_load_indicators(facial_data),
                'truthfulness_probability': self.calculate_truthfulness_probability(facial_data)
            }
            
            return deception_scores
        except Exception as e:
            print(f"Error analyzing deception indicators: {e}")
            return None

    def update_frame(self):
        """Process video frame and update analysis in real-time"""
        if not self.is_analyzing:
            return
            
        try:
            # Read frame
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                print("Warning: Invalid frame received")
                self.root.after(10, self.update_frame)
                return
            
            # Ensure frame is valid
            if frame.shape[0] == 0 or frame.shape[1] == 0:
                print("Warning: Invalid frame dimensions")
                self.root.after(10, self.update_frame)
                return
            
            # Calculate elapsed time
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # Update elapsed time display
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            self.time_label.config(text=f"Elapsed Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # Create a copy of the frame for processing
            process_frame = frame.copy()
            
            # Process frame
            try:
                rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                results = self.process_frame(process_frame, rgb_frame, current_time)
            except cv2.error as e:
                print(f"OpenCV error during processing: {str(e)}")
                results = {}
            
            # Update video display
            try:
                self.update_video_display(frame)
            except Exception as e:
                print(f"Error updating video display: {str(e)}")
            
            # Update analysis display
            if results:
                try:
                    self.update_analysis_display(results)
                except Exception as e:
                    print(f"Error updating analysis display: {str(e)}")
            
            # Schedule next frame
            self.root.after(10, self.update_frame)
            
        except Exception as e:
            print(f"Error in frame processing: {str(e)}")
            self.root.after(10, self.update_frame)

    def update_video_display(self, frame):
        """Update the video display"""
        try:
            if frame is None or frame.size == 0:
                return
            
            # Make a copy for display
            display_frame = frame.copy()
            
            # Resize frame for display
            display_width = 640
            display_height = 480
            
            # Ensure proper aspect ratio
            aspect_ratio = frame.shape[1] / frame.shape[0]
            display_height = int(display_width / aspect_ratio)
            
            try:
                display_frame = cv2.resize(display_frame, (display_width, display_height))
            except cv2.error as e:
                print(f"Error resizing frame: {str(e)}")
                return
            
            # Convert frame for tkinter
            try:
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = ImageTk.PhotoImage(image=img)
                
                # Update label
                self.video_label.img = img
                self.video_label.configure(image=img)
            except Exception as e:
                print(f"Error converting frame for display: {str(e)}")
            
        except Exception as e:
            print(f"Error in video display update: {str(e)}")

    def update_analysis_display(self, results):
        """Update the analysis display with results"""
        try:
            # Update emotion labels
            if 'emotion' in results:
                emotions = results['emotion']['emotion']
                for emotion, score in emotions.items():
                    if emotion in self.emotion_labels:
                        self.emotion_labels[emotion].config(
                            text=f"{emotion.title()}: {score:.1f}%"
                        )
            
            # Update other analysis sections
            for section in ['personality', 'posture', 'mental_health', 
                           'attention', 'stress']:
                if section in results:
                    label_dict = getattr(self, f"{section}_labels")
                    if 'main' in label_dict:
                        label_dict['main'].config(
                            text=str(results[section])
                        )
                    
        except Exception as e:
            print(f"Error updating analysis display: {str(e)}")

    def process_frame(self, frame, rgb_frame, current_time):
        """Process a single frame and return analysis results"""
        results = {}
        
        try:
            # Process with MediaPipe
            holistic_results = self.holistic.process(rgb_frame)
            face_mesh_results = self.face_mesh.process(rgb_frame)
            
            # Detect faces with dlib
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray)
            
            if len(faces) > 0:
                # Analyze emotions with DeepFace
                emotion_data = DeepFace.analyze(frame, actions=['emotion'], 
                                              enforce_detection=False)
                if emotion_data:
                    self.emotion_history.append(emotion_data[0]['dominant_emotion'])
                    self.emotion_intensities.append(emotion_data[0]['emotion'])
                    results['emotion'] = emotion_data[0]
                
                # Analyze posture
                if holistic_results.pose_landmarks:
                    posture_data = self.analyze_posture(holistic_results.pose_landmarks)
                    self.posture_history.append(posture_data)
                    results['posture'] = posture_data
                
                # Analyze gestures
                gestures = self.analyze_gestures(holistic_results)
                if gestures:
                    self.gesture_history.extend(gestures)
                    results['gestures'] = gestures
                
                # Analyze micro-expressions
                for face in faces:
                    landmarks = self.predictor(gray, face)
                    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
                    micro_expr = self.analyze_micro_expressions(landmarks)
                    if micro_expr:
                        self.micro_expressions.append(micro_expr)
                        results['micro_expressions'] = micro_expr
                
                # Analyze confidence indicators
                confidence_score = self.analyze_confidence(holistic_results, landmarks)
                self.confidence_indicators.append(confidence_score)
                results['confidence'] = confidence_score
                
                # Update eye contact tracking
                if face_mesh_results.multi_face_landmarks:
                    eye_contact = self.analyze_eye_contact(face_mesh_results.multi_face_landmarks[0])
                    self.eye_contact_history.append(eye_contact)
                    results['eye_contact'] = eye_contact
            
            # Update data points
            self.update_data_points(results, current_time)
            
            return results
            
        except Exception as e:
            print(f"Error in frame processing: {str(e)}")
            return {}

    def analyze_gestures(self, results):
        """Analyze gestures and body language"""
        gestures = []
        
        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Check for crossed arms
                left_wrist = landmarks[15]
                right_wrist = landmarks[16]
                if abs(left_wrist.x - right_wrist.x) < 0.1:
                    gestures.append("crossed_arms")
                
                # Check for hand touching face
                nose = landmarks[0]
                left_hand = landmarks[19]
                right_hand = landmarks[20]
                if (abs(left_hand.x - nose.x) < 0.1 or 
                    abs(right_hand.x - nose.x) < 0.1):
                    gestures.append("touching_face")
                
                # Check for leaning forward
                head_pos = landmarks[0].z
                if head_pos < -0.15:  # Threshold for forward leaning
                    gestures.append("leaning_forward")
                
                # Check for hand fidgeting
                if self.detect_hand_movement(results):
                    gestures.append("hand_fidgeting")
            
            return gestures
            
        except Exception as e:
            print(f"Error analyzing gestures: {str(e)}")
            return []

    def analyze_micro_expressions(self, landmarks):
        """Analyze micro-expressions from facial landmarks"""
        try:
            # Calculate facial muscle movements
            brow_movement = self.calculate_brow_movement(landmarks)
            mouth_movement = self.calculate_mouth_movement(landmarks)
            eye_movement = self.calculate_eye_movement(landmarks)
            
            # Detect micro-expressions
            expressions = []
            if brow_movement > 0.2:
                expressions.append("brow_raise")
            if mouth_movement > 0.15:
                expressions.append("lip_tightening")
            if eye_movement > 0.1:
                expressions.append("eye_narrowing")
            
            return expressions
            
        except Exception as e:
            print(f"Error analyzing micro-expressions: {str(e)}")
            return []

    def analyze_confidence(self, holistic_results, landmarks):
        """Analyze confidence indicators"""
        try:
            confidence_score = 0
            
            # Analyze posture contribution to confidence
            if holistic_results.pose_landmarks:
                posture_score = self.analyze_posture(holistic_results.pose_landmarks)
                if posture_score.get('is_straight', False):
                    confidence_score += 0.4
            
            # Analyze facial expressions
            if len(self.emotion_history) > 0:
                if self.emotion_history[-1] in ['happy', 'neutral']:
                    confidence_score += 0.3
            
            # Analyze eye contact
            if len(self.eye_contact_history) > 0:
                confidence_score += self.eye_contact_history[-1] * 0.3
            
            return confidence_score
            
        except Exception as e:
            print(f"Error analyzing confidence: {str(e)}")
            return 0.5

    def stop_analysis(self):
        """Stop analysis and generate final report"""
        self.is_analyzing = False
        print("\nAnalysis stopped. Generating comprehensive report...")
        self.generate_final_report()
        self.root.destroy()

    def generate_final_report(self):
        """Generate comprehensive psychological analysis report"""
        try:
            report = "\n🔍 Advanced Psychological Analysis Report\n"
            report += "=" * 50 + "\n\n"
            
            # Add AI-driven deductions
            report += "Detective's Psychological Profile:\n"
            report += self.make_ai_deductions()
            
            # Add detailed psychological analysis
            report += "\n\n🧠 Deep Psychological Analysis:\n"
            report += "=" * 30 + "\n"
            
            # Add personality insights
            report += self.generate_personality_insights()
            
            # Add cognitive patterns
            report += self.analyze_cognitive_patterns()
            
            # Add behavioral tendencies
            report += self.analyze_behavioral_tendencies()
            
            # Add emotional intelligence assessment
            report += self.assess_emotional_intelligence()
            
            # Add social interaction style
            report += self.analyze_social_interaction_style()
            
            # Add stress response patterns
            report += self.analyze_stress_patterns()
            
            # Add defense mechanisms
            report += self.identify_defense_mechanisms()
            
            # Add recommendations
            report += "\n\n📋 Professional Recommendations:\n"
            report += "=" * 30 + "\n"
            report += self.generate_recommendations()
            
            return report
            
        except Exception as e:
            print(f"Error generating final report: {str(e)}")
            return "Error generating comprehensive analysis"

    def generate_temporal_analysis(self):
        """Generate analysis of patterns over time"""
        report = "Temporal Analysis\n" + "-" * 20 + "\n"
        
        # Analyze emotional changes over time
        if self.data_points['emotions']:
            report += "\nEmotional Progression:\n"
            report += self.analyze_emotional_progression()
        
        # Analyze attention patterns
        if self.data_points['attention_patterns']:
            report += "\nAttention Pattern Changes:\n"
            report += self.analyze_attention_progression()
        
        # Analyze stress level changes
        if self.data_points['stress_indicators']:
            report += "\nStress Level Progression:\n"
            report += self.analyze_stress_progression()
        
        return report

    def analyze_emotional_progression(self):
        """Analyze how emotions changed over time"""
        emotions = self.data_points['emotions']
        report = ""
        
        if len(emotions) > 1:
            # Analyze emotional stability
            emotion_changes = sum(1 for i in range(1, len(emotions))
                                if emotions[i][1] != emotions[i-1][1])
            stability = 1 - (emotion_changes / len(emotions))
            
            report += f"• Emotional Stability: {stability:.2%}\n"
            report += "• Notable Emotional Shifts:\n"
            
            # Identify significant emotional shifts
            for i in range(1, len(emotions)):
                if emotions[i][1] != emotions[i-1][1]:
                    time_diff = emotions[i][0] - emotions[i-1][0]
                    report += f"   - {emotions[i-1][1]} → {emotions[i][1]} "
                    report += f"(after {time_diff:.1f} seconds)\n"
        
        return report

    def analyze_attention_progression(self):
        """Analyze changes in attention patterns over time"""
        attention_patterns = self.data_points['attention_patterns']
        report = ""
        
        if len(attention_patterns) > 1:
            # Analyze attention stability
            attention_changes = sum(1 for i in range(1, len(attention_patterns))
                                if attention_patterns[i][1] != attention_patterns[i-1][1])
            stability = 1 - (attention_changes / len(attention_patterns))
            
            report += f"• Attention Stability: {stability:.2%}\n"
            report += "• Notable Attention Pattern Changes:\n"
            
            # Identify significant attention pattern shifts
            for i in range(1, len(attention_patterns)):
                if attention_patterns[i][1] != attention_patterns[i-1][1]:
                    time_diff = attention_patterns[i][0] - attention_patterns[i-1][0]
                    report += f"   - {attention_patterns[i-1][1]} → {attention_patterns[i][1]} "
                    report += f"(after {time_diff:.1f} seconds)\n"
        
        return report

    def analyze_stress_progression(self):
        """Analyze changes in stress level over time"""
        stress_indicators = self.data_points['stress_indicators']
        report = ""
        
        if len(stress_indicators) > 1:
            # Analyze stress level stability
            stress_changes = sum(1 for i in range(1, len(stress_indicators))
                                if stress_indicators[i][1] != stress_indicators[i-1][1])
            stability = 1 - (stress_changes / len(stress_indicators))
            
            report += f"• Stress Level Stability: {stability:.2%}\n"
            report += "• Notable Stress Level Changes:\n"
            
            # Identify significant stress level shifts
            for i in range(1, len(stress_indicators)):
                if stress_indicators[i][1] != stress_indicators[i-1][1]:
                    time_diff = stress_indicators[i][0] - stress_indicators[i-1][0]
                    report += f"   - {stress_indicators[i-1][1]} → {stress_indicators[i][1]} "
                    report += f"(after {time_diff:.1f} seconds)\n"
        
        return report

    def generate_pattern_analysis(self):
        """Generate analysis of behavioral patterns"""
        report = "Pattern Analysis\n" + "-" * 20 + "\n"
        
        # Analyze facial patterns
        if self.data_points['facial_movements']:
            report += "\nFacial Movement Patterns:\n"
            report += self.analyze_facial_movement_patterns()
        
        # Analyze eye movement patterns
        if self.data_points['eye_movements']:
            report += "\nEye Movement Patterns:\n"
            report += self.analyze_eye_movement_patterns()
        
        # Analyze mouth movement patterns
        if self.data_points['mouth_patterns']:
            report += "\nMouth Movement Patterns:\n"
            report += self.analyze_mouth_movement_patterns()
        
        return report

    def analyze_facial_movement_patterns(self):
        """Analyze patterns in facial movements"""
        facial_movements = self.data_points['facial_movements']
        report = ""
        
        if len(facial_movements) > 1:
            # Analyze facial movement stability
            movement_changes = sum(1 for i in range(1, len(facial_movements))
                                if facial_movements[i][1] != facial_movements[i-1][1])
            stability = 1 - (movement_changes / len(facial_movements))
            
            report += f"• Facial Movement Stability: {stability:.2%}\n"
            report += "• Notable Facial Movement Changes:\n"
            
            # Identify significant facial movement shifts
            for i in range(1, len(facial_movements)):
                if facial_movements[i][1] != facial_movements[i-1][1]:
                    time_diff = facial_movements[i][0] - facial_movements[i-1][0]
                    report += f"   - {facial_movements[i-1][1]} → {facial_movements[i][1]} "
                    report += f"(after {time_diff:.1f} seconds)\n"
        
        return report

    def analyze_eye_movement_patterns(self):
        """Analyze patterns in eye movements"""
        eye_movements = self.data_points['eye_movements']
        report = ""
        
        if len(eye_movements) > 1:
            # Analyze eye movement stability
            movement_changes = sum(1 for i in range(1, len(eye_movements))
                                if eye_movements[i][1] != eye_movements[i-1][1])
            stability = 1 - (movement_changes / len(eye_movements))
            
            report += f"• Eye Movement Stability: {stability:.2%}\n"
            report += "• Notable Eye Movement Changes:\n"
            
            # Identify significant eye movement shifts
            for i in range(1, len(eye_movements)):
                if eye_movements[i][1] != eye_movements[i-1][1]:
                    time_diff = eye_movements[i][0] - eye_movements[i-1][0]
                    report += f"   - {eye_movements[i-1][1]} → {eye_movements[i][1]} "
                    report += f"(after {time_diff:.1f} seconds)\n"
        
        return report

    def analyze_mouth_movement_patterns(self):
        """Analyze patterns in mouth movements"""
        mouth_movements = self.data_points['mouth_patterns']
        report = ""
        
        if len(mouth_movements) > 1:
            # Analyze mouth movement stability
            movement_changes = sum(1 for i in range(1, len(mouth_movements))
                                if mouth_movements[i][1] != mouth_movements[i-1][1])
            stability = 1 - (movement_changes / len(mouth_movements))
            
            report += f"• Mouth Movement Stability: {stability:.2%}\n"
            report += "• Notable Mouth Movement Changes:\n"
            
            # Identify significant mouth movement shifts
            for i in range(1, len(mouth_movements)):
                if mouth_movements[i][1] != mouth_movements[i-1][1]:
                    time_diff = mouth_movements[i][0] - mouth_movements[i-1][0]
                    report += f"   - {mouth_movements[i-1][1]} → {mouth_movements[i][1]} "
                    report += f"(after {time_diff:.1f} seconds)\n"
        
        return report

    def generate_mental_health_section(self):
        """Generate mental health analysis section"""
        report = "Mental Health Analysis\n" + "-" * 20 + "\n"
        
        # Analyze mental health indicators
        if self.data_points['mental_health_indicators']:
            report += "\nMental Health Indicators:\n"
            for timestamp, data in self.data_points['mental_health_indicators']:
                report += f"• {data}\n"
        
        return report

    def analyze_mental_health_patterns(self, facial_data, emotion_data, data_points):
        """Analyze mental health patterns based on facial expressions and behavioral indicators"""
        try:
            # Implement mental health pattern analysis logic here
            # This is a placeholder and should be replaced with actual implementation
            return "Healthy mental state"
        except Exception as e:
            print(f"Error analyzing mental health patterns: {str(e)}")
            return None

    def update_data_points(self, results, timestamp):
        """Update data points with analysis results"""
        for key, value in results.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    self.data_points[f"{key}_{subkey}"].append((timestamp, subvalue))
            else:
                self.data_points[key].append((timestamp, value))

    def make_ai_deductions(self):
        """Generate AI-driven detective-style behavioral analysis"""
        try:
            deductions = []
            confidence_levels = {
                'high': 'Strong evidence suggests',
                'medium': 'Observable patterns indicate',
                'low': 'Subtle indicators suggest'
            }

            # Analyze emotional patterns
            if self.emotion_history:
                emotional_stability = self.calculate_emotional_stability()
                dominant_emotions = self.get_dominant_emotions(top_n=3)
                emotion_transitions = self.analyze_emotion_transitions()
                
                deductions.append("\n🔍 Emotional Pattern Analysis:")
                deductions.append(f"Primary observation: {confidence_levels['high']} that the subject "
                                f"exhibits {emotional_stability['pattern']}.")
                deductions.append(f"• Emotional Fingerprint: {dominant_emotions['interpretation']}")
                deductions.append(f"• Transition Analysis: {emotion_transitions['interpretation']}")
                
                if emotional_stability['score'] < 0.4:
                    deductions.append("🎯 Notable Insight: The rapid emotional shifts potentially "
                                    "indicate underlying psychological pressure or situational stress.")

            # Analyze body language patterns
            if self.gesture_history:
                gesture_patterns = self.analyze_gesture_patterns()
                deductions.append("\n🔍 Body Language Decode:")
                for pattern in gesture_patterns['patterns']:
                    deductions.append(f"• {confidence_levels[pattern['confidence_level']]} "
                                    f"{pattern['interpretation']}")
                
                if gesture_patterns.get('clusters', []):
                    deductions.append("\n🎯 Behavioral Clusters Detected:")
                    for cluster in gesture_patterns['clusters']:
                        deductions.append(f"• {cluster}")

            # Analyze micro-expressions
            if self.micro_expressions:
                micro_expr_analysis = self.analyze_micro_expression_patterns()
                deductions.append("\n🔍 Micro-Expression Intelligence:")
                deductions.append(f"• {micro_expr_analysis['primary_observation']}")
                for insight in micro_expr_analysis['insights']:
                    deductions.append(f"  - {insight}")

            # Analyze confidence and power dynamics
            if self.confidence_indicators:
                confidence_analysis = self.analyze_confidence_patterns()
                deductions.append("\n🔍 Power Dynamic Assessment:")
                deductions.append(f"• Confidence Baseline: {confidence_analysis['baseline']}")
                deductions.append(f"• Authority Indicators: {confidence_analysis['authority_markers']}")
                
                if confidence_analysis.get('incongruencies'):
                    deductions.append("\n🎯 Notable Incongruencies:")
                    for incongruency in confidence_analysis['incongruencies']:
                        deductions.append(f"• {incongruency}")

            # Generate psychological profile
            profile = self.generate_psychological_profile()
            deductions.append("\n🔍 Psychological Profile Synthesis:")
            deductions.append(profile['summary'])
            for trait in profile['key_traits']:
                deductions.append(f"• {trait}")

            # Add investigative recommendations
            recommendations = self.generate_investigative_recommendations()
            deductions.append("\n🎯 Investigative Recommendations:")
            for rec in recommendations:
                deductions.append(f"• {rec}")

            # Final analysis timestamp
            deductions.append(f"\nAnalysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            deductions.append("Note: This analysis is based on behavioral observation patterns and "
                             "should be considered as part of a broader investigative context.")

            return "\n".join(deductions)

        except Exception as e:
            print(f"Error in AI deductions: {str(e)}")
            return "Error generating deductions"

    def calculate_emotional_stability(self):
        """Calculate emotional stability patterns"""
        try:
            changes = sum(1 for i in range(1, len(self.emotion_history))
                         if self.emotion_history[i] != self.emotion_history[i-1])
            stability_score = 1 - (changes / len(self.emotion_history))
            
            if stability_score > 0.7:
                pattern = "remarkable emotional control, suggesting trained composure"
            elif stability_score > 0.4:
                pattern = "natural emotional fluidity with controlled transitions"
            else:
                pattern = "heightened emotional responsiveness to environmental stimuli"
            
            return {
                'score': stability_score,
                'pattern': pattern
            }
        except Exception as e:
            print(f"Error calculating emotional stability: {str(e)}")
            return {'score': 0.5, 'pattern': "inconclusive emotional pattern"}

    def analyze_gesture_patterns(self):
        """Analyze complex gesture patterns and their psychological implications"""
        try:
            patterns = []
            gesture_counts = {}
            
            for gesture in self.gesture_history:
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
            
            # Analyze gesture clusters
            clusters = []
            if 'touching_face' in gesture_counts and 'hand_fidgeting' in gesture_counts:
                clusters.append("Self-soothing behavioral cluster detected: "
                              "Combination of face-touching and hand movements "
                              "suggests heightened internal processing")
            
            # Analyze individual patterns
            for gesture, count in gesture_counts.items():
                if count > len(self.gesture_history) * 0.3:
                    patterns.append({
                        'confidence_level': 'high',
                        'interpretation': self.interpret_frequent_gesture(gesture, count)
                    })
            
            return {
                'patterns': patterns,
                'clusters': clusters
            }
        except Exception as e:
            print(f"Error analyzing gesture patterns: {str(e)}")
            return {'patterns': [], 'clusters': []}

    def generate_psychological_profile(self):
        """Generate comprehensive psychological profile based on behavioral data"""
        try:
            # Analyze all available data points
            emotional_data = self.analyze_emotional_data()
            behavioral_data = self.analyze_behavioral_data()
            confidence_data = self.analyze_confidence_data()
            
            # Generate profile
            profile = {
                'summary': self.synthesize_profile_summary(emotional_data, 
                                                         behavioral_data, 
                                                         confidence_data),
                'key_traits': self.identify_key_traits(emotional_data, 
                                                     behavioral_data, 
                                                     confidence_data)
            }
            
            return profile
        except Exception as e:
            print(f"Error generating psychological profile: {str(e)}")
            return {'summary': "Profile generation incomplete", 'key_traits': []}

    def analyze_emotional_data(self):
        """Analyze emotional patterns and stability"""
        try:
            if not self.emotion_history:
                return {'stability': 0.5, 'patterns': []}
            
            # Calculate emotional stability
            changes = sum(1 for i in range(1, len(self.emotion_history))
                         if self.emotion_history[i] != self.emotion_history[i-1])
            stability = 1 - (changes / len(self.emotion_history))
            
            # Identify emotional patterns
            patterns = []
            emotion_counts = Counter(self.emotion_history)
            
            # Analyze dominant emotions
            dominant = emotion_counts.most_common(1)[0]
            if dominant[1] / len(self.emotion_history) > 0.6:
                patterns.append(f"Strong {dominant[0]} dominance")
            
            # Analyze emotional transitions
            transitions = self.analyze_emotional_transitions()
            if transitions:
                patterns.extend(transitions)
            
            return {
                'stability': stability,
                'patterns': patterns,
                'dominant_emotion': dominant[0],
                'emotion_distribution': dict(emotion_counts)
            }
        except Exception as e:
            print(f"Error analyzing emotional data: {str(e)}")
            return {'stability': 0.5, 'patterns': []}

    def analyze_behavioral_data(self):
        """Analyze behavioral patterns and gestures"""
        try:
            behavioral_data = {
                'gestures': self.analyze_gesture_patterns(),
                'posture': self.analyze_posture_trends(),
                'micro_expressions': self.analyze_micro_expression_patterns(),
                'attention': self.analyze_attention_patterns()
            }
            return behavioral_data
        except Exception as e:
            print(f"Error analyzing behavioral data: {str(e)}")
            return {}

    def analyze_confidence_data(self):
        """Analyze confidence indicators and patterns"""
        try:
            if not self.confidence_indicators:
                return {'average': 0.5, 'stability': 0.5, 'patterns': []}
            
            avg_confidence = sum(self.confidence_indicators) / len(self.confidence_indicators)
            stability = 1 - np.std(self.confidence_indicators)
            
            patterns = []
            if avg_confidence > 0.7:
                patterns.append("High consistent confidence")
            elif avg_confidence < 0.3:
                patterns.append("Notable confidence deficiency")
            
            return {
                'average': avg_confidence,
                'stability': stability,
                'patterns': patterns
            }
        except Exception as e:
            print(f"Error analyzing confidence data: {str(e)}")
            return {'average': 0.5, 'stability': 0.5, 'patterns': []}

    def generate_investigative_recommendations(self):
        """Generate professional recommendations based on analysis"""
        try:
            recommendations = []
            
            # Analyze emotional patterns
            emotional_data = self.analyze_emotional_data()
            if emotional_data['stability'] < 0.4:
                recommendations.append("Further investigation into emotional triggers recommended")
            
            # Analyze behavioral patterns
            behavioral_data = self.analyze_behavioral_data()
            if behavioral_data.get('gestures', {}).get('patterns'):
                recommendations.append("Notable behavioral patterns warrant deeper investigation")
            
            # Analyze confidence patterns
            confidence_data = self.analyze_confidence_data()
            if confidence_data['stability'] < 0.3:
                recommendations.append("Investigate factors affecting confidence fluctuations")
            
            return recommendations
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return ["Unable to generate recommendations due to insufficient data"]

    def generate_personality_insights(self):
        """Generate detailed personality insights"""
        try:
            insights = "\n📊 Personality Analysis:\n"
            insights += "=" * 25 + "\n"
            
            # Analyze emotional patterns
            emotional_data = self.analyze_emotional_data()
            insights += "\nEmotional Disposition:\n"
            insights += f"• Emotional Stability: {emotional_data['stability']:.2f}\n"
            if emotional_data['patterns']:
                insights += "• Notable Patterns:\n"
                for pattern in emotional_data['patterns']:
                    insights += f"  - {pattern}\n"
                
            # Analyze behavioral patterns
            behavioral_data = self.analyze_behavioral_data()
            insights += "\nBehavioral Tendencies:\n"
            if behavioral_data.get('gestures', {}).get('patterns'):
                for pattern in behavioral_data['gestures']['patterns']:
                    insights += f"• {pattern}\n"
                
            # Add confidence analysis
            confidence_data = self.analyze_confidence_data()
            insights += f"\nConfidence Profile:\n"
            insights += f"• Baseline Confidence: {confidence_data['average']:.2f}\n"
            insights += f"• Stability: {confidence_data['stability']:.2f}\n"
            
            return insights
        except Exception as e:
            print(f"Error generating personality insights: {str(e)}")
            return "\nPersonality analysis incomplete"

    def analyze_emotional_transitions(self):
        """Analyze patterns in emotional transitions"""
        try:
            if len(self.emotion_history) < 2:
                return []
            
            transitions = []
            for i in range(1, len(self.emotion_history)):
                if self.emotion_history[i] != self.emotion_history[i-1]:
                    transitions.append((self.emotion_history[i-1], 
                                     self.emotion_history[i]))
                
            patterns = []
            if transitions:
                # Analyze rapid switches
                rapid_switches = [t for t in transitions 
                                if t[0] in ['happy', 'neutral'] and 
                                t[1] in ['angry', 'fearful']]
                if rapid_switches:
                    patterns.append("Notable emotional volatility detected")
                
                # Analyze emotional suppression
                suppression = [t for t in transitions 
                             if t[0] in ['angry', 'fearful'] and 
                             t[1] == 'neutral']
                if suppression:
                    patterns.append("Potential emotional suppression observed")
                
            return patterns
        except Exception as e:
            print(f"Error analyzing emotional transitions: {str(e)}")
            return []

    def analyze_posture_trends(self):
        """Analyze posture patterns over time"""
        try:
            if not self.posture_history:
                return {'stability': 0.5, 'patterns': []}
            
            patterns = []
            posture_scores = [p.get('score', 0.5) for p in self.posture_history]
            avg_posture = sum(posture_scores) / len(posture_scores)
            
            if avg_posture > 0.7:
                patterns.append("Consistently upright posture indicating confidence")
            elif avg_posture < 0.3:
                patterns.append("Notable posture issues suggesting discomfort")
            
            return {
                'average': avg_posture,
                'stability': 1 - np.std(posture_scores),
                'patterns': patterns
            }
        except Exception as e:
            print(f"Error analyzing posture trends: {str(e)}")
            return {'stability': 0.5, 'patterns': []}

def configure_environment():
    # Suppress TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel(logging.ERROR)
    
    # Suppress MediaPipe logging
    logging.getLogger("mediapipe").setLevel(logging.ERROR)
    
    # Configure XNNPACK delegate
    os.environ["XNNPACK_DELEGATE"] = "1"
    
    # Disable feedback tensor warnings
    os.environ["MEDIAPIPE_DISABLE_FEEDBACK_TENSORS"] = "1"

if __name__ == "__main__":
    analyzer = BehaviorAnalyzer()
    analyzer.root.mainloop() 