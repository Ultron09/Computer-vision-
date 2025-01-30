from deepface import DeepFace
import cv2

# Load a test image (this will trigger the model download)
img = cv2.imread("test.jpg")  # You can use any image

# First run will automatically download the model
try:
    result = DeepFace.analyze(img_path = img, actions = ['emotion'])
    print("Model downloaded successfully!")
except Exception as e:
    print(f"Error: {e}") 