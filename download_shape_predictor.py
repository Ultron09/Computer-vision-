import urllib.request
import bz2
import os

def download_shape_predictor():
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    filename = "shape_predictor_68_face_landmarks.dat.bz2"
    
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print("Downloading shape predictor...")
        urllib.request.urlretrieve(url, filename)
        
        print("Extracting file...")
        with bz2.BZ2File(filename) as fr, open("shape_predictor_68_face_landmarks.dat", "wb") as fw:
            fw.write(fr.read())
        
        os.remove(filename)
        print("Done!")

if __name__ == "__main__":
    download_shape_predictor() 