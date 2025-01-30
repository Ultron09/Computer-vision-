import urllib.request
import os

def download_models():
    # URLs for the model files
    urls = {
        'age_deploy.prototxt': 'https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt',
        'age_net.caffemodel': 'https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel',
        'gender_deploy.prototxt': 'https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt',
        'gender_net.caffemodel': 'https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel'
    }

    for filename, url in urls.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"Downloaded {filename} successfully!")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    download_models() 