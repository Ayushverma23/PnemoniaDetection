import requests
import sys
import os

# Configuration
API_URL = "http://localhost:8000/api/v1/predict/pneumonia"
TEST_IMAGE_PATH = "d:/Medical/pneumonia_detection/data/stage_2_train_images/003d8fa0-6bf1-40ed-b54c-ac657f8495c5.dcm" # Modify this to a valid path if needed

def test_prediction():
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Test image not found at {TEST_IMAGE_PATH}")
        # Try to find any dcm file in the directory
        data_dir = "d:/Medical/pneumonia_detection/data/stage_2_train_images/"
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith('.dcm')]
            if files:
                global TEST_IMAGE_PATH
                TEST_IMAGE_PATH = os.path.join(data_dir, files[0])
                print(f"Using alternative test image: {TEST_IMAGE_PATH}")
            else:
                print("No DICOM files found for testing.")
                return

    print(f"Testing API at {API_URL} with image {TEST_IMAGE_PATH}...")
    
    try:
        with open(TEST_IMAGE_PATH, "rb") as f:
            files = {"file": (os.path.basename(TEST_IMAGE_PATH), f, "application/dicom")}
            response = requests.post(API_URL, files=files)
            
        if response.status_code == 200:
            print("✅ Success!")
            print("Response:", response.json())
        else:
            print("❌ Failed!")
            print(f"Status Code: {response.status_code}")
            print("Response:", response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Connecton Error. Is the backend running?")
        print("Run 'uvicorn backend.main:app --reload' from 'd:/Medical/pneumonia_detection/'")

if __name__ == "__main__":
    test_prediction()
