import requests
import io
from PIL import Image
import base64

def create_dummy_image():
    img = Image.new('RGB', (224, 224), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_endpoint(url, filename="test.png"):
    print(f"Testing {url}...")
    try:
        img_data = create_dummy_image()
        files = {'file': (filename, img_data, 'image/png')}
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            print("SUCCESS")
            print("Response keys:", response.json().keys())
            return True
        else:
            print(f"FAILED: {response.status_code}")
            print(response.text)
            return False
    except requests.exceptions.ConnectionError:
        print("FAILED: Connection refused. Is the backend running?")
        return False
    except Exception as e:
        print(f"FAILED: {e}")
        return False

if __name__ == "__main__":
    print("--- Starting Verification ---")
    
    # Check Pneumonia Endpoint
    pneumonia_url = "http://localhost:8000/api/v1/predict/pneumonia"
    p_result = test_endpoint(pneumonia_url)
    
    # Check Tuberculosis Endpoint
    tb_url = "http://localhost:8000/api/v1/predict/tuberculosis"
    tb_result = test_endpoint(tb_url)
    
    if p_result and tb_result:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")
