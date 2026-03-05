import requests
import cv2
import numpy as np
import io

def create_test_image():
    # Create a simple test image (300x300 white square)
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img.fill(255)
    # Add some colored rectangles
    cv2.rectangle(img, (50, 50), (100, 100), (0, 0, 255), -1)
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', img)
    return io.BytesIO(buffer)

def test_detection():
    url = "http://127.0.0.1:5000/api/detect/image"
    
    try:
        img_buffer = create_test_image()
        files = {'image': ('test.jpg', img_buffer, 'image/jpeg')}
        
        print(f"Sending request to {url}...")
        response = requests.post(url, files=files)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response JSON:", response.json())
        else:
            print("Error Response:", response.text)
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_detection()
