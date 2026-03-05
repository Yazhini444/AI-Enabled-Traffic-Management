import requests

def check_page():
    url = "http://127.0.0.1:5000/input"
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Success: Page loaded correctly.")
        else:
            print("Error: Page returned status code", response.status_code)
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    check_page()
