# test_client.py
import requests
import json
from pprint import pprint

# The URL of the API endpoint
API_URL = "http://localhost:8000/predict"

# The path to the image you want to test
IMAGE_PATH = "test.jpg" # Make sure you have an image with this name in the same folder

def test_detection_api(image_path):
    """Sends an image to the detection API and prints the response."""
    try:
        with open(image_path, "rb") as image_file:
            files = {"file": (image_path, image_file, "image/jpeg")}
            print(f"Sending request for image: {image_path}")
            response = requests.post(API_URL, files=files)
            
            # Check for successful response
            response.raise_for_status()
            
            # Parse and print the JSON response
            data = response.json()
            print("\n--- Detection Results ---")
            pprint(data)
            print("-------------------------\n")
            
    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred while communicating with the API: {e}")
    except FileNotFoundError:
        print(f"\nError: The file '{image_path}' was not found.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    test_detection_api(IMAGE_PATH)
