import os
import random
import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
# Path to your trained model (adjust folder name if different)
MODEL_PATH = 'my_final_model.pt'

# Path to your TEST images folder
TEST_IMAGES_DIR = 'datasets/my_mini_dataset/test/images'

def predict_random():
    # 1. Load your custom trained model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    model = YOLO(MODEL_PATH)

    # 2. Pick a random image from the test folder
    images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('.jpg', '.png'))]
    if not images:
        print("No images found in test folder!")
        return
        
    random_image_name = random.choice(images)
    image_path = os.path.join(TEST_IMAGES_DIR, random_image_name)
    print(f"Testing on: {random_image_name}")

    # 3. Run Prediction
    # save=True automatically saves the image with boxes/labels to 'runs/detect/predict'
    # conf=0.25 sets the minimum confidence threshold
    results = model.predict(source=image_path, save=True, conf=0.25, device='cpu')

    # 4. Optional: Save a copy directly to root for easier viewing in VS Code
    # The 'plot()' method returns the image as a numpy array with boxes drawn
    annotated_frame = results[0].plot()
    
    output_filename = "result_prediction.jpg"
    cv2.imwrite(output_filename, annotated_frame)
    print(f"✅ Prediction saved! Open '{output_filename}' in your VS Code explorer to see it.")

if __name__ == "__main__":
    predict_random()