import os
import shutil
import random

# --- CONFIGURATION ---
SOURCE_DIR = "datasets/my_dataset"       # Your huge dataset
TARGET_DIR = "datasets/my_mini_dataset"  # New small dataset location
NUM_IMAGES = 200                         # How many images you want

def create_subset():
    # Define paths
    src_img_path = os.path.join(SOURCE_DIR, "train/images")
    src_lbl_path = os.path.join(SOURCE_DIR, "train/labels")
    
    tgt_img_path = os.path.join(TARGET_DIR, "train/images")
    tgt_lbl_path = os.path.join(TARGET_DIR, "train/labels")

    # Create directories (clears old mini-dataset if it exists)
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.makedirs(tgt_img_path)
    os.makedirs(tgt_lbl_path)

    # Get list of all images
    all_images = [f for f in os.listdir(src_img_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Randomly select images
    selected_images = random.sample(all_images, min(len(all_images), NUM_IMAGES))
    
    print(f"Creating subset with {len(selected_images)} images...")

    for img_name in selected_images:

        shutil.copy(os.path.join(src_img_path, img_name), os.path.join(tgt_img_path, img_name))
        
        label_name = os.path.splitext(img_name)[0] + ".txt"
        
        if os.path.exists(os.path.join(src_lbl_path, label_name)):
            shutil.copy(os.path.join(src_lbl_path, label_name), os.path.join(tgt_lbl_path, label_name))

    print(f"Done! New dataset created at: {TARGET_DIR}")
    print("REMINDER: Create a new 'mini_data.yaml' pointing to this new folder.")

if __name__ == "__main__":
    create_subset()