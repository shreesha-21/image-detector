import os
import shutil
import random

# --- CONFIGURATION ---
SOURCE_DIR = "datasets/my_dataset"      
TARGET_DIR = "datasets/my_mini_dataset"  

# Define how many images you want for each split
LIMITS = {
    'train': 750,
    'valid': 125,   
    'test': 400  
}

def create_subset():
    # 1. Clean start: Delete target directory if it exists
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
        print(f"Removed old {TARGET_DIR}")

    # 2. Loop through each split (train, val, test)
    for split, limit in LIMITS.items():
        if limit == 0:
            continue
            
        # Define paths for this split
        src_img_dir = os.path.join(SOURCE_DIR, split, "images")
        src_lbl_dir = os.path.join(SOURCE_DIR, split, "labels")
        
        tgt_img_dir = os.path.join(TARGET_DIR, split, "images")
        tgt_lbl_dir = os.path.join(TARGET_DIR, split, "labels")

        # Check if source exists (e.g., if you don't have a 'test' folder)
        if not os.path.exists(src_img_dir):
            print(f"Skipping '{split}': Source folder not found.")
            continue

        # Create target directories
        os.makedirs(tgt_img_dir, exist_ok=True)
        os.makedirs(tgt_lbl_dir, exist_ok=True)

        # List all images
        all_images = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Randomly sample
        selected_images = random.sample(all_images, min(len(all_images), limit))
        
        print(f"Processing '{split}': Copying {len(selected_images)} images...")

        for img_name in selected_images:
            # Copy Image
            shutil.copy(os.path.join(src_img_dir, img_name), os.path.join(tgt_img_dir, img_name))
            
            # Copy Label (if it exists)
            # Assumes label has same name as image but with .txt extension
            label_name = os.path.splitext(img_name)[0] + ".txt"
            src_label_path = os.path.join(src_lbl_dir, label_name)
            
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, os.path.join(tgt_lbl_dir, label_name))

    print(f"\nSuccess! Mini dataset created at: {TARGET_DIR}")

if __name__ == "__main__":
    create_subset()