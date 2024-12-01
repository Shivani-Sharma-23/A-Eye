import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


dataset_root = "Doors-and-walls-object-detection-1"  
output_root = "./mobilenet_output"
os.makedirs(output_root, exist_ok=True)

model = MobileNet(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    images_folder = os.path.join(input_folder, "images")
    if not os.path.exists(images_folder):
        print(f"Images folder not found at: {images_folder}")
        return
    
    for img_file in os.listdir(images_folder):
        if img_file.lower().endswith(('png', 'jpg', 'jpeg')):
            img_path = os.path.join(images_folder, img_file)
          
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
      
            features = model.predict(img_array)
     
            feature_file = os.path.join(output_folder, os.path.splitext(img_file)[0] + ".npy")
            np.save(feature_file, features)

for folder in ["train", "test", "valid"]:
    input_path = os.path.join(dataset_root, folder)
    output_path = os.path.join(output_root, folder)
    process_folder(input_path, output_path)

print(f"Feature extraction completed. Features are saved in: {output_root}")
