import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

model = MobileNet(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

output_root = "./mobilenet_ext_output"
os.makedirs(output_root, exist_ok=True)

def process_image_with_visualization(img_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(img_path):
        print(f"Image file not found at: {img_path}")
        return

    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array)

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(output_folder, img_name + "_features.png")
    visualize_features(features[0], save_path)
    
    print(f"Feature visualization saved at: {save_path}")
    return features

def visualize_features(features, save_path):
    num_features = features.shape[-1] 
    size = int(np.sqrt(num_features))  
    plt.figure(figsize=(12, 12))

    for i in range(min(size * size, num_features)): 
        ax = plt.subplot(size, size, i + 1)
        plt.imshow(features[:, :, i], cmap="viridis")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

example_image_path = r"C:\Users\shiva\Downloads\Blindman_project\custom_model\Doors-and-walls-object-detection-1\train\images\VID-20241014-WA0002_mp4-0006_jpg.rf.9853928d63d526f3188c01be859fe210.jpg"  # Replace with the actual image path
features = process_image_with_visualization(example_image_path, output_root)

if features is not None:
    print(f"Extracted features shape: {features.shape}")
