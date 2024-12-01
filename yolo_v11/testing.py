import cv2
from ultralytics import YOLO

# Generate random colors for each class
import random
def generate_class_colors(num_classes):
    random.seed(42)  # For reproducibility
    return {i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(num_classes)}

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=4, text_thickness=2):
    # Run predictions
    results = predict(chosen_model, img, classes, conf=conf)
    
    # Access class names from the model
    class_names = chosen_model.names
    
    # Generate colors for the classes
    class_colors = generate_class_colors(len(class_names))
    
    for result in results:
        for box in result.boxes:
            # Get the class index
            class_id = int(box.cls[0])
            # Get the class name and its corresponding color
            class_name = class_names[class_id]
            color = class_colors[class_id]
            
            # Draw the rectangle around the object
            cv2.rectangle(img, 
                          (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), 
                          color, rectangle_thickness)
            
            # Draw the label text above the rectangle
            cv2.putText(img, f"{class_name}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, color, text_thickness)
    return img, results


def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))
    return writer

# Model and video setup
model = YOLO(r"C:\Users\shiva\Downloads\Blindman_project\final\yolo\runs\detect\Yolov5_new_results\weights\best.pt")
output_filename = "custom_videoresult2.mp4"
video_path = r"C:\Users\shiva\Downloads\Blindman_project\final\Doors-and-walls-object-detection-1\WhatsApp Video 2024-11-12 at 13.32.36_7dcd1a24.mp4"

cap = cv2.VideoCapture(video_path)
writer = create_video_writer(cap, output_filename)

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Process the frame
    result_img, _ = predict_and_detect(model, img, classes=[], conf=0.5)
    
    # Write to the video
    writer.write(result_img)
    cv2.imshow("Image", result_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release resources
cap.release()
writer.release()
cv2.destroyAllWindows()
