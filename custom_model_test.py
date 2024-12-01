import cv2
from ultralytics import YOLO

import random
def generate_class_colors(num_classes):
    random.seed(42) 
    return {i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(num_classes)}

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=4, text_thickness=2):
    results = predict(chosen_model, img, classes, conf=conf)
    class_names = chosen_model.names
    class_colors = generate_class_colors(len(class_names))
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            color = class_colors[class_id]
            cv2.rectangle(img, 
                          (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), 
                          color, rectangle_thickness)

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


model = YOLO(r"C:\Users\shiva\Downloads\Blindman_project\final\custom_model\model_pt\best.pt")
output_filename = "custom_videoresult_new2.mp4"
video_path = "WhatsApp Video 2024-11-21 at 12.33.19_99d2a019.mp4"
cap = cv2.VideoCapture(video_path)
writer = create_video_writer(cap, output_filename)

while True:
    success, img = cap.read()
    if not success:
        break
    result_img, _ = predict_and_detect(model, img, classes=[], conf=0.5)

    writer.write(result_img)
    cv2.imshow("Image", result_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
