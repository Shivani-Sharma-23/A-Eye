import cv2
from ultralytics import YOLO

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=4, text_thickness=2):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))
    return writer

model = YOLO(r"C:\Users\shiva\Downloads\Blindman_project\final\yolo\runs\detect\Yolo11_new_results\weights\best.pt")

output_filename = "Yolo11_new_videoresult2.mp4"

video_path = r"C:\Users\shiva\Downloads\Blindman_project\final\Doors-and-walls-object-detection-1\WhatsApp Video 2024-11-12 at 13.32.36_7dcd1a24.mp4"
cap = cv2.VideoCapture(video_path)
writer = create_video_writer(cap, output_filename)
while True:
    success, img = cap.read()
    if not success:
        break
    result_img, _ = predict_and_detect(model, img, classes=[], conf=0.5)
    writer.write(result_img)
    cv2.imshow("Image", result_img)
    
    cv2.waitKey(1)
writer.release()