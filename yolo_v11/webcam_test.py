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

# Load the trained YOLO model
model = YOLO(r"C:\Users\shiva\Downloads\Blindman_project\final\yolo\runs\detect\Yolo11_new_results\weights\best.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or change to 1, 2, etc., for external cameras

# Check if webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam.")
        break

    # Perform prediction and draw bounding boxes
    result_img, _ = predict_and_detect(model, img, classes=[], conf=0.5)

    # Display the output
    cv2.imshow("Webcam Live Detection", result_img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
