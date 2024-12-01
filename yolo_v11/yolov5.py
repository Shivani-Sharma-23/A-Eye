if __name__ == "__main__":
    from ultralytics import YOLO

    model = YOLO('yolov5n.pt')

    results = model.train(
        data= "C:/Users/shiva/Downloads/Blindman_project/final/Doors-and-walls-object-detection-1/blindman_project-1/data.yaml",
        epochs=50,
        batch=32,
        imgsz=320,
        name='Yolov5_new_results'
    )