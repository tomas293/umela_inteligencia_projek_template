from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def train():
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        name="moj_model"
    )

if __name__ == "__main__":
    train()
    print("Tréning dokončený")