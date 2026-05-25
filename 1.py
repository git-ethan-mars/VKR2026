from ultralytics import YOLO

model = YOLO('experiments/weights/best.pt')
model('recognition_dataset/1_22b.jpg')[0].save()
