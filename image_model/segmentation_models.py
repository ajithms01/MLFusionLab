from ultralytics import YOLO


def segNano():
    return YOLO("yolov8n-seg.pt")

def segMed():
    return YOLO("yolov8m-seg.pt")

def segSmall():
    return YOLO("yolov8s-seg.pt")