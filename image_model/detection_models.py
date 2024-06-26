import torchvision
from ultralytics import YOLO

def fasterRCNN(output_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, output_classes)
    return model

def yolonano():
    return YOLO("yolov8n.pt")

def yolosmall():
    return YOLO("yolov8s.pt")

def yolomed():
    return YOLO("yolov8m.pt")