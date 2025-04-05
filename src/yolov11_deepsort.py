#!/usr/bin/env python3

import sys
import os
sys.path.append("/home/a/catkin_ws/src/yolov11_deepsort/ultralytics")  # 예시: deep_sort가 이 안에 있음
os.system("source /home/a/anaconda3/bin/activate yolov11")
sys.path.append("/home/a/anaconda3/envs/yolov11/lib/python3.8/site-packages")

import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import torch
from ultralytics import YOLO
from deep_sort.deep_sort import DeepSort
import numpy as np
from collections import deque

# Initialize ROS node
rospy.init_node('yolov11_deepsort_node')

# Initialize YOLOv11 model
yolo_model = YOLO("yolo11n.pt")

# Initialize DeepSORT tracker
deep_sort_tracker = DeepSort(
    model_path="/home/a/catkin_ws/src/yolov11_deepsort/ultralytics/deep_sort/deep_sort/deep/checkpoint/ckpt.t7",  # 정확한 .t7 모델 파일로 수정
    max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0,
    max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100,
    use_cuda=torch.cuda.is_available()
)

data_deque = {}

# Drawing function
def draw_boxes(img, bbox, object_id, identities=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        id = int(identities[i]) if identities is not None else 0
        label = f"ID {id}"
        color = (0, 255, 0)

        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)

        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        data_deque[id].appendleft(center)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for j in range(1, len(data_deque[id])):
            if data_deque[id][j - 1] is None or data_deque[id][j] is None:
                continue
            cv2.line(img, data_deque[id][j - 1], data_deque[id][j], color, 2)

        cv2.circle(img, center, 5, (0, 0, 255), -1)
    return img

# Image callback
def image_callback(msg):
    try:
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        rospy.logerr(f"CompressedImage decode error: {e}")
        return

    frame = cv2.flip(frame, 1)

    results = yolo_model(frame)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        cv2.imshow("YOLOv11 + DeepSORT", frame)
        cv2.waitKey(1)
        return

    bbox_xyxy = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()

    xywh_bboxs = []
    confs = []
    oids = []
    for i, bbox in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x_c = x1 + w / 2
        y_c = y1 + h / 2
        xywh_bboxs.append([x_c, y_c, w, h])
        confs.append([confidences[i]])
        oids.append(int(class_ids[i]))

    xywhs = torch.Tensor(xywh_bboxs)
    confss = torch.Tensor(confs)

    outputs = deep_sort_tracker.update(xywhs, confss, oids, frame)

    if len(outputs) > 0:
        outputs = np.array(outputs)
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -2]
        object_id = outputs[:, -1]
        draw_boxes(frame, bbox_xyxy, object_id, identities)

    cv2.imshow("YOLOv11 + DeepSORT", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key to exit
        rospy.signal_shutdown("User exited.")

# ROS Subscriber
image_sub = rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, image_callback)

rospy.loginfo("✅ YOLOv11 + DeepSORT ROS Node Started.")
rospy.spin()
cv2.destroyAllWindows()
