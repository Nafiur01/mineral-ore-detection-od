import openvino as ov
import os
import cv2
import time
import numpy as np 
import json


def read_and_compile_model(xml_path:str,bin_path:str):
    core = ov.Core()
    model = core.read_model(xml_path,bin_path)
    compiled_model = core.compile_model(model=model,device_name='CPU')
    print(compiled_model.inputs)
    print(compiled_model.outputs)
    return compiled_model

def load_labels(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    labels_str = config["model_parameters"]["labels"]
    labels = labels_str.split()          # ['gold', 'copper', 'silver', 'iron']
    
    return labels

def infer_frame(img, model, label_list, conf_threshold=0.1):
    if img is None:
        return []
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    N, C, H, W = model.inputs[0].shape
    resized = cv2.resize(img_rgb, dsize=(W, H))
    input_tensor = np.expand_dims(resized.transpose(2, 0, 1).astype(np.float32) / 255.0, 0)

    results = model([input_tensor])
    boxes = results[model.output("boxes")][0]    # (N, 5)
    class_ids = results[model.output("labels")][0]  # (N,)

    scores = boxes[:, 4]
    mask = scores >= conf_threshold

    kept_boxes = boxes[mask]
    kept_classes = class_ids[mask]

    scale_x = orig_w / W
    scale_y = orig_h / H
    kept_boxes[:, [0, 2]] *= scale_x
    kept_boxes[:, [1, 3]] *= scale_y

    detections = []
    for box, cls in zip(kept_boxes, kept_classes):
        x_min, y_min, x_max, y_max, score = map(float, box)
        detections.append({
            "bbox": [x_min, y_min, x_max, y_max],
            "label": label_list[int(cls)],
            "confidence": score
        })

    return detections


def infer_model(img_path, model, label_list, conf_threshold=0.1):
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        np_arr = np.frombuffer(img_path, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return infer_frame(img, model, label_list, conf_threshold)


def draw_detections(img, detections):
    annotated = img.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d['label']} {d['confidence']:.2f}"
        cv2.putText(
            annotated, label, (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
        )
    return annotated


def stream_video_frames(video_path, model, label_list, conf_threshold=0.1, preserve_fps=True):
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps = max(fps, 1.0)
        interval = 1.0 / fps
        start = time.perf_counter()
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = infer_frame(frame, model, label_list, conf_threshold)
            annotated = draw_detections(frame, detections)
            _, buf = cv2.imencode(".jpg", annotated)
            yield buf.tobytes(), detections

            if preserve_fps:
                target = start + (frame_idx + 1) * interval
                sleep_for = target - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
            frame_idx += 1
    finally:
        cap.release()





# read_and_compile_model(MODEL_XML,MODEL_BIN)
# infer_model(model=dt_model,img_path="./extracted/merged/Images/copper_12.jpg",label_list=labels)
# infer_model(model=dt_model,img_path="https://cdn.britannica.com/41/142441-050-7DFCF3B8/nuggets-flakes-dust.jpg")


