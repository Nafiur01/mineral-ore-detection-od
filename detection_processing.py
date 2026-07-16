import openvino as ov
import os
import cv2
import time
import numpy as np 
import json


ROCK_CLASSES = {
    "yellow_siltstone": {
        "name": "Yellow Siltstone",
        "value_tier": "low",
        "value_rank": 5,
        "visual_cue": "yellow-tan, fine-grained, slabby",
        "description": "A fine-grained, yellow-tan rock that breaks into slabs and blocks. Its color comes from limonite (iron hydroxide) staining. Too soft for serious construction; mostly decorative filler.",
        "key_characteristics": {
            "appearance": "Mustard-yellow to tan, uniform and dull; smooth to slightly gritty surface",
            "texture": "Grain size between shale and sandstone; scratches easily with a knife",
            "origin": "Compacted silt from lakes, floodplains, and quiet marine settings"
        },
        "common_uses": [
            "Landscaping: garden fill, rockery accents, pathway edging",
            "Pigment: strongly colored pieces can be ground into yellow ochre"
        ],
        "price_usd": {
            "bulk_landscaping_per_ton": [20, 60],
            "individual_piece": [0, 0],
            "yellow_ochre_pigment_per_kg": [1, 4]
        }
    },
    "red_breccia": {
        "name": "Red Breccia",
        "value_tier": "high",
        "value_rank": 1,
        "visual_cue": "red-purple matrix with angular light clasts",
        "description": "A striking rock formed of angular fragments cemented in a fine matrix, featuring rich red, burgundy, and terracotta hues. Widely traded as a luxury decorative 'marble' in the stone industry.",
        "key_characteristics": {
            "appearance": "Bold mix of crimson, rose, and rust tones with contrasting angular light clasts",
            "texture": "Sharp, broken rock pieces locked in a finer matrix, not smooth rounded pebbles",
            "origin": "Forms from fault zones, landslides, or collapse debris; commercially quarried in Italy, China, and Iran"
        },
        "common_uses": [
            "Interiors: feature walls, hotel lobby cladding, fireplace surrounds",
            "Surfaces: luxury flooring, tabletops, statement countertops"
        ],
        "price_usd": {
            "polished_slab_per_sqft": [20, 60],
            "tile_per_sqft": [5, 15],
            "rough_decorative_boulder_per_ton": [150, 400],
            "unpolished_field_piece": [0, 5]
        }
    },
    "white_limestone": {
        "name": "White Limestone",
        "value_tier": "medium",
        "value_rank": 2,
        "visual_cue": "pale cream/white, chalky, sometimes veined",
        "description": "A pale calcium-carbonate rock with a chalky to compact surface, sometimes crossed by thin calcite veins. One of the most economically important rocks on earth due to industrial demand rather than beauty.",
        "key_characteristics": {
            "appearance": "Cream to white with an earthy or chalky finish; may show fine white veining",
            "texture": "Fine-grained and uniform; fizzes with dilute acid (definitive test)",
            "origin": "Forms in shallow marine environments from shell and coral debris; quarried worldwide"
        },
        "common_uses": [
            "Industry: cement production, lime, steelmaking flux, agricultural soil treatment",
            "Construction and decor: building blocks, landscaping stone, aquarium rock, floor tiles"
        ],
        "price_usd": {
            "crushed_aggregate_per_ton": [10, 40],
            "landscaping_dimension_grade_per_ton": [100, 300],
            "cut_tile_per_sqft": [3, 10],
            "individual_rough_piece": [0, 2]
        }
    },
    "shale": {
        "name": "Shale",
        "value_tier": "lowest",
        "value_rank": 6,
        "visual_cue": "pale, flaky, splits into thin sheets",
        "description": "A pale, flaky rock that splits into thin sheets along bedding planes. The most abundant sedimentary rock on Earth and the cheapest per kilogram, yet the most likely to hide a fossil.",
        "key_characteristics": {
            "appearance": "Pale gray to tan, dull, with visible thin layering",
            "texture": "Fissile: splits by hand or with a light tap into flat flakes; soft and crumbly",
            "origin": "Compacted clay and mud from deep, still water; found virtually everywhere"
        },
        "common_uses": [
            "Industry: raw feedstock for bricks, cement, and expanded lightweight aggregate",
            "Collecting: splitting shale layers is the classic method of fossil hunting"
        ],
        "price_usd": {
            "raw_per_ton": [5, 25],
            "individual_piece": [0, 0],
            "fossil_specimen_if_found": [5, 300]
        }
    },
    "iron_mudstone": {
        "name": "Iron Mudstone",
        "value_tier": "low_medium",
        "value_rank": 4,
        "visual_cue": "red-brown, iron-stained, lumpy",
        "description": "A red-brown, iron-stained (ferruginous) rock with a lumpy, sometimes concretionary form. Nearly worthless as stone but carries two niche opportunities: red ochre pigment and possible fossils inside concretions.",
        "key_characteristics": {
            "appearance": "Rusty red-brown to maroon, often with a cracked, nodular surface",
            "texture": "Fine-grained mud base hardened and stained by iron oxides; heavier than it looks",
            "origin": "Iron-rich sediment deposited in swamps, deltas, or oxidizing environments"
        },
        "common_uses": [
            "Pigment: ground into red ochre for paints and coatings",
            "Collecting: concretions occasionally split open to reveal fossils"
        ],
        "price_usd": {
            "raw_fill_per_ton": [5, 15],
            "red_ochre_pigment_per_kg": [1, 5],
            "fossil_in_concretion_if_found": [10, 300]
        }
    },
    "gray_sandstone": {
        "name": "Gray Sandstone",
        "value_tier": "medium",
        "value_rank": 3,
        "visual_cue": "gray-brown, granular weathered surface",
        "description": "A granular rock of cemented sand grains with a gray-brown weathered surface. The workhorse of natural building stone: unglamorous but consistently in demand.",
        "key_characteristics": {
            "appearance": "Gray to gray-brown, matte, visibly granular like compacted sand",
            "texture": "Gritty to the touch; grains can sometimes be rubbed loose on weathered surfaces",
            "origin": "Compacted river, beach, or desert sand; quarried on every continent"
        },
        "common_uses": [
            "Construction: paving slabs, garden walls, flagstone paths, wall cladding",
            "Landscaping: steps, edging, retaining walls"
        ],
        "price_usd": {
            "flagstone_per_ton": [200, 600],
            "paving_per_sqft": [2, 8],
            "crushed_per_ton": [15, 50],
            "individual_field_piece": [0, 2]
        }
    },
    "iron_ore":{
        "name": "Iron Ore",
        "value_tier": "medium",
        "value_rank": 3,
        "visual_cue": "dark red-brown to black, heavy, metallic luster or earthy red streak",
        "description": "A dense, iron-rich rock mined primarily for smelting into metallic iron and steel. Common forms include hematite (reddish, earthy to metallic) and magnetite (black, strongly magnetic). Valued industrially rather than as a collector's or decorative stone.",
        "key_characteristics": {
            "appearance": "Ranges from dull brick-red (hematite) to shiny steel-gray/black (magnetite); often banded or botryoidal (kidney-shaped) surfaces",
            "texture": "Heavy for its size (high specific gravity, ~4.5-5.3); hematite gives a red-brown streak, magnetite is strongly attracted to a magnet",
            "origin": "Formed in Precambrian banded iron formations (BIFs) from ancient ocean chemistry, or in sedimentary/hydrothermal deposits where iron oxides precipitated and concentrated over geologic time"
        },
        "common_uses": [
            "Industrial: primary feedstock for iron and steel production via blast furnaces",
            "Pigment: hematite varieties ground into red ochre for paints and dyes",
            "Ballast and heavy aggregate in specialized construction applications"
        ],
        "price_usd": {
            "bulk_industrial_per_ton": [80, 130],
            "individual_piece": [0, 5],
            "red_ochre_pigment_per_kg": [2, 6]
        }
    }
}

def get_rock_info(label: str) -> dict:
    return ROCK_CLASSES.get(label, {
        "name": label.replace("_", " ").title(),
        "value_tier": "unknown",
        "value_rank": "-",
        "visual_cue": "",
        "description": f"No detailed data available for '{label}'.",
        "key_characteristics": {"appearance": "-", "texture": "-", "origin": "-"},
        "common_uses": [],
        "price_usd": {},
    })


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


