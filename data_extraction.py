#!/usr/bin/env python3
"""
Extract images + YOLO annotations from the mineral5k CSV, organized by matched
mineral keyword, with class ids assigned at the ORE level (so multiple mineral
sub-folders belonging to the same ore share one class id and merge cleanly).

Folder / file naming:
    - If the matched keyword equals the ore name itself      -> "<ore>"
    - If the matched keyword is a specific mineral of the ore -> "<ore>(<mineral>)"
      e.g. "copper(bornite)", "copper(chalcopyrite)", "iron(magnetite)", "gold"

Output layout:
    <OUT_DIR>/
        copper(bornite)/
            images/copper(bornite)_1.jpg
            ann/copper(bornite)_1.txt      # "<ore_class_id> cx cy w h" per line
        copper/
            images/copper_1.jpg
            ann/copper_1.txt
        gold/
            images/gold_1.jpg
            ann/gold_1.txt
        ...

Matching is plain case-insensitive substring match against en_name (same as your
`if v in stem` logic), first match wins, checked in ore_name dict order.

Just edit CONFIG below and run:  python extract_by_ore.py
"""

import csv
import json
import shutil
from pathlib import Path

# ============================== CONFIG ==============================

CSV_PATH = "minerals_full.csv"                 # path to the CSV file
IMAGES_ROOT = "./mineral_images"       # root that the CSV 'path' column is relative to
OUT_DIR = "./extracted"        # where to write the extracted dataset
PIXEL_BOXES = False                        # True if mineral_boxes coords are pixels, not normalized 0-1
COPY_IMAGES = True                         # True = copy files, False = hardlink (falls back to copy)

ore_name = {
    "iron": ['magnetite','hematite'],
    "gold": ['gold'],
    "silver": ['silver','argentite'],
    "copper": ['copper','chalcopyrite','chalcocite','covellite','bornite'],
}

target_ore_classes = {
    "iron": 0,
    "gold": 1,
    "silver": 2,
    "copper": 3,
}

# Only include CSV rows whose 'path' contains one of these top-level data folders.
# Set to None to disable filtering and process every row.
TARGET_DIRS = [
    "1_syst",
    "2_mest",
    "3_op",
    "4_cryst",
    "5_PDK",
    "7_stepanov",
    "10_meteor",
]

# ======================================================================


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def box_to_yolo(box, img_w, img_h, pixel_boxes):
    xmin, ymin, xmax, ymax = box
    if pixel_boxes:
        xmin, xmax = xmin / img_w, xmax / img_w
        ymin, ymax = ymin / img_h, ymax / img_h

    xmin, xmax = sorted((clamp01(xmin), clamp01(xmax)))
    ymin, ymax = sorted((clamp01(ymin), clamp01(ymax)))

    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return xc, yc, w, h


def path_in_target_dirs(rel_path: str) -> bool:
    if not TARGET_DIRS:
        return True
    parts = Path(rel_path).parts
    return any(td in parts for td in TARGET_DIRS)


def find_match(en_name: str):
    """First (ore, mineral) whose keyword is a substring of en_name, checked in dict order."""
    stem = en_name.lower()
    for ore, minerals in ore_name.items():
        for mineral in minerals:
            if mineral.lower() in stem:
                return ore, mineral
    return None


def label_name(ore: str, mineral: str) -> str:
    return ore if mineral == ore else f"{ore}({mineral})"


def main():
    csv_path = Path(CSV_PATH)
    images_root = Path(IMAGES_ROOT)
    out_dir = Path(OUT_DIR)

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    if not images_root.exists():
        raise SystemExit(f"images root not found: {images_root}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("No rows found in CSV.")

    out_dir.mkdir(parents=True, exist_ok=True)

    counters = {}       # label_name -> running counter
    n_ok = n_missing_img = n_bad_json = n_no_boxes = n_skipped_dir = n_unmatched = 0

    for row in rows:
        rid = row["id"].strip()
        rel_path = row["path"].strip()
        en_name = row.get("en_name", "").strip()

        if not path_in_target_dirs(rel_path):
            n_skipped_dir += 1
            continue

        match = find_match(en_name)
        if not match:
            n_unmatched += 1
            continue

        ore, mineral = match
        class_id = target_ore_classes[ore]
        lbl = label_name(ore, mineral)

        try:
            img_h = int(float(row["height"]))
            img_w = int(float(row["width"]))
        except (ValueError, KeyError):
            img_h = img_w = None

        src_img = images_root / rel_path
        if not src_img.exists():
            n_missing_img += 1
            print(f"  [missing image] {rid}: {src_img}")
            continue

        counters[lbl] = counters.get(lbl, 0) + 1 if counters.get(lbl,0) < 500 else counters.get(lbl, 0) + 0
        counter = counters[lbl]

        ext = src_img.suffix
        new_stem = f"{lbl}_{counter}"

        mineral_dir = out_dir / lbl
        img_out_dir = mineral_dir / "images"
        ann_out_dir = mineral_dir / "ann"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        ann_out_dir.mkdir(parents=True, exist_ok=True)

        dst_img = img_out_dir / f"{new_stem}{ext}"
        dst_ann = ann_out_dir / f"{new_stem}.txt"

        try:
            mineral_boxes = json.loads(row.get("mineral_boxes", "") or "[]")
        except json.JSONDecodeError:
            mineral_boxes = []
            n_bad_json += 1
            print(f"  [bad mineral_boxes JSON] {rid}")

        lines = []
        for entry in mineral_boxes:
            box = entry.get("box")
            if not box or len(box) != 4:
                continue
            xc, yc, w, h = box_to_yolo(box, img_w, img_h, PIXEL_BOXES)
            if w <= 0 or h <= 0:
                continue
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if not lines:
            n_no_boxes += 1


        if COPY_IMAGES and counter < 500:
            shutil.copy2(src_img, dst_img)
        # else:
        #     try:
        #         if dst_img.exists():
        #             dst_img.unlink()
        #         dst_img.hardlink_to(src_img)
        #     except (OSError, AttributeError):
        #         shutil.copy2(src_img, dst_img)

            with open(dst_ann, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

            n_ok += 1

    print("\nDone.")
    print(f"  Extracted OK: {n_ok}")
    for lbl, c in sorted(counters.items()):
        print(f"    {lbl}: {c}")
    print(f"  Skipped (outside TARGET_DIRS): {n_skipped_dir}")
    print(f"  Unmatched en_name (no ore/mineral keyword found): {n_unmatched}")
    print(f"  Missing images: {n_missing_img}")
    print(f"  Rows with no boxes (empty ann file): {n_no_boxes}")
    print(f"  Rows with bad mineral_boxes JSON: {n_bad_json}")


if __name__ == "__main__":
    main()