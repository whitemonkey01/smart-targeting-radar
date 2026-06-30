import modal
from pathlib import Path

app = modal.App("yolo-military-training")
vol = modal.Volume.from_name("yolo-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "ultralytics",
        "datasets",
        "huggingface_hub",
        "pyyaml",
        "Pillow>=10.0.0",
        "scikit-learn",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
    )
)

# Weapon class remap: clean categories from messy 29 classes
WEAPON_CLASS_MAP = {
    "person": 0, "Aggressor": 0, "Victim": 0,
    "gun": 1, "Guns": 1, "guns": 1, "handgun": 1, "Pistol": 1, "pistol": 1, "pistols": 1,
    "rifle": 2, "Rifle": 2, "Shotgun": 2, "shotgun": 2, "Long guns": 2, "long guns": 2, "Heavy Gun": 2, "heavyweapon": 2, "heavy weapon": 2,
    "knife": 3, "Knife": 3, "Knife_Deploy": 3, "Knife_Weapon": 3, "Stabbing": 3,
}

CLASS_NAMES = {0: "person", 1: "gun", 2: "rifle", 3: "knife"}

@app.function(gpu="T4", timeout=7200, image=image, volumes={"/models": vol})
def train():
    import shutil
    import yaml
    import os
    from PIL import Image
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    base = Path("/data")
    base.mkdir(parents=True, exist_ok=True)
    dataset_dir = base / "military_dataset"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    for d in [images_dir, labels_dir, train_dir / "images", train_dir / "labels",
              val_dir / "images", val_dir / "labels"]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Download weapon detection dataset from Hugging Face
    print("Downloading weapon detection dataset...")
    weapon_ds = load_dataset("Subh775/WeaponDetection", split="train")

    # Build class ID -> name map from features
    cat_feature = weapon_ds.features["objects"]["category"]
    id_to_name = {i: n for i, n in enumerate(cat_feature.feature.names)}

    img_id = 0
    for item in weapon_ds:
        img = item["image"]
        w, h = img.size
        objects = item["objects"]
        if len(objects["category"]) == 0:
            continue

        labels = []
        for bbox, cat_id_val in zip(objects["bbox"], objects["category"]):
            cat_name = id_to_name[cat_id_val]
            new_id = WEAPON_CLASS_MAP.get(cat_name)
            if new_id is None:
                continue
            x, y, bw, bh = bbox
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bw_norm = bw / w
            bh_norm = bh / h
            labels.append(f"{new_id} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}")

        if not labels:
            continue

        fname = f"img_{img_id:06d}.jpg"
        img.save(str(images_dir / fname), "JPEG")
        with open(labels_dir / f"{Path(fname).stem}.txt", "w") as f:
            f.write("\n".join(labels))
        img_id += 1

    print(f"Saved {img_id} weapon detection images")

    # 2. Download military aircraft dataset (snapshot → pseudo-labels)
    print("Downloading military aircraft dataset...")
    try:
        aircraft_dir = Path("/data/aircraft_raw")
        aircraft_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="Ahnuf/Military_Aircraft_Detection_Classification_Image_Dataset",
            repo_type="dataset",
            local_dir=str(aircraft_dir),
            max_workers=4,
        )

        # Find image files recursively
        ext_map = {".jpg": 1, ".jpeg": 1, ".png": 1, ".webp": 1}
        image_files = [p for p in aircraft_dir.rglob("*") if p.suffix.lower() in ext_map]
        print(f"Found {len(image_files)} aircraft images in snapshot")

        aircraft_count = 0
        aircraft_class_offset = 4
        class_names_extended = CLASS_NAMES.copy()

        # Try to infer class from directory structure (subdirectories = class names)
        dir_to_class = {}
        for img_path in image_files:
            parent = img_path.parent
            class_name = parent.name
            if class_name not in dir_to_class:
                dir_to_class[class_name] = len(dir_to_class)

        for img_path in image_files:
            parent = img_path.parent
            class_name = parent.name
            class_id = dir_to_class[class_name]
            new_class = class_id + aircraft_class_offset
            class_names_extended[new_class] = f"aircraft_{class_name}"

            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            margin = 0.05
            x, y, bw, bh = margin, margin, 1 - 2 * margin, 1 - 2 * margin
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bw_norm = bw / w
            bh_norm = bh / h
            label = f"{new_class} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}"

            fname = f"aircraft_{aircraft_count:06d}.jpg"
            img.save(str(images_dir / fname), "JPEG")
            with open(labels_dir / f"{Path(fname).stem}.txt", "w") as f:
                f.write(label + "\n")
            aircraft_count += 1

        CLASS_NAMES.update(class_names_extended)
        nc = max(CLASS_NAMES.keys()) + 1
        print(f"Added {aircraft_count} aircraft images across {len(dir_to_class)} classes, total classes: {nc}")

    except Exception as e:
        print(f"Warning: Could not load aircraft dataset ({e}), continuing with weapon data only")

    nc = max(CLASS_NAMES.keys()) + 1
    names = [CLASS_NAMES[i] for i in range(nc)]

    # 3. Split into train/val
    from sklearn.model_selection import train_test_split
    all_images = sorted(images_dir.glob("*"))
    train_imgs, val_imgs = train_test_split(all_images, test_size=0.15, random_state=42)

    for img in train_imgs:
        shutil.copy(str(img), str(train_dir / "images" / img.name))
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy(str(lbl), str(train_dir / "labels" / lbl.name))

    for img in val_imgs:
        shutil.copy(str(img), str(val_dir / "images" / img.name))
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy(str(lbl), str(val_dir / "labels" / lbl.name))

    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Classes: {nc}")

    # 4. Create data.yaml
    data_yaml = {
        "train": str(train_dir / "images"),
        "val": str(val_dir / "images"),
        "nc": nc,
        "names": names,
    }
    yaml_path = base / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    # 5. Train YOLOv8n
    print("Starting YOLOv8n training...")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=640,
        batch=32,
        patience=20,
        device=0,
        workers=4,
        lr0=0.01,
        cos_lr=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        project="/data/runs",
        name="military",
        exist_ok=True,
    )

    # 6. Export to TFLite float32
    print("Exporting to TFLite...")
    best_model_path = Path("/data/runs/military/weights/best.pt")
    if not best_model_path.exists():
        print(f"Model not found at {best_model_path}, checking runs...")
        import glob as g
        found = g.glob("/data/runs/**/best.pt", recursive=True)
        if found:
            best_model_path = Path(found[0])
        else:
            raise Exception("Training failed - no best.pt found")

    exported = model.export(format="tflite", imgsz=640, int8=False)
    print(f"TFLite exported at: {exported}")

    # 7. Also export labels
    labels_path = base / "custom_labels.txt"
    with open(labels_path, "w") as f:
        for i in range(nc):
            f.write(names[i] + "\n")

    # 8. Copy outputs to volume
    import shutil
    models_dir = Path("/models")
    models_dir.mkdir(exist_ok=True)
    shutil.copy(str(exported), str(models_dir / "model.tflite"))
    shutil.copy(str(labels_path), str(models_dir / "labels.txt"))
    vol.commit()

    tflite_size = Path(exported).stat().st_size
    print(f"TFLite size: {tflite_size} bytes")
    print(f"Done! Model saved to volume.")

    return {
        "tflite_size": tflite_size,
        "num_classes": nc,
        "class_names": names,
        "train_images": len(train_imgs),
        "val_images": len(val_imgs),
    }

@app.local_entrypoint()
def download_model():
    import shutil
    print("Downloading trained model from Modal volume...")
    models_dir = Path("/models")
    if not models_dir.exists():
        models_dir.mkdir(parents=True)
        vol.reload()

    tflite_path = models_dir / "model.tflite"
    labels_path = models_dir / "labels.txt"
    labels_data = labels_path.read_text()

    local_dir = Path("/tmp/trained_model")
    local_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(str(tflite_path), str(local_dir / "model.tflite"))
    shutil.copy(str(labels_path), str(local_dir / "labels.txt"))
    print(f"Model downloaded to {local_dir}")
    print(f"labels.txt content:\n{labels_data}")

@app.local_entrypoint()
def main():
    print("Starting YOLO training on Modal with T4 GPU...")
    result = train.remote()
    print(f"Training complete!")
    print(f"Classes: {result['num_classes']}")
    print(f"Class names: {result['class_names']}")
    print(f"Train images: {result['train_images']}")
    print(f"Val images: {result['val_images']}")
    print(f"TFLite model size: {result['tflite_size']} bytes")
    print("Model saved to volume 'yolo-models'.")
