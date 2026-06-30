#!/usr/bin/env python3
"""Download trained model from Modal volume."""
import modal
from pathlib import Path

app = modal.App("download-tool")
vol = modal.Volume.from_name("yolo-models", create_if_missing=True)

@app.function(volumes={"/models": vol})
def download():
    tflite_bytes = Path("/models/model.tflite").read_bytes()
    labels_data = Path("/models/labels.txt").read_text()
    size = len(tflite_bytes)
    print(f"Model size: {size} bytes")
    print(f"labels:\n{labels_data}")
    return {"tflite": tflite_bytes, "labels": labels_data, "size": size}

@app.local_entrypoint()
def main():
    result = download.remote()
    local_dir = Path("/tmp/trained_model")
    local_dir.mkdir(parents=True, exist_ok=True)
    
    (local_dir / "model.tflite").write_bytes(result["tflite"])
    (local_dir / "labels.txt").write_text(result["labels"])
    
    print(f"Model ({result['size']} bytes) saved to {local_dir}")
    print(f"Labels:\n{result['labels']}")
    print("Done!")

if __name__ == "__main__":
    main()
