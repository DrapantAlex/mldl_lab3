import os, zipfile, urllib.request, shutil
from pathlib import Path

URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

def prepare_tiny_imagenet(out_root="data"):
    out_root = Path(out_root)
    data_dir = out_root / "tiny-imagenet-200"
    zip_path = out_root / "tiny-imagenet-200.zip"
    out_root.mkdir(parents=True, exist_ok=True)

    # Scarica/estrai solo se mancano le cartelle chiave
    if not ((data_dir/"train").is_dir() and (data_dir/"val").is_dir()):
        if not zip_path.exists():
            urllib.request.urlretrieve(URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_root)
        zip_path.unlink(missing_ok=True)

    # Riorganizza val solo se serve (idempotente)
    val_dir = data_dir / "val"
    images_dir = val_dir / "images"
    ann_file = val_dir / "val_annotations.txt"
    if images_dir.is_dir() and ann_file.is_file():
        with ann_file.open("r") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2: 
                    continue
                fn, cls = parts[0], parts[1]
                cls_dir = val_dir / cls
                cls_dir.mkdir(exist_ok=True)
                src = images_dir / fn
                dst = cls_dir / fn
                if src.exists() and not dst.exists():
                    shutil.move(str(src), str(dst))
        try:
            images_dir.rmdir()
        except OSError:
            shutil.rmtree(images_dir, ignore_errors=True)

    # Sanity check
    if not (data_dir/"train").is_dir() or not any((val_dir).iterdir()):
        raise RuntimeError(f"Dataset incompleto in {data_dir}")

    return str(data_dir)

# ESEMPIO USO:
# root = prepare_tiny_imagenet("data")
# poi lancia: python train.py --data-root data/tiny-imagenet-200
