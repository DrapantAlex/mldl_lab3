import os
import shutil
import urllib.request
import zipfile

def prepare_tiny_imagenet(data_dir="data/tiny-imagenet-200"):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join("data", "tiny-imagenet-200.zip")

    if not os.path.exists(data_dir):
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data")

    # reorganize validation set
    val_dir = os.path.join(data_dir, "val")
    with open(os.path.join(val_dir, "val_annotations.txt")) as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
            shutil.copyfile(
                os.path.join(val_dir, "images", fn),
                os.path.join(val_dir, cls, fn)
            )
    shutil.rmtree(os.path.join(val_dir, "images"))
