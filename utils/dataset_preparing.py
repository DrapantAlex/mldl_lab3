import os
import shutil
import urllib.request
import zipfile

def prepare_tiny_imagenet(data_dir="dataset/tiny-imagenet-200"):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join("dataset", "tiny-imagenet-200.zip")

    if not os.path.exists(data_dir):
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data")

    with open('dataset/tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt') as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            os.makedirs(f'dataset/tiny-imagenet/tiny-imagenet-200/val/{cls}', exist_ok=True)
            shutil.copyfile(f'datset/tiny-imagenet/tiny-imagenet-200/val/images/{fn}', f'dataset/tiny-imagenet/tiny-imagenet-200/val/{cls}/{fn}')
    shutil.rmtree('dataset/tiny-imagenet/tiny-imagenet-200/val/images')
