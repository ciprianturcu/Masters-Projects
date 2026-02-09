import os
import json
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

SOURCE_DIR = os.path.join(
    os.path.dirname(__file__),
    "datasets", "nih-chest-xrays", "data", "versions", "3_resized",
)
DEST_DIR = os.path.join(
    os.path.dirname(__file__),
    "datasets", "nih-chest-xrays", "data", "versions", "3_memmap",
)
IMAGE_SIZE = 224


def collect_images(source_dir):
    images = {}
    for folder in sorted(os.listdir(source_dir)):
        img_dir = os.path.join(source_dir, folder, "images")
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                images[fname] = os.path.join(img_dir, fname)
    return images


def load_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8).copy()
    img.close()
    return arr


def main():
    parser = argparse.ArgumentParser(description="Build memmap from preprocessed images")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--batch", type=int, default=500,
                        help="Images per batch (controls peak RAM usage)")
    args = parser.parse_args()

    print(f"Source:  {SOURCE_DIR}")
    print(f"Dest:    {DEST_DIR}")
    print(f"Threads: {args.threads}, Batch: {args.batch}")

    images = collect_images(SOURCE_DIR)
    total = len(images)
    print(f"\nFound {total} images")

    if total == 0:
        print("No images found. Run preprocess.py first.")
        return

    filenames = sorted(images.keys())
    fname_to_idx = {fname: i for i, fname in enumerate(filenames)}

    os.makedirs(DEST_DIR, exist_ok=True)

    memmap_path = os.path.join(DEST_DIR, "images.dat")
    shape = (total, IMAGE_SIZE, IMAGE_SIZE, 3)
    nbytes = total * IMAGE_SIZE * IMAGE_SIZE * 3
    print(f"Memmap shape: {shape}")
    print(f"Memmap size:  {nbytes / 1e9:.2f} GB")

    mmap = np.memmap(memmap_path, dtype=np.uint8, mode="w+", shape=shape)

    print(f"\nPacking images into memmap...\n")
    failed = 0

    for batch_start in range(0, total, args.batch):
        batch_end = min(batch_start + args.batch, total)
        batch_fnames = filenames[batch_start:batch_end]
        batch_paths = [images[fn] for fn in batch_fnames]

        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            results = list(executor.map(load_image, batch_paths))

        for i, arr in enumerate(results):
            if arr is not None:
                mmap[batch_start + i] = arr
            else:
                failed += 1

        mmap.flush()
        pct = batch_end / total * 100
        print(f"  Progress: {batch_end}/{total} ({pct:.1f}%) - {failed} failures")

    del mmap

    index_path = os.path.join(DEST_DIR, "index.json")
    metadata = {
        "shape": list(shape),
        "dtype": "uint8",
        "fname_to_idx": fname_to_idx,
    }
    with open(index_path, "w") as f:
        json.dump(metadata, f)

    import shutil
    for fname in os.listdir(SOURCE_DIR):
        src = os.path.join(SOURCE_DIR, fname)
        dst = os.path.join(DEST_DIR, fname)
        if os.path.isfile(src) and not fname.endswith((".png", ".jpg", ".jpeg", ".dat")):
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"  Copied: {fname}")

    print(f"\nDone. {total - failed}/{total} images packed.")
    print(f"Memmap: {memmap_path}")
    print(f"Index:  {index_path}")


if __name__ == "__main__":
    main()
