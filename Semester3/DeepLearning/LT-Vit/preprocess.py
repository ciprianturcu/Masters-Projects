import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

SOURCE_DIR = os.path.join(
    os.path.dirname(__file__),
    "datasets", "nih-chest-xrays", "data", "versions", "3",
)
DEST_DIR = os.path.join(
    os.path.dirname(__file__),
    "datasets", "nih-chest-xrays", "data", "versions", "3_resized",
)
DEST_DIR_JPG = os.path.join(
    os.path.dirname(__file__),
    "datasets", "nih-chest-xrays", "data", "versions", "3_jpg",
)


def process_single_image(src_path, dest_path, target_size, fmt="png"):
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        img = Image.open(src_path).convert("RGB")
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        if fmt == "jpg":
            img.save(dest_path, format="JPEG", optimize=True, quality=95)
        else:
            img.save(dest_path, optimize=True)
        return True
    except Exception as e:
        print(f"  ERROR: {src_path} -> {e}")
        return False


def collect_image_tasks(source_dir, dest_dir, fmt="png"):
    tasks = []
    for folder in sorted(os.listdir(source_dir)):
        img_dir = os.path.join(source_dir, folder, "images")
        if not os.path.isdir(img_dir):
            continue
        dest_img_dir = os.path.join(dest_dir, folder, "images")
        for fname in os.listdir(img_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                src = os.path.join(img_dir, fname)
                if fmt == "jpg":
                    dst_fname = os.path.splitext(fname)[0] + ".jpg"
                else:
                    dst_fname = fname
                dst = os.path.join(dest_img_dir, dst_fname)
                if not os.path.exists(dst):
                    tasks.append((src, dst))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Preprocess NIH-CXR14 images")
    parser.add_argument("--size", type=int, default=224,
                        help="Target image size (default 224)")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4,
                        help="Number of threads for parallel processing")
    parser.add_argument("--fmt", choices=["png", "jpg"], default="png",
                        help="Output format: png (default) or jpg (faster decode)")
    args = parser.parse_args()

    target_size = (args.size, args.size)
    dest_dir = DEST_DIR_JPG if args.fmt == "jpg" else DEST_DIR
    print(f"Source:  {SOURCE_DIR}")
    print(f"Dest:    {dest_dir}")
    print(f"Format:  {args.fmt}")
    print(f"Size:    {target_size}")
    print(f"Threads: {args.threads}")

    import shutil
    for fname in os.listdir(SOURCE_DIR):
        src = os.path.join(SOURCE_DIR, fname)
        dst = os.path.join(dest_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  Copied: {fname}")

    tasks = collect_image_tasks(SOURCE_DIR, dest_dir, fmt=args.fmt)
    total = len(tasks)

    if total == 0:
        print("\nAll images already preprocessed. Nothing to do.")
        return

    print(f"\nResizing {total} images...\n")
    done = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(process_single_image, src, dst, target_size, args.fmt): (src, dst)
            for src, dst in tasks
        }
        for future in as_completed(futures):
            done += 1
            if not future.result():
                failed += 1
            if done % 500 == 0 or done == total:
                pct = done / total * 100
                print(f"  Progress: {done}/{total} ({pct:.1f}%) - {failed} failures")

    print(f"\nDone. {done - failed}/{total} images resized successfully.")
    print(f"Output: {dest_dir}")


if __name__ == "__main__":
    main()
