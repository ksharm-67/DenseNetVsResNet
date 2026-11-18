import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image

SIZE = 224


def load(path):
    img = Image.open(path).convert("L").resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/kaggle/input/datasets/nih-chest-xrays/data")
    p.add_argument("--out_dir", default="/kaggle/temp/cache")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_root, "images_*", "images", "*.png")))
    if not paths:
        raise FileNotFoundError("No images found, check --data_root")
    os.makedirs(args.out_dir, exist_ok=True)
    names = [os.path.basename(x) for x in paths]
    with open(os.path.join(args.out_dir, "names.json"), "w") as f:
        json.dump(names, f)

    arr = np.lib.format.open_memmap(
        os.path.join(args.out_dir, "images.npy"), mode="w+", dtype=np.uint8, shape=(len(paths), SIZE, SIZE)
    )
    with Pool(args.workers) as pool:
        for i, a in enumerate(pool.imap(load, paths, chunksize=64)):
            arr[i] = a
            if i % 5000 == 0:
                print(f"{i}/{len(paths)}", flush=True)
    arr.flush()
    print(f"Done, {len(paths)} images cached to {args.out_dir}")


if __name__ == "__main__":
    main()
