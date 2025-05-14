import os
import argparse
from pathlib import Path
import pydicom
import numpy as np
import cv2
from tqdm import tqdm

def load_dicom_series(series_dir):
    dicom_files = sorted([f for f in os.listdir(series_dir) if f.endswith(".dcm")])
    slices = [pydicom.dcmread(os.path.join(series_dir, f)) for f in dicom_files]
    images = [s.pixel_array for s in slices]
    return images

def save_image(img: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)

def process_series(series_dir, output_dir, resize_factor=2):
    patient_id = Path(series_dir).parts[-2]
    series_uid = Path(series_dir).name
    images = load_dicom_series(series_dir)

    for idx, img in enumerate(images):
        img = img.astype(np.int16)

        # Contrast normalization (robust window)
        min_val, max_val = np.percentile(img, [1, 99])
        img_norm = np.clip((img - min_val) / (max_val - min_val), 0, 1)
        img_8bit = (img_norm * 255).astype(np.uint8)

        img_name = f"{series_uid}_{idx:03d}.png"

        # Save high-res normalized image under patient folder
        hr_path = output_dir / "images_hr" / patient_id / img_name
        save_image(img_8bit, hr_path)

        # Downsample and save low-res version
        h, w = img_8bit.shape
        lr_img = cv2.resize(img_8bit, (w // resize_factor, h // resize_factor), interpolation=cv2.INTER_CUBIC)
        lr_path = output_dir / f"images_lr_x{resize_factor}" / patient_id / img_name
        save_image(lr_img, lr_path)

def main():
    parser = argparse.ArgumentParser(description="Process CT DICOM series into HR/LR PNGs.")
    parser.add_argument('--collection', type=str, required=True, help='Name of the dataset/collection')
    parser.add_argument('--resize_factor', type=int, default=2, help='Downsampling factor (default: 2)')
    args = parser.parse_args()

    input_root = Path("/mnt/tcia_data/raw") / args.collection
    output_root = Path("/mnt/tcia_data/processed") / args.collection

    print(f"[INFO] Scanning collection: {args.collection}")
    series_dirs = [d for d in input_root.glob("*/*") if d.is_dir()]  # patient/series_uid

    for series_dir in tqdm(series_dirs, desc="Processing series"):
        try:
            process_series(series_dir, output_root, resize_factor=args.resize_factor)
        except Exception as e:
            print(f"[ERROR] Failed to process {series_dir}: {e}")

if __name__ == "__main__":
    main()
