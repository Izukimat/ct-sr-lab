import argparse
import random
from pathlib import Path

def get_patient_ids(images_dir: Path):
    return sorted([p.name for p in images_dir.iterdir() if p.is_dir()])

def write_split(paths, out_file):
    with open(out_file, "w") as f:
        for path in paths:
            f.write(str(path) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", type=str, default="/mnt/tcia_data/processed", help="Root processed folder")
    parser.add_argument("--collection", type=str, default="NSCLC-Radiomics", help="Collection name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    hr_root = Path(args.processed_root) / args.collection / "images_hr"
    lr_root = Path(args.processed_root) / args.collection / "images_lr_x2"
    split_dir = Path(args.processed_root) / args.collection / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    patients = get_patient_ids(hr_root)
    random.seed(args.seed)
    random.shuffle(patients)

    num_total = len(patients)
    num_train = int(0.7 * num_total)
    num_val = int(0.1 * num_total)

    train_patients = patients[:num_train]
    val_patients = patients[num_train:num_train + num_val]
    test_patients = patients[num_train + num_val:]

    for split_name, patient_list in [("train", train_patients), ("val", val_patients), ("test", test_patients)]:
        hr_paths = []
        lr_paths = []

        for pid in patient_list:
            hr_paths += list((hr_root / pid).glob("*.png"))
            lr_paths += list((lr_root / pid).glob("*.png"))

        write_split(hr_paths, split_dir / f"{split_name}_hr.txt")
        write_split(lr_paths, split_dir / f"{split_name}_lr.txt")
        print(f"{split_name.upper()} - {len(patient_list)} patients | {len(hr_paths)} HR slices")

if __name__ == "__main__":
    main()
