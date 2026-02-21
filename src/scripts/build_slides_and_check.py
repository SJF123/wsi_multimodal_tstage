import argparse
from pathlib import Path
import pandas as pd

def parse_ids(name: str):
    # name: TCGA-4N-A93T-01A-01-TS1.xxxxx.png
    stem = Path(name).stem
    slide_id = stem.split(".")[0]
    patient_id = "-".join(slide_id.split("-")[:3])
    return patient_id, slide_id

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["confirm", "write"])
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--clinical_csv", required=True)
    ap.add_argument("--out_csv", default="data/slides.csv")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    imgs = sorted([p for p in img_dir.glob("*.png")])

    rows = []
    for p in imgs:
        pid, sid = parse_ids(p.name)
        rows.append({"patient_id": pid, "slide_id": sid, "png_path": str(p.resolve())})

    slides = pd.DataFrame(rows)
    print("Found PNGs:", len(slides))
    print("Unique patients in PNGs:", slides["patient_id"].nunique())
    print("Unique slides in PNGs:", slides["slide_id"].nunique())

    clinical = pd.read_csv(args.clinical_csv_confirm if False else args.clinical_csv)
    # 你的 clinical 主键：case_submitter_id
    clinical_pids = set(clinical["case_submitter_id"].astype(str).tolist())

    png_pids = set(slides["patient_id"].astype(str).tolist())
    matched = png_pids & clinical_pids
    print("\nAlignment check (patient_id):")
    print("  PNG patient_id matched in clinical:", len(matched), f"({len(matched)/max(len(png_pids),1):.2%})")
    print("  PNG patient_id NOT in clinical:", len(png_pids - clinical_pids))
    print("  examples:", list(sorted(png_pids - clinical_pids))[:10])

    if args.mode == "write":
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        slides.to_csv(args.out_csv, index=False)
        print("\nSaved:", args.out_csv)

if __name__ == "__main__":
    main()
#python -m src.scripts.build_slides_and_check confirm --img_dir "E:\CRC-DL\02-生存预测\TCGA-DATA(png)" --clinical_csv data/cleaned_data.csv
