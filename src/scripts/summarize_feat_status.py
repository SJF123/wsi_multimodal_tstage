from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Summarize ViT feature extraction status by comparing slides.csv with generated .pt files."
    )
    parser.add_argument(
        "--slides_csv",
        type=str,
        default="data/slides_400_matched.csv",
        help="CSV that contains at least a 'slide_id' column (default: data/slides_400_matched.csv)",
    )
    parser.add_argument(
        "--pt_dir",
        type=str,
        default="features/vit_patch_feats",
        help="Directory containing generated .pt files (default: features/vit_patch_feats)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="logs",
        help="Output directory for reports (default: logs)",
    )
    args = parser.parse_args()

    slides_csv = Path(args.slides_csv)
    pt_dir = Path(args.pt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load slide list
    if not slides_csv.exists():
        raise FileNotFoundError(f"slides_csv not found: {slides_csv}")

    slides = pd.read_csv(slides_csv)
    if "slide_id" not in slides.columns:
        raise ValueError("slides_csv must contain a column named 'slide_id'")

    slide_ids = slides["slide_id"].astype(str).tolist()

    # 2) list pt files
    if not pt_dir.exists():
        raise FileNotFoundError(f"pt_dir not found: {pt_dir}")

    pt_ids = set(p.stem for p in pt_dir.glob("*.pt"))

    # 3) compare
    missing = [sid for sid in slide_ids if sid not in pt_ids]
    existing = [sid for sid in slide_ids if sid in pt_ids]

    # 4) save reports
    missing_path = out_dir / "missing_slides.csv"
    existing_path = out_dir / "existing_slides.csv"

    pd.DataFrame({"slide_id": missing}).to_csv(missing_path, index=False)
    pd.DataFrame({"slide_id": existing}).to_csv(existing_path, index=False)

    # 5) print summary
    print("=== Feature Extraction Status ===")
    print(f"Slides in CSV:   {len(slide_ids)}")
    print(f"PT generated:    {len(existing)}")
    print(f"Missing PT:      {len(missing)}")
    if len(slide_ids) > 0:
        print(f"Completion rate: {len(existing)/len(slide_ids):.2%}")

    print("\nSaved:")
    print(f"  Missing list  -> {missing_path}")
    print(f"  Existing list -> {existing_path}")

    if missing:
        print("\nExamples of missing slides:")
        for sid in missing[:10]:
            print(" ", sid)


if __name__ == "__main__":
    main()
