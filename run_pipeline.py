"""
End-to-end runner for ArtistAppraisal:
1. Downloads free online WikiArt paintings for 10 iconic artists (`src/data_fetcher.py`).
2. Trains `ArtistStyleNet` (ResNet50 + Multi-scale Gatys Gram Matrix Style Head) (`src/train.py`).
3. Evaluates held-out test accuracy, fits style probes, computes 2D t-SNE manifolds,
   and saves Grad-CAM attribution figures (`src/evaluate.py`).
"""

import argparse
from src.data_fetcher import fetch_and_prepare_dataset
from src.train import train_model
from src.evaluate import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run the ArtistAppraisal DNN pipeline.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-per-artist", type=int, default=130, help="Max paintings per artist")
    args = parser.parse_args()

    print("=" * 72)
    print("STEP 1/3: Fetching & Preparing Curated WikiArt Dataset")
    print("=" * 72)
    fetch_and_prepare_dataset(max_per_artist=args.max_per_artist)

    print("\n" + "=" * 72)
    print("STEP 2/3: Training ArtistStyleNet (CNN + Multi-Scale Gram Style Head)")
    print("=" * 72)
    train_model(epochs=args.epochs, batch_size=args.batch_size)

    print("\n" + "=" * 72)
    print("STEP 3/3: Evaluating Implicit Style Learning & Generating Visuals")
    print("=" * 72)
    run_evaluation(batch_size=args.batch_size)


if __name__ == "__main__":
    main()

