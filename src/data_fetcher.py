"""
Download and curate a dataset of paintings from iconic artists using Hugging Face's
free public `Artificio/WikiArt` dataset (Parquet format).

Saves images to `data/paintings/<artist_slug>/` and metadata (including deterministic
train/val/test splits) to `data/metadata.csv`.
"""

import io
import json
import os
import random
from pathlib import Path
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

# Curated list of iconic, recognizable artists spanning diverse art movements & brushwork styles.
# The fetcher will select the top `num_artists` from this priority list that have at least `min_per_artist` works
# in the downloaded shard (falling back to highest-count artists if needed).
ICONIC_ARTISTS_PRIORITY = [
    "Vincent van Gogh",        # Post-Impressionism (impasto swirls, bold yellows/blues)
    "Claude Monet",            # Impressionism (broken light, pastel water lilies/landscapes)
    "Pablo Picasso",           # Cubism / Expressionism (geometric facets, bold contours)
    "Rembrandt",               # Baroque (chiaroscuro lighting, dark earthy tones, portraiture)
    "Salvador Dali",           # Surrealism (hyper-smooth dreamscapes, sharp shadows)
    "Pierre-Auguste Renoir",   # Impressionism (warm dappled brushwork, figures)
    "Albrecht Durer",          # Northern Renaissance (fine linework, woodcuts/engravings, detail)
    "Paul Gauguin",            # Post-Impressionism / Synthetism (flat bold color fields, tropics)
    "Marc Chagall",            # Modernism / Surrealism (dreamlike saturated folk colors)
    "John Singer Sargent",     # Realism / Impressionism (fluid virtuoso bravura brushstrokes)
    "Paul Cezanne",            # Post-Impressionism (structured geometric planes of color)
    "Camille Pissarro",        # Impressionism / Pointillism
    "Ivan Aivazovsky",         # Romanticism (luminous seascapes and translucent waves)
    "Ivan Shishkin",           # Realism (hyper-detailed forest landscapes)
    "Nicholas Roerich",        # Symbolism (tempera mountain landscapes, vibrant flat gradients)
]


def slugify(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "")
        .replace("'", "")
    )


def fetch_and_prepare_dataset(
    data_dir: str = "data",
    num_artists: int = 10,
    max_per_artist: int = 130,
    min_per_artist: int = 90,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Downloads the first Parquet shard of `Artificio/WikiArt` from Hugging Face,
    extracts `num_artists` iconic artists, saves RGB images to disk, and creates
    a stratified train/val/test split (70% / 15% / 15%).
    """
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    random.seed(seed)

    data_path = Path(data_dir)
    paintings_dir = data_path / "paintings"
    metadata_csv = data_path / "metadata.csv"

    if metadata_csv.exists():
        df = pd.read_csv(metadata_csv)
        if len(df) > 0 and all((data_path.parent / p).exists() for p in df["filepath"].head(10)):
            print(f"[DataFetcher] Existing dataset found at {metadata_csv} ({len(df)} paintings across {df['artist'].nunique()} artists).")
            return df

    data_path.mkdir(parents=True, exist_ok=True)
    paintings_dir.mkdir(parents=True, exist_ok=True)

    print("[DataFetcher] Downloading free WikiArt Parquet shard from Hugging Face (Artificio/WikiArt)...")
    parquet_path = hf_hub_download(
        repo_id="Artificio/WikiArt",
        filename="default/train/0000.parquet",
        repo_type="dataset",
        revision="refs/convert/parquet",
    )
    print(f"[DataFetcher] Parquet shard cached at: {parquet_path}")

    pf = pq.ParquetFile(parquet_path)
    meta_table = pf.read(columns=["artist", "style", "genre", "title", "date"])
    meta_df = meta_table.to_pandas()

    artist_counts = Counter(meta_df["artist"])
    selected_artists = []
    for artist in ICONIC_ARTISTS_PRIORITY:
        if artist_counts[artist] >= min_per_artist:
            selected_artists.append(artist)
        if len(selected_artists) >= num_artists:
            break

    if len(selected_artists) < num_artists:
        for artist, count in artist_counts.most_common():
            if artist not in selected_artists and count >= min_per_artist:
                selected_artists.append(artist)
            if len(selected_artists) >= num_artists:
                break

    print(f"[DataFetcher] Selected {len(selected_artists)} iconic artists:")
    for a in selected_artists:
        print(f"  - {a}: {artist_counts[a]} available in shard (sampling up to {max_per_artist})")

    # Determine row indices to extract
    selected_set = set(selected_artists)
    artist_to_indices = {a: [] for a in selected_artists}
    for idx, artist_name in enumerate(meta_df["artist"]):
        if artist_name in selected_set:
            artist_to_indices[artist_name].append(idx)

    sampled_indices = set()
    for artist, indices in artist_to_indices.items():
        random.shuffle(indices)
        sampled_indices.update(indices[:max_per_artist])

    print(f"[DataFetcher] Extracting {len(sampled_indices)} paintings and saving RGB JPEGs...")
    records = []
    artist_saved_counts = Counter()

    # Read row groups and extract only sampled indices
    global_row_idx = 0
    for rg_idx in range(pf.num_row_groups):
        rg_Meta = pf.metadata.row_group(rg_idx)
        rg_len = rg_Meta.num_rows
        rg_range = range(global_row_idx, global_row_idx + rg_len)

        # Check if any sampled index falls in this row group
        local_targets = [i - global_row_idx for i in rg_range if i in sampled_indices]
        if local_targets:
            rg_table = pf.read_row_group(
                rg_idx,
                columns=["title", "artist", "date", "genre", "style", "image"],
            )
            rg_dict = rg_table.to_pydict()
            for local_i in local_targets:
                artist = rg_dict["artist"][local_i]
                title = str(rg_dict["title"][local_i] or "Untitled")
                date_str = str(rg_dict["date"][local_i] or "Unknown")
                genre = str(rg_dict["genre"][local_i] or "Unknown")
                style = str(rg_dict["style"][local_i] or "Unknown")
                img_struct = rg_dict["image"][local_i]

                if img_struct is None or not img_struct.get("bytes"):
                    continue

                try:
                    img = Image.open(io.BytesIO(img_struct["bytes"])).convert("RGB")
                except Exception:
                    continue

                artist_slug = slugify(artist)
                artist_folder = paintings_dir / artist_slug
                artist_folder.mkdir(parents=True, exist_ok=True)

                img_num = artist_saved_counts[artist]
                artist_saved_counts[artist] += 1
                rel_path = f"data/paintings/{artist_slug}/{artist_slug}_{img_num:04d}.jpg"
                full_path = data_path.parent / rel_path
                img.save(full_path, quality=92)

                records.append(
                    {
                        "id": f"{artist_slug}_{img_num:04d}",
                        "artist": artist,
                        "artist_slug": artist_slug,
                        "title": title,
                        "date": date_str,
                        "genre": genre,
                        "style": style,
                        "filepath": rel_path,
                    }
                )
        global_row_idx += rg_len

    df = pd.DataFrame(records)

    # Create deterministic stratified train (70%), val (15%), test (15%) splits per artist
    splits = []
    for artist, group in df.groupby("artist", sort=False):
        n = len(group)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        artist_splits = (
            ["train"] * n_train
            + ["val"] * n_val
            + ["test"] * (n - n_train - n_val)
        )
        rng = random.Random(seed + hash(artist) % 10000)
        rng.shuffle(artist_splits)
        df.loc[group.index, "split"] = artist_splits

    # Assign integer class labels sorted alphabetically by artist name
    unique_artists = sorted(df["artist"].unique())
    artist_to_idx = {a: i for i, a in enumerate(unique_artists)}
    df["label"] = df["artist"].map(artist_to_idx)

    df.to_csv(metadata_csv, index=False)
    with open(data_path / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "artist_to_idx": artist_to_idx,
                "idx_to_artist": {int(i): a for a, i in artist_to_idx.items()},
            },
            f,
            indent=2,
        )

    print(
        f"[DataFetcher] Saved {len(df)} paintings across {len(unique_artists)} artists to {metadata_csv}."
    )
    print(df.groupby(["artist", "split"]).size().unstack(fill_value=0))
    return df


if __name__ == "__main__":
    fetch_and_prepare_dataset()

