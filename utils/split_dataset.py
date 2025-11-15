import argparse
import os
import numpy as np
import pandas as pd


def extract_speaker(segid: str) -> str:
    """
    Extract speaker ID from a segment ID like:
      '502_001' -> '502'
      'fa_025'  -> 'fa'

    If there is no '_', just return the whole segid.
    """
    if not isinstance(segid, str):
        segid = str(segid)
    parts = segid.split("_")
    if len(parts) >= 2:
        return parts[0]
    return segid


def balanced_speaker_split(
    metadata_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    output_dir: str = "./splits",
    segid_column: str = "segid",
):
    # --- sanity check on ratios ---
    total_ratio = train_ratio + val_ratio + test_ratio
    assert abs(total_ratio - 1.0) < 1e-6, "train_ratio + val_ratio + test_ratio must sum to 1.0"

    # --- load metadata ---
    if metadata_path.endswith(".csv"):
        df = pd.read_csv(metadata_path)
    else:
        df = pd.read_excel(metadata_path)

    if segid_column not in df.columns:
        raise ValueError(f"Expected metadata to contain column '{segid_column}'.")

    # --- get speaker for each segment ---
    df["speaker"] = df[segid_column].apply(extract_speaker)

    # --- count segments per speaker ---
    spk_counts = df["speaker"].value_counts().to_dict()
    speakers = list(spk_counts.keys())

    # shuffle speakers for randomness
    rng = np.random.RandomState(seed)
    rng.shuffle(speakers)

    total_segments = len(df)
    target = {
        "train": total_segments * train_ratio,
        "val":   total_segments * val_ratio,
        "test":  total_segments * test_ratio,
    }

    # we will sort speakers by descending segment count (greedy bin packing)
    speakers_sorted = sorted(speakers, key=lambda s: spk_counts[s], reverse=True)

    split_speakers = {"train": [], "val": [], "test": []}
    split_counts   = {"train": 0,    "val": 0,    "test": 0}

    for spk in speakers_sorted:
        spk_n = spk_counts[spk]

        # compute "fill ratio" for each split: current / target
        # if target is 0 (ratio 0), skip that split
        best_split = None
        best_fill = None

        for split_name in ["train", "val", "test"]:
            tgt = target[split_name]
            if tgt <= 0:
                continue
            fill = split_counts[split_name] / tgt  # how full this split is
            if (best_fill is None) or (fill < best_fill):
                best_fill = fill
                best_split = split_name

        # assign this speaker to the least-filled split (relative to its target)
        if best_split is None:
            # all targets are zero (weird case), just assign to train
            best_split = "train"

        split_speakers[best_split].append(spk)
        split_counts[best_split] += spk_n

    # --- build dataframes for each split ---
    df_train = df[df["speaker"].isin(split_speakers["train"])].drop(columns=["speaker"])
    df_val   = df[df["speaker"].isin(split_speakers["val"])].drop(columns=["speaker"])
    df_test  = df[df["speaker"].isin(split_speakers["test"])].drop(columns=["speaker"])

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_metadata.csv")
    val_path   = os.path.join(output_dir, "val_metadata.csv")
    test_path  = os.path.join(output_dir, "test_metadata.csv")

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    print("Done splitting by speaker (prefix before '_'), balanced by segment counts.\n")

    print(f"Total segments: {total_segments}")
    print(f"Target segments: "
          f"train ~ {int(target['train'])}, "
          f"val ~ {int(target['val'])}, "
          f"test ~ {int(target['test'])}")
    print()

    print("Speakers per split:")
    print(f"  Train speakers: {len(split_speakers['train'])}")
    print(f"  Val speakers:   {len(split_speakers['val'])}")
    print(f"  Test speakers:  {len(split_speakers['test'])}")
    print()

    print("Segments per split:")
    print(f"  Train: {len(df_train)}")
    print(f"  Val:   {len(df_val)}")
    print(f"  Test:  {len(df_test)}")
    print(f"\nSaved to:\n  {train_path}\n  {val_path}\n  {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker-level split of FluencyBank metadata, balanced by segment counts.")
    parser.add_argument("--metadata_path", type=str, required=True,
                        help="Path to metadata (.csv or .xlsx). Must contain segid column.")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./splits")
    parser.add_argument("--segid_column", type=str, default="segid",
                        help="Name of the column that contains segment IDs like '502_001'.")
    args = parser.parse_args()

    balanced_speaker_split(
        metadata_path=args.metadata_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
        segid_column=args.segid_column,
    )
