import os
import glob
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn as nn
from typing import Dict, Any, Tuple, List

from transformers import WavLMModel, Wav2Vec2FeatureExtractor

# ================== CONFIG ================== #

LABELS = ["FP", "RP", "RV", "RS", "PW"]
SAMPLE_RATE = 16000  # WavLM expects 16k


# ================== MODEL ================== #

class AcousticModel(nn.Module):
    def __init__(self):
        super(AcousticModel, self).__init__()
        self.basemodel = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.linear = nn.Linear(768, 5)

    def forward(self, x):
        # x: [B, T_samples] after feature extractor
        feats = self.basemodel.feature_extractor(x)      # [B, feature_dim, T']
        feats = feats.transpose(1, 2)                    # [B, T', feature_dim]
        feats, _ = self.basemodel.feature_projection(feats)
        emb = self.basemodel.encoder(feats, return_dict=True)[0]  # [B, T_model, 768]
        out = self.linear(emb)                           # [B, T_model, 5]
        return emb, out


# ================== HELPERS ================== #

def extract_speaker_from_filename(audio_path: str) -> str:
    base = os.path.basename(audio_path)
    speaker_id = os.path.splitext(base)[0]
    return speaker_id


def load_full_audio(audio_path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, sr = torchaudio.load(audio_path)  # [channels, samples]

    # Convert to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    return waveform  # [1, T]


def load_metadata_for_speaker(metadata_path: str, speaker_id: str) -> pd.DataFrame:
    if metadata_path.endswith(".csv"):
        meta = pd.read_csv(metadata_path)
    else:
        meta = pd.read_excel(metadata_path)

    required_cols = ["segid", "segstart", "segend"]
    for col in required_cols:
        if col not in meta.columns:
            raise ValueError(f"Metadata must contain column '{col}'")

    mask = meta["segid"].astype(str).str.startswith(speaker_id + "_")
    speaker_meta = meta[mask].copy()

    if speaker_meta.empty:
        raise ValueError(f"No segments found in metadata for speaker_id='{speaker_id}'")

    speaker_meta.sort_values(by="segstart", inplace=True)
    speaker_meta.reset_index(drop=True, inplace=True)

    return speaker_meta


# ========== GT on MODEL FRAME GRID ========== #

def build_frame_labels_from_words(
    word_dir: str,
    segid: str,
    seg_start: float,
    seg_end: float,
    n_frames_model: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build frame-level labels directly on the model's frame grid.

    - n_frames_model = T_model from WavLM output for that segment.
    - We map frame k to a time in [0, duration] (segment-relative).
    - If frame time ∈ [wordstart, wordend), assign that word's labels.
    """

    csv_path = os.path.join(word_dir, f"{segid}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing word-level CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    for col in ["wordstart", "wordend"]:
        if col not in df.columns:
            raise ValueError(f"{csv_path} must contain '{col}' column.")
    for col in ["fp", "rp", "rv", "pw"]:
        if col not in df.columns:
            raise ValueError(f"{csv_path} must contain '{col}' column.")

    word_starts = df["wordstart"].astype(float).to_numpy()
    word_ends = df["wordend"].astype(float).to_numpy()
    fp = df["fp"].fillna(0).astype(int).to_numpy()
    rp = df["rp"].fillna(0).astype(int).to_numpy()
    rv = df["rv"].fillna(0).astype(int).to_numpy()
    pw = df["pw"].fillna(0).astype(int).to_numpy()
    rs = np.zeros_like(fp, dtype=int)  # RS not annotated

    duration = seg_end - seg_start

    # Model frame times: midpoints over [0, duration]
    frame_times_seg = (np.arange(n_frames_model) + 0.5) * duration / n_frames_model

    # Detect relative vs absolute word times
    max_word_end = float(np.max(word_ends)) if len(word_ends) > 0 else 0.0
    eps = 0.05
    if max_word_end <= duration + eps:
        w_starts_seg = word_starts
        w_ends_seg = word_ends
    else:
        w_starts_seg = word_starts - seg_start
        w_ends_seg = word_ends - seg_start

    labels = np.zeros((n_frames_model, len(LABELS)), dtype=np.float32)

    for i in range(len(df)):
        w_start = w_starts_seg[i]
        w_end = w_ends_seg[i]

        # skip words outside segment
        if w_end <= 0.0 or w_start >= duration:
            continue

        # clamp
        w_start_c = max(w_start, 0.0)
        w_end_c = min(w_end, duration)

        mask = (frame_times_seg >= w_start_c) & (frame_times_seg < w_end_c)
        if not np.any(mask):
            continue

        lbl_vec = np.array([fp[i], rp[i], rv[i], rs[i], pw[i]], dtype=np.float32)
        labels[mask] = np.maximum(labels[mask], lbl_vec)

    return labels, frame_times_seg, df


# ================== METRICS ================== #

def compute_metrics(y_true: np.ndarray, y_logits: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    if y_true.size == 0:
        print("Warning: no frames to evaluate.")
        return {}

    y_true = np.asarray(y_true)
    y_logits = np.asarray(y_logits)
    y_prob = 1.0 / (1.0 + np.exp(-y_logits))
    y_pred = (y_prob >= threshold).astype(int)

    num_labels = y_true.shape[1]
    results: Dict[str, Any] = {}

    def prf1(t, p):
        t = t.astype(int)
        p = p.astype(int)
        tp = np.logical_and(t == 1, p == 1).sum()
        fn = np.logical_and(t == 1, p == 0).sum()
        fp = np.logical_and(t == 0, p == 1).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    recalls = []
    f1s = []

    for i, lbl in enumerate(LABELS):
        prec, rec, f1 = prf1(y_true[:, i], y_pred[:, i])
        results[lbl] = {"precision": prec, "recall": rec, "f1": f1}
        recalls.append(rec)
        f1s.append(f1)

    results["UAR"] = float(np.mean(recalls))
    results["macro_f1"] = float(np.mean(f1s))
    return results


# ================== PER-SPEAKER EVAL ================== #

def evaluate_single_speaker(
    audio_path: str,
    metadata_path: str,
    word_dir: str,
    model: AcousticModel,
    feature_extractor: Wav2Vec2FeatureExtractor,
    device: str = "cuda",
    threshold: float = 0.5,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:

    speaker_id = extract_speaker_from_filename(audio_path)
    print(f"\n=== Evaluating speaker: {speaker_id} ===")

    if not os.path.exists(audio_path):
        print(f"Audio file not found, skipping speaker: {audio_path}")
        return (
            np.zeros((0, len(LABELS))),
            np.zeros((0, len(LABELS))),
            {}
        )

    # Load audio
    waveform = load_full_audio(audio_path, target_sr=SAMPLE_RATE).to(device)

    # Load metadata rows for this speaker
    try:
        meta_spk = load_metadata_for_speaker(metadata_path, speaker_id)
    except ValueError:
        print(f"  No segments found in metadata for speaker_id='{speaker_id}', skipping this file.")
        return (
            np.zeros((0, len(LABELS))),
            np.zeros((0, len(LABELS))),
            {}
        )

    all_true = []
    all_logits = []

    with torch.no_grad():
        for _, row in meta_spk.iterrows():

            segid = str(row["segid"])
            seg_start = float(row["segstart"])
            seg_end = float(row["segend"])

            # Slice segment
            start_sample = int(seg_start * SAMPLE_RATE)
            end_sample = int(seg_end * SAMPLE_RATE)
            seg_wave = waveform[:, start_sample:end_sample]
            if seg_wave.shape[-1] == 0:
                continue

            # Feature extraction
            seg_wave_mono = seg_wave[0].detach().cpu()
            audio_feats = feature_extractor(seg_wave_mono, sampling_rate=SAMPLE_RATE).input_values[0]
            audio_feats = torch.tensor(audio_feats, dtype=torch.float32).unsqueeze(0).to(device)

            # Model forward
            emb, output = model(audio_feats)
            logits = output[0]  # [T_model, 5]

            # Build GT aligned to model frames
            labels, _, _ = build_frame_labels_from_words(
                word_dir=word_dir,
                segid=segid,
                seg_start=seg_start,
                seg_end=seg_end,
                n_frames_model=logits.shape[0],
            )

            all_true.append(labels)
            all_logits.append(logits.cpu().numpy())

    # Speaker-level concatenation
    if not all_true:
        print(f"  No valid segments for speaker {speaker_id}.")
        return np.zeros((0, len(LABELS))), np.zeros((0, len(LABELS))), {}

    y_true_spk = np.concatenate(all_true, axis=0)
    y_logits_spk = np.concatenate(all_logits, axis=0)

    # Compute speaker-level metrics
    metrics_spk = compute_metrics(y_true_spk, y_logits_spk, threshold)

    # Print speaker summary ONLY (no segment/frame logs)
    print(f"\n--- Speaker-level Metrics: {speaker_id} ---")
    for lbl in LABELS:
        m = metrics_spk[lbl]
        print(f"{lbl}: precision={m['precision']:.4f}, recall={m['recall']:.4f}, f1={m['f1']:.4f}")
    print(f"UAR       = {metrics_spk['UAR']:.4f}")
    print(f"macro F1  = {metrics_spk['macro_f1']:.4f}")

    return y_true_spk, y_logits_spk, metrics_spk



# ================== FOLDER-LEVEL TEST ================== #

def test_folder(
    audio_dir: str,
    metadata_path: str,
    word_dir: str,
    weights_path: str,
    device: str = "cuda",
    threshold: float = 0.5,
    verbose: bool = False,
):
    """
    Evaluate all .wav files in a folder.
    """

    # 1) Collect audio files
    audio_paths = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    if not audio_paths:
        raise ValueError(f"No .wav files found in folder: {audio_dir}")

    print(f"Found {len(audio_paths)} audio files in {audio_dir}.")

    # 2) Shared feature extractor + model
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLE_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )

    print(f"Loading acoustic model from: {weights_path}")
    model = AcousticModel()
    state_dict = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("  Missing keys in state_dict:", missing)
    if unexpected:
        print("  Unexpected keys in state_dict:", unexpected)

    model.to(device)
    model.eval()
    print("Model loaded.\n")

    # 3) Evaluate each file & also build global metrics
    all_true_global: List[np.ndarray] = []
    all_logits_global: List[np.ndarray] = []

    for audio_path in audio_paths:
        y_true_spk, y_logits_spk, _ = evaluate_single_speaker(
            audio_path=audio_path,
            metadata_path=metadata_path,
            word_dir=word_dir,
            model=model,
            feature_extractor=feature_extractor,
            device=device,
            threshold=threshold,
            verbose=verbose,
        )

        if y_true_spk.size > 0:
            all_true_global.append(y_true_spk)
            all_logits_global.append(y_logits_spk)

    if not all_true_global:
        print("\nNo data to compute global metrics.")
        return

    y_true_global = np.concatenate(all_true_global, axis=0)
    y_logits_global = np.concatenate(all_logits_global, axis=0)
    metrics_global = compute_metrics(y_true_global, y_logits_global, threshold)

    print("\n=== GLOBAL METRICS OVER ALL FILES ===")
    for lbl in LABELS:
        m = metrics_global[lbl]
        print(f"{lbl}: precision={m['precision']:.4f}, recall={m['recall']:.4f}, f1={m['f1']:.4f}")
    print(f"UAR       = {metrics_global['UAR']:.4f}")
    print(f"macro F1  = {metrics_global['macro_f1']:.4f}")


# ================== EXAMPLE CALL ================== #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate WavLM acoustic disfluency model on a folder of WAV files."
    )

    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing .wav files (e.g., 24fb.wav).")

    parser.add_argument("--metadata_path", type=str, required=True,
                        help="Path to metadata CSV (with segid, segstart, segend).")

    parser.add_argument("--word_dir", type=str, required=True,
                        help="Directory containing word-level CSV files (segid.csv).")

    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to acoustic model weights (acoustic.pt).")

    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")

    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold.")

    parser.add_argument("--verbose", action="store_true",
                        help="Print frame-level predictions (default: off).")

    args = parser.parse_args()

    test_folder(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata_path,
        word_dir=args.word_dir,
        weights_path=args.weights_path,
        device=args.device,
        threshold=args.threshold,
        verbose=args.verbose,
    )


