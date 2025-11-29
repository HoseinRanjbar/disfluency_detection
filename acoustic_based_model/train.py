import os
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from transformers import WavLMModel


# ================== GLOBAL CONFIG ================== #

LABELS = ["FP", "RP", "RV", "RS", "PW"]
SAMPLE_RATE = 16000  # WavLM expects 16k

# ================== MODEL DEFINITION ================== #

class AcousticModel(nn.Module):
    """
    WavLM-base + linear classifier -> frame-level 5-label logits.
    This MUST match the architecture used to train acoustic.pt.
    """

    def __init__(self):
        super(AcousticModel, self).__init__()
        # Load pretrained WavLM-base weights
        self.basemodel = WavLMModel.from_pretrained("microsoft/wavlm-base")
        # Linear classifier head (will be overwritten by acoustic.pt weights)
        self.linear = nn.Linear(768, len(LABELS))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T_samples] raw mono audio at 16kHz.

        Returns:
          emb: [B, T_frames, 768]  (encoder hidden states)
          out: [B, T_frames, 5]    (logits for FP, RP, RV, RS, PW)
        """
        # Convolutional feature extractor (frozen during fine-tuning)
        feats = self.basemodel.feature_extractor(x)      # [B, feat_dim, T']
        feats = feats.transpose(1, 2)                    # [B, T', feat_dim]
        feats, _ = self.basemodel.feature_projection(feats)
        emb = self.basemodel.encoder(feats, return_dict=True)[0]  # [B, T_frames, 768]
        out = self.linear(emb)                           # [B, T_frames, 5]
        return emb, out


# ================== DATA STRUCTURES ================== #

@dataclass
class SegmentInfo:
    segid: str
    audio_path: str
    seg_start: float
    seg_end: float


# ================== HELPERS ================== #

def extract_speaker_from_segid(segid: str) -> str:
    """
    '24fb_014' -> '24fb'
    """
    return segid.split("_")[0]


def load_full_audio(audio_path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Load mono waveform at target_sr: [1, T]
    """
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


def load_segments_from_metadata(
    metadata_path: str,
    audio_dir: str,
) -> List[SegmentInfo]:
    """
    metadata CSV columns: segid, segstart, segend
    audio files: <speaker>.wav (e.g. 24fb.wav)
    """
    if metadata_path.endswith(".csv"):
        meta = pd.read_csv(metadata_path)
    else:
        meta = pd.read_excel(metadata_path)

    required_cols = ["segid", "segstart", "segend"]
    for col in required_cols:
        if col not in meta.columns:
            raise ValueError(f"Metadata must contain column '{col}'")

    segments: List[SegmentInfo] = []
    missing_speakers = set()

    for _, row in meta.iterrows():
        segid = str(row["segid"])
        seg_start = float(row["segstart"])
        seg_end = float(row["segend"])
        spk_id = extract_speaker_from_segid(segid)
        audio_path = os.path.join(audio_dir, f"{spk_id}.wav")

        # ✅ check if audio file exists, otherwise skip this segment
        if not os.path.exists(audio_path):
            if spk_id not in missing_speakers:
                print(
                    f"[WARN] Audio for speaker '{spk_id}' not found at {audio_path}. "
                    f"Skipping all segments for this speaker."
                )
                missing_speakers.add(spk_id)
            continue

        segments.append(
            SegmentInfo(
                segid=segid,
                audio_path=audio_path,
                seg_start=seg_start,
                seg_end=seg_end,
            )
        )

    print(f"Loaded {len(segments)} segments from {metadata_path} (after skipping missing audio).")
    return segments

def estimate_pos_weights(
    train_segments: List[SegmentInfo],
    word_dir: str,
    alpha: float = 0.3,      # how strong to push rare classes (>0, <1)
    max_weight: float = 5.0  # upper limit on pos_weight
) -> np.ndarray:
    """
    Estimate pos_weight for BCEWithLogitsLoss:
      pos_weight[c] = (#negative_frames_c / #positive_frames_c)

    using the SAME label-building function as training/eval.
    """
    total_pos = np.zeros(len(LABELS), dtype=np.float64)
    total_frames = 0

    for seg in train_segments:
        # fake n_frames_model just to use label builder:
        # we only care about the relative distribution of labels,
        # so pick some T (e.g. 100) and time span seg.seg_end-seg.seg_start
        duration = seg.seg_end - seg.seg_start
        if duration <= 0:
            continue

        # use T=100 frames for counting (resolution doesn't matter much here)
        T = 100
        labels = build_frame_labels_from_words(
            word_dir=word_dir,
            segid=seg.segid,
            seg_start=seg.seg_start,
            seg_end=seg.seg_end,
            n_frames_model=T,
        )  # [T, 5]

        total_pos += labels.sum(axis=0)
        total_frames += labels.shape[0]

    # avoid division by zero
    total_pos = np.maximum(total_pos, 1.0)
    total_neg = total_frames - total_pos
    total_neg = np.maximum(total_neg, 1.0)

    raw_ratio = total_neg / total_pos

    # soften the effect: sqrt + blend toward 1 with factor alpha
    w_raw = np.sqrt(raw_ratio)
    pos_weight = 1.0 + alpha * (w_raw - 1.0)

    # clip so nothing explodes
    pos_weight = np.clip(pos_weight, 1.0, max_weight)

    # RS is always zero -> keep neutral
    RS_index = LABELS.index("RS")
    pos_weight[RS_index] = 1.0

    print("[DEBUG] raw_ratio per label:", dict(zip(LABELS, raw_ratio)))
    print("[DEBUG] Final pos_weight per label:", dict(zip(LABELS, pos_weight)))
    return pos_weight

# ========== BUILD FRAME LABELS ON MODEL GRID ========== #

def build_frame_labels_from_words(
    word_dir: str,
    segid: str,
    seg_start: float,
    seg_end: float,
    n_frames_model: int,
) -> np.ndarray:
    """
    Build frame-level labels directly on the model's frame grid.

    - n_frames_model = T_frames from WavLM output for that segment.
    - We map frame k to a time in [0, duration] (segment-relative).
    - If a frame time is inside [wordstart, wordend), assign that word's labels.

    word_dir contains {segid}.csv with columns:
      wordstart, wordend, fp, rp, rv, pw
    RS is not annotated (assumed 0).
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
    n_frames_model = max(n_frames_model, 1)

    # Frame times as midpoints over [0, duration]
    frame_times_seg = (np.arange(n_frames_model) + 0.5) * duration / n_frames_model

    # Detect relative vs absolute word times
    max_word_end = float(np.max(word_ends)) if len(word_ends) > 0 else 0.0
    eps = 0.05
    if max_word_end <= duration + eps:
        # Already relative [0, duration]
        w_starts_seg = word_starts
        w_ends_seg = word_ends
    else:
        # Absolute -> convert to segment-relative
        w_starts_seg = word_starts - seg_start
        w_ends_seg = word_ends - seg_start

    labels = np.zeros((n_frames_model, len(LABELS)), dtype=np.float32)

    for i in range(len(df)):
        w_start = w_starts_seg[i]
        w_end = w_ends_seg[i]

        # Skip words outside [0, duration]
        if w_end <= 0.0 or w_start >= duration:
            continue

        # Clamp
        w_start_c = max(w_start, 0.0)
        w_end_c = min(w_end, duration)

        mask = (frame_times_seg >= w_start_c) & (frame_times_seg < w_end_c)
        if not np.any(mask):
            continue

        lbl_vec = np.array([fp[i], rp[i], rv[i], rs[i], pw[i]], dtype=np.float32)
        labels[mask] = np.maximum(labels[mask], lbl_vec)

    return labels  # [T_frames, 5]


# ================== METRICS ================== #

def compute_metrics(
    y_true: np.ndarray,
    y_logits: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Frame-level metrics: per-label precision/recall/F1 + UAR + macro F1.
    """
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


# ================== EVAL LOOP ================== #

def evaluate(
    model: AcousticModel,
    segments: List[SegmentInfo],
    word_dir: str,
    device: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    model.eval()
    all_true: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []

    with torch.no_grad():
        for seg in segments:
            waveform = load_full_audio(seg.audio_path, target_sr=SAMPLE_RATE).to(device)

            start_sample = int(seg.seg_start * SAMPLE_RATE)
            end_sample = int(seg.seg_end * SAMPLE_RATE)
            seg_wave = waveform[:, start_sample:end_sample]  # [1, T_seg]

            if seg_wave.shape[-1] == 0:
                continue

            _, logits = model(seg_wave)   # [1, T_frames, 5]
            logits = logits[0]            # [T_frames, 5]

            labels = build_frame_labels_from_words(
                word_dir=word_dir,
                segid=seg.segid,
                seg_start=seg.seg_start,
                seg_end=seg.seg_end,
                n_frames_model=logits.shape[0],
            )  # [T_frames, 5]

            all_true.append(labels)
            all_logits.append(logits.cpu().numpy())

    if not all_true:
        print("Warning: no valid segments in evaluation.")
        return {}

    y_true = np.concatenate(all_true, axis=0)
    y_logits = np.concatenate(all_logits, axis=0)
    # 🔍 DEBUG: check how many positives we have in GT and predictions
    y_prob = 1.0 / (1.0 + np.exp(-y_logits))
    y_pred = (y_prob >= threshold).astype(int)

    print("\n[DEBUG] Dev label / prediction stats:")
    for i, lbl in enumerate(LABELS):
        gt_pos = int(y_true[:, i].sum())
        pred_pos = int(y_pred[:, i].sum())
        print(f"  {lbl}: GT positives={gt_pos}, Pred positives={pred_pos}")
        
    return compute_metrics(y_true, y_logits, threshold=threshold)


# ================== TRAIN LOOP ================== #

def train_fluencybank(
    train_metadata_path: str,
    dev_metadata_path: str,
    audio_dir: str,
    word_dir: str,
    acoustic_checkpoint: str,    # e.g. "acoustic.pt" from original repo
    device: str = "cuda",
    num_epochs: int = 15,        # as in paper
    lr: float = 1e-5,            # you can try 1e-4, 5e-5, 1e-5
    threshold: float = 0.5,
    seed: int = 42,
    patience_epochs: int = 5,    # early stopping after N epochs with no UAR improvement
    save_best_path: str = "best_fluencybank_wavlm.pt",
):
    """
    Train WavLM-base acoustic model on FluencyBank, starting from
    a Switchboard-fine-tuned acoustic.pt (which already includes the classifier).
    Evaluation is done once at the end of each epoch.
    """

    # ----- seeds -----
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
    print(f"Using device: {device}")

    # ----- load segments -----
    train_segments = load_segments_from_metadata(train_metadata_path, audio_dir)
    dev_segments = load_segments_from_metadata(dev_metadata_path, audio_dir)
    print(f"Train segments: {len(train_segments)} | Dev segments: {len(dev_segments)}")

    # Estimate pos_weight from train labels
    pos_weight_np = estimate_pos_weights(train_segments=train_segments, word_dir=word_dir, alpha=0.3, max_weight=10)

    # ----- model -----
    model = AcousticModel()

    if not os.path.exists(acoustic_checkpoint):
        raise FileNotFoundError(f"acoustic_checkpoint not found: {acoustic_checkpoint}")
    print(f"Loading pretrained acoustic model from: {acoustic_checkpoint}")
    state = torch.load(acoustic_checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("  Missing keys in checkpoint:", missing)
    if unexpected:
        print("  Unexpected keys in checkpoint:", unexpected)

    # Freeze conv layers
    for p in model.basemodel.feature_extractor.parameters():
        p.requires_grad = False

    model.to(device)


    # Only train parameters that require grad (encoder + classifier)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    ####### for solve unbalance data #################
    pos_weight_tensor = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    # criterion = nn.BCEWithLogitsLoss()

    global_step = 0
    best_uar = -1.0
    best_metrics: Dict[str, Any] = {}
    best_epoch = 0
    epochs_without_improve = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")
        model.train()
        random.shuffle(train_segments)

        running_loss = 0.0
        num_updates = 0

        progress = tqdm(train_segments, desc=f"Epoch {epoch}/{num_epochs} - training", ncols=100)

        for seg in progress:
            global_step += 1

            waveform = load_full_audio(seg.audio_path, target_sr=SAMPLE_RATE).to(device)
            start_sample = int(seg.seg_start * SAMPLE_RATE)
            end_sample = int(seg.seg_end * SAMPLE_RATE)
            seg_wave = waveform[:, start_sample:end_sample]

            if seg_wave.shape[-1] == 0:
                continue

            emb, logits = model(seg_wave)   # [1, T_frames, 5]
            logits = logits[0]

            labels_np = build_frame_labels_from_words(
                word_dir=word_dir,
                segid=seg.segid,
                seg_start=seg.seg_start,
                seg_end=seg.seg_end,
                n_frames_model=logits.shape[0],
            )
            labels = torch.from_numpy(labels_np).to(device)

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_updates += 1

            # update progress bar text
            avg_loss = running_loss / max(1, num_updates)
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{avg_loss:.4f}"
            })

        # ----- end of epoch: report training loss -----
        if num_updates > 0:
            avg_loss = running_loss / num_updates
        else:
            avg_loss = float("nan")
        print(f"\n[Epoch {epoch}] avg training loss = {avg_loss:.4f}")

        # ----- evaluate on dev set -----
        print(f"[Epoch {epoch}] Evaluating on dev set...")
        metrics_dev = evaluate(
            model=model,
            segments=dev_segments,
            word_dir=word_dir,
            device=device,
            threshold=threshold,
        )

        if not metrics_dev:
            print("  (no metrics from dev – check your data)")
            continue

        uar = metrics_dev["UAR"]
        macro_f1 = metrics_dev["macro_f1"]
        print(f"[Epoch {epoch}] Dev UAR = {uar:.4f}, macro F1 = {macro_f1:.4f}")

        # ----- early stopping based on UAR -----
        if uar > best_uar:
            best_uar = uar
            best_metrics = metrics_dev
            best_epoch = epoch
            epochs_without_improve = 0

            # 1) Save locally
            torch.save(model.state_dict(), save_best_path)
            print(f"  --> New best model saved to {save_best_path}")


        else:
            epochs_without_improve += 1
            print(f"  No improvement in UAR. epochs_without_improve = {epochs_without_improve}")

        if patience_epochs is not None and epochs_without_improve >= patience_epochs:
            print("\nEarly stopping: no UAR improvement for "
                  f"{patience_epochs} consecutive epochs.")
            break

    print("\nTraining finished.")
    if best_uar >= 0:
        print(f"Best UAR={best_uar:.4f} at epoch {best_epoch}")
        print("Best dev metrics:")
        for lbl in LABELS:
            m = best_metrics[lbl]
            print(
                f"  {lbl}: precision={m['precision']:.4f}, "
                f"recall={m['recall']:.4f}, f1={m['f1']:.4f}"
            )
        print(f"  UAR={best_metrics['UAR']:.4f}, macro F1={best_metrics['macro_f1']:.4f}")
    else:
        print("No valid evaluation happened (check your data).")


# ================== EXAMPLE MAIN ================== #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train WavLM-based acoustic model on the FluencyBank dataset."
    )

    parser.add_argument("--train_meta", type=str, required=True,
                        help="Path to train metadata CSV (segid, segstart, segend).")

    parser.add_argument("--dev_meta", type=str, required=True,
                        help="Path to dev/validation metadata CSV.")

    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing speaker audio files (e.g., 24fb.wav or .mp3).")

    parser.add_argument("--word_dir", type=str, required=True,
                        help="Directory containing per-segment word-level CSV files (e.g., 24fb_000.csv).")

    parser.add_argument("--acoustic_checkpoint", type=str, required=True,
                        help="Path to pretrained acoustic model (acoustic.pt from original repo).")

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device,
                        help="Device to use (cuda or cpu).")

    parser.add_argument("--num_epochs", type=int, default=15,
                        help="Maximum number of epochs.")

    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (try 1e-4, 5e-5, 1e-5).")

    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for evaluation metrics.")

    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility.")

    parser.add_argument("--patience_epochs", type=int, default=5,
                        help="Early stopping patience (in epochs without UAR improvement). "
                             "Set to 0 or negative to disable early stopping.")

    parser.add_argument("--save_best_path", type=str, default="best_fluencybank_wavlm.pt",
                        help="Where to save the best model (based on dev UAR).")

    args = parser.parse_args()

    # disable early stopping if user sets patience <= 0
    patience = args.patience_epochs if args.patience_epochs > 0 else None

    train_fluencybank(
        train_metadata_path=args.train_meta,
        dev_metadata_path=args.dev_meta,
        audio_dir=args.audio_dir,
        word_dir=args.word_dir,
        acoustic_checkpoint=args.acoustic_checkpoint,
        device=args.device,
        num_epochs=args.num_epochs,
        lr=args.lr,
        threshold=args.threshold,
        seed=args.seed,
        patience_epochs=patience,
        save_best_path=args.save_best_path,
    )
