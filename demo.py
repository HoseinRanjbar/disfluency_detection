import os, sys
import warnings
import argparse
import logging
import numpy as np
import pandas as pd

import torch, torchaudio

warnings.filterwarnings("ignore")
from transformers import BertTokenizerFast, BertForTokenClassification, Wav2Vec2FeatureExtractor
import whisper_timestamped as whisper

from models import AcousticModel, MultimodalModel

labels = ['FP', 'RP', 'RV', 'RS', 'PW']

def run_asr(audio_file, device):

    # Load audio file and resample to 16 kHz
    audio, orgnl_sr = torchaudio.load(audio_file)
    audio_rs = torchaudio.functional.resample(audio, orgnl_sr, 16000)[0, :]
    audio_rs.to(device)

    # Load in Whisper model that has been fine-tuned for verbatim speech transcription
    model = whisper.load_model('./weights/asr', device='cpu')
    model.to(device)
    print('loaded finetuned whisper asr') 

    # Get Whisper output
    result = whisper.transcribe(model, audio_rs, language='en', beam_size=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))

    print(result)
    # Convert output dictionary to a dataframe
    words = []
    for segment in result['segments']:
        words += segment['words']
    text_df = pd.DataFrame(words)
    text_df['text'] = text_df['text'].str.lower()

    return text_df

def run_language_based(audio_file, text_df, device):
    # --- 1) Tokenize as a list of words so we can map tokens -> words
    words_list = text_df["text"].tolist()

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    enc = tokenizer(
        words_list,
        is_split_into_words=True,      # <— important
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
        truncation=True,               # keep inputs <= 512 tokens
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # --- 2) Load model weights
    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=5)
    state_dict = torch.load("./weights/language.pt", map_location="cpu")
    # Some checkpoints include a static position_ids that conflict with HF models
    if "bert.embeddings.position_ids" in state_dict:
        del state_dict["bert.embeddings.position_ids"]
    model.load_state_dict(state_dict, strict=False)
    print("loaded finetuned language model")
    model.config.output_hidden_states = True
    model.to(device)
    model.eval()

    # --- 3) Forward
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        token_logits = output.logits[0]             # [seq_len, 5]
        token_hidden = output.hidden_states[-1][0]  # [seq_len, 768]
        token_probs = torch.sigmoid(token_logits)

    # --- 4) Pool token-level to word-level using tokenizer’s word_ids mapping
    word_ids = enc.word_ids(batch_index=0)  # list of length seq_len, values: 0..(len(words_list)-1) or None
    num_words = len(words_list)
    num_labels = token_logits.shape[-1]

    # containers
    word_logits_sum = torch.zeros((num_words, num_labels), device=token_logits.device)
    word_counts = torch.zeros((num_words, 1), device=token_logits.device)
    word_hidden_sum = torch.zeros((num_words, token_hidden.shape[-1]), device=token_hidden.device)

    for t_idx, w_id in enumerate(word_ids):
        if w_id is None:
            continue  # special tokens
        word_logits_sum[w_id] += token_logits[t_idx]
        word_hidden_sum[w_id] += token_hidden[t_idx]
        word_counts[w_id] += 1

    # avoid div-by-zero if some words were dropped/truncated
    mask = (word_counts.squeeze(-1) > 0)
    word_logits = torch.zeros_like(word_logits_sum)
    word_hidden = torch.zeros_like(word_hidden_sum)
    word_logits[mask] = word_logits_sum[mask] / word_counts[mask]
    word_hidden[mask] = word_hidden_sum[mask] / word_counts[mask]

    # --- 5) Convert to preds/emb for exactly len(words_list) rows
    probs = torch.sigmoid(word_logits)
    preds = (probs > 0.5).int().cpu().numpy()       # [num_words, 5]
    emb = word_hidden.cpu().numpy()                 # [num_words, 768]

    pred_columns = [f"pred{i}" for i in range(preds.shape[1])]
    emb_columns = [f"emb{i}" for i in range(emb.shape[1])]
    pred_df = pd.DataFrame(preds, columns=pred_columns)
    emb_df  = pd.DataFrame(emb,  columns=emb_columns)

    # --- 6) Make sure start/end are real floats and no NaNs remain
    text_df = text_df.copy()
    text_df["start"] = pd.to_numeric(text_df["start"], errors="coerce")
    text_df["end"]   = pd.to_numeric(text_df["end"],   errors="coerce")

    # if any words were truncated away by BERT (long audio), trim all to a common length
    N = min(len(text_df), len(pred_df), len(emb_df))
    text_df = text_df.iloc[:N].reset_index(drop=True)
    pred_df = pred_df.iloc[:N].reset_index(drop=True)
    emb_df  = emb_df.iloc[:N].reset_index(drop=True)

    df = pd.concat([text_df, pred_df, emb_df], axis=1)

    # final safety: drop rows with bad times
    df = df.dropna(subset=["start", "end"])
    df = df[df["end"] >= df["start"]]

    # --- 7) Convert to frame-level
    frame_emb, frame_pred = convert_word_to_framelevel(audio_file, df)
    return frame_emb, frame_pred

def convert_word_to_framelevel(audio_file, df):
    # ensure numeric
    df = df.copy()
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"]   = pd.to_numeric(df["end"],   errors="coerce")
    df = df.dropna(subset=["start", "end"])
    df = df[df["end"] >= df["start"]]

    # small padding
    df["end"] = df["end"] + 0.01

    info = torchaudio.info(audio_file)
    end = info.num_frames / info.sample_rate

    frame_time = np.arange(0, end, 0.01).tolist()
    num_labels = len(labels)
    frame_pred = [[0] * num_labels for _ in range(len(frame_time))]
    frame_emb  = [[0] * 768       for _ in range(len(frame_time))]

    for _, row in df.iterrows():
        start_idx = int(round(float(row["start"]) * 100))
        end_idx   = int(round(float(row["end"])   * 100))
        start_idx = max(0, min(start_idx, len(frame_time)))
        end_idx   = max(0, min(end_idx,   len(frame_time)))
        if end_idx <= start_idx:
            continue

        frame_pred[start_idx:end_idx] = [[row[f"pred{i}"] for i in range(num_labels)]] * (end_idx - start_idx)
        frame_emb[start_idx:end_idx]  = [[row[f"emb{i}"]  for i in range(768)]]       * (end_idx - start_idx)

    # frame_emb  = torch.tensor(np.array(frame_emb)[::2])
    # frame_pred = torch.tensor(np.array(frame_pred)[::2])
    frame_emb  = torch.tensor(np.array(frame_emb,  dtype=np.float32)[::2], dtype=torch.float32)
    frame_pred = torch.tensor(np.array(frame_pred, dtype=np.float32)[::2], dtype=torch.float32)

    return frame_emb, frame_pred


def run_acoustic_based(audio_file, device):

    # Load audio file and resample to 16 kHz
    audio, orgnl_sr = torchaudio.load(audio_file)
    audio_rs = torchaudio.functional.resample(audio, orgnl_sr, 16000)[0, :]
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                 sampling_rate=16000,
                                                 padding_value=0.0,
                                                 do_normalize=True,
                                                 return_attention_mask=False)
    audio_feats = feature_extractor(audio_rs, sampling_rate=16000).input_values[0]
    audio_feats = torch.Tensor(audio_feats).unsqueeze(0)
    audio_feats = audio_feats.to(device)

    # Initialize WavLM model and load in pre-trained weights
    model = AcousticModel()
    model.load_state_dict(torch.load('./weights/acoustic.pt', map_location='cpu'))
    model.to(device)
    print('loaded finetuned acoustic model') 

    # Get WavLM output
    emb, output = model(audio_feats)
    probs = torch.sigmoid(output)
    preds = (probs > 0.5).int()[0]
    emb = emb[0]

    return emb, preds

def run_multimodal(language, acoustic, device):

    # Rounding differences may result in slightly different embedding sizes
    # Adjust so they're both the same size
    min_size = min(language.size(0), acoustic.size(0))
    language = language[:min_size].unsqueeze(0)
    acoustic = acoustic[:min_size].unsqueeze(0)

    language = language.to(device)
    acoustic = acoustic.to(device)

    # Initialize multimodal model and load in pre-trained weights
    model = MultimodalModel()
    model.load_state_dict(torch.load('./weights/multimodal.pt', map_location='cpu'))
    model.to(device)
    print('loaded finetuned multimodal model') 

    # Get multimodal output
    output = model(language, acoustic)
    probs = torch.sigmoid(output)
    preds = (probs > 0.5).int()[0]

    return preds

def setup_log(log_file):

    # Set up a logger
    logger = logging.getLogger("demo_log")
    logger.setLevel(logging.INFO)

    # Create a file handler to write log messages to a file
    file_handler = logging.FileHandler(log_file)

    # Create a stream handler to display log messages on the screen
    stream_handler = logging.StreamHandler(sys.stdout)

    # Define the log format
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    stream_handler.setFormatter(log_format)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Redirect stdout and stderr to the logger
    sys.stdout = logger
    sys.stderr = logger

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--audio_file', type=str, default=None, required=True, help='path to 8k .wav file')
    parser.add_argument('--output_trans', type=str, default=None, required=False, help='path to intermediate .csv with asr transcript')
    parser.add_argument('--output_file', type=str, default=None, required=True, help='path to output .csv')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--modality', type=str, default='multimodal', choices=['language', 'acoustic', 'multimodal'],
                        help='modality can be language, acoustic, or multimodal')

    args = parser.parse_args()

    # Setup log
    #setup_log(args.output_file.replace('.csv', '.log'))

    # Get predictions
    text_df = None
    if args.modality == 'language' or args.modality == 'multimodal':
        text_df = run_asr(args.audio_file, args.device)
        if args.output_trans is not None: 
            text_df.to_csv(args.output_trans)
        language_emb, preds = run_language_based(args.audio_file, text_df, args.device)
    if args.modality == 'acoustic' or args.modality == 'multimodal':
        acoustic_emb, preds = run_acoustic_based(args.audio_file, args.device)
    if args.modality == 'multimodal':
        preds = run_multimodal(language_emb, acoustic_emb, args.device)

    # Save output
    pred_df = pd.DataFrame(preds.cpu(), columns=labels).astype(int)
    pred_df['frame_time'] = [round(i * 0.02, 2) for i in range(pred_df.shape[0])]
    pred_df = pred_df.set_index('frame_time')
    pred_df.to_csv(args.output_file)

