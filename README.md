# Disfluency Detection on FluencyBank

This repository extends the work from  
**[Disfluency Detection from Untranscribed Speech](https://github.com/amritkromana/disfluency_detection_from_audio)**  
by **Kumar et al.**, which introduces a multimodal framework for detecting disfluencies using both audio and text signals.

## 📘 Overview

Disfluency detection plays an important role in studying conversational patterns, speech disorders, hesitations, and natural dialogue modeling.  
FluencyBank contains detailed manual annotations of several disfluency types at the word level, making it suitable for fine-grained token classification tasks.

## 🗂️ Disfluency Labels Used in This Work

We follow the same five disfluency categories defined in the original paper:

- **FP — Filled Pause:**  
  Hesitations such as *“uh”, “um”*.

- **RP — Repetition:**  
  Immediate word or phrase repetitions (e.g., *“I I think…”*).

- **RV — Revision:**  
  Corrections or alterations to previous speech (e.g., *“Take the red—uh, blue one”*).

- **RS — Restart:**  
  Sentence restarts where the speaker abandons an initial structure.

- **PW — Partial Word:**  
  Word fragments or cut-off beginnings (e.g., *“go—going”*).

### 📌 Note  
Although all five labels are supported by our model, **the FluencyBank test set contains only four of them** — FP, RP, RV, and PW.  
The **RS** (restart) class has *no positive examples* in this dataset split, so its evaluation score remains zero.

## Text_Based_Model (language model)

The original project provides a BERT-based text model trained on the **Switchboard** corpus.  
🚀 Our Contribution: In this work, we adapt and fine-tune this text model on the **FluencyBank** dataset to improve cross-domain generalization.

---

### ✔️ 1. Clone Repository
```bash
git clone https://github.com/HoseinRanjbar/disfluency_detection.git
```

### ✔️ 2. Loaded Original Switchboard Weights
The model is initialized using the publicly released **Switchboard-trained language model weights**, ensuring continuity with the original methodology.
```bash
!gdown --id 1GQIXgCSF3Usiuy5hkxgOl483RPX3f_SX -O checkpoints/language.pt
```


### ✔️ 3. Prepared FluencyBank for Fine-Tuning
- Unzip the dataset  
- Mapped word-level labels to BERT subwords  
- Created **speaker-independent** train/validation/test splits  
- Handled variable-length segments and alignment issues

```bash
!unzip ./data/FluencyBank_TimeStamped.zip -d ./data/

!python utils/split_dataset.py \
    --metadata_path data/FluencyBank_TimeStamped/metadata.csv \
    --train_ratio 0.80 \
    --test_ratio 0.10 \
    --val_ratio 0.10
```

This creates:
```bash
data/split/train_metadata.csv
data/split/val_metadata.csv
data/split/test_metadata.csv
```

### ✔️ 4. Fine-Tuned on FluencyBank
Training design:
- BCEWithLogitsLoss for multi-label prediction  
- Step-based evaluation (every 100 training steps)  
- Model selection based on **Unweighted Average Recall (UAR)**  
- Early stopping using patience in evaluation steps  
- Final checkpoint selected using best dev UAR

```bash
!python text_based_model/train.py \
    --train_metadata_path data/split/train_metadata.csv \
    --val_metadata_path data/split/val_metadata.csv \
    --word_dir data/FluencyBank_TimeStamped/csvs/csvs \
    --output_weights checkpoints/language_fluencybank.pt \
    --init_weights checkpoints/language.pt \
    --batch_size 16 \
    --lr 5e-5 \
    --eval_every 100 \
    --patience_evals 10
```

📥 Download the Fine-Tuned Model Weights

```bash
from huggingface_hub import hf_hub_download

checkpoint_path = hf_hub_download(
    repo_id="HosseinRanjbar/disfluency_detection",
    filename="language_model_fluencybank.pth"
)
```

### ✔️ 5. Evaluated the Model
On the held-out FluencyBank test set, we compute:
- Precision  
- Recall  
- F1 per disfluency category  
- Macro-F1  
- UAR (Unweighted Average Recall)

```bash
!python text_based_model/test.py \
    --metadata_path data/split/test_metadata.csv \
    --word_dir data/FluencyBank_TimeStamped/csvs/csvs \
    --weights_path checkpoints/language_fluencybank.pt \
    --batch_size 16
```

---

## Evaluation Results on FluencyBank Test Set

Fine-tuning on FluencyBank yielded consistent improvements across most disfluency categories, especially for high-frequency classes.

### 🔍 Per-Class Performance

| Class | Precision (Before) | Recall (Before) | F1 (Before) | Precision (After) | Recall (After) | F1 (After) |
|-------|--------------------|------------------|--------------|--------------------|-----------------|-------------|
| **FP** | 0.9726 | 1.0000 | 0.9861 | **0.9930** | **1.0000** | **0.9965** |
| **RP** | 0.9545 | 0.5833 | 0.7241 | **0.8976** | **0.9293** | **0.9132** |
| **RV** | 0.3197 | 0.6258 | 0.4232 | **0.5798** | **0.4233** | **0.4894** |
| **PW** | 0.8000 | 0.7926 | 0.7963 | **0.9393** | **0.9263** | **0.9327** |
| **ND** | 0.9821 | 0.9739 | 0.9780 | **0.9819** | **0.9885** | **0.9852** |

<img width="1979" height="1180" alt="output" src="https://github.com/user-attachments/assets/a0fde465-178b-4d13-b0bb-7786ac02160a" />

---

### 📈 Macro Averages

| Metric | Before Fine-Tuning | After Fine-Tuning |
|--------|---------------------|--------------------|
| **Macro Recall** | 0.6626 | **0.7112** |
| **Macro F1** | 0.6513 | **0.7195** |

## Acoustic_Based_Model

### ✔️ 1. Loaded Original Switchboard Weights
The model is initialized using the publicly released **Switchboard-trained acoustic model weights**, ensuring continuity with the original methodology.
```bash
!gdown --id 1wWrmopvvdhlBw-cL7EDyih9zn_IJu5Wr -O checkpoints/acoustic.pt
```

### ✔️ 2. Demo - Test a single audio file

```bash
!python /acoustic_based_model/demo.py --audio_path /voice-example/24fb.wav \
 --metadata_path data/split/test_metadata.csv \
 --word_dir data/FluencyBank_TimeStamped/csvs/csvs \
 --weights_path checkpoints/language_fluencybank.pt 
```

### ✔️ 3. Fine-Tuned on FluencyBank
Training design:
- BCEWithLogitsLoss for multi-label prediction  
- Step-based evaluation (every N training steps)  
- Model selection based on **UAR**  
- Early stopping using patience in evaluation steps  
- Final checkpoint selected using best dev UAR

```bash
!python acoustic_model/train.py \
    --train_metadata_path data/split/train_metadata.csv \
    --dev_metadata_path data/split/val_metadata.csv \
    --audio_dir data/FluencyBank_Wav \
    --word_dir data/FluencyBank_TimeStamped/csvs/csvs \
    --output_weights checkpoints/acoustic_fluencybank.pt \
    --num_epochs 15 \
    --lr 1e-4 \
    --patience_epochs 5 \
    --device cuda
```

### ✔️ 4. Evaluated the Model

```bash
!python acoustic_based_model/test.py \
    --audio_dir data/FluencyBank_Wav \
    --metadata_path data/split/test_metadata.csv \
    --word_dir data/FluencyBank_TimeStamped/csvs/csvs \
    --weights_path checkpoints/acoustic_fluencybank.pt \
    --device cuda \
    --threshold 0.5
```

