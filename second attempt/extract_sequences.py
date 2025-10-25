# extract_sequences.py
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# -------- CONFIG (edit only these if needed) --------
DATASET_ROOT = r"C:\Users\GG\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen"
SPLITS_DIR   = os.path.join(DATASET_ROOT, "splits")
VIDEOS_DIR   = os.path.join(DATASET_ROOT, "videos")
OUTPUT_ROOT  = os.path.join(DATASET_ROOT, "sequences")  # where sequences will be written

FRACTION = 0.10              # use 10% of each CSV
FRAMES_PER_CLIP = 6          # how many frames to save per video
IMG_SIZE = (112, 112)        # (width, height) of saved images

VIDEO_COL = "Video file"     # CSV column name containing filenames
LABEL_COL = "Gloss"          # CSV column name containing labels

# -------- helpers --------
def uniform_indices(num_frames, k):
    if num_frames <= 0:
        return []
    if num_frames >= k:
        return np.linspace(0, num_frames - 1, k, dtype=int).tolist()
    idx = np.linspace(0, num_frames - 1, num_frames, dtype=int).tolist()
    idx += [idx[-1]] * (k - num_frames)
    return idx[:k]

def sanitize_stem(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    return "".join(ch for ch in stem if ch.isalnum() or ch in ("-", "_"))

def save_frames_from_video_sequential(video_path, out_dir, indices_set, img_size):
    """
    Read video sequentially and save frames whose indices are in indices_set.
    This avoids repeated random seeks (faster overall for extraction).
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "cannot open video"

    saved = 0
    fr_idx = 0
    wanted = sorted(indices_set)
    w_idx = 0
    W, H = img_size
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fr_idx == wanted[w_idx]:
            frame_resized = cv2.resize(frame, (W, H))
            out_path = os.path.join(out_dir, f"frame_{saved+1:04d}.jpg")
            ok = cv2.imwrite(out_path, frame_resized)
            if not ok:
                cap.release()
                return False, f"failed write {out_path}"
            saved += 1
            w_idx += 1
            if w_idx >= len(wanted):
                break
        fr_idx += 1

    cap.release()
    # If not enough saved (very short video), pad by repeating last frame if any
    if saved < len(wanted):
        if saved == 0:
            return False, "no frames captured"
        last_path = os.path.join(out_dir, f"frame_{saved:04d}.jpg")
        last_img = cv2.imread(last_path)
        for i in range(saved, len(wanted)):
            out_path = os.path.join(out_dir, f"frame_{i+1:04d}.jpg")
            cv2.imwrite(out_path, last_img)
        saved = len(wanted)
    return True, "ok"

# -------- main processing --------
def process_split(split_name):
    csv_path = os.path.join(SPLITS_DIR, f"{split_name}.csv")
    if not os.path.isfile(csv_path):
        print(f"[WARN] CSV missing: {csv_path}")
        return

    print(f"[INFO] Reading {csv_path}")
    df = pd.read_csv(csv_path)

    if VIDEO_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"CSV must contain columns: '{VIDEO_COL}', '{LABEL_COL}'. Found: {df.columns.tolist()}")

    # sample fraction (global sample, faster & simpler)
    if 0 < FRACTION < 1.0:
        df = df.sample(frac=FRACTION, random_state=42).reset_index(drop=True)

    out_split = os.path.join(OUTPUT_ROOT, split_name)
    os.makedirs(out_split, exist_ok=True)

    errors = 0
    processed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extract {split_name}"):
        video_val = str(row[VIDEO_COL])
        label = str(row[LABEL_COL]).strip()
        video_filename = os.path.basename(video_val)
        video_path = os.path.join(VIDEOS_DIR, video_filename)

        if not os.path.isfile(video_path):
            errors += 1
            continue

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total <= 0:
            # fallback: read all frames to count
            cap2 = cv2.VideoCapture(video_path)
            cnt = 0
            while True:
                r, _ = cap2.read()
                if not r:
                    break
                cnt += 1
            cap2.release()
            total = cnt

        idxs = uniform_indices(total, FRAMES_PER_CLIP)
        seq_name = f"{sanitize_stem(video_filename)}_clip01"
        out_dir = os.path.join(out_split, label, seq_name)
        ok, msg = save_frames_from_video_sequential(video_path, out_dir, set(idxs), IMG_SIZE)
        if not ok:
            errors += 1
        processed += 1

    print(f"[DONE] {split_name}: processed={processed}, errors={errors}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    process_split("train")
    process_split("val")
    process_split("test")
    print("All finished.")
