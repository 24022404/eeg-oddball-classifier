import os
import pickle
import numpy as np
import pandas as pd
import warnings
import mne
from mne.preprocessing import Xdawn
from sklearn.preprocessing import StandardScaler

try:
    from osl_dynamics.models.hmm import Config, Model
    from osl_dynamics.data import Data
    OSL_AVAILABLE = True
except ImportError:
    OSL_AVAILABLE = False
    print("Warning: osl-dynamics not found.")

warnings.filterwarnings("ignore")

INPUT_EPOCHS_PATH = "../processed/processed_data.pkl"
INPUT_RAW_DIR     = "../processed/raw_filtered" 
OUTPUT_CLASSICAL  = "../processed/optimal_xdawn_dataset.csv"
OUTPUT_HMM        = "../processed/osl_raw_continuous_features.csv"

P300_WIN_TMIN = 0.25  
P300_WIN_TMAX = 0.60  

N_XDAWN_COMPS = 3     

HMM_STATES     = 6    
HMM_LAGE       = 5    
HMM_SEQ_LEN    = 100  
HMM_BATCH_SIZE = 32   
HMM_EPOCHS     = 30   

def load_data(path):
    if not os.path.exists(path):
        alts = [
            "../processed/processed_data.pkl",
            "./processed/processed_data.pkl",
        ]
        for alt in alts:
            if os.path.exists(alt):
                return load_data(alt)
        raise FileNotFoundError(f"File not found: {path}")
    
    print(f"Loading data from: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)

def sanitize_data(data):
    if np.isnan(data).any() or np.isinf(data).any():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    std_val = np.std(data, axis=0)
    mean_val = np.mean(data, axis=0)
    std_val[std_val == 0] = 1.0
    
    limit = 20 * std_val
    return np.clip(data, mean_val - limit, mean_val + limit)

def find_raw_file(subject_id, search_dir):
    candidates = [
        f"{subject_id}_raw_filtered.fif",
        f"{subject_id}_raw_filtered"
    ]
    if os.path.exists(search_dir):
        for fname in candidates:
            fpath = os.path.join(search_dir, fname)
            if os.path.exists(fpath): return fpath
            
    kaggle_path = "/kaggle/input/data-processing/processed/raw_filtered"
    if os.path.exists(kaggle_path):
        for fname in candidates:
            fpath = os.path.join(kaggle_path, fname)
            if os.path.exists(fpath): return fpath
            
    return None

def extract_classical_features(all_subjects):
    print("-" * 60)
    print("PART 1: CLASSICAL FEATURE EXTRACTION")
    print("-" * 60)

    final_data = []

    for subj_id, epochs in all_subjects.items():
        n_targets = np.sum(epochs.events[:, 2] == 6)
        if n_targets < 5:
            continue
            
        print(f"Processing {subj_id}...")
        
        epochs_work = epochs.copy()
        epochs_work.pick_types(eeg=True, verbose=False)
        epochs_work.crop(tmin=0.2, tmax=0.7)
        
        try:
            xdawn = Xdawn(n_components=N_XDAWN_COMPS, correct_overlap=False)
            xdawn.fit(epochs_work)
            epochs_denoised = xdawn.apply(epochs_work)['Stimulus/S  5', 'Stimulus/S  6']
        except Exception:
            continue
            
        X = epochs_denoised.get_data()
        y = epochs_denoised.events[:, 2]
        times = epochs_denoised.times
        
        t_idx_start = np.abs(times - P300_WIN_TMIN).argmin()
        t_idx_end = np.abs(times - P300_WIN_TMAX).argmin()
        
        for i in range(len(X)):
            label = 1 if y[i] == 6 else 0
            
            row = {'subject_id': subj_id, 'epoch_idx': i, 'label': label}
            
            for c in range(N_XDAWN_COMPS):
                signal = X[i, c, t_idx_start:t_idx_end]
                
                row[f'Comp{c}_Mean'] = np.mean(signal)
                row[f'Comp{c}_Max']  = np.max(signal)
                row[f'Comp{c}_Eng']  = np.sum(signal ** 2)
                
                peak_ix = np.argmax(signal)
                row[f'Comp{c}_Lat']  = times[t_idx_start + peak_ix]
            
            final_data.append(row)
            
    if final_data:
        df = pd.DataFrame(final_data)
        df.to_csv(OUTPUT_CLASSICAL, index=False)
        print(f"Saved Classical features: {OUTPUT_CLASSICAL} | Shape: {df.shape}")
    else:
        print("Failed to extract classical features.")

def extract_hmm_features(all_subjects):
    print("-" * 60)
    print("PART 2: CONTINUOUS HMM EXTRACTION")
    print("-" * 60)
    
    if not OSL_AVAILABLE:
        print("OSL Dynamics not available.")
        return

    continuous_data_list = []
    subject_meta = []

    print("Step 1: Preparing Data...")
    
    for subj_id, epochs_ref in all_subjects.items():
        n_targets = np.sum(epochs_ref.events[:, 2] == 6)
        if n_targets < 5: continue
        
        raw_path = find_raw_file(subj_id, INPUT_RAW_DIR)
        if not raw_path:
            continue
            
        print(f"Processing {subj_id}...")
        
        try:
            try:
                raw = mne.io.read_raw_fif(raw_path, preload=True, verbose='error')
            except:
                raw = mne.io.read_raw_fif(raw_path + ".fif", preload=True, verbose='error')
        except Exception:
            continue

        raw.pick_types(eeg=True)
        
        if raw.info['sfreq'] > 100: raw.resample(100, verbose=False)
        if epochs_ref.info['sfreq'] > 100: 
            epochs_ref = epochs_ref.copy().resample(100, verbose=False)
            
        try:
            xdawn = Xdawn(n_components=N_XDAWN_COMPS, correct_overlap=False)
            xdawn.fit(epochs_ref)
            raw_denoised = xdawn.apply(raw)['Stimulus/S  6']
        except Exception:
            continue

        data_cont = raw_denoised.get_data().T.astype(np.float32)
        data_clean = sanitize_data(data_cont)
        continuous_data_list.append(data_clean)
        
        subject_meta.append({
            'sid': subj_id,
            'events': epochs_ref.events, 
            'sfreq': raw.info['sfreq']
        })

    if not continuous_data_list:
        print("No raw data processed.")
        return

    print("Step 2: Training HMM...")
    
    training_data = Data(continuous_data_list)
    training_data.tde(n_embeddings=HMM_LAGE)
    training_data.standardize()
    
    print(f"Input channels: {training_data.n_channels}")

    config = Config(
        n_states=HMM_STATES,
        n_channels=training_data.n_channels,
        sequence_length=HMM_SEQ_LEN,
        learn_means=True,
        learn_covariances=True,
        batch_size=HMM_BATCH_SIZE,
        learning_rate=0.005, 
        n_epochs=HMM_EPOCHS
    )
    
    model = Model(config)
    model.fit(training_data)
    
    print("Step 3: Extracting Features...")
    
    alphas = model.get_alpha(training_data)
    final_feats = []
    t_offset = HMM_LAGE // 2
    
    for idx, alpha_cont in enumerate(alphas):
        meta = subject_meta[idx]
        events = meta['events']
        sfreq = meta['sfreq']
        
        win_start = int(P300_WIN_TMIN * sfreq)
        win_end = int(P300_WIN_TMAX * sfreq)
        
        for i in range(len(events)):
            evt_time = events[i, 0] 
            code = events[i, 2]
            
            if code == 6: label = 1
            elif code == 5: label = 0
            else: continue
            
            start_idx = evt_time + win_start - t_offset
            end_idx = evt_time + win_end - t_offset
            
            if start_idx < 0 or end_idx > alpha_cont.shape[0]: continue
            
            alpha_window = alpha_cont[start_idx:end_idx, :]
            
            if alpha_window.size == 0: continue
            fo = np.mean(alpha_window, axis=0)
            
            row = {'subject_id': meta['sid'], 'epoch_idx': i, 'label': label}
            for s in range(HMM_STATES):
                row[f'State_{s}_FO'] = fo[s]
                
            final_feats.append(row)
        
    if final_feats:
        df = pd.DataFrame(final_feats).fillna(0)
        df.to_csv(OUTPUT_HMM, index=False)
        print(f"Saved HMM features: {OUTPUT_HMM} | Shape: {df.shape}")
        print(df.groupby('label').mean(numeric_only=True))
    else:
        print("Failed to extract HMM features.")

if __name__ == "__main__":
    try:
        data = load_data(INPUT_EPOCHS_PATH)
    except Exception as e:
        print(e)
        exit()
        
    extract_classical_features(data)
    extract_hmm_features(data)
    
    print("-" * 60)
    print("DONE")
    print("-" * 60)