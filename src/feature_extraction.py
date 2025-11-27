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
    print("⚠️ CẢNH BÁO: Chưa cài đặt 'osl-dynamics'. Phần HMM sẽ bị bỏ qua.")

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
    """Load dữ liệu an toàn với đường dẫn dự phòng."""
    if not os.path.exists(path):
        alts = [
            "../processed/processed_data.pkl",
            "./processed/processed_data.pkl"
        ]
        for alt in alts:
            if os.path.exists(alt):
                return load_data(alt)
        raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu tại: {path}")
    
    print(f"Đang tải dữ liệu Epochs từ: {path}")
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
    """Tìm file raw .fif linh hoạt"""
    candidates = [
        f"{subject_id}_raw_filtered.fif",
        f"{subject_id}_raw_filtered"
    ]
    if os.path.exists(search_dir):
        for fname in candidates:
            fpath = os.path.join(search_dir, fname)
            if os.path.exists(fpath): return fpath
            
    kaggle_path = "../processed/raw_filtered"
    if os.path.exists(kaggle_path):
        for fname in candidates:
            fpath = os.path.join(kaggle_path, fname)
            if os.path.exists(fpath): return fpath
            
    return None

def extract_classical_features(all_subjects):
    print(f"\n{'='*60}")
    print(f"💎 PHẦN 1: TRÍCH XUẤT ĐẶC TRƯNG CỔ ĐIỂN (xDAWN + STATS)")
    print(f"{'='*60}")

    final_data = []

    for subj_id, epochs in all_subjects.items():
        n_targets = np.sum(epochs.events[:, 2] == 6)
        if n_targets < 5:
            print(f"⚠️ Bỏ qua {subj_id}: Dữ liệu quá ít ({n_targets} targets).")
            continue
            
        print(f"⚡ Classical: Xử lý {subj_id}...")
        
        epochs_work = epochs.copy()
        epochs_work.pick_types(eeg=True, verbose=False)
        epochs_work.crop(tmin=0.2, tmax=0.7)
        
        try:
            xdawn = Xdawn(n_components=N_XDAWN_COMPS, correct_overlap=False)
            xdawn.fit(epochs_work)
            epochs_denoised = xdawn.apply(epochs_work)['Stimulus/S  5', 'Stimulus/S  6']
        except Exception as e:
            print(f"   ❌ Lỗi xDAWN: {e}. Bỏ qua.")
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
        print(f"✅ Đã lưu file Classical: {OUTPUT_CLASSICAL} | Shape: {df.shape}")
    else:
        print("❌ Thất bại phần Classical.")

def extract_hmm_features(all_subjects):
    print(f"\n{'='*60}")
    print(f"PHẦN 2: TRÍCH XUẤT HMM TỪ DỮ LIỆU THÔ (CONTINUOUS RAW)")
    print(f"   Chiến thuật: Fit Epochs -> Apply Raw -> Train HMM -> Slice")
    print(f"{'='*60}")
    
    if not OSL_AVAILABLE:
        print("Không tìm thấy thư viện osl-dynamics. Dừng phần 2."); return

    continuous_data_list = []
    subject_meta = []

    print(f"[1/3] Chuẩn bị dữ liệu Raw & xDAWN Hybrid...")
    
    for subj_id, epochs_ref in all_subjects.items():
        n_targets = np.sum(epochs_ref.events[:, 2] == 6)
        if n_targets < 5: continue
        raw_path = find_raw_file(subj_id, INPUT_RAW_DIR)
        if not raw_path:
            print(f"Bỏ qua {subj_id}: Không tìm thấy file Raw (.fif)")
            continue
            
        print(f"HMM: Xử lý {subj_id} (Raw found)...")
        
        try:
            try:
                raw = mne.io.read_raw_fif(raw_path, preload=True, verbose='error')
            except:
                raw = mne.io.read_raw_fif(raw_path + ".fif", preload=True, verbose='error')
        except Exception as e:
            print(f"      ❌ Lỗi đọc file Raw: {e}"); continue

        raw.pick_types(eeg=True)
        
        if raw.info['sfreq'] > 100: raw.resample(100, verbose=False)
        if epochs_ref.info['sfreq'] > 100: 
            epochs_ref = epochs_ref.copy().resample(100, verbose=False)
            
        try:
            xdawn = Xdawn(n_components=N_XDAWN_COMPS, correct_overlap=False)
            xdawn.fit(epochs_ref)
            
            raw_denoised = xdawn.apply(raw)['Stimulus/S  6']
            
        except Exception as e:
            print(f"Lỗi xDAWN: {e}. Bỏ qua."); continue

        data_cont = raw_denoised.get_data().T.astype(np.float32)
        data_clean = sanitize_data(data_cont)
        continuous_data_list.append(data_clean)
        
        subject_meta.append({
            'sid': subj_id,
            'events': epochs_ref.events, 
            'sfreq': raw.info['sfreq']
        })

    if not continuous_data_list:
        print("❌ Không có dữ liệu Raw nào được xử lý thành công."); return

    print(f"\n[2/3] Huấn luyện HMM trên Raw (OSL-Dynamics)...")
    training_data = Data(continuous_data_list)
    
    print(f" Applying TDE (lags={HMM_LAGE})...")
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
    
    print(f"Training Model...")
    model = Model(config)
    model.fit(training_data)
    
    print(f"\n[3/3] Cắt chuỗi trạng thái (Slicing)...")
    
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
        print(f"✅ Đã lưu file HMM: {OUTPUT_HMM} | Shape: {df.shape}")
        
        print(df.groupby('label').mean(numeric_only=True))
    else:
        print("❌ Thất bại phần HMM.")


if __name__ == "__main__":    
    # 1. Load dữ liệu Epochs
    try:
        data = load_data(INPUT_EPOCHS_PATH)
    except Exception as e:
        print(e)
        exit()
        
    # 2. Chạy Classical
    extract_classical_features(data)
    
    # 3. Chạy HMM 
    extract_hmm_features(data)
    
    print(f"\n{'='*60}")
    print(f"🎉 HOÀN TẤT TOÀN BỘ QUY TRÌNH!")
    print(f"{'='*60}")