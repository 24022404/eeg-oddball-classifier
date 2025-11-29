import os
import pickle
import numpy as np
import pandas as pd
import warnings
import mne
from sklearn.preprocessing import StandardScaler
from mne.preprocessing import Xdawn

try:
    from osl_dynamics.models.hmm import Config, Model
    from osl_dynamics.data import Data
    OSL_AVAILABLE = True
except ImportError:
    OSL_AVAILABLE = False
    print("⚠️ CẢNH BÁO: Chưa cài đặt 'osl-dynamics'.")

warnings.filterwarnings("ignore")

INPUT_PATH = "/kaggle/input/data-processing/processed/processed_data.pkl"
INPUT_RAW_DIR = "/kaggle/input/data-processing/processed/raw_filtered" 

OUT_CLASSICAL = "./eeg.csv"
OUT_HMM       = "./hmm.csv"

ROI_CHANNELS = ['Pz', 'Cz', 'CPz', 'P3', 'P4', 'Fz', 'C3', 'C4']

HMM_STATES     = 6    
HMM_LAGS       = 5    
HMM_PCA_COMPS  = 12   
HMM_SEQ_LEN    = 32   
HMM_BATCH_SIZE = 16   
HMM_EPOCHS     = 10  
N_XDAWN_COMPS = 3
P300_WIN_TMIN = 0.25  
P300_WIN_TMAX = 0.60 

def load_data(path):
    if not os.path.exists(path):
        alts = ["../processed/processed_data.pkl", "./processed/processed_data.pkl"]
        for alt in alts:
            if os.path.exists(alt):
                print(f"⚠️ Dùng đường dẫn dự phòng: {alt}")
                return load_data(alt)
        raise FileNotFoundError(f"❌ Không tìm thấy file: {path}")
    
    print(f"📂 Loading data: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)

def find_raw_file(subject_id, search_dir):
    candidates = [
        f"{subject_id}_raw_filtered.fif",
        f"{subject_id}_raw_filtered"
    ]
    if not os.path.exists(search_dir): return None
    
    for fname in candidates:
        full_path = os.path.join(search_dir, fname)
        if os.path.exists(full_path): return full_path
    return None

def sanitize_data(data):
    if np.isnan(data).any():
        data = np.nan_to_num(data, nan=0.0)
    
    std_val = np.std(data, axis=0)
    mean_val = np.mean(data, axis=0)
    std_val[std_val == 0] = 1.0
    
    limit = 20 * std_val
    return np.clip(data, mean_val - limit, mean_val + limit)


def extract_classical(all_subjects):
    print(f"\n{'='*50}")
    print(f"💎 CLASSICAL FEATURE EXTRACTION (ROI-BASED)")
    print(f"{'='*50}")
    
    final_data = []
    
    for subj_id, epochs in all_subjects.items():
        if np.sum(epochs.events[:, 2] == 6) < 3:
            print(f"⚠️ Bỏ qua {subj_id} (Dữ liệu < 3 targets)")
            continue
            
        print(f"Processing: {subj_id}...")
        
        picks = [ch for ch in ROI_CHANNELS if ch in epochs.ch_names]
        if not picks: picks = epochs.ch_names[:5] 
        
        ep_crop = epochs.copy().pick(picks).crop(P300_WIN_TMIN, P300_WIN_TMAX) 
        
        X = ep_crop.get_data() 
        y = ep_crop.events[:, 2]
        times = ep_crop.times
        
        for i in range(len(X)):
            label = 1 if y[i] == 6 else 0 if y[i] == 5 else -1
            if label == -1: continue
            
            row = {'subject_id': subj_id, 'epoch_idx': i, 'label': label}
            
            roi_mean = np.mean(X[i]) 
            row['ROI_Mean_Amp'] = roi_mean
            target_ch = 'Pz' if 'Pz' in picks else picks[0]
            ch_idx = picks.index(target_ch)
            signal = X[i, ch_idx, :]
            
            row['Peak_Amp'] = np.max(signal)
            row['Peak_Latency'] = times[np.argmax(signal)]
            row['Peak_to_Peak'] = np.ptp(signal)
            
            final_data.append(row)
            
    if final_data:
        df = pd.DataFrame(final_data)
        df.to_csv(OUT_CLASSICAL, index=False)
        print(f"✅ ĐÃ LƯU: {OUT_CLASSICAL} | Shape: {df.shape}")
    else:
        print("❌ Lỗi: Không tạo được eeg.csv")

def extract_hmm_features(all_subjects):
    print(f"\n{'='*60}")
    print(f"HMM ADVANCED (RAW CONTINUOUS DATA)")
    print(f"{'='*60}")
    
    if not OSL_AVAILABLE: return

    continuous_data_list = []
    subject_meta = []

    print(f"[1/3] Chuẩn bị dữ liệu Raw & Giảm chiều thủ công...")
    
    for subj_id, epochs_ref in all_subjects.items():
        if np.sum(epochs_ref.events[:, 2] == 6) < 3: continue
            
        # Tìm file Raw
        raw_path = find_raw_file(subj_id, INPUT_RAW_DIR)
        if not raw_path:
            print(f"⚠️ Bỏ qua {subj_id}: Không tìm thấy file Raw.")
            continue
            
        print(f"HMM: Xử lý {subj_id}...", end=" ")
        
        try:
            try:
                raw = mne.io.read_raw_fif(raw_path, preload=True, verbose='error')
            except:
                raw = mne.io.read_raw_fif(raw_path + ".fif", preload=True, verbose='error')
        except:
            print("Lỗi đọc file."); continue
        raw.pick_types(eeg=True)
        if raw.info['sfreq'] > 100: raw.resample(100, verbose=False)
        if epochs_ref.info['sfreq'] > 100: 
            epochs_ref = epochs_ref.copy().resample(100, verbose=False)
            
        try:
            xdawn = Xdawn(n_components=N_XDAWN_COMPS, correct_overlap=False)
            xdawn.fit(epochs_ref)
            
            all_filters = xdawn.filters_['Stimulus/S  6']
            best_filters = all_filters[:N_XDAWN_COMPS, :]
            
            raw_data = raw.get_data()
            projected_data = np.dot(best_filters, raw_data)
            
            print(f"Done. Shape: {projected_data.shape}")
            
        except Exception as e:
            print(f"Lỗi Projection: {e}"); continue

        data_cont = projected_data.T.astype(np.float32)
        data_clean = sanitize_data(data_cont)
        
        continuous_data_list.append(data_clean)
        
        subject_meta.append({
            'sid': subj_id,
            'events': epochs_ref.events, 
            'sfreq': raw.info['sfreq']
        })

    if not continuous_data_list:
        print("❌ Không có dữ liệu Raw nào hợp lệ."); return

    print(f"\n[2/3] Huấn luyện HMM (OSL-Dynamics)...")

    training_data = Data(continuous_data_list)
    
    training_data.tde(n_embeddings=HMM_LAGS)
    training_data.standardize()
    
    print(f"Input Channels cho HMM: {training_data.n_channels}")

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
    
    print(f"\n[3/3] Trích xuất Fractional Occupancy...")
    
    alphas = model.get_alpha(training_data)
    final_feats = []
    t_offset = HMM_LAGS // 2
    
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
            for s in range(HMM_STATES): row[f'State_{s}_FO'] = fo[s]
                
            final_feats.append(row)
        
    if final_feats:
        df = pd.DataFrame(final_feats).fillna(0)
        df.to_csv(OUT_HMM, index=False)
        print(f"✅ Đã lưu file HMM: {OUT_HMM} | Shape: {df.shape}")
        
        print("\n🔍 Khác biệt State (Target vs Standard):")
        print(df.groupby('label').mean(numeric_only=True))
    else:
        print("❌ Thất bại phần HMM.")

if __name__ == "__main__":
    print(f"🚀 BẮT ĐẦU TRÍCH XUẤT DỮ LIỆU...")
    try:
        data = load_data(INPUT_PATH)
        extract_classical(data)
        extract_hmm_features(data)
        print(f"\n🎉 HOÀN TẤT!")
    except Exception as e:
        print(f"\n❌ LỖI NGHIÊM TRỌNG: {e}")