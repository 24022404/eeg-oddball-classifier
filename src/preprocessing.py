"""
Module tiền xử lý dữ liệu EEG cho Oddball task.
"""

import os
import pickle
import mne
import numpy as np

def preprocess_subject_eeg(subject_id, data_dir="./dataset", save_raw_filtered=False, raw_output_dir="./processed/raw_filtered"):
    """
    Preprocess EEG data cho một subject (Oddball task).
    
    Args:
        subject_id (str): ID của subject, ví dụ: 'sub-01'
        data_dir (str): Thư mục chứa file EEG
        save_raw_filtered (bool): Có lưu raw filtered data không (cho HMM)
        raw_output_dir (str): Thư mục lưu raw filtered data
    
    Returns:
        tuple: (mne.Epochs, str hoặc None) - epochs và đường dẫn file raw filtered
    """
    vhdr_file = os.path.join(data_dir, f"{subject_id}_task-oddball_eeg.vhdr")
    eeg_file  = os.path.join(data_dir, f"{subject_id}_task-oddball_eeg.eeg")
    vmrk_file = os.path.join(data_dir, f"{subject_id}_task-oddball_eeg.vmrk")

    # Kiểm tra file tồn tại
    if not os.path.exists(vhdr_file):
        raise FileNotFoundError(f"Thiếu file .vhdr cho {subject_id}")
    if not os.path.exists(eeg_file):
        raise FileNotFoundError(f"Thiếu file .eeg cho {subject_id}")
    if not os.path.exists(vmrk_file):
        raise FileNotFoundError(f"Thiếu file .vmrk cho {subject_id}")

    print(f"\n{'='*60}")
    print(f"Đang xử lý: {subject_id}")
    print(f"{'='*60}")
    
    # Load raw data
    print(f"[1/6] Đang load dữ liệu raw...")
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True, verbose=False)
    print(f"      ✓ Loaded {len(raw.ch_names)} kênh, {raw.n_times} samples, sfreq={raw.info['sfreq']} Hz")

    # Lọc tín hiệu
    print(f"[2/6] Đang lọc tín hiệu (1-40 Hz + notch 50 Hz)...")
    raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
    raw.notch_filter(freqs=50.0, verbose=False)
    print(f"      ✓ Hoàn thành lọc tín hiệu")

    # Lưu raw filtered cho HMM
    raw_filtered_path = None
    if save_raw_filtered:
        print(f"[3/6] Đang lưu raw filtered data cho HMM...")
        if not os.path.exists(raw_output_dir):
            os.makedirs(raw_output_dir)
        raw_filtered_path = os.path.join(raw_output_dir, f"{subject_id}_raw_filtered.fif")
        raw.save(raw_filtered_path, overwrite=True, verbose=False)
        file_size = os.path.getsize(raw_filtered_path) / (1024**2)
        print(f"      ✓ Đã lưu raw filtered tại: {raw_filtered_path}")
        print(f"      ✓ Kích thước file: {file_size:.1f} MB")

    # Trích xuất events
    print(f"[4/6] Đang trích xuất events...")
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    
    # Oddball paradigm
    oddball_event_id = {
        'Stimulus/S  5': 5,  # Standard
        'Stimulus/S  6': 6,  # Target
        'Stimulus/S  7': 7   # Novel
    }
    
    oddball_codes = [5, 6, 7]
    oddball_events = events[np.isin(events[:, 2], oddball_codes)]
    
    print(f"      ✓ Tìm thấy {len(oddball_events)} oddball events")
    print(f"        - Standard (S 5): {np.sum(oddball_events[:, 2] == 5)}")
    print(f"        - Target (S 6):   {np.sum(oddball_events[:, 2] == 6)}")
    print(f"        - Novel (S 7):    {np.sum(oddball_events[:, 2] == 7)}")

    # Tạo epochs
    print(f"[5/6] Đang tạo epochs (-0.2 đến 0.8s)...")
    epochs = mne.Epochs(
        raw,
        oddball_events,
        event_id=oddball_event_id,
        tmin=-0.2,
        tmax=0.8,
        baseline=(-0.2, 0),
        preload=True,
        verbose=False,
        reject_by_annotation=True
    )
    
    initial_count = len(epochs)
    print(f"      ✓ Tạo được {initial_count} epochs")

    # Loại bỏ artifacts
    print(f"[6/6] Đang loại bỏ artifacts (threshold: 100 µV)...")
    epochs.drop_bad(reject={'eeg': 100e-6}, verbose=False)
    
    final_count = len(epochs)
    rejected = initial_count - final_count
    rejection_rate = (rejected / initial_count) * 100 if initial_count > 0 else 0
    
    print(f"      ✓ Epochs sau khi làm sạch: {final_count}")
    print(f"      ✓ Đã loại bỏ: {rejected} epochs ({rejection_rate:.1f}%)")
    
    return epochs, raw_filtered_path


def run_all_subjects(subject_list, data_dir="./dataset", output_dir="./processed", save_raw=True):
    """
    Chạy tiền xử lý cho tất cả subjects và lưu kết quả.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Đã tạo thư mục: {output_dir}\n")
    
    raw_output_dir = os.path.join(output_dir, "raw_filtered")
    
    processed_data = {}
    raw_filtered_info = {}
    success_count = 0
    fail_count = 0
    
    print(f"\n{'='*60}")
    print(f"BẮT ĐẦU TIỀN XỬ LÝ {len(subject_list)} SUBJECTS")
    print(f"{'='*60}\n")
    
    for idx, subject_id in enumerate(subject_list, 1):
        try:
            print(f"\n[{idx}/{len(subject_list)}] Processing {subject_id}...")
            epochs, raw_path = preprocess_subject_eeg(
                subject_id, 
                data_dir,
                save_raw_filtered=save_raw,
                raw_output_dir=raw_output_dir
            )
            processed_data[subject_id] = epochs
            if raw_path:
                raw_filtered_info[subject_id] = raw_path
            success_count += 1
            print(f"✅ THÀNH CÔNG: {subject_id}")
            
        except Exception as e:
            print(f"❌ LỖI cho {subject_id}: {e}")
            fail_count += 1
            continue
    
    if processed_data:
        output_file = os.path.join(output_dir, "processed_data.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"\n{'='*60}")
        print(f"KẾT QUẢ TIỀN XỬ LÝ")
        print(f"{'='*60}")
        print(f"✅ Thành công: {success_count}/{len(subject_list)} subjects")
        print(f"❌ Thất bại:   {fail_count}/{len(subject_list)} subjects")
        print(f"\n📦 Đã lưu processed_data.pkl tại: {output_file}")
        
        if save_raw and raw_filtered_info:
            print(f"\n📦 RAW FILTERED DATA (cho HMM):")
            print(f"   Đã lưu {len(raw_filtered_info)} files tại: {raw_output_dir}/")
        
        print(f"{'='*60}\n")
    else:
        print("\n❌ KHÔNG SUBJECT NÀO XỬ LÝ THÀNH CÔNG!")