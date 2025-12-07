# EEG Oddball Classifier

Dự án phân loại tín hiệu EEG từ paradigm Oddball sử dụng Machine Learning và Hidden Markov Models.

## 📋 Mô tả

Project này thực hiện tiền xử lý và phân tích dữ liệu EEG từ thí nghiệm Oddball task, bao gồm:

- **Standard stimuli (S 5)**: Kích thích thường gặp
- **Target stimuli (S 6)**: Kích thích hiếm (deviant)
- **Novel stimuli (S 7)**: Kích thích mới lạ

## 🎯 Mục tiêu

1. Tiền xử lý dữ liệu EEG (filtering, artifact rejection)
2. Trích xuất đặc trưng từ epochs
3. Phân loại các loại kích thích sử dụng ML/HMM
4. Trực quan hóa kết quả

## 📁 Cấu trúc Project

```
eeg-oddball-classifier/
├── dataset/                    # Dữ liệu raw EEG (không upload lên git)
│   ├── sub-01_task-oddball_eeg.vhdr
│   ├── sub-01_task-oddball_eeg.eeg
│   └── ...
├── processed/                  # Dữ liệu đã xử lý (không upload lên git)
│   ├── processed_data.pkl     # Epochs data
│   └── raw_filtered/          # Raw filtered data cho HMM
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_feature_extraction.ipynb
├── src/                        # Source code
│   ├── __init__.py
│   ├── preprocessing.py       # Tiền xử lý
│   ├── feature_extraction.py  # Trích xuất đặc trưng
│   ├── hmm_training.py        # HMM training
│   └── visualization.py       # Trực quan hóa
├── results/                    # Kết quả, figures
├── tests/                      # Unit tests
├── .gitignore
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/24022404/eeg-oddball-classifier.git
cd eeg-oddball-classifier
```

### 2. Tạo virtual environment

```bash
python -m venv eeg_env
source eeg_env/bin/activate  # Linux/Mac
# hoặc
eeg_env\Scripts\activate  # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## 📊 Dataset

Dataset bao gồm dữ liệu EEG từ 5 subjects:

- **Format**: Brain Vision (.vhdr, .eeg, .vmrk)
- **Channels**: 127 EEG channels
- **Sampling rate**: 1000 Hz
- **Task**: Visual Oddball paradigm

### Tải dataset

Dataset có thể được tải từ: [Link đến dataset của bạn]

Giải nén vào thư mục `dataset/`:

```bash
mkdir dataset
# Copy các file .vhdr, .eeg, .vmrk vào đây
```

## 🔧 Sử dụng

### 1. Tiền xử lý dữ liệu

```bash
python preprocess.py
```

Hoặc sử dụng module:

```python
from src.preprocessing import preprocess_subject_eeg, run_all_subjects

# Xử lý 1 subject
epochs = preprocess_subject_eeg('sub-01', data_dir='./dataset')

# Xử lý tất cả subjects
subject_list = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']
run_all_subjects(subject_list, save_raw=True)
```

### 2. Trích xuất đặc trưng

```python
# Sẽ cập nhật sau khi Nhóm 3 hoàn thành
```

### 3. Training HMM

```python
# Sẽ cập nhật sau khi Nhóm 3 hoàn thành
```

## 📈 Kết quả

### Preprocessing Statistics

| Subject | Initial Epochs | After Cleaning | Rejection Rate |
| ------- | -------------- | -------------- | -------------- |
| sub-01  | 579            | 556            | 4.0%           |
| sub-02  | 579            | 548            | 5.4%           |
| sub-04  | 579            | 542            | 6.4%           |
| sub-05  | 579            | 551            | 4.8%           |

### Event Distribution

- **Standard (S 5)**: ~80% của tổng events
- **Target (S 6)**: ~10% của tổng events
- **Novel (S 7)**: ~10% của tổng events

## 🛠️ Quy trình Preprocessing

1. **Load raw data** (.vhdr format)
2. **Filtering**:
   - Bandpass filter: 1-40 Hz
   - Notch filter: 50 Hz (line noise)
3. **Event extraction**: Lấy S 5, S 6, S 7 từ .vmrk
4. **Epoching**: -0.2 đến 0.8s quanh stimulus
5. **Artifact rejection**: Threshold 100 µV
6. **Save**:
   - `processed_data.pkl`: Epochs (cho feature extraction)
   - `raw_filtered/*.fif`: Raw continuous (cho HMM)

## 👥 Nhóm phát triển

- **Nhóm 1**: Tìm hiểu lý thuyết & Dataset
- **Nhóm 2**: Tiền xử lý dữ liệu ✅
- **Nhóm 3**: Trích xuất đặc trưng & HMM
- **Nhóm 4**: Huấn luyện & Đánh giá
- **Nhóm 5**: Trực quan hóa

## 📄 Báo cáo

**📝 Technical Report (LaTeX)**: [View on Overleaf]([https://www.overleaf.com/project/6934c066d10f4dd7b24edbb5](https://www.overleaf.com/9331923581dhbrqjjmbywn#3c46f1))

Báo cáo chi tiết về phương pháp, kết quả và phân tích được viết bằng LaTeX và có thể xem/chỉnh sửa trên Overleaf.

## 📚 Tài liệu tham khảo

- [MNE-Python Documentation](https://mne.tools/stable/index.html)
- [Brain Vision Data Exchange Format](https://www.brainproducts.com/support-resources/brainvision-core-data-format-1-0/)
- [Oddball Paradigm](https://en.wikipedia.org/wiki/Oddball_paradigm)

## 📝 License

MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng:

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📧 Liên hệ

- GitHub: [@24022404](https://github.com/24022404)
- Repository: [eeg-oddball-classifier](https://github.com/24022404/eeg-oddball-classifier)

---

**Last updated**: December 2025
