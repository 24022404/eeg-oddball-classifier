# EEG Oddball Classification

Phân loại tín hiệu EEG sử dụng Machine Learning và Hidden Markov Models.

## 👥 Team Members

1. Đồng Mạnh Hùng - 23020370
2. Phạm Anh Quân - 22022625
3. Cao Đặng Quốc Vương - 22022601
4. Nguyễn Văn Thắn - 22022596
5. Nguyễn Đức Minh - 24022404
6. Nguyễn Xuân Hiệp - 22022591
7. Lương Minh Trí - 23020440

## 📋 Overview

Classification of EEG signals from Oddball paradigm:
- **Frequent (S 5)**: Standard stimuli (~80%)
- **Target (S 6)**: Deviant stimuli (~10%)

**Dataset**: 5 subjects, 127 EEG channels, 1000 Hz sampling rate

## 🚀 Quick Start

```bash
git clone https://github.com/24022404/eeg-oddball-classifier.git
cd eeg-oddball-classifier
pip install -r requirements.txt
```

## 📁 Structure

```
├── dataset/         # Raw EEG data (.vhdr, .eeg, .vmrk)
├── data/            # Processed features (eeg.csv, hmm.csv)
├── results/         # Output figures and models
├── src/             # Source code
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── visualization.ipynb
│   └── model.ipynb
└── README.md
```

## 📊 Results

**Approach 1: Pooled SMOTE**
- EEG: 82% accuracy (Random Forest)
- HMM: 90% accuracy (Neural Network)
- HMM improvement: +9.8%

**Approach 2: Per-Subject SMOTE**
- EEG: 84% accuracy (Random Forest)
- HMM: 91% accuracy (Neural Network)
- HMM improvement: +8.3%

## 🔗 Links

- **Report**: [Overleaf](https://www.overleaf.com/9331923581dhbrqjjmbywn#3c46f1)
- **Repository**: [GitHub](https://github.com/24022404/eeg-oddball-classifier)

## 📄 License

MIT License