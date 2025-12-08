# EEG Oddball Classification

Classification of EEG signals from Oddball paradigm using Machine Learning and Hidden Markov Models.

## Team Members

1. Đồng Mạnh Hùng - 23020370
2. Phạm Anh Quân - 22022625
3. Cao Đặng Quốc Vương - 22022601
4. Nguyễn Văn Thắn - 22022596
5. Nguyễn Đức Minh - 24022404
6. Nguyễn Xuân Hiệp - 22022591
7. Lương Minh Trí - 23020440

## Overview

Classification task:
- **Frequent (S5)**: Standard stimuli (~80%)
- **Target (S6)**: Deviant stimuli (~10%)

Dataset: 4 subjects, 127 EEG channels, 1000 Hz, 847 trials (imbalance ratio 7.1:1)

## Quick Start

```bash
git clone https://github.com/24022404/eeg-oddball-classifier.git
cd eeg-oddball-classifier
pip install -r requirements.txt

# Run preprocessing and feature extraction
python src/preprocessing.py
python src/feature_extraction.py

# Train models
jupyter notebook src/model.ipynb
```

## Project Structure

```
├── dataset/         # Raw EEG data (.vhdr, .eeg, .vmrk)
├── data/            # Processed features (eeg.csv, hmm.csv)
├── results/         # Output figures and comparison tables
├── src/             # Source code
└── requirements.txt
```

## Results

### Approach 1: Pooled SMOTE

| Features | Best Model | F1-Score | Accuracy |
|----------|-----------|----------|----------|
| EEG | Random Forest | 0.7987 | 0.7912 |
| HMM | Random Forest | **0.8837** | **0.8822** |

**HMM Improvement: +10.64%**

### Approach 2: Per-Subject SMOTE

| Features | Best Model | F1-Score | Accuracy |
|----------|-----------|----------|----------|
| EEG | SVM | 0.8160 | 0.7980 |
| HMM | Random Forest | **0.8571** | **0.8586** |

**HMM Improvement: +5.05%**

## Key Findings

- HMM features consistently outperform traditional EEG features by 5-10%
- Random Forest achieved best performance across both approaches
- Successfully handled 7.1:1 class imbalance using SMOTE

## Links

- **Full Report**: [Overleaf](https://www.overleaf.com/9331923581dhbrqjjmbywn#3c46f1)
- **Repository**: [GitHub](https://github.com/24022404/eeg-oddball-classifier)

## License

MIT License
