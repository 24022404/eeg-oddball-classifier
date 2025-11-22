import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

SUBJECT_COLUMN_NAME = 'subject_id'
LABEL_COLUMN_NAME = 'label'
EPOCH_ID_COLUMN_NAME = 'epoch_id' 

RESULTS_DIR = '../results'

def train_and_evaluate(df_features, subject_id, model_type):
    feature_type = 'features'

    df_subject = df_features[df_features[SUBJECT_COLUMN_NAME] == subject_id].copy()
    df_subject_filtered = df_subject[df_subject[LABEL_COLUMN_NAME].map(df_subject[LABEL_COLUMN_NAME].value_counts()) >= 2]
    
    X = df_subject_filtered.drop(columns=[LABEL_COLUMN_NAME, SUBJECT_COLUMN_NAME, EPOCH_ID_COLUMN_NAME])
    y = df_subject_filtered[LABEL_COLUMN_NAME]

    if len(np.unique(y)) < 2:
        message = "Skipped: Only one class present in the data."
        print(f"Warning for Subject {subject_id}: {message}")
        return 0.0, None, message

    if y.value_counts().min() < 2:
        message = f"Warning: Evaluated on training data (not enough samples for split)."
        print(f"Warning for Subject {subject_id}: Minority class has only {y.value_counts().min()} sample. Training on full data.")
        
        X_train, y_train = X, y 
        
        if model_type == 'svm':
            model = SVC(probability=True, random_state=42, C=1, gamma=0.1)
        else: # rf
            model = RandomForestClassifier(random_state=42, n_estimators=100) 
        
        pipeline = ImbPipeline([('scaler', StandardScaler()), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        
        return accuracy, pipeline, message
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n_samples_minority_train = y_train.value_counts().min()
    n_splits = min(5, n_samples_minority_train)

    if n_splits < 2:
        message = f"Skipped: Minority class in training set has only {n_samples_minority_train} sample(s), not enough for Cross-Validation."
        print(f"Warning for Subject {subject_id}: {message}")
        return 0.0, None, message

    if model_type == 'svm':
        model = SVC(probability=True, random_state=42)
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]}
    
    samples_in_val_fold = int(np.ceil(n_samples_minority_train / n_splits))
    samples_in_train_fold = n_samples_minority_train - samples_in_val_fold

    if samples_in_train_fold < 2:
        pipeline = ImbPipeline([('scaler', StandardScaler()), ('classifier', model)])
    else:
        k_safe = min(samples_in_train_fold - 1, 5) 
        smote = SMOTE(random_state=42, k_neighbors=k_safe)
        pipeline = ImbPipeline([('scaler', StandardScaler()), ('smote', smote), ('classifier', model)])
    
    pipeline_param_grid = {f'classifier__{key}': value for key, value in param_grid.items()}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=pipeline, param_grid=pipeline_param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0, error_score='raise')
    
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, "Success"

def process_feature_file(feature_name, filepath):
    print(f"\n======================= PROCESSING {feature_name} =======================")

    try:
        df_features = pd.read_csv(filepath)
        print(f"{feature_name} data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found. Skipping {feature_name}.")
        return None
    
    try:
        subjects = sorted(df_features[SUBJECT_COLUMN_NAME].unique())
    except KeyError as e:
        print(f"Column {e} missing in {feature_name}. Skipping.")
        return None

    comparison_results = []

    for subject in tqdm(subjects, desc=f"Training {feature_name}"):
        subject_results = {'Subject': subject}
        
        # Train SVM
        try:
            acc_svm, status_svm = train_and_evaluate(df_features, subject, 'svm')
        except Exception as e:
            acc_svm = 0.0
            status_svm = f"Critical Error: {e}"
        
        subject_results['SVM_Accuracy'] = acc_svm
        subject_results['SVM_Status'] = status_svm

        # Train RF
        try:
            acc_rf, status_rf = train_and_evaluate(df_features, subject, 'rf')
        except Exception as e:
            acc_rf = 0.0
            status_rf = f"Critical Error: {e}"
        
        subject_results['RF_Accuracy'] = acc_rf
        subject_results['RF_Status'] = status_rf

        comparison_results.append(subject_results)

    df_comparison = pd.DataFrame(comparison_results)
    
    for col in df_comparison.columns:
        if "Accuracy" in col:
            df_comparison[col] = df_comparison[col].map(lambda x: float(f"{x:.4f}"))

    output_path = os.path.join(RESULTS_DIR, f"final_accuracy_{feature_name}.csv")
    df_comparison.to_csv(output_path, index=False)
    print(df_comparison)
    print(f"\nSaved {feature_name} accuracy file → {output_path}")
    return output_path


def compute_final_summary(eeg_path, hmm_path):
    print("\n======================= FINAL SUMMARY TABLE =======================")

    df_eeg = pd.read_csv(eeg_path)
    df_hmm = pd.read_csv(hmm_path)

    summary = pd.DataFrame({
        'EEG': [
            df_eeg['SVM_Accuracy'].mean(),
            df_eeg['RF_Accuracy'].mean()
        ],
        'HMM': [
            df_hmm['SVM_Accuracy'].mean(),
            df_hmm['RF_Accuracy'].mean()
        ]
    }, index=['Support Vector Machine', 'Random Forest'])

    print(summary)

    summary_path = os.path.join(RESULTS_DIR, "final_summary_eeg_hmm.csv")
    summary.to_csv(summary_path, index=False)

    print(f"Summary saved → {summary_path}")
    return summary_path



def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    INPUT_FILES = {
        "EEG": "../data/eeg.csv",
        "HMM": "../data/hmm.csv"
    }

    output_eeg = process_feature_file("EEG", INPUT_FILES["EEG"])
    output_hmm = process_feature_file("HMM", INPUT_FILES["HMM"])

    if output_eeg and output_hmm:
        compute_final_summary(output_eeg, output_hmm)
    else:
        print("Not enough output files to compute summary.")

if __name__ == "__main__":
    main()
