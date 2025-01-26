import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import resample
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from google.colab import files

uploadedfile=files.upload()
fname=list(uploadedfile.keys())[0]
df=pd.read_csv(fname)

bootstrap_sample = resample(df, replace=True, n_samples=int(0.3 * len(df)), random_state=42)
print(bootstrap_sample['Class'].value_counts())
print(bootstrap_sample.head())
X_bootstrap = bootstrap_sample.drop(columns='Class')
y_bootstrap = bootstrap_sample['Class']
X_train_bootstrap, X_test_bootstrap, y_train_bootstrap, y_test_bootstrap = train_test_split(X_bootstrap, y_bootstrap, test_size=0.2, stratify=y_bootstrap, random_state=42)
xgb_model_bootstrap_imbalanced = XGBClassifier(random_state=42)
xgb_model_bootstrap_imbalanced.fit(X_train_bootstrap, y_train_bootstrap)
y_pred_bootstrap_imbalanced = xgb_model_bootstrap_imbalanced.predict(X_test_bootstrap)
print(accuracy_score(y_test_bootstrap, y_pred_bootstrap_imbalanced))
smote = SMOTE(random_state=42,k_neighbors=3)
X_resampled_bootstrap, y_resampled_bootstrap = smote.fit_resample(X_train_bootstrap, y_train_bootstrap)
xgb_model_bootstrap_balanced = XGBClassifier(random_state=42)
xgb_model_bootstrap_balanced.fit(X_resampled_bootstrap, y_resampled_bootstrap)
y_pred_bootstrap_balanced = xgb_model_bootstrap_balanced.predict(X_test_bootstrap)
print(accuracy_score(y_test_bootstrap, y_pred_bootstrap_balanced))

balanced_sample_file_name = 'balanced_sample1_creditcard_data.csv'
balanced_sample_df = pd.concat([pd.DataFrame(X_resampled_sample, columns=X.columns), pd.Series(y_resampled_sample, name='Class')], axis=1)
balanced_sample_df.to_csv(balanced_sample_file_name, index=False)
print(f"Balanced sampled dataset saved to {balanced_sample_file_name}")
files.download(balanced_sample_file_name)

SRS = df.sample(frac=0.5, random_state=42)
print(SRS.head())
X_sample = SRS.drop(columns='Class')
y_sample = SRS['Class']
X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=42)
xgb_model_sample_imbalanced = XGBClassifier(random_state=42)
xgb_model_sample_imbalanced.fit(X_train_sample, y_train_sample)
y_pred_sample_imbalanced = xgb_model_sample_imbalanced.predict(X_test_sample)
print(accuracy_score(y_test_sample, y_pred_sample_imbalanced))
smote = SMOTE(random_state=42)
X_resampled_sample, y_resampled_sample = smote.fit_resample(X_train_sample, y_train_sample)
xgb_model_sample_balanced = XGBClassifier(random_state=42)
xgb_model_sample_balanced.fit(X_resampled_sample, y_resampled_sample)
y_pred_sample_balanced = xgb_model_sample_balanced.predict(X_test_sample)
print(accuracy_score(y_test_sample, y_pred_sample_balanced))
balanced_sample_file_name = 'balanced_sample2_creditcard_data.csv'
balanced_sample_df = pd.concat([pd.DataFrame(X_resampled_sample, columns=X.columns), pd.Series(y_resampled_sample, name='Class')], axis=1)
balanced_sample_df.to_csv(balanced_sample_file_name, index=False)
print(f"Balanced sampled dataset saved to {balanced_sample_file_name}")
files.download(balanced_sample_file_name)

X_cluster = cluster_sample.drop(columns='Class')
y_cluster = cluster_sample['Class']

X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = train_test_split(
    X_cluster, y_cluster, test_size=0.2, stratify=y_cluster, random_state=42
)
smote = SMOTE(random_state=42)
X_resampled_cluster, y_resampled_cluster = smote.fit_resample(X_train_cluster, y_train_cluster)
xgb_model_cluster = XGBClassifier(random_state=42)
xgb_model_cluster.fit(X_resampled_cluster, y_resampled_cluster)
y_pred_cluster = xgb_model_cluster.predict(X_test_cluster)
print("Accuracy:", accuracy_score(y_test_cluster, y_pred_cluster))

balanced_sample_file_name = 'balanced_sample3_creditcard_data.csv'
balanced_sample_df = pd.concat([pd.DataFrame(X_resampled_sample, columns=X.columns), pd.Series(y_resampled_sample, name='Class')], axis=1)
balanced_sample_df.to_csv(balanced_sample_file_name, index=False)
print(f"Balanced sampled dataset saved to {balanced_sample_file_name}")
files.download(balanced_sample_file_name)

