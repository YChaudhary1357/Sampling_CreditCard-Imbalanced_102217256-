from google.colab import files
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

uploadedfile=files.upload()
fname=list(uploadedfile.keys())[0]
df=pd.read_csv(fname)

X = df.drop(columns='Class')
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

xgb_model_imbalanced = XGBClassifier(random_state=42)
xgb_model_imbalanced.fit(X_train, y_train)
y_pred_imbalanced = xgb_model_imbalanced.predict(X_test)
print(accuracy_score(y_test, y_pred_imbalanced))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

xgb_model_balanced = XGBClassifier(random_state=42)
xgb_model_balanced.fit(X_resampled, y_resampled)
y_pred_balanced = xgb_model_balanced.predict(X_test)
print(accuracy_score(y_test, y_pred_balanced))
df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='Class')], axis=1)

df_resampled
df_resampled.to_csv('df_resampled.csv', index=False) 
files.download('df_resampled.csv')
