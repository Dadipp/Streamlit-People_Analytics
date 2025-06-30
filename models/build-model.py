import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle

# Load data
df = pd.read_excel("HANDSON_32B/my_portfolio/People-Analytics/data/EmployeeSurvey.xlsx", engine='openpyxl')

# Drop kolom yang tidak dibutuhkan
df = df.drop(columns=["emp_id", "num_companies"])

# --- Encoding fitur kategorikal ---
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# --- Pisahkan fitur dan target ---
X = df.drop("job_satisfaction", axis=1)
y = df["job_satisfaction"]

# --- Oversampling SMOTE ---
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# --- Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# --- Evaluasi ---
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# --- Simpan Model, Fitur, dan Encoder ---
with open("HANDSON_32B/my_portfolio/People-Analytics/models/people_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("HANDSON_32B/my_portfolio/People-Analytics/models/feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

with open("HANDSON_32B/my_portfolio/People-Analytics/models/label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
