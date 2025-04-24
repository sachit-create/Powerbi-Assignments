import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")  # Make sure this file is in the same directory

# Separate features and label
X = df.drop("label", axis=1)
y = df["label"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=le.classes_)
print("Classification Report:\n", report)

# Save model and label encoder
joblib.dump(model, "crop_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("sModel and encoder saved as 'crop_model.pkl' and 'label_encoder.pkl'")
